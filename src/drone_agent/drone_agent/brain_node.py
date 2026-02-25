#!/usr/bin/env python3
"""Brain Node — the main orchestrator for the AI-controlled drone.

This node:
1. Subscribes to PX4 telemetry (odometry, GPS, status, battery)
2. Subscribes to YOLO detections
3. Accepts user natural language commands
4. Calls Gemini LLM for decision-making (throttled, not every frame)
5. Translates LLM commands to PX4 setpoints
6. Publishes offboard control at 10Hz continuously
"""

import json
import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from std_msgs.msg import String
from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleOdometry,
    SensorGps,
    VehicleStatus,
    BatteryStatus,
)

from drone_agent.llm_client import LLMClient
from drone_agent.command_translator import CommandTranslator


class BrainNode(Node):
    def __init__(self):
        super().__init__('brain_node')

        # Parameters
        self.declare_parameter('llm_interval_sec', 7.0)
        self.declare_parameter('offboard_rate_hz', 10.0)
        self.declare_parameter('ollama_url', 'http://localhost:11434')
        self.declare_parameter('ollama_model', 'qwen2.5:32b')

        self.llm_interval = self.get_parameter('llm_interval_sec').value
        offboard_rate = self.get_parameter('offboard_rate_hz').value
        ollama_url = self.get_parameter('ollama_url').value
        ollama_model = self.get_parameter('ollama_model').value

        # Initialize components
        self.llm = LLMClient(model=ollama_model, ollama_url=ollama_url)
        self.translator = CommandTranslator()

        # State
        self.current_command = ''
        self.latest_detections = '[]'
        self.odometry = None
        self.gps = None
        self.vehicle_status = None
        self.battery = None
        self.last_llm_call = 0.0
        self.last_detection_classes = set()
        self.armed = False
        self.offboard_active = False
        # Object to search for during orbit (set from LLM orbit target_class field)
        self.search_target: str | None = None

        # QoS for PX4 topics (best-effort, volatile — matches PX4 uXRCE-DDS)
        px4_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # --- Subscribers ---

        # PX4 telemetry
        self.create_subscription(
            VehicleOdometry, '/fmu/out/vehicle_odometry',
            self._odom_cb, px4_qos,
        )
        self.create_subscription(
            SensorGps, '/fmu/out/vehicle_gps_position',
            self._gps_cb, px4_qos,
        )
        self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status',
            self._status_cb, px4_qos,
        )
        self.create_subscription(
            BatteryStatus, '/fmu/out/battery_status',
            self._battery_cb, px4_qos,
        )

        # YOLO detections
        self.create_subscription(
            String, '/yolo/detections',
            self._yolo_cb, 10,
        )

        # User commands
        self.create_subscription(
            String, '/user_command',
            self._user_cmd_cb, 10,
        )

        # --- Publishers ---
        self.offboard_pub = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', 10,
        )
        self.setpoint_pub = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', 10,
        )
        self.command_pub = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', 10,
        )

        # --- Timers ---

        # 10Hz offboard control loop (MUST be continuous for PX4 OFFBOARD mode)
        period = 1.0 / offboard_rate
        self.create_timer(period, self._offboard_loop)

        # LLM decision timer
        self.create_timer(self.llm_interval, self._llm_decision_tick)

        self.get_logger().info('Brain node started. Waiting for user command on /user_command ...')

    # ─── Subscriber callbacks ──────────────────────────────────────────

    def _odom_cb(self, msg: VehicleOdometry):
        self.odometry = msg
        pos = msg.position
        self.translator.update_position(pos[0], pos[1], pos[2])
        # Fallback: if vehicle_status is not arriving and the drone is more
        # than 2 m above home, it must be airborne and therefore armed.
        if self.vehicle_status is None and abs(float(pos[2])) > 2.0:
            self.armed = True

    def _gps_cb(self, msg: SensorGps):
        self.gps = msg
        # Set home position on first GPS fix
        if not self.translator.home_set and msg.fix_type >= 3:
            lat = msg.latitude_deg
            lon = msg.longitude_deg
            alt = msg.altitude_msl_m
            self.translator.set_home(lat, lon, alt)
            self.get_logger().info(
                f'Home set: lat={lat:.6f}, lon={lon:.6f}, alt={alt:.1f}m'
            )

    def _status_cb(self, msg: VehicleStatus):
        self.vehicle_status = msg
        self.armed = msg.arming_state == 2  # ARMED

    def _battery_cb(self, msg: BatteryStatus):
        self.battery = msg

    def _yolo_cb(self, msg: String):
        self.latest_detections = msg.data

        try:
            detections = json.loads(msg.data)
            new_classes = {d['class'] for d in detections}
        except (json.JSONDecodeError, KeyError):
            new_classes = set()
            detections = []

        # Target object spotted while in search orbit — stop and descend toward it
        if self.translator.orbiting and self.search_target and self.search_target in new_classes:
            self.last_detection_classes = new_classes
            self.get_logger().info(
                f'"{self.search_target}" detected during orbit — transitioning to hover'
            )
            self._target_found(detections, self.search_target)
            return

        # Otherwise re-trigger LLM on detection class changes
        if new_classes != self.last_detection_classes and self.current_command:
            self.last_detection_classes = new_classes
            if new_classes:
                self.get_logger().info(f'Detection change: {new_classes} — triggering LLM')
                self._call_llm()

    def _target_found(self, detections: list, target_class: str):
        """Stop orbiting and descend toward a detected target object.

        Uses the YOLO bbox_center x-coordinate and the drone's current yaw
        to estimate the bearing toward the target, then flies a short distance
        in that direction and descends to 40% of current altitude (min 5 m).

        Args:
            detections: Parsed list of YOLO detection dicts.
            target_class: YOLO class name that was found (e.g. "person", "car").
        """
        self.translator.orbiting = False
        self.translator.square_active = False
        self.search_target = None  # search complete — clear the target

        # Current position from odometry or fall back to last known target
        if self.odometry:
            cur_x = float(self.odometry.position[0])
            cur_y = float(self.odometry.position[1])
            current_alt = abs(float(self.odometry.position[2]))
        else:
            cur_x = self.translator.target_x
            cur_y = self.translator.target_y
            current_alt = abs(self.translator.target_z)

        # Descend to 40% of current altitude, never below 5 m
        hover_alt_z = -max(5.0, current_alt * 0.4)
        target_x = cur_x
        target_y = cur_y
        target_yaw = float('nan')

        target_obj = next((d for d in detections if d.get('class') == target_class), None)
        if target_obj and self.odometry:
            # bbox_center[0]: 0 = left edge, 0.5 = centre, 1 = right edge
            bbox_cx = max(0.1, min(0.9, target_obj['bbox_center'][0]))

            # Extract yaw from VehicleOdometry quaternion [w, x, y, z]
            q = self.odometry.q
            yaw = math.atan2(
                2.0 * (float(q[0]) * float(q[3]) + float(q[1]) * float(q[2])),
                1.0 - 2.0 * (float(q[2]) ** 2 + float(q[3]) ** 2),
            )

            # Horizontal FOV ~90 deg; offset bearing left/right by pixel position
            bearing_offset = (bbox_cx - 0.5) * math.radians(90)
            target_yaw = yaw + bearing_offset

            # Fly a short distance toward the estimated target position
            move_dist = min(current_alt * 0.3, 8.0)
            target_x = cur_x + move_dist * math.cos(target_yaw)
            target_y = cur_y + move_dist * math.sin(target_yaw)

        self.translator.target_x = target_x
        self.translator.target_y = target_y
        self.translator.target_z = hover_alt_z
        self.translator.target_yaw = target_yaw

        self.get_logger().info(
            f'TARGET FOUND: "{target_class}"! Orbit stopped. '
            f'Descending to {abs(hover_alt_z):.1f} m AGL, '
            f'moving to NED=({target_x:.1f}, {target_y:.1f})'
        )

    def _user_cmd_cb(self, msg: String):
        command = msg.data.strip()

        # Special commands handled locally — no LLM call needed
        if command.lower() in ('reset', 'reset memory', 'clear memory', 'new mission'):
            self.llm.reset_memory()
            self.search_target = None
            self.current_command = ''
            self.get_logger().info('Memory and search target cleared. Ready for new mission.')
            return

        self.current_command = command
        self.get_logger().info(f'New user command: "{self.current_command}"')
        # Immediately call LLM on new user command
        self._call_llm()

    # ─── Core loops ────────────────────────────────────────────────────

    def _offboard_loop(self):
        """10Hz loop: publish offboard heartbeat + trajectory setpoint."""
        # Use the ROS2 clock (synced to PX4 via uXRCE-DDS) for timestamps.
        # time.time() gives Unix epoch μs (~1.7×10^15) but PX4 expects μs
        # since boot (~10^6–10^9), so the messages would be silently dropped.
        ts = int(self.get_clock().now().nanoseconds / 1000)

        offboard_msg = self.translator.get_offboard_control_mode()
        offboard_msg.timestamp = ts
        self.offboard_pub.publish(offboard_msg)

        setpoint_msg = self.translator.get_setpoint_tick()
        setpoint_msg.timestamp = ts
        self.setpoint_pub.publish(setpoint_msg)

    def _llm_decision_tick(self):
        """Periodic LLM re-evaluation — disabled, LLM called on demand only.

        The LLM is called on new user commands or YOLO detection changes.
        """
        # if not self.current_command:
        #     return
        # self._call_llm()
        pass

    def _call_llm(self):
        """Format state and call the LLM for a new command."""
        now = time.time()
        # Rate limit: don't call more than once per 3 seconds
        if now - self.last_llm_call < 3.0:
            return
        self.last_llm_call = now

        drone_state = self._format_drone_state()

        self.get_logger().info('=' * 60)
        self.get_logger().info('>>> LLM INPUT <<<')
        self.get_logger().info(f'User command: {self.current_command}')
        self.get_logger().info(f'Drone state:\n{drone_state}')
        self.get_logger().info(f'YOLO detections: {self.latest_detections}')
        self.get_logger().info('Calling Ollama LLM...')

        try:
            cmd = self.llm.ask(drone_state, self.current_command, self.latest_detections)
            self.get_logger().info('>>> LLM OUTPUT <<<')

            thought = cmd.pop('thought', None)
            if thought:
                self.get_logger().info(f'LLM reasoning: {thought}')
            self.get_logger().info(f'LLM action: {cmd}')

            action = cmd.get('action')

            # Orbit with a search goal: extract the YOLO class to watch for
            if action == 'orbit':
                self.search_target = cmd.get('target_class', None)
                if self.search_target:
                    self.get_logger().info(
                        f'Search orbit: will stop when YOLO detects "{self.search_target}"'
                    )
                else:
                    self.get_logger().info('Pure loiter orbit (no search target)')

            # Hold while orbiting/square: stop the pattern and freeze at current position
            elif action == 'hold' and (
                self.translator.orbiting or self.translator.square_active
            ):
                self.translator.orbiting = False
                self.translator.square_active = False
                self.search_target = None
                if self.odometry:
                    pos = self.odometry.position
                    self.translator.target_x = float(pos[0])
                    self.translator.target_y = float(pos[1])
                    self.translator.target_z = float(pos[2])
                self.get_logger().info('Hold: pattern stopped, holding current position')

            extra_msgs = self.translator.process_command(cmd)
            self.get_logger().info(f'>>> COMMAND TRANSLATOR <<<')
            self.get_logger().info(
                f'Action: {cmd.get("action")} | '
                f'Target NED: x={self.translator.target_x:.1f}, '
                f'y={self.translator.target_y:.1f}, '
                f'z={self.translator.target_z:.1f} | '
                f'Orbiting: {self.translator.orbiting} | '
                f'Square: {self.translator.square_active} | '
                f'Extra PX4 msgs: {len(extra_msgs)}'
            )
            self.get_logger().info('=' * 60)

            ts = int(self.get_clock().now().nanoseconds / 1000)
            for _, msg in extra_msgs:
                msg.timestamp = ts
                self.command_pub.publish(msg)

        except Exception as e:
            self.get_logger().error(f'LLM call failed: {e}')

    def _format_drone_state(self) -> str:
        """Format current telemetry into a human-readable string for the LLM."""
        lines = []

        if self.gps:
            lines.append(
                f'GPS: lat={self.gps.latitude_deg:.6f}, '
                f'lon={self.gps.longitude_deg:.6f}, '
                f'alt={self.gps.altitude_msl_m:.1f}m MSL, '
                f'fix={self.gps.fix_type}'
            )
        else:
            lines.append('GPS: No fix')

        if self.odometry:
            pos = self.odometry.position
            vel = self.odometry.velocity
            lines.append(
                f'Local position (NED): x={pos[0]:.1f}m, y={pos[1]:.1f}m, z={pos[2]:.1f}m'
            )
            lines.append(
                f'Velocity (NED): vx={vel[0]:.1f}, vy={vel[1]:.1f}, vz={vel[2]:.1f} m/s'
            )
        else:
            lines.append('Odometry: Not available')

        if self.battery:
            lines.append(
                f'Battery: {self.battery.remaining * 100:.0f}% '
                f'({self.battery.voltage_v:.1f}V)'
            )

        if self.vehicle_status:
            lines.append(f'Armed: {self.armed} (from vehicle_status arming_state={self.vehicle_status.arming_state})')
            lines.append(f'Nav state: {self.vehicle_status.nav_state}')
        else:
            lines.append(f'Armed: {self.armed} (vehicle_status not received — inferred from altitude)')

        if self.translator.home_set:
            lines.append(
                f'Home GPS (NED origin): lat={self.translator.home_lat:.6f}, '
                f'lon={self.translator.home_lon:.6f}, '
                f'alt={self.translator.home_alt:.1f}m MSL'
            )
        else:
            lines.append('Home GPS: not set yet')

        lines.append(f'Orbiting: {self.translator.orbiting}')
        if self.search_target:
            lines.append(f'Search mode: actively scanning for "{self.search_target}"')
        lines.append(
            f'Current target (NED): '
            f'x={self.translator.target_x:.1f}, '
            f'y={self.translator.target_y:.1f}, '
            f'z={self.translator.target_z:.1f}'
        )

        return '\n'.join(lines)


def main(args=None):
    rclpy.init(args=args)
    node = BrainNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
