#!/usr/bin/env python3
"""Translates LLM JSON commands into PX4 messages.

Converts high-level drone commands (takeoff, goto, orbit, land, etc.)
into PX4 OffboardControlMode, TrajectorySetpoint, and VehicleCommand messages.
"""

import math
import time

from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
)

from drone_agent.shape_generator import generate_shape_waypoints


# PX4 MAV_CMD constants
MAV_CMD_NAV_TAKEOFF = 22
MAV_CMD_NAV_LAND = 21
MAV_CMD_NAV_RETURN_TO_LAUNCH = 20
MAV_CMD_COMPONENT_ARM_DISARM = 400
MAV_CMD_DO_SET_MODE = 176
MAV_CMD_DO_SET_ROI_LOCATION = 195
MAV_CMD_DO_CHANGE_SPEED = 178

# PX4 navigation modes
PX4_CUSTOM_MAIN_MODE_OFFBOARD = 6.0


class CommandTranslator:
    """Translates LLM JSON commands to PX4 ROS2 messages."""

    def __init__(
        self,
        home_lat: float = 0.0,
        home_lon: float = 0.0,
        home_alt: float = 0.0,
        max_speed_m_s: float = 0.0,
    ):
        self.home_lat = home_lat
        self.home_lon = home_lon
        self.home_alt = home_alt
        self.home_set = False
        self.max_speed_m_s = max(0.0, float(max_speed_m_s))
        self.translation_speed_cap_override_m_s = 0.0

        # Current target setpoint (updated by commands)
        self.target_x = 0.0  # North (meters, NED)
        self.target_y = 0.0  # East (meters, NED)
        self.target_z = -10.0  # Down (meters, NED — negative = up)
        self.target_yaw = float('nan')  # NaN = don't care
        self.target_speed = 5.0

        # Orbit state
        self.orbiting = False
        self.orbit_center_x = 0.0
        self.orbit_center_y = 0.0
        self.orbit_radius = 20.0
        self.orbit_speed = 5.0
        self.orbit_alt_z = -40.0
        self.orbit_angle = 0.0  # Current angle in radians

        # Square pattern state
        self.square_active = False
        self.square_waypoints = []  # List of (x, y) NED waypoints
        self.square_wp_index = 0
        self.square_alt_z = -10.0
        self.square_speed = 2.0
        self.square_threshold = 1.0  # meters — how close before moving to next wp

        # Generic waypoint path state
        self.path_active = False
        self.path_waypoints = []  # List of (x, y, z) NED waypoints
        self.path_wp_index = 0
        self.path_speed = 2.0
        self.path_threshold = 1.0  # meters — how close before advancing
        self.path_loop = False
        self.path_sequence_id = 0
        self.path_completed_sequence_id = 0

        # Latest measured vehicle position from odometry (used for speed limiting)
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_z = 0.0
        self.have_position = False
        self._last_tick_time = None
        self._last_setpoint_x = self.target_x
        self._last_setpoint_y = self.target_y
        self._last_setpoint_z = self.target_z
        self._have_last_setpoint = False

    def update_position(self, x: float, y: float, z: float):
        """Update the drone's current local position (NED) for waypoint tracking."""
        self.current_x = float(x)
        self.current_y = float(y)
        self.current_z = float(z)
        self.have_position = True

        if self.square_active and self.square_waypoints:
            wp = self.square_waypoints[self.square_wp_index]
            dist = math.sqrt((x - wp[0]) ** 2 + (y - wp[1]) ** 2)
            if dist < self.square_threshold:
                self.square_wp_index = (self.square_wp_index + 1) % len(self.square_waypoints)
                wp = self.square_waypoints[self.square_wp_index]
                self.target_x = wp[0]
                self.target_y = wp[1]

        if self.path_active and self.path_waypoints:
            wp = self.path_waypoints[self.path_wp_index]
            dist = math.sqrt(
                (x - wp[0]) ** 2 +
                (y - wp[1]) ** 2 +
                (z - wp[2]) ** 2
            )
            if dist < self.path_threshold:
                if self.path_wp_index + 1 < len(self.path_waypoints):
                    self.path_wp_index += 1
                    wp = self.path_waypoints[self.path_wp_index]
                    self.target_x = wp[0]
                    self.target_y = wp[1]
                    self.target_z = wp[2]
                elif self.path_loop and len(self.path_waypoints) > 1:
                    self.path_wp_index = 0
                    wp = self.path_waypoints[self.path_wp_index]
                    self.target_x = wp[0]
                    self.target_y = wp[1]
                    self.target_z = wp[2]
                else:
                    self.path_active = False
                    self.path_completed_sequence_id = self.path_sequence_id
                    self.target_x = wp[0]
                    self.target_y = wp[1]
                    self.target_z = wp[2]

    def set_home(self, lat: float, lon: float, alt: float):
        """Set the home/reference position for GPS-to-local conversions."""
        self.home_lat = lat
        self.home_lon = lon
        self.home_alt = alt
        self.home_set = True

    def gps_to_local(self, lat: float, lon: float, alt: float):
        """Convert GPS coordinates to local NED frame relative to home.

        Returns:
            Tuple of (north_m, east_m, down_m) in NED frame.
        """
        if not self.home_set:
            return 0.0, 0.0, -(alt - self.home_alt)

        dlat = lat - self.home_lat
        dlon = lon - self.home_lon

        north_m = dlat * 111_139.0
        east_m = dlon * 111_139.0 * math.cos(math.radians(self.home_lat))
        down_m = -(alt - self.home_alt)  # NED: negative altitude = up

        return north_m, east_m, down_m

    def process_command(self, cmd: dict) -> list:
        """Translate an LLM command dict into a list of PX4 messages to publish.

        Args:
            cmd: Dict with 'action' key and associated parameters.

        Returns:
            List of (topic_name, message) tuples to publish.
        """
        action = cmd.get('action', 'hold')
        messages = []

        # ── Arm + OFFBOARD mode ──────────────────────────────────────────
        if action in ('arm_offboard', 'takeoff'):
            # takeoff kept as alias: arm, switch mode, set target altitude
            self.orbiting = False
            self.square_active = False
            self.path_active = False
            if action == 'takeoff':
                self.target_z = -abs(cmd.get('alt', 10.0))
            messages.extend(self._arm_and_offboard())

        # ── Direct NED position (primary navigation primitive) ───────────
        elif action == 'position_ned':
            self.target_x = float(cmd.get('x', self.target_x))
            self.target_y = float(cmd.get('y', self.target_y))
            self.target_z = float(cmd.get('z', self.target_z))
            yaw = cmd.get('yaw_rad')
            self.target_yaw = float(yaw) if yaw is not None else float('nan')
            self.orbiting = False
            self.square_active = False
            self.path_active = False

        # ── GPS-based goto (legacy alias) ────────────────────────────────
        elif action == 'goto':
            lat = cmd.get('lat', self.home_lat)
            lon = cmd.get('lon', self.home_lon)
            alt = cmd.get('alt', abs(self.target_z))
            n, e, d = self.gps_to_local(lat, lon, alt)
            self.target_x = n
            self.target_y = e
            self.target_z = d
            self.orbiting = False
            self.square_active = False
            self.path_active = False

        # ── Circular orbit ───────────────────────────────────────────────
        elif action == 'orbit':
            self.orbit_radius = float(cmd.get('radius', 20.0))
            self.orbit_speed = self._cap_speed(cmd.get('speed', 5.0))
            self.orbit_angle = 0.0
            self.square_active = False
            self.path_active = False

            if 'cx' in cmd:
                # New style: LLM computed NED centre directly
                self.orbit_center_x = float(cmd['cx'])
                self.orbit_center_y = float(cmd['cy'])
                self.orbit_alt_z = float(cmd.get('alt_z', self.target_z))
            else:
                # Legacy style: GPS lat/lon centre
                lat = cmd.get('lat', self.home_lat)
                lon = cmd.get('lon', self.home_lon)
                alt = cmd.get('alt', abs(self.target_z))
                n, e, _ = self.gps_to_local(lat, lon, alt)
                self.orbit_center_x = n
                self.orbit_center_y = e
                self.orbit_alt_z = -abs(alt)

            self.orbiting = True

        # ── Square / rectangular survey ──────────────────────────────────
        elif action in ('square_survey', 'square'):
            side = float(cmd.get('side', 10.0))
            speed = self._cap_speed(cmd.get('speed', 2.0))
            # alt_z (new style, already negative) or alt (legacy, positive)
            if 'alt_z' in cmd:
                alt_z = float(cmd['alt_z'])
            else:
                alt_z = -abs(float(cmd.get('alt', abs(self.target_z))))
            cx = self.target_x
            cy = self.target_y
            half = side / 2.0
            self.square_waypoints = [
                (cx + half, cy + half),
                (cx + half, cy - half),
                (cx - half, cy - half),
                (cx - half, cy + half),
            ]
            self.square_wp_index = 0
            self.square_alt_z = alt_z
            self.square_speed = speed
            self.square_active = True
            self.orbiting = False
            self.path_active = False
            self.target_x = self.square_waypoints[0][0]
            self.target_y = self.square_waypoints[0][1]
            self.target_z = self.square_alt_z

        # ── Generic waypoint path ───────────────────────────────────────
        elif action == 'path':
            points_raw = cmd.get('points')
            if not isinstance(points_raw, list) or len(points_raw) < 2:
                raise ValueError('path action requires at least two waypoints')

            default_z = self.target_z
            waypoints = [self._normalize_path_point(point, default_z) for point in points_raw]
            closed = bool(cmd.get('closed', False))
            if closed and self._path_points_differ(waypoints[0], waypoints[-1]):
                waypoints.append(waypoints[0])

            self._activate_path(
                waypoints,
                speed=cmd.get('speed', self.target_speed),
                threshold_m=cmd.get('threshold_m'),
                loop=cmd.get('loop', False),
            )

        # ── Deterministic named shape path ──────────────────────────────
        elif action == 'shape_path':
            if self.have_position:
                origin_x = self.current_x
                origin_y = self.current_y
                origin_z = self.current_z
            else:
                origin_x = self.target_x
                origin_y = self.target_y
                origin_z = self.target_z

            waypoints, _ = generate_shape_waypoints(
                cmd,
                origin_x=origin_x,
                origin_y=origin_y,
                origin_z=float(cmd.get('alt_z', origin_z)),
            )
            self._activate_path(
                waypoints,
                speed=cmd.get('speed', cmd.get('speed_m_s', self.target_speed)),
                threshold_m=cmd.get('threshold_m'),
                loop=cmd.get('loop', False),
            )

        # ── Camera ROI ───────────────────────────────────────────────────
        elif action in ('look_at_gps', 'look_at'):
            lat = cmd.get('lat', self.home_lat)
            lon = cmd.get('lon', self.home_lon)
            alt = cmd.get('alt', 0.0)
            msg = self._make_vehicle_command(
                MAV_CMD_DO_SET_ROI_LOCATION,
                param5=lat,
                param6=lon,
                param7=alt,
            )
            messages.append(('/fmu/in/vehicle_command', msg))

        # ── Speed / heading ──────────────────────────────────────────────
        elif action == 'set_speed':
            self.target_speed = self._cap_speed(cmd.get('speed', 5.0))
            self.orbit_speed = self.target_speed

        elif action == 'set_heading':
            # Accept both "heading_deg" (new) and "heading" (legacy)
            heading_deg = cmd.get('heading_deg', cmd.get('heading', 0.0))
            self.target_yaw = math.radians(float(heading_deg))

        # ── Land / RTL ───────────────────────────────────────────────────
        elif action == 'land':
            self.orbiting = False
            self.square_active = False
            self.path_active = False
            msg = self._make_vehicle_command(MAV_CMD_NAV_LAND)
            messages.append(('/fmu/in/vehicle_command', msg))

        elif action == 'rtl':
            self.orbiting = False
            self.square_active = False
            self.path_active = False
            msg = self._make_vehicle_command(MAV_CMD_NAV_RETURN_TO_LAUNCH)
            messages.append(('/fmu/in/vehicle_command', msg))

        # ── Hold: stop all patterns, brain_node sets position ────────────
        elif action == 'hold':
            self.orbiting = False
            self.square_active = False
            self.path_active = False

        return messages

    def get_setpoint_tick(self) -> TrajectorySetpoint:
        """Generate the current TrajectorySetpoint for the offboard publish loop.

        If orbiting, advances the orbit angle. Otherwise returns the
        current static target position.
        """
        msg = TrajectorySetpoint()
        msg.timestamp = int(time.time() * 1e6)
        now = time.monotonic()
        if self._last_tick_time is None:
            dt = 0.0
        else:
            dt = max(0.0, now - self._last_tick_time)
        self._last_tick_time = now

        target_x = self.target_x
        target_y = self.target_y
        target_z = self.target_z
        target_yaw = self.target_yaw

        if self.square_active:
            wp = self.square_waypoints[self.square_wp_index]
            target_x = wp[0]
            target_y = wp[1]
            target_z = self.square_alt_z
            target_yaw = float('nan')

        elif self.path_active:
            wp = self.path_waypoints[self.path_wp_index]
            target_x = wp[0]
            target_y = wp[1]
            target_z = wp[2]
            if self.have_position:
                dx = target_x - self.current_x
                dy = target_y - self.current_y
                if abs(dx) > 1e-6 or abs(dy) > 1e-6:
                    target_yaw = math.atan2(dy, dx)
            elif self._have_last_setpoint:
                dx = target_x - self._last_setpoint_x
                dy = target_y - self._last_setpoint_y
                if abs(dx) > 1e-6 or abs(dy) > 1e-6:
                    target_yaw = math.atan2(dy, dx)

        elif self.orbiting:
            angular_vel = self.orbit_speed / max(self.orbit_radius, 1.0)
            self.orbit_angle += angular_vel * dt

            target_x = self.orbit_center_x + self.orbit_radius * math.cos(self.orbit_angle)
            target_y = self.orbit_center_y + self.orbit_radius * math.sin(self.orbit_angle)
            target_z = self.orbit_alt_z

            # Point yaw toward orbit center
            dx = self.orbit_center_x - target_x
            dy = self.orbit_center_y - target_y
            target_yaw = math.atan2(dy, dx)

        target_x, target_y, target_z = self._limit_translation_speed(
            target_x, target_y, target_z, dt
        )
        msg.position[0] = target_x
        msg.position[1] = target_y
        msg.position[2] = target_z
        msg.yaw = target_yaw
        self._last_setpoint_x = target_x
        self._last_setpoint_y = target_y
        self._last_setpoint_z = target_z
        self._have_last_setpoint = True

        return msg

    def _activate_path(
        self,
        waypoints: list[tuple[float, float, float]],
        *,
        speed: float,
        threshold_m: float | None,
        loop: bool,
    ):
        """Activate the shared waypoint-path follower state."""
        self.path_speed = self._cap_speed(speed)
        self.path_loop = bool(loop)
        self.path_waypoints = list(waypoints)
        self.path_threshold = self._compute_path_threshold(
            self.path_waypoints,
            threshold_m,
        )
        self.path_wp_index = 0
        self.path_active = True
        self.path_sequence_id += 1
        self.orbiting = False
        self.square_active = False
        self.target_x = self.path_waypoints[0][0]
        self.target_y = self.path_waypoints[0][1]
        self.target_z = self.path_waypoints[0][2]
        self.target_yaw = float('nan')

    def _cap_speed(self, speed: float) -> float:
        """Clamp requested mission speed to the configured hard cap."""
        speed = max(0.0, float(speed))
        if self.max_speed_m_s > 0.0:
            return min(speed, self.max_speed_m_s)
        return speed

    def _compute_path_threshold(
        self,
        waypoints: list[tuple[float, float, float]],
        requested_threshold_m: float | None,
    ) -> float:
        """Choose a waypoint-advance threshold that matches the path density.

        Dense generated shapes like spirals can have sub-meter waypoint spacing.
        A fixed 1.0 m threshold causes the controller to skip many early points
        before the vehicle has moved. When the caller does not explicitly set a
        threshold, derive one from the median segment length instead.
        """
        if requested_threshold_m is not None:
            return max(0.15, float(requested_threshold_m))

        if len(waypoints) < 2:
            return 0.3

        segment_lengths = []
        for idx in range(1, len(waypoints)):
            prev = waypoints[idx - 1]
            curr = waypoints[idx]
            segment_lengths.append(
                math.sqrt(
                    (curr[0] - prev[0]) ** 2 +
                    (curr[1] - prev[1]) ** 2 +
                    (curr[2] - prev[2]) ** 2
                )
            )

        segment_lengths = [length for length in segment_lengths if length > 1e-6]
        if not segment_lengths:
            return 0.3

        segment_lengths.sort()
        mid = len(segment_lengths) // 2
        if len(segment_lengths) % 2 == 1:
            median_length = segment_lengths[mid]
        else:
            median_length = 0.5 * (segment_lengths[mid - 1] + segment_lengths[mid])

        return max(0.15, min(0.75, 0.4 * median_length))

    def _limit_translation_speed(
        self,
        target_x: float,
        target_y: float,
        target_z: float,
        dt: float,
    ) -> tuple[float, float, float]:
        """Limit horizontal setpoint motion so the setpoint itself never jumps faster than the cap.

        Altitude is left unconstrained here so takeoff and climb commands can
        present a meaningful vertical target to PX4 immediately.
        """
        speed_cap = self.translation_speed_cap_override_m_s
        if speed_cap <= 0.0 and self.path_active:
            speed_cap = self.path_speed
        if speed_cap <= 0.0 and self.square_active:
            speed_cap = self.square_speed
        if speed_cap <= 0.0:
            speed_cap = self.max_speed_m_s

        if speed_cap <= 0.0 or dt <= 0.0:
            return target_x, target_y, target_z

        if self._have_last_setpoint:
            base_x = self._last_setpoint_x
            base_y = self._last_setpoint_y
        elif self.have_position:
            base_x = self.current_x
            base_y = self.current_y
        else:
            return target_x, target_y, target_z

        max_step = speed_cap * dt
        dx = target_x - base_x
        dy = target_y - base_y
        dist_xy = math.hypot(dx, dy)
        if dist_xy <= max_step or dist_xy == 0.0:
            return target_x, target_y, target_z

        scale = max_step / dist_xy
        return (
            base_x + dx * scale,
            base_y + dy * scale,
            target_z,
        )

    def get_offboard_control_mode(self) -> OffboardControlMode:
        """Generate an OffboardControlMode message (position control)."""
        msg = OffboardControlMode()
        msg.timestamp = int(time.time() * 1e6)
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        return msg

    def _normalize_path_point(self, point, default_z: float) -> tuple[float, float, float]:
        """Convert one raw LLM waypoint into a safe absolute NED tuple."""
        if isinstance(point, dict):
            x = point.get('x')
            y = point.get('y')
            z = point.get('z', default_z)
        elif isinstance(point, (list, tuple)):
            if len(point) == 2:
                x, y = point
                z = default_z
            elif len(point) >= 3:
                x, y, z = point[:3]
            else:
                raise ValueError('path waypoint lists must contain at least x and y')
        else:
            raise ValueError('path waypoints must be objects or [x, y, z] lists')

        x = float(x)
        y = float(y)
        z = -abs(float(z))
        z = max(-120.0, min(-1.0, z))
        return (x, y, z)

    def _path_points_differ(
        self,
        first: tuple[float, float, float],
        second: tuple[float, float, float],
    ) -> bool:
        """Return True when two path points are meaningfully different."""
        return (
            abs(first[0] - second[0]) > 1e-3 or
            abs(first[1] - second[1]) > 1e-3 or
            abs(first[2] - second[2]) > 1e-3
        )

    def _arm_and_offboard(self) -> list:
        """Generate messages to arm the drone and switch to OFFBOARD mode."""
        messages = []

        # Arm
        arm_cmd = self._make_vehicle_command(
            MAV_CMD_COMPONENT_ARM_DISARM,
            param1=1.0,  # 1 = arm
        )
        messages.append(('/fmu/in/vehicle_command', arm_cmd))

        # Switch to OFFBOARD mode
        mode_cmd = self._make_vehicle_command(
            MAV_CMD_DO_SET_MODE,
            param1=1.0,  # base mode
            param2=PX4_CUSTOM_MAIN_MODE_OFFBOARD,
        )
        messages.append(('/fmu/in/vehicle_command', mode_cmd))

        return messages

    def _make_vehicle_command(
        self,
        command: int,
        param1: float = 0.0,
        param2: float = 0.0,
        param3: float = 0.0,
        param4: float = 0.0,
        param5: float = 0.0,
        param6: float = 0.0,
        param7: float = 0.0,
    ) -> VehicleCommand:
        """Create a VehicleCommand message."""
        msg = VehicleCommand()
        msg.timestamp = int(time.time() * 1e6)
        msg.command = command
        msg.param1 = param1
        msg.param2 = param2
        msg.param3 = param3
        msg.param4 = param4
        msg.param5 = param5
        msg.param6 = param6
        msg.param7 = param7
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        return msg
