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

    def __init__(self, home_lat: float = 0.0, home_lon: float = 0.0, home_alt: float = 0.0):
        self.home_lat = home_lat
        self.home_lon = home_lon
        self.home_alt = home_alt
        self.home_set = False

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

    def update_position(self, x: float, y: float, z: float):
        """Update the drone's current local position (NED) for waypoint tracking."""
        if self.square_active and self.square_waypoints:
            wp = self.square_waypoints[self.square_wp_index]
            dist = math.sqrt((x - wp[0]) ** 2 + (y - wp[1]) ** 2)
            if dist < self.square_threshold:
                self.square_wp_index = (self.square_wp_index + 1) % len(self.square_waypoints)
                wp = self.square_waypoints[self.square_wp_index]
                self.target_x = wp[0]
                self.target_y = wp[1]

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

        # ── Circular orbit ───────────────────────────────────────────────
        elif action == 'orbit':
            self.orbit_radius = float(cmd.get('radius', 20.0))
            self.orbit_speed = float(cmd.get('speed', 5.0))
            self.orbit_angle = 0.0
            self.square_active = False

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
            speed = float(cmd.get('speed', 2.0))
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
            self.target_x = self.square_waypoints[0][0]
            self.target_y = self.square_waypoints[0][1]
            self.target_z = self.square_alt_z

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
            self.target_speed = float(cmd.get('speed', 5.0))
            self.orbit_speed = self.target_speed

        elif action == 'set_heading':
            # Accept both "heading_deg" (new) and "heading" (legacy)
            heading_deg = cmd.get('heading_deg', cmd.get('heading', 0.0))
            self.target_yaw = math.radians(float(heading_deg))

        # ── Land / RTL ───────────────────────────────────────────────────
        elif action == 'land':
            self.orbiting = False
            self.square_active = False
            msg = self._make_vehicle_command(MAV_CMD_NAV_LAND)
            messages.append(('/fmu/in/vehicle_command', msg))

        elif action == 'rtl':
            self.orbiting = False
            self.square_active = False
            msg = self._make_vehicle_command(MAV_CMD_NAV_RETURN_TO_LAUNCH)
            messages.append(('/fmu/in/vehicle_command', msg))

        # ── Hold: stop all patterns, brain_node sets position ────────────
        elif action == 'hold':
            self.orbiting = False
            self.square_active = False

        return messages

    def get_setpoint_tick(self) -> TrajectorySetpoint:
        """Generate the current TrajectorySetpoint for the 10Hz publish loop.

        If orbiting, advances the orbit angle. Otherwise returns the
        current static target position.
        """
        msg = TrajectorySetpoint()
        msg.timestamp = int(time.time() * 1e6)

        if self.square_active:
            # Check if we reached the current waypoint
            wp = self.square_waypoints[self.square_wp_index]
            dist = math.sqrt(
                (self.target_x - wp[0]) ** 2 + (self.target_y - wp[1]) ** 2
            )
            # target is already set to current wp, check if drone is near it
            # (We approximate by checking if setpoint hasn't changed recently)
            # Advance to next waypoint each tick cycle based on position feedback
            msg.position[0] = wp[0]
            msg.position[1] = wp[1]
            msg.position[2] = self.square_alt_z
            msg.yaw = float('nan')
            return msg

        elif self.orbiting:
            # Advance orbit angle based on speed and radius
            # angular_velocity = linear_speed / radius
            dt = 0.1  # 10Hz tick
            angular_vel = self.orbit_speed / max(self.orbit_radius, 1.0)
            self.orbit_angle += angular_vel * dt

            msg.position[0] = self.orbit_center_x + self.orbit_radius * math.cos(self.orbit_angle)
            msg.position[1] = self.orbit_center_y + self.orbit_radius * math.sin(self.orbit_angle)
            msg.position[2] = self.orbit_alt_z

            # Point yaw toward orbit center
            dx = self.orbit_center_x - msg.position[0]
            dy = self.orbit_center_y - msg.position[1]
            msg.yaw = math.atan2(dy, dx)
        else:
            msg.position[0] = self.target_x
            msg.position[1] = self.target_y
            msg.position[2] = self.target_z
            msg.yaw = self.target_yaw

        return msg

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
