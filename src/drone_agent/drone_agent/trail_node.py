#!/usr/bin/env python3
"""Publish an RViz-friendly trail from PX4 vehicle odometry."""

from collections import deque
import math

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from px4_msgs.msg import VehicleOdometry
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy


class TrailNode(Node):
    """Convert PX4 odometry into a persistent RViz path trail."""

    def __init__(self):
        super().__init__('trail_node')

        self.declare_parameter('odometry_topic', '/fmu/out/vehicle_odometry')
        self.declare_parameter('path_topic', '/drone/path')
        self.declare_parameter('frame_id', 'map')
        self.declare_parameter('max_points', 2000)
        self.declare_parameter('min_point_spacing_m', 0.15)
        self.declare_parameter('jump_reset_distance_m', 20.0)

        self.frame_id = self.get_parameter('frame_id').get_parameter_value().string_value
        self.max_points = max(
            10,
            self.get_parameter('max_points').get_parameter_value().integer_value,
        )
        self.min_point_spacing_m = max(
            0.01,
            self.get_parameter('min_point_spacing_m').get_parameter_value().double_value,
        )
        self.jump_reset_distance_m = max(
            1.0,
            self.get_parameter('jump_reset_distance_m').get_parameter_value().double_value,
        )

        self.path_topic = self.get_parameter('path_topic').get_parameter_value().string_value
        self.path_pub = self.create_publisher(
            Path,
            self.path_topic,
            10,
        )
        self.pose_pub = self.create_publisher(PoseStamped, '/drone/pose', 10)
        odom_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.odom_sub = self.create_subscription(
            VehicleOdometry,
            self.get_parameter('odometry_topic').get_parameter_value().string_value,
            self._odom_cb,
            odom_qos,
        )

        self.poses = deque(maxlen=self.max_points)
        self.last_position = None
        self.last_reset_counter = None
        self.get_logger().info(
            f'Trail node started. Publishing RViz path on '
            f'{self.path_topic} in frame "{self.frame_id}".'
        )

    def _odom_cb(self, msg: VehicleOdometry):
        """Append the current pose to the trail in ENU coordinates."""
        now = self.get_clock().now().to_msg()
        pose = PoseStamped()
        pose.header.stamp = now
        pose.header.frame_id = self.frame_id

        # PX4 odometry uses local NED. RViz expects a conventional ENU-like map.
        north = float(msg.position[0])
        east = float(msg.position[1])
        down = float(msg.position[2])
        position = (east, north, -down)
        pose.pose.position.x = position[0]
        pose.pose.position.y = position[1]
        pose.pose.position.z = position[2]
        pose.pose.orientation.w = 1.0

        if self.last_reset_counter is None:
            self.last_reset_counter = int(msg.reset_counter)
        elif int(msg.reset_counter) != self.last_reset_counter:
            self.poses.clear()
            self.last_position = None
            self.last_reset_counter = int(msg.reset_counter)

        if self.last_position is not None:
            dx = position[0] - self.last_position[0]
            dy = position[1] - self.last_position[1]
            dz = position[2] - self.last_position[2]
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            if dist > self.jump_reset_distance_m:
                self.poses.clear()
                self.last_position = None
            elif dist < self.min_point_spacing_m:
                self.pose_pub.publish(pose)
                return

        self.poses.append(pose)
        self.last_position = position

        path = Path()
        path.header.stamp = now
        path.header.frame_id = self.frame_id
        path.poses = list(self.poses)

        self.pose_pub.publish(pose)
        self.path_pub.publish(path)


def main(args=None):
    rclpy.init(args=args)
    node = TrailNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
