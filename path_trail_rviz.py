#!/usr/bin/env python3
import math
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from px4_msgs.msg import VehicleOdometry


class Px4TrailNode(Node):
    def __init__(self) -> None:
        super().__init__("px4_trail_node")

        self.declare_parameter("odom_topic", "/fmu/out/vehicle_odometry")
        self.declare_parameter("path_topic", "/drone_path")
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("max_points", 5000)
        self.declare_parameter("min_distance", 0.15)

        odom_topic = self.get_parameter("odom_topic").value
        path_topic = self.get_parameter("path_topic").value
        self.frame_id = self.get_parameter("frame_id").value
        self.max_points = int(self.get_parameter("max_points").value)
        self.min_distance = float(self.get_parameter("min_distance").value)

        self.path_pub = self.create_publisher(Path, path_topic, 10)

        px4_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.sub = self.create_subscription(
            VehicleOdometry,
            odom_topic,
            self.odom_cb,
            px4_qos,
        )

        self.path_msg = Path()
        self.path_msg.header.frame_id = self.frame_id
        self.poses = deque(maxlen=self.max_points)

        self.last_x = None
        self.last_y = None
        self.last_z = None

        self.get_logger().info(f"Listening on {odom_topic}")
        self.get_logger().info(f"Publishing trail on {path_topic}")
        self.get_logger().info("Using ENU display frame: x=east, y=north, z=up")

    def odom_cb(self, msg: VehicleOdometry) -> None:
        n = float(msg.position[0])
        e = float(msg.position[1])
        d = float(msg.position[2])

        x = e
        y = n
        z = -d

        if self.last_x is not None:
            dist = math.sqrt(
                (x - self.last_x) ** 2 +
                (y - self.last_y) ** 2 +
                (z - self.last_z) ** 2
            )
            if dist < self.min_distance:
                return

        self.last_x = x
        self.last_y = y
        self.last_z = z

        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = self.frame_id
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z
        pose.pose.orientation.w = 1.0

        self.poses.append(pose)

        self.path_msg.header.stamp = pose.header.stamp
        self.path_msg.header.frame_id = self.frame_id
        self.path_msg.poses = list(self.poses)

        self.path_pub.publish(self.path_msg)


def main() -> None:
    rclpy.init()
    node = Px4TrailNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
