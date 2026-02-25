#!/usr/bin/env python3
"""YOLO detection ROS2 node.

Subscribes to /camera images from Gazebo (via ros_gz_bridge),
runs YOLOv8 inference, and publishes structured detections as JSON.
"""

import json

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from ultralytics import YOLO


class YoloDetector(Node):
    def __init__(self):
        super().__init__('yolo_detector')

        # Parameters
        self.declare_parameter('model_path', 'yolov8n.pt')
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('camera_topic', '/camera')
        self.declare_parameter('skip_frames', 2)

        model_path = self.get_parameter('model_path').value
        self.conf_threshold = self.get_parameter('confidence_threshold').value
        camera_topic = self.get_parameter('camera_topic').value
        self.skip_frames = self.get_parameter('skip_frames').value

        self.get_logger().info(f'Loading YOLO model: {model_path}')
        self.model = YOLO(model_path)
        self.bridge = CvBridge()
        self.frame_count = 0

        # Store latest detections for brain node polling
        self.latest_detections = []

        # QoS for sensor data
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.sub = self.create_subscription(
            Image, camera_topic, self.image_callback, sensor_qos
        )
        self.pub = self.create_publisher(String, '/yolo/detections', 10)
        # Annotated image with bounding boxes — view with:
        #   ros2 run rqt_image_view rqt_image_view   (select /yolo/image_annotated)
        self.img_pub = self.create_publisher(Image, '/yolo/image_annotated', sensor_qos)

        self.get_logger().info('YOLO detector node started')

    def image_callback(self, msg: Image):
        # Skip frames to reduce CPU load
        self.frame_count += 1
        if self.frame_count % (self.skip_frames + 1) != 0:
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge conversion failed: {e}')
            return

        results = self.model(frame, verbose=False, conf=self.conf_threshold)[0]

        detections = []
        h, w = frame.shape[:2]
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append({
                'class': results.names[int(box.cls[0])],
                'confidence': round(float(box.conf[0]), 2),
                'bbox_center': [
                    round((x1 + x2) / (2 * w), 2),
                    round((y1 + y2) / (2 * h), 2),
                ],
                'bbox_area': round(((x2 - x1) * (y2 - y1)) / (w * h), 4),
            })

        self.latest_detections = detections

        out = String()
        out.data = json.dumps(detections)
        self.pub.publish(out)

        if detections:
            classes = [d['class'] for d in detections]
            self.get_logger().info(f'Detected: {classes}')

        # Draw bounding boxes and publish annotated image
        annotated = frame.copy()
        for box in results.boxes:
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            label = results.names[int(box.cls[0])]
            conf = float(box.conf[0])
            # Green box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Label background
            text = f'{label} {conf:.2f}'
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw, y1), (0, 255, 0), -1)
            cv2.putText(annotated, text, (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

        try:
            img_msg = self.bridge.cv2_to_imgmsg(annotated, 'bgr8')
            img_msg.header = msg.header
            self.img_pub.publish(img_msg)
        except Exception as e:
            self.get_logger().error(f'Annotated image publish failed: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = YoloDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
