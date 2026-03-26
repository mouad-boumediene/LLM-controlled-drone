#!/usr/bin/env python3
"""YOLO detection ROS 2 node for Gazebo camera images."""

import json

from cv_bridge import CvBridge
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from ultralytics import YOLO


class YoloDetector(Node):
    def __init__(self):
        super().__init__('yolo_detector')

        # Parameters
        self.declare_parameter('model_path', 'yolov8s.pt')
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('camera_topic', '/x500/camera/image_raw')
        self.declare_parameter('skip_frames', 2)

        model_path = self.get_parameter('model_path').value
        self.conf_threshold = self.get_parameter('confidence_threshold').value
        camera_topic = self.get_parameter('camera_topic').value
        self.skip_frames = self.get_parameter('skip_frames').value

        self.get_logger().info(f'Loading YOLO model: {model_path}')
        self.model = YOLO(model_path)
        self.bridge = CvBridge()
        self.camera_topic = camera_topic
        self.frame_count = 0
        self.logged_camera_format = False
        self.airplane_min_confidence = max(self.conf_threshold, 0.75)
        self.airplane_min_bbox_area = 0.06
        self.airplane_corner_margin = 0.22
        self.airplane_top_margin = 0.35

        # Store latest detections for brain node polling
        self.latest_detections = []

        # QoS for sensor data
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        image_pub_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.sub = self.create_subscription(
            Image, camera_topic, self.image_callback, sensor_qos
        )
        self.pub = self.create_publisher(String, '/yolo/detections', 10)
        # Annotated image with bounding boxes — view with:
        #   ros2 run rqt_image_view rqt_image_view   (select /yolo/image_annotated)
        self.img_pub = self.create_publisher(
            Image, '/yolo/image_annotated', image_pub_qos
        )

        self.get_logger().info(
            f'YOLO detector node started on camera topic {camera_topic}'
        )

    def _image_msg_to_bgr(self, msg: Image) -> np.ndarray:
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        frame = np.ascontiguousarray(frame)
        encoding = (msg.encoding or '').lower()

        if frame.ndim == 2 or encoding in {'mono8', '8uc1'}:
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if encoding == 'rgb8':
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if encoding == 'rgba8':
            return cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        if encoding in {'bgra8', '8uc4'}:
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        if encoding in {'bgr8', '8uc3', ''} and frame.ndim == 3 and frame.shape[2] == 3:
            return frame

        raise ValueError(
            f'unsupported image encoding {msg.encoding!r} with shape {frame.shape}'
        )

    def _keep_detection(
        self,
        label: str,
        confidence: float,
        bbox_center: tuple[float, float],
        bbox_area: float,
    ) -> bool:
        """Filter classes that need stricter rules in the mono-cam view."""
        if label != 'airplane':
            return True

        if confidence < self.airplane_min_confidence:
            return False

        if bbox_area < self.airplane_min_bbox_area:
            return False

        cx, cy = bbox_center
        near_corner = (
            cy < self.airplane_top_margin and
            (cx < self.airplane_corner_margin or cx > 1.0 - self.airplane_corner_margin)
        )
        if near_corner:
            return False

        return True

    def image_callback(self, msg: Image):
        # Skip frames to reduce CPU load
        self.frame_count += 1
        if self.frame_count % (self.skip_frames + 1) != 0:
            return

        if not self.logged_camera_format:
            self.get_logger().info(
                f'Received frames from {self.camera_topic}: '
                f'encoding={msg.encoding} size={msg.width}x{msg.height}'
            )
            self.logged_camera_format = True

        try:
            frame = self._image_msg_to_bgr(msg)
        except Exception as e:
            self.get_logger().error(f'Image conversion failed: {e}')
            return

        try:
            results = self.model(frame, verbose=False, conf=self.conf_threshold)[0]
        except Exception as e:
            self.get_logger().error(f'YOLO inference failed: {e}')
            return

        detections = []
        filtered_boxes = []
        h, w = frame.shape[:2]
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            label = results.names[int(box.cls[0])]
            confidence = float(box.conf[0])
            bbox_center = ((x1 + x2) / (2 * w), (y1 + y2) / (2 * h))
            bbox_area = ((x2 - x1) * (y2 - y1)) / (w * h)

            if not self._keep_detection(label, confidence, bbox_center, bbox_area):
                continue

            detections.append({
                'class': label,
                'confidence': round(confidence, 2),
                'bbox_center': [
                    round(bbox_center[0], 2),
                    round(bbox_center[1], 2),
                ],
                'bbox_area': round(bbox_area, 4),
            })
            filtered_boxes.append((box, label, confidence))

        self.latest_detections = detections

        out = String()
        out.data = json.dumps(detections)
        self.pub.publish(out)

        if detections:
            classes = [d['class'] for d in detections]
            self.get_logger().info(f'Detected: {classes}')

        # Draw bounding boxes and publish annotated image
        annotated = frame.copy()
        for box, label, conf in filtered_boxes:
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
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
