"""Launch file for the drone_agent pipeline.

Launches:
1. ros_gz_bridge — bridges Gazebo camera to ROS2
2. yolo_detector — runs YOLOv8 on camera images
3. brain_node — orchestrates LLM + PX4 control

Prerequisites (run in separate terminals):
- PX4 SITL: cd PX4-Autopilot && make px4_sitl gz_x500_mono_cam
- uXRCE-DDS: MicroXRCEAgent udp4 -p 8888
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        # --- Arguments ---
        DeclareLaunchArgument(
            'camera_topic', default_value='/camera',
            description='Gazebo camera topic name',
        ),
        DeclareLaunchArgument(
            'yolo_model', default_value='yolov8n.pt',
            description='YOLO model file (e.g. yolov8n.pt, yolov8s.pt)',
        ),
        DeclareLaunchArgument(
            'llm_interval', default_value='7.0',
            description='Seconds between periodic LLM re-evaluation calls',
        ),

        # --- Gazebo camera bridge ---
        ExecuteProcess(
            cmd=[
                'ros2', 'run', 'ros_gz_bridge', 'parameter_bridge',
                '/camera@sensor_msgs/msg/Image@gz.msgs.Image',
            ],
            output='screen',
        ),

        # --- YOLO detection node ---
        Node(
            package='drone_agent',
            executable='yolo_detector',
            name='yolo_detector',
            output='screen',
            parameters=[{
                'model_path': LaunchConfiguration('yolo_model'),
                'confidence_threshold': 0.5,
                'camera_topic': LaunchConfiguration('camera_topic'),
                'skip_frames': 2,
            }],
        ),

        # --- Brain node ---
        Node(
            package='drone_agent',
            executable='brain_node',
            name='brain_node',
            output='screen',
            parameters=[{
                'llm_interval_sec': LaunchConfiguration('llm_interval'),
                'offboard_rate_hz': 10.0,
            }],
        ),
    ])
