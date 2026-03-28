"""Launch file for the drone_agent pipeline.

Launches:
1. Optional ros_gz_bridge camera bridge
2. yolo_detector — runs YOLOv8 on camera images
3. brain_node — orchestrates LLM + PX4 control
4. trail_node — publishes an RViz path trail from PX4 odometry

Prerequisites (run in separate terminals):
- PX4 SITL: cd PX4-Autopilot && make px4_sitl gz_x500_mono_cam
- uXRCE-DDS: MicroXRCEAgent udp4 -p 8888
"""

from launch import LaunchDescription
from launch.conditions import IfCondition
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        # --- Arguments ---
        DeclareLaunchArgument(
            'camera_topic', default_value='/x500/camera/image_raw',
            description='ROS camera topic consumed by the YOLO node',
        ),
        DeclareLaunchArgument(
            'gazebo_camera_topic',
            default_value='/world/default/model/x500_mono_cam_0/link/camera_link/sensor/camera/image',
            description='Gazebo camera topic to bridge into ROS',
        ),
        DeclareLaunchArgument(
            'start_camera_bridge', default_value='false',
            description='Start the Gazebo camera bridge inside this launch file',
        ),
        DeclareLaunchArgument(
            'yolo_model', default_value='yolov8s.pt',
            description='YOLO model file (e.g. yolov8s.pt, yolov8m.pt)',
        ),
        DeclareLaunchArgument(
            'llm_interval', default_value='7.0',
            description='Seconds between periodic LLM re-evaluation calls',
        ),
        DeclareLaunchArgument(
            'ollama_model', default_value='qwen3:4b',
            description='Ollama model used by brain_node',
        ),
        DeclareLaunchArgument(
            'offboard_rate_hz', default_value='30.0',
            description='Brain-node offboard setpoint publish rate in Hz',
        ),
        DeclareLaunchArgument(
            'max_speed_m_s', default_value='2.0',
            description='Hard translational speed cap in metres per second (0 disables cap)',
        ),
        DeclareLaunchArgument(
            'enable_custom_shape_fallback', default_value='true',
            description='Use a lightweight FunctionGemma fallback for unsupported custom shapes',
        ),
        DeclareLaunchArgument(
            'custom_shape_model', default_value='functiongemma',
            description='Ollama model used for unsupported custom shape prompts',
        ),
        DeclareLaunchArgument(
            'start_trail_node', default_value='true',
            description='Publish an RViz-friendly nav_msgs/Path trail from PX4 odometry',
        ),

        # --- Gazebo camera bridge ---
        ExecuteProcess(
            condition=IfCondition(LaunchConfiguration('start_camera_bridge')),
            cmd=['bash', '-lc', [
                'ros2 run ros_gz_bridge parameter_bridge ',
                LaunchConfiguration('gazebo_camera_topic'),
                '@sensor_msgs/msg/Image[gz.msgs.Image --ros-args -r ',
                LaunchConfiguration('gazebo_camera_topic'),
                ':=',
                LaunchConfiguration('camera_topic'),
            ]],
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
                'ollama_model': LaunchConfiguration('ollama_model'),
                'offboard_rate_hz': LaunchConfiguration('offboard_rate_hz'),
                'max_speed_m_s': LaunchConfiguration('max_speed_m_s'),
                'enable_custom_shape_fallback': LaunchConfiguration('enable_custom_shape_fallback'),
                'custom_shape_model': LaunchConfiguration('custom_shape_model'),
            }],
        ),

        # --- RViz trail node ---
        Node(
            condition=IfCondition(LaunchConfiguration('start_trail_node')),
            package='drone_agent',
            executable='trail_node',
            name='trail_node',
            output='screen',
            parameters=[{
                'odometry_topic': '/fmu/out/vehicle_odometry',
                'path_topic': '/drone/path',
                'frame_id': 'map',
                'max_points': 2000,
                'min_point_spacing_m': 0.15,
                'jump_reset_distance_m': 20.0,
            }],
        ),
    ])
