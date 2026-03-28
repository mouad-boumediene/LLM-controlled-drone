# LLM-Controlled Drone

A ROS 2 + PX4 + Gazebo drone stack that accepts natural-language prompts, uses a local LLM to choose flight actions, runs YOLO object detection on the monocular camera feed, and flies through PX4 Offboard setpoints.

This fork extends the upstream Gazebo-based project with a more capable mission system, GPS-aware commands, local visual search/follow behaviors, deterministic custom shape generation, a lightweight custom-shape fallback via FunctionGemma, and RViz trail visualization.

## Attribution

- Forked from [pratikPhadte/LLM-controlled-drone](https://github.com/pratikPhadte/LLM-controlled-drone)
- Related paper: [Taking Flight with Dialogue: Enabling Natural Language Control for PX4-based Drone Agent](https://arxiv.org/abs/2506.07509)
- Official paper codebase: [limshoonkit/ros2-px4-agent-ws](https://github.com/limshoonkit/ros2-px4-agent-ws)
- Upstream demo video: [YouTube demo](https://www.youtube.com/watch?v=cimnMgLYCnY&t=406s)

This repository is not the official paper repository. It is a fork of the Gazebo/ROS 2 implementation above, with additional features added here.

## What The Current Stack Does

- Natural-language control through a local Ollama model
- PX4 Offboard flight in Gazebo using `gz_x500_mono_cam`
- YOLOv8 detection from the forward camera
- Prompt-driven object search, approach, and follow behaviors
- Multi-step missions that decompose one prompt into sequential step prompts
- GPS-aware actions like `goto`, `look_at_gps`, and `rtl` when PX4 has a valid fix
- Deterministic built-in shape paths:
  - `circle`
  - `square`
  - `rectangle`
  - `triangle`
  - `polygon`
  - `star`
  - `figure_eight`
  - `zigzag`
  - `spiral`
  - `heart`
- Best-effort custom shape fallback through local Ollama `functiongemma`
- RViz path trail publishing via `/drone/path`

## Current Architecture

### Main Nodes

| Node | Role |
|---|---|
| `brain_node` | Main orchestrator. Receives prompts, calls the LLM, runs the mission queue, handles visual search/follow, and publishes PX4 Offboard setpoints |
| `yolo_detector` | Runs YOLOv8 on the camera feed and publishes detections plus an annotated image |
| `trail_node` | Converts PX4 odometry into RViz-friendly `nav_msgs/Path` and `PoseStamped` topics |
| `ros_gz_bridge` | Bridges Gazebo camera images into ROS 2 |
| `prompt_chat` | Small terminal chat client that publishes prompts to `/user_command` |

### Important Topics

| Topic | Type | Purpose |
|---|---|---|
| `/user_command` | `std_msgs/msg/String` | Natural-language user prompt input |
| `/x500/camera/image_raw` | `sensor_msgs/msg/Image` | Default ROS camera topic consumed by YOLO |
| `/yolo/detections` | `std_msgs/msg/String` | YOLO detections as JSON |
| `/yolo/image_annotated` | `sensor_msgs/msg/Image` | Annotated camera image with bounding boxes |
| `/drone/path` | `nav_msgs/msg/Path` | RViz trail path |
| `/drone/pose` | `geometry_msgs/msg/PoseStamped` | Current pose for RViz overlays |
| `/fmu/out/vehicle_odometry` | `px4_msgs/msg/VehicleOdometry` | PX4 local pose and velocity |
| `/fmu/out/vehicle_gps_position` | `px4_msgs/msg/SensorGps` | PX4 GPS telemetry |
| `/fmu/out/vehicle_status` | `px4_msgs/msg/VehicleStatus` | PX4 vehicle state |
| `/fmu/in/offboard_control_mode` | `px4_msgs/msg/OffboardControlMode` | Offboard heartbeat to PX4 |
| `/fmu/in/trajectory_setpoint` | `px4_msgs/msg/TrajectorySetpoint` | Main position/yaw setpoints |
| `/fmu/in/vehicle_command` | `px4_msgs/msg/VehicleCommand` | Arm/land/RTL/offboard commands |

## Requirements

- Ubuntu 24.04
- ROS 2 Jazzy
- PX4-Autopilot built with `gz_x500_mono_cam`
- `MicroXRCEAgent`
- Ollama
- Python dependencies used by the workspace

### Recommended Local Models

Current defaults:

- Main controller model: `qwen3:4b`
- Custom unsupported-shape fallback: `functiongemma`
- YOLO model: `yolov8s.pt`

Pull the Ollama models before running:

```bash
ollama pull qwen3:4b
ollama pull functiongemma
```

If Ollama is not already running as a background service on your machine, start it with:

```bash
ollama serve
```

## Workspace Setup

Clone your fork:

```bash
git clone https://github.com/mouad-boumediene/LLM-controlled-drone.git
cd LLM-controlled-drone
```

The repository includes a convenience setup script:

```bash
bash setup_workspace.sh
```

That bootstraps the workspace, installs base ROS/Python dependencies, clones `px4_msgs` if needed, and builds with `colcon`.

You can also rebuild manually at any time:

```bash
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --packages-select drone_agent
source install/setup.bash
```

## Launching The Full Stack

If any terminal shows `(base)`, deactivate Conda first:

```bash
conda deactivate
conda deactivate
```

### Terminal 1: PX4 + Gazebo

```bash
cd ~/PX4-Autopilot
make px4_sitl gz_x500_mono_cam
```

### Terminal 2: Micro XRCE-DDS

```bash
/usr/local/bin/MicroXRCEAgent udp4 -p 8888
```

### Terminal 3: Gazebo Camera Bridge

```bash
source /opt/ros/jazzy/setup.bash
ros2 run ros_gz_bridge parameter_bridge /world/default/model/x500_mono_cam_0/link/camera_link/sensor/camera/image@sensor_msgs/msg/Image[gz.msgs.Image --ros-args -r /world/default/model/x500_mono_cam_0/link/camera_link/sensor/camera/image:=/x500/camera/image_raw
```

### Terminal 4: Drone Stack

```bash
source /opt/ros/jazzy/setup.bash
source ~/LLM-controlled-drone/install/setup.bash
ros2 launch drone_agent drone_agent.launch.py camera_topic:=/x500/camera/image_raw
```

### Terminal 5: Prompt Terminal

```bash
bash ~/LLM-controlled-drone/run_prompt_chat.sh
```

### Optional Terminal 6: YOLO Visualization

Annotated detections:

```bash
source /opt/ros/jazzy/setup.bash
ros2 run rqt_image_view rqt_image_view
```

Then select `/yolo/image_annotated`.

Detection JSON:

```bash
source /opt/ros/jazzy/setup.bash
ros2 topic echo /yolo/detections
```

### Optional Terminal 7: RViz Trail

```bash
source /opt/ros/jazzy/setup.bash
source ~/LLM-controlled-drone/install/setup.bash
rviz2
```

In RViz:

- Set `Fixed Frame` to `map`
- Add `Path` with topic `/drone/path`
- Optionally add `Pose` with topic `/drone/pose`

### Optional Single-Launch Camera Bridge

If you prefer, `drone_agent.launch.py` can also start the camera bridge itself:

```bash
source /opt/ros/jazzy/setup.bash
source ~/LLM-controlled-drone/install/setup.bash
ros2 launch drone_agent drone_agent.launch.py start_camera_bridge:=true
```

When you do that, you do not need the separate bridge terminal.

## First Prompt Sequence

In the prompt terminal:

```text
reset memory
take off and hover at 5 meters
```

Then try one of:

```text
fly an upward spiral
```

```text
search for a person and stop above them when found
```

```text
search for a bus then approach it and hold for 10 seconds then search for a person and approach it
```

To stop the current mission:

```text
hold
```

## Supported Prompt Behaviors

### Core Flight Commands

- `take off and hover at 5 meters`
- `go 10 meters north`
- `set altitude to 20 meters`
- `face east`
- `set speed to 1 meter per second`
- `hold`
- `land`
- `return to launch`

### GPS-Aware Commands

These require PX4 to be publishing a valid GPS fix.

- `go to latitude ... longitude ... altitude ...`
- `look at GPS latitude ... longitude ...`
- `return to launch`

A valid GPS fix typically means `fix_type >= 3` on `/fmu/out/vehicle_gps_position`.

### Search / Visual Missions

- `search for a person`
- `approach the bus`
- `move slowly toward the person`
- `follow the person`

The current implementation includes:

- target search orbit
- local visual approach/follow
- local reacquire/search when the target leaves frame
- multi-frame confirmation before switching back from search to follow

### Multi-Step Missions

Multi-step prompts are supported. The planner converts one large prompt into explicit step prompts, then the stack executes them one by one through the normal controller path.

Examples:

```text
takeoff, then set the altitude to 2 meters, then fly in a square for 1 minute, then hold for 10 seconds, then fly in a circle for 1 minute
```

```text
search for a bus then approach it and hold for 10 seconds then search for a person and approach it
```

Completion rules currently cover:

- `airborne`
- `altitude_reached`
- `heading_reached`
- `position_reached`
- `duration`
- `path_complete`
- `target_found`
- `approach_complete`

Entering a new prompt interrupts the current mission and starts the new one. The safest manual interrupt prompt is:

```text
hold
```

## Shape And Path Support

### Deterministic Built-In Shapes

The stack generates these shapes in code and then flies them through the generic waypoint path follower:

- `circle`
- `square`
- `rectangle`
- `triangle`
- `polygon`
- `star`
- `figure_eight`
- `zigzag`
- `spiral`
- `heart`

Examples:

```text
fly a heart shape
```

```text
fly a five-point star
```

```text
fly an upward spiral
```

### Unsupported Custom Shapes

Unsupported named shapes do not silently get approximated as a random polygon anymore. Instead, the LLM can request a custom `path` through the lightweight local `functiongemma` fallback.

Current best-fit procedural families include:

- clover / flower / quatrefoil / trefoil
- crescent / moon
- butterfly
- arrow / pointer
- diamond / rhombus / kite
- cloud / blob

Examples:

```text
fly a clover shape
```

```text
fly a crescent moon shape
```

```text
fly a butterfly outline
```

Important limitation:

- Built-in named shapes are the most reliable
- Unsupported custom shapes are best-effort
- Truly arbitrary logos, letters, or highly specific outlines are still experimental

Custom fallback shapes are smoothed before flight using corner-cutting and resampling so they fly as denser curves instead of a coarse polygon.

## Default Runtime Configuration

Current launch defaults from [`src/drone_agent/launch/drone_agent.launch.py`](src/drone_agent/launch/drone_agent.launch.py):

| Setting | Default |
|---|---|
| `camera_topic` | `/x500/camera/image_raw` |
| `gazebo_camera_topic` | `/world/default/model/x500_mono_cam_0/link/camera_link/sensor/camera/image` |
| `start_camera_bridge` | `false` |
| `yolo_model` | `yolov8s.pt` |
| `llm_interval` | `7.0` |
| `ollama_model` | `qwen3:4b` |
| `offboard_rate_hz` | `30.0` |
| `max_speed_m_s` | `2.0` |
| `enable_custom_shape_fallback` | `true` |
| `custom_shape_model` | `functiongemma` |
| `start_trail_node` | `true` |

The parameter template at [`src/drone_agent/config/params.yaml.template`](src/drone_agent/config/params.yaml.template) matches these defaults for the main stack.

## Project Layout

```text
LLM-controlled-drone/
├── README.md
├── prompt_chat.py
├── run_prompt_chat.sh
├── path_trail_rviz.py
├── setup_workspace.sh
├── paper.pdf
├── worlds/
├── yolov8n.pt
├── yolov8s.pt
└── src/
    ├── px4_msgs/
    └── drone_agent/
        ├── package.xml
        ├── setup.py
        ├── config/
        │   └── params.yaml.template
        ├── launch/
        │   └── drone_agent.launch.py
        └── drone_agent/
            ├── brain_node.py
            ├── command_translator.py
            ├── llm_client.py
            ├── yolo_detector.py
            ├── trail_node.py
            ├── shape_generator.py
            ├── functiongemma_path_generator.py
            └── starvector_path_generator.py
```

## Troubleshooting

### `PX4 server already running for instance 0`

A previous PX4 SITL instance is still running. Stop the old stack before launching a new one.

### No movement after a prompt

Check:

- `MicroXRCEAgent` is running
- only one `brain_node` is publishing
- only one `yolo_detector` is publishing
- PX4 has subscribers on `/fmu/in/trajectory_setpoint`

### YOLO Works But The Drone Ignores A Plain Motion Prompt

Detection-driven replans are now limited, but if behavior looks strange, confirm that only one `brain_node` and one `yolo_detector` are active.

### GPS Commands Do Not Work

Inspect:

```bash
source /opt/ros/jazzy/setup.bash
ros2 topic echo --once /fmu/out/vehicle_gps_position
```

You need a real fix, typically `fix_type >= 3`.

### RViz Trail Is Empty

Check:

- `trail_node` is running
- RViz `Fixed Frame` is `map`
- topic `/drone/path` exists

### LLM Timeouts

Queued missions make multiple LLM calls, so slow local inference can still stall a mission. Current code already increases timeouts and retries some queued steps, but the simplest improvement is to use a model that responds comfortably on your machine.

### NumPy / `cv_bridge` ABI Errors

If `cv_bridge` complains about NumPy ABI mismatches:

```bash
/usr/bin/python3 -m pip install "numpy<2"
```

### Conda / Non-System Python Build Problems

Deactivate Conda and rebuild with the system Python:

```bash
conda deactivate
conda deactivate
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install --packages-select drone_agent
```

## Notes

- The launch file starts `trail_node` by default
- The launch file can optionally start the Gazebo camera bridge
- `run_prompt_chat.sh` is the easiest way to interact with the stack from a terminal
- `path_trail_rviz.py` is still included as a standalone helper, but the main pipeline now uses `trail_node`
