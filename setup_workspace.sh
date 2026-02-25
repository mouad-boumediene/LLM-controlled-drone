#!/bin/bash
# Setup script for the AI-Controlled Drone workspace.
#
# Prerequisites:
#   - Ubuntu 24.04
#   - ROS2 Humble installed (sudo apt install ros-jazzy-desktop)
#   - PX4-Autopilot cloned and built separately
#
# Usage:
#   cd /home/user/Ai_controlled_drone
#   bash setup_workspace.sh

set -e

echo "=== AI-Controlled Drone Workspace Setup ==="

# Source ROS2
source /opt/ros/jazzy/setup.bash

# Install ROS2 dependencies
echo "[1/4] Installing ROS2 apt dependencies..."
sudo apt update
sudo apt install -y \
    ros-jazzy-ros-gz-bridge \
    ros-jazzy-ros-gz-image \
    ros-jazzy-cv-bridge \
    python3-colcon-common-extensions

# Install Python dependencies
echo "[2/4] Installing Python dependencies..."
pip install ultralytics opencv-python google-genai

# Clone px4_msgs if not present
echo "[3/4] Setting up px4_msgs..."
if [ ! -d "src/px4_msgs" ]; then
    cd src
    git clone https://github.com/PX4/px4_msgs.git
    cd ..
else
    echo "  px4_msgs already exists, skipping clone"
fi

# Build workspace
echo "[4/4] Building workspace with colcon..."
colcon build --symlink-install

echo ""
echo "=== Setup complete! ==="
echo ""
echo "To use the workspace:"
echo "  source install/setup.bash"
echo ""
echo "Full launch sequence (each in a separate terminal):"
echo ""
echo "  Terminal 1 — PX4 + Gazebo:"
echo "    cd PX4-Autopilot && make px4_sitl gz_x500_mono_cam"
echo ""
echo "  Terminal 2 — uXRCE-DDS agent:"
echo "    MicroXRCEAgent udp4 -p 8888"
echo ""
echo "  Terminal 3 — Drone agent (all nodes):"
echo "    source install/setup.bash"
echo "    export GEMINI_API_KEY='your-key-here'"
echo "    ros2 launch drone_agent drone_agent.launch.py"
echo ""
echo "  Terminal 4 — Send a command:"
echo "    ros2 topic pub /user_command std_msgs/msg/String \\"
echo "      \"data: 'Fly to 40m and circle the area'\" --once"
