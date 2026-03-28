# Course 2: Control A PX4 Drone With An LLM

From a completely fresh Ubuntu 24.04 installation to a full ROS 2 + PX4 + Gazebo + Ollama + YOLO workflow.

These notes assume a completely fresh Ubuntu 24.04 Desktop machine. Nothing is assumed to be preinstalled except Ubuntu itself and an internet connection.

Goal: by the end of this handout, students will be able to launch PX4 SITL in Gazebo, bridge the camera into ROS 2, run the LLM-controlled drone stack, visualize YOLO detections, and command the drone with natural-language prompts.

## 1. Fresh Ubuntu Preparation

Open a terminal and run:

```bash
sudo apt update
sudo apt upgrade -y
sudo apt install -y \
  software-properties-common \
  curl \
  git \
  wget \
  gnupg \
  lsb-release \
  locales \
  build-essential \
  cmake \
  python3-pip \
  python3-colcon-common-extensions
```

Set a UTF-8 locale:

```bash
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8
locale
```

## 2. Install ROS 2 Jazzy

Enable the Ubuntu universe repository:

```bash
sudo apt install -y software-properties-common
sudo add-apt-repository universe -y
```

Add the official ROS 2 apt source:

```bash
sudo apt update && sudo apt install curl -y
export ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F '"tag_name"' | awk -F'"' '{print $4}')
curl -L -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo ${UBUNTU_CODENAME:-${VERSION_CODENAME}})_all.deb"
sudo dpkg -i /tmp/ros2-apt-source.deb
```

Install ROS 2 Jazzy Desktop and development tools:

```bash
sudo apt update
sudo apt install -y ros-jazzy-desktop ros-dev-tools
```

Add ROS 2 to your shell and verify it:

```bash
echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc
source /opt/ros/jazzy/setup.bash
ros2 --version
```

## 3. Install Gazebo ROS Integration And Tools

```bash
sudo apt update
sudo apt install -y \
  ros-jazzy-ros-gz \
  ros-jazzy-ros-gz-bridge \
  ros-jazzy-ros-gz-image \
  ros-jazzy-cv-bridge \
  ros-jazzy-rqt-image-view
```

Optional but useful shell settings:

```bash
echo 'export PATH=/usr/bin:$PATH' >> ~/.bashrc
echo 'export GZ_CONFIG_PATH=/usr/share/gz' >> ~/.bashrc
source ~/.bashrc
```

## 4. Install PX4-Autopilot

Clone PX4:

```bash
cd ~
git clone --recursive https://github.com/PX4/PX4-Autopilot.git
```

Run the PX4 Ubuntu setup script:

```bash
cd ~/PX4-Autopilot
bash ./Tools/setup/ubuntu.sh
```

When the script finishes, reboot the machine:

```bash
sudo reboot
```

## 5. After Reboot: Re-Source ROS 2

```bash
source /opt/ros/jazzy/setup.bash
```

## 6. Test PX4 + Gazebo

Launch the camera-equipped PX4 SITL model:

```bash
cd ~/PX4-Autopilot
make px4_sitl gz_x500_mono_cam
```

Inside the PX4 shell, apply these parameters once:

```text
param set NAV_DLL_ACT 0
param set COM_RC_IN_MODE 1
param set COM_ARM_SDCARD 0
param set CBRK_SUPPLY_CHK 894281
param save
```

Optional quick test in the PX4 shell:

```text
commander arm
commander takeoff
```

Stop PX4 with `Ctrl+C` before continuing.

## 7. Optional: Install QGroundControl

This is optional for the LLM course, but useful for monitoring vehicle state.

Prepare the system:

```bash
sudo usermod -aG dialout "$(id -un)"
sudo systemctl mask --now ModemManager.service
sudo apt install -y \
  gstreamer1.0-plugins-bad \
  gstreamer1.0-libav \
  gstreamer1.0-gl \
  python3-gi \
  python3-gst-1.0 \
  libfuse2 \
  libxcb-xinerama0 \
  libxkbcommon-x11-0 \
  libxcb-cursor-dev
```

Now log out and log back in once so the `dialout` group takes effect.

Download `QGroundControl-x86_64.AppImage` from the official QGroundControl site into `~/Downloads`, then run:

```bash
cd ~/Downloads
chmod +x QGroundControl-*.AppImage
./QGroundControl-*.AppImage
```

## 8. Install The PX4 DDS Bridge Agent

Build `MicroXRCEAgent` from source:

```bash
cd ~
git clone -b v2.4.3 https://github.com/eProsima/Micro-XRCE-DDS-Agent.git
cd ~/Micro-XRCE-DDS-Agent
mkdir build
cd build
cmake ..
make -j$(nproc)
sudo make install
sudo ldconfig /usr/local/lib/
```

Verify it exists:

```bash
MicroXRCEAgent --help
```

## 9. Install Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Start Ollama in a dedicated terminal:

```bash
ollama serve
```

Leave that terminal open. In a second terminal, pull the models used by this project:

```bash
ollama pull qwen3:4b
ollama pull functiongemma
ollama list
```

## 10. Clone The LLM Drone Project

```bash
cd ~
git clone https://github.com/mouad-boumediene/LLM-controlled-drone.git
cd ~/LLM-controlled-drone
```

Replace the embedded `px4_msgs` placeholder with a real clone before building:

```bash
rm -rf ~/LLM-controlled-drone/src/px4_msgs
git clone https://github.com/PX4/px4_msgs.git ~/LLM-controlled-drone/src/px4_msgs
```

## 11. Install Python Packages Used By The Project

Ubuntu 24.04 may block plain `pip install` into the system Python, so use:

```bash
/usr/bin/python3 -m pip install --break-system-packages "numpy<2" ultralytics opencv-python
```

## 12. Build The Workspace

Initialize `rosdep` once on a fresh machine:

```bash
sudo rosdep init
```

Then build the workspace:

```bash
source /opt/ros/jazzy/setup.bash
cd ~/LLM-controlled-drone
rosdep update
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install --packages-select drone_agent
source ~/LLM-controlled-drone/install/setup.bash
```

Add the workspace to your shell:

```bash
echo "source ~/LLM-controlled-drone/install/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

## 13. Launch The Full System

If any terminal shows `(base)`, deactivate Conda first:

```bash
conda deactivate
conda deactivate
```

### Terminal 1: Ollama

```bash
ollama serve
```

If it says Ollama is already running, that is fine.

### Terminal 2: PX4 + Gazebo

```bash
cd ~/PX4-Autopilot
make px4_sitl gz_x500_mono_cam
```

### Terminal 3: Micro XRCE DDS Agent

```bash
MicroXRCEAgent udp4 -p 8888
```

### Terminal 4: Gazebo Camera Bridge

```bash
source /opt/ros/jazzy/setup.bash
ros2 run ros_gz_bridge parameter_bridge /world/default/model/x500_mono_cam_0/link/camera_link/sensor/camera/image@sensor_msgs/msg/Image[gz.msgs.Image --ros-args -r /world/default/model/x500_mono_cam_0/link/camera_link/sensor/camera/image:=/x500/camera/image_raw
```

### Terminal 5: LLM Drone Stack

```bash
source /opt/ros/jazzy/setup.bash
source ~/LLM-controlled-drone/install/setup.bash
ros2 launch drone_agent drone_agent.launch.py camera_topic:=/x500/camera/image_raw
```

### Terminal 6: Prompt Terminal

```bash
bash ~/LLM-controlled-drone/run_prompt_chat.sh
```

### Terminal 7: See The Camera / YOLO Output

```bash
source /opt/ros/jazzy/setup.bash
ros2 run rqt_image_view rqt_image_view
```

In `rqt_image_view`, select `/x500/camera/image_raw` for the raw camera or `/yolo/image_annotated` for the YOLO overlay.

Optional detection text terminal:

```bash
source /opt/ros/jazzy/setup.bash
ros2 topic echo /yolo/detections
```

### Terminal 8: RViz Trail

```bash
source /opt/ros/jazzy/setup.bash
source ~/LLM-controlled-drone/install/setup.bash
rviz2
```

In RViz:
1. Set `Fixed Frame` to `map`
2. Add `Path` and choose `/drone/path`
3. Add `Pose` and choose `/drone/pose`

## 14. Quick Health Checks

Check the ROS nodes:

```bash
source /opt/ros/jazzy/setup.bash
ros2 node list
```

You should see at least `/brain_node`, `/yolo_detector`, and `/trail_node`.

Check the main LLM model:

```bash
source /opt/ros/jazzy/setup.bash
ros2 param get /brain_node ollama_model
```

Expected:

```text
String value is: qwen3:4b
```

Check GPS:

```bash
source /opt/ros/jazzy/setup.bash
ros2 topic echo --once /fmu/out/vehicle_gps_position
```

If `fix_type` is `3` or higher, GPS-aware prompts should work.

## 15. First Prompt Sequence

```text
reset memory
```

```text
take off and hover at 5 meters
```

```text
fly a heart shape
```

```text
hold
```

```text
fly a clover shape
```

```text
hold
```

```text
search for a person and stop above them when found
```

```text
hold
```

```text
search for a bus then approach it and hold for 10 seconds then search for a person and approach it
```

```text
land
```

## 16. Good Prompt Examples For The Course

### Basic control

```text
take off and hover at 3 meters
go 5 meters north
set altitude to 10 meters
face east
set speed to 1 meter per second
hold
land
return to launch
```

### Shapes

```text
fly in a circle of 5 meter radius
fly a square pattern
fly an upward spiral
fly a figure eight
fly a heart shape
fly a clover shape
```

### Search and follow

```text
search for a person
approach the person
move slowly toward the person
follow the person
search for a bus
approach the bus
```

### Multi-step missions

```text
takeoff, then set the altitude to 2 meters, then fly in a square for 1 minute, then hold for 10 seconds, then fly in a circle for 1 minute
```

```text
search for a bus then approach it and hold for 10 seconds then search for a person and approach it
```

## 17. Rules To Tell Students

- Run only one `drone_agent.launch.py` at a time
- Run only one camera bridge at a time
- Run only one `MicroXRCEAgent` at a time
- If the drone is doing the wrong thing, type `hold`
- A new prompt interrupts the current mission
- If YOLO is not seeing the target, search/follow prompts will not behave well
- Keep Ollama running the whole time

## 18. Full Cleanup Command

If the system gets messy and you want to kill everything:

```bash
pkill -f 'ros2 launch drone_agent drone_agent.launch.py' || true
pkill -f '/install/drone_agent/lib/drone_agent/brain_node' || true
pkill -f '/install/drone_agent/lib/drone_agent/yolo_detector' || true
pkill -f '/install/drone_agent/lib/drone_agent/trail_node' || true
pkill -f 'ros2 run ros_gz_bridge parameter_bridge /world/default/model/x500_mono_cam_0/link/camera_link/sensor/camera/image' || true
pkill -f 'MicroXRCEAgent udp4 -p 8888' || true
pkill -f '/PX4-Autopilot/build/px4_sitl_default/bin/px4' || true
pkill -f 'gz sim' || true
pkill -f 'ollama serve' || true
```

Then launch again from Section 13.

## Official References

- ROS 2 Jazzy Ubuntu install: https://docs.ros.org/en/jazzy/Installation/Alternatives/Ubuntu-Install-Binary.html
- PX4 Ubuntu development environment: https://docs.px4.io/main/en/dev_setup/dev_env_linux_ubuntu
- PX4 uXRCE-DDS bridge: https://docs.px4.io/main/en/middleware/uxrce_dds
- Ollama Linux install: https://docs.ollama.com/linux
- QGroundControl Ubuntu install: https://docs.qgroundcontrol.com/master/en/qgc-user-guide/getting_started/download_and_install.html
