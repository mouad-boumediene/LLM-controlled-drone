#!/usr/bin/env bash
set -eo pipefail

source /opt/ros/jazzy/setup.bash
source "$HOME/LLM-controlled-drone/install/setup.bash"

exec /usr/bin/python3 "$HOME/LLM-controlled-drone/prompt_chat.py"
