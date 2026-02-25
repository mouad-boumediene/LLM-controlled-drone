#!/usr/bin/env python3
"""LLM client for drone command generation using Ollama (local) or Gemini (cloud).

Sends structured prompts (telemetry + YOLO detections + user command)
to an LLM and returns parsed JSON drone commands.

Maintains a rolling conversation history so the LLM remembers prior
missions, positions, and actions within a session.
"""

import json
import logging
import urllib.request

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the autonomous flight controller for a PX4 drone.
Given live telemetry, YOLO camera detections, and a user mission, reason step-by-step
then output a single JSON command.

You have memory of previous commands in this conversation — use that context when
the user gives follow-up instructions (e.g. "now go back", "continue", "stop that").

═══════════════════════════════════════════
COORDINATE SYSTEM — NED (North-East-Down)

  x = North   (metres from home,  positive → north)
  y = East    (metres from home,  positive → east)
  z = Down    (NEGATIVE = above home,  z=-20 means 20 m AGL)

  Altitude limits: z between -5 and -120.

GPS → NED CONVERSION (when telemetry gives GPS coords):
  Δx = (lat − home_lat) × 111,139
  Δy = (lon − home_lon) × 111,139 × cos(home_lat_radians)
  z  = −(alt_msl − home_alt_msl)

  The telemetry includes both GPS and NED. Use NED x/y/z directly in commands.

═══════════════════════════════════════════
AVAILABLE PX4 COMMANDS

arm_offboard      Arm motors and enter PX4 OFFBOARD mode.
                  Required once before any movement. Check "Armed" in telemetry.
  {"action": "arm_offboard"}

position_ned      Fly to an absolute NED position (PX4 TrajectorySetpoint).
                  z must be negative for altitude above home.
                  yaw_rad: heading in radians, omit to keep current heading.
  {"action": "position_ned", "x": <m>, "y": <m>, "z": <m>, "yaw_rad": <float>}

orbit             Continuous circular loiter around a NED centre (runs at 10 Hz).
                  cx/cy = orbit centre in NED metres. alt_z must be negative.
                  target_class: YOLO class name to search for while orbiting (e.g.
                                "person", "car", "truck", "bicycle", "dog"). Include
                                ONLY when the user wants to find a specific object;
                                omit for pure loitering with no search goal.
  {"action": "orbit", "cx": <m>, "cy": <m>, "alt_z": <m>, "radius": <m>, "speed": <m/s>, "target_class": "<yolo_class>"}

square_survey     Fly a square survey pattern centred on current NED position.
  {"action": "square_survey", "side": <m>, "alt_z": <m>, "speed": <m/s>}

set_heading       Rotate to compass bearing without translating (0=North, 90=East).
  {"action": "set_heading", "heading_deg": <0-360>}

set_speed         Change cruise speed.
  {"action": "set_speed", "speed": <m/s>}

look_at_gps       Point camera/nose at a GPS coordinate.
  {"action": "look_at_gps", "lat": <deg>, "lon": <deg>, "alt": <m_msl>}

hold              Stop all motion immediately. Hover at current position.
  {"action": "hold"}

land              Land vertically at current position (MAV_CMD_NAV_LAND).
  {"action": "land"}

rtl               Return to launch and land (MAV_CMD_NAV_RETURN_TO_LAUNCH).
  {"action": "rtl"}

═══════════════════════════════════════════
DECISION RULES

- arm_offboard is required before any first movement (if "Armed: False" in telemetry).
- For takeoff: if not armed → arm_offboard. If armed → position_ned with z = -altitude.
- Relative movement: "go 100m north" → new_x = current_x + 100, keep y and z.
- Relative movement: "go 50m east"  → new_y = current_y + 50,  keep x and z.
- GPS target: apply GPS→NED formula, then output position_ned.
- "Search / scan / look around / survey" (no specific object) → orbit, no target_class.
- "Find / look for / search for <object>" → orbit with target_class = YOLO class name.
  Common YOLO class names: "person", "car", "truck", "motorcycle", "bicycle",
                            "bus", "dog", "cat", "boat", "airplane".
- "Stop / cancel / abort / wait / freeze" → hold.
- Default orbit radius: 20 m. Default speed: 5 m/s.
- If intent is unclear or mission is done → hold.
- Use conversation history to handle follow-up commands like "go back", "repeat",
  "now land", "continue scanning", etc.

═══════════════════════════════════════════
OUTPUT FORMAT — always this exact shape, no markdown:

{
  "thought": "<step-by-step: what user wants, NED maths, which command and why>",
  "action": "<action name>",
  <parameters for that action>
}
"""


class LLMClient:
    """LLM client using Ollama local API with rolling conversation memory."""

    def __init__(
        self,
        model: str = 'qwen2.5:32b',
        ollama_url: str = 'http://localhost:11434',
        max_history_turns: int = 10,
    ):
        self.model = model
        self.ollama_url = ollama_url.rstrip('/')
        self.max_history_turns = max_history_turns
        # Rolling conversation history: list of {'role': ..., 'content': ...} dicts.
        # Does NOT include the system prompt (that is always prepended separately).
        self.history: list = []
        logger.info(
            f'Using Ollama model: {self.model} at {self.ollama_url} '
            f'(memory: {self.max_history_turns} turns)'
        )

    def reset_memory(self):
        """Clear the conversation history (call at the start of a new mission)."""
        self.history = []
        logger.info('LLM memory cleared.')

    def ask(self, drone_state: str, user_command: str, yolo_detections: str) -> dict:
        """Send telemetry + command to Ollama and return parsed JSON command.

        The full conversation history is included in every request so the LLM
        has context from prior exchanges.

        Args:
            drone_state: Formatted string of current telemetry.
            user_command: Natural language mission from the user.
            yolo_detections: JSON string of YOLO detection results.

        Returns:
            dict with an 'action' key and associated parameters.

        Raises:
            ValueError: If the LLM response cannot be parsed as valid JSON.
        """
        user_content = (
            f"USER MISSION: {user_command}\n\n"
            f"CURRENT DRONE STATE:\n{drone_state}\n\n"
            f"YOLO DETECTIONS:\n{yolo_detections}\n\n"
            "Respond with ONLY a JSON command."
        )

        messages = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            *self.history,
            {'role': 'user', 'content': user_content},
        ]

        payload = json.dumps({
            'model': self.model,
            'messages': messages,
            'stream': False,
            'options': {
                'temperature': 0.2,
            },
            'format': 'json',
        }).encode('utf-8')

        text = ''
        try:
            req = urllib.request.Request(
                f'{self.ollama_url}/api/chat',
                data=payload,
                headers={
                    'Content-Type': 'application/json',
                    'ngrok-skip-browser-warning': 'true',
                },
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read().decode('utf-8'))

            text = result['message']['content'].strip()
            # Strip markdown code fences if present
            text = text.replace('```json', '').replace('```', '').strip()

            command = json.loads(text)
            if 'action' not in command:
                raise ValueError(f"LLM response missing 'action' key: {command}")

            # Store user turn and assistant response in history
            self.history.append({'role': 'user', 'content': user_content})
            self.history.append({'role': 'assistant', 'content': text})

            # Trim history to max_history_turns (each turn = 1 user + 1 assistant msg)
            max_msgs = self.max_history_turns * 2
            if len(self.history) > max_msgs:
                self.history = self.history[-max_msgs:]

            logger.info(
                f'LLM command: {command}  '
                f'[memory: {len(self.history) // 2}/{self.max_history_turns} turns]'
            )
            return command

        except json.JSONDecodeError as e:
            logger.error(f'Failed to parse LLM response as JSON: {text!r}')
            raise ValueError(f'LLM returned invalid JSON: {e}') from e
        except Exception as e:
            logger.error(f'LLM API call failed: {e}')
            raise
