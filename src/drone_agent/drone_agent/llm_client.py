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

shape_path        Fly a deterministic named shape generated in code.
                  Use this ONLY for these built-in shapes:
                  circle, square, rectangle, triangle, polygon, star,
                  figure_eight, zigzag, spiral, heart.
                  The controller generates the actual waypoints from these parameters.
                  In NED, "upward / climb / ascend" means z becomes MORE NEGATIVE.
                  "downward / descend" means z becomes LESS NEGATIVE.
  {"action": "shape_path", "shape": "<name>", "radius": <m>, "width": <m>, "height": <m>, "side": <m>, "sides": <int>, "turns": <count>, "climb_m": <m>, "forward_m": <m>, "amplitude_m": <m>, "point_count": <int>, "speed": <m/s>, "closed": <bool>, "loop": <bool>}

path              Fly an arbitrary finite waypoint path in absolute NED coordinates.
                  Use this only when the user explicitly provides or implies custom
                  free-form waypoints rather than a supported built-in shape.
                  For unsupported named shapes such as clover, arrow, crescent,
                  letters, logos, or any other custom outline, output a path
                  request for the lightweight FunctionGemma fallback generator
                  instead of guessing bad waypoints yourself.
  Explicit waypoint form:
  {"action": "path", "points": [[<x>, <y>, <z>], ...], "speed": <m/s>, "closed": <bool>, "loop": <bool>}
  FunctionGemma custom-shape form:
  {"action": "path", "generator": "functiongemma", "shape_prompt": "<original custom shape request>", "size_m": <m>, "point_count": <int>, "speed": <m/s>, "closed": <bool>, "loop": <bool>}

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
- If the user does not explicitly ask to change altitude, preserve the current altitude.
- GPS target: apply GPS→NED formula, then output position_ned.
- "Search / scan / look around / survey" (no specific object) → orbit, no target_class.
- "Find / look for / search for <object>" → orbit with target_class = YOLO class name.
  Common YOLO class names: "person", "car", "truck", "motorcycle", "bicycle",
                            "bus", "dog", "cat", "boat", "airplane".
- Prefer orbit for simple circles and loitering.
- Prefer square_survey for simple square/box patterns.
- Use shape_path only for supported built-in shapes: circle, square, rectangle,
  triangle, polygon, star, figure_eight, zigzag, spiral, heart.
- If the requested shape is NOT in that list, use path with
  `"generator": "functiongemma"` and preserve the original shape request in
  `"shape_prompt"`.
- Do NOT approximate unsupported named shapes with a polygon, circle, or other
  different built-in shape unless the user explicitly asks for an approximation.
- Use raw path only for genuinely free-form routes or explicit waypoint-style requests.
- For shape_path, choose sensible dimensions if the user does not specify them:
  usually 8-20 metres, not tiny 1-metre shapes.
- If the user did not ask to change altitude, keep the shape at the current altitude.
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

MISSION_PLANNER_PROMPT = """Convert a sequenced drone mission into a list of explicit step prompts.

Return JSON only in this exact shape:

{
  "steps":[
    {
      "command":"set altitude to 20 meters",
      "completion":{"type":"altitude_reached","target_m":20.0,"tolerance_m":0.8}
    },
    {
      "command":"fly in a rectangle",
      "completion":{"type":"duration","seconds":60}
    },
    {
      "command":"fly an upward spiral",
      "completion":{"type":"path_complete"}
    },
    {
      "command":"takeoff",
      "completion":{"type":"airborne","min_altitude_m":0.8}
    },
    {
      "command":"face east",
      "completion":{"type":"heading_reached","tolerance_deg":8.0}
    },
    {
      "command":"fly forward 10 meters",
      "completion":{"type":"position_reached","tolerance_m":1.0}
    },
    {
      "command":"search for the bus",
      "completion":{"type":"target_found","target_class":"bus"}
    },
    {
      "command":"approach the bus",
      "completion":{"type":"approach_complete","target_class":"bus"}
    }
  ]
}

Rules:
- Always return a non-empty "steps" list.
- Each step's "command" must be a short standalone prompt that the normal controller can execute by itself.
- Resolve references like "it", "them", and "that one" into explicit command text and explicit target_class values.
- Supported completion types:
  - duration
  - altitude_reached
  - airborne
  - heading_reached
  - position_reached
  - path_complete
  - target_found
  - approach_complete
- For heading-only steps like "face east" or "set heading to 90 degrees", use "heading_reached", not duration 0.
- For one-shot movement steps like "fly forward 10 meters" or "go 20 meters north", use "position_reached", not duration 0.
- For custom finite shapes like spirals, triangles, zigzags, figure-eights, stars, and other arbitrary
  routes, use "path_complete".
- target_class must be one of: person, car, truck, bus, bicycle, motorcycle, dog, cat, boat, airplane.
- Use "takeoff" only when the user explicitly wants takeoff as a separate step. If the next step already sets altitude, you may still keep "takeoff" as its own step with completion type "airborne".
- Do not include explanations or markdown.

Example:
Input: "takeoff, then set the altitude to 2 meters, then fly in a square for 1 minute then hold for 10 seconds, then fly in a circle for 1 minute"
Output:
{
  "steps":[
    {"command":"takeoff","completion":{"type":"airborne","min_altitude_m":0.8}},
    {"command":"set altitude to 2 meters","completion":{"type":"altitude_reached","target_m":2.0,"tolerance_m":0.5}},
    {"command":"fly in a square","completion":{"type":"duration","seconds":60}},
    {"command":"hold for 10 seconds","completion":{"type":"duration","seconds":10}},
    {"command":"fly in a circle","completion":{"type":"duration","seconds":60}}
  ]
}
"""


class LLMClient:
    """LLM client using Ollama local API with rolling conversation memory."""

    def __init__(
        self,
        model: str = 'qwen2.5:32b',
        ollama_url: str = 'http://localhost:11434',
        max_history_turns: int = 10,
        request_timeout_sec: float = 60.0,
        planner_timeout_sec: float = 45.0,
    ):
        self.model = model
        self.ollama_url = ollama_url.rstrip('/')
        self.max_history_turns = max_history_turns
        self.request_timeout_sec = float(request_timeout_sec)
        self.planner_timeout_sec = float(planner_timeout_sec)
        # Rolling conversation history: list of {'role': ..., 'content': ...} dicts.
        # Does NOT include the system prompt (that is always prepended separately).
        self.history: list = []
        logger.info(
            f'Using Ollama model: {self.model} at {self.ollama_url} '
            f'(memory: {self.max_history_turns} turns, '
            f'command timeout: {self.request_timeout_sec:.0f}s, '
            f'planner timeout: {self.planner_timeout_sec:.0f}s)'
        )

    def reset_memory(self):
        """Clear the conversation history (call at the start of a new mission)."""
        self.history = []
        logger.info('LLM memory cleared.')

    def ask(
        self,
        drone_state: str,
        user_command: str,
        yolo_detections: str,
        *,
        include_history: bool = True,
    ) -> dict:
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

        history = self.history if include_history else []

        messages = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            *history,
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
            with urllib.request.urlopen(req, timeout=self.request_timeout_sec) as resp:
                result = json.loads(resp.read().decode('utf-8'))

            text = result['message']['content'].strip()
            # Strip markdown code fences if present
            text = text.replace('```json', '').replace('```', '').strip()

            command = json.loads(text)
            if 'action' not in command:
                raise ValueError(f"LLM response missing 'action' key: {command}")

            if include_history:
                # Store user turn and assistant response in history
                self.history.append({'role': 'user', 'content': user_content})
                self.history.append({'role': 'assistant', 'content': text})

                # Trim history to max_history_turns (each turn = 1 user + 1 assistant msg)
                max_msgs = self.max_history_turns * 2
                if len(self.history) > max_msgs:
                    self.history = self.history[-max_msgs:]

            logger.info(
                f'LLM command: {command}  '
                f'[memory: {len(self.history) // 2}/{self.max_history_turns} turns'
                f'{"; no history used" if not include_history else ""}]'
            )
            return command

        except json.JSONDecodeError as e:
            logger.error(f'Failed to parse LLM response as JSON: {text!r}')
            raise ValueError(f'LLM returned invalid JSON: {e}') from e
        except Exception as e:
            logger.error(f'LLM API call failed: {e}')
            raise

    def plan_mission(self, user_command: str) -> dict:
        """Return a structured list of step prompts for a sequenced mission."""
        messages = [
            {'role': 'system', 'content': MISSION_PLANNER_PROMPT},
            {'role': 'user', 'content': user_command},
        ]

        payload = json.dumps({
            'model': self.model,
            'messages': messages,
            'stream': False,
            'options': {
                'temperature': 0.0,
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
            with urllib.request.urlopen(req, timeout=self.planner_timeout_sec) as resp:
                result = json.loads(resp.read().decode('utf-8'))

            text = result['message']['content'].strip()
            text = text.replace('```json', '').replace('```', '').strip()
            plan = json.loads(text)

            if not isinstance(plan.get('steps'), list) or len(plan['steps']) == 0:
                raise ValueError(f"Planner response missing non-empty 'steps': {plan}")

            logger.info(f'LLM mission planner output: {plan}')
            return plan

        except json.JSONDecodeError as e:
            logger.error(f'Failed to parse planner response as JSON: {text!r}')
            raise ValueError(f'Planner returned invalid JSON: {e}') from e
        except Exception as e:
            logger.error(f'Planner API call failed: {e}')
            raise
