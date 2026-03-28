#!/usr/bin/env python3
"""Brain Node — the main orchestrator for the AI-controlled drone.

This node:
1. Subscribes to PX4 telemetry (odometry, GPS, status, battery)
2. Subscribes to YOLO detections
3. Accepts user natural language commands
4. Calls Gemini LLM for decision-making (throttled, not every frame)
5. Translates LLM commands to PX4 setpoints
6. Publishes offboard control at 10Hz continuously
"""

import json
import math
import re
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from std_msgs.msg import String
from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleOdometry,
    SensorGps,
    VehicleStatus,
    BatteryStatus,
)

from drone_agent.llm_client import LLMClient
from drone_agent.command_translator import CommandTranslator
from drone_agent.shape_generator import is_supported_shape_name
from drone_agent.functiongemma_path_generator import FunctionGemmaPathGenerator


class BrainNode(Node):
    def __init__(self):
        super().__init__('brain_node')

        # Parameters
        self.declare_parameter('llm_interval_sec', 7.0)
        self.declare_parameter('offboard_rate_hz', 30.0)
        self.declare_parameter('max_speed_m_s', 2.0)
        self.declare_parameter('ollama_url', 'http://localhost:11434')
        self.declare_parameter('ollama_model', 'qwen3:4b')
        self.declare_parameter('enable_custom_shape_fallback', True)
        self.declare_parameter('custom_shape_model', 'functiongemma')

        self.llm_interval = self.get_parameter('llm_interval_sec').value
        offboard_rate = self.get_parameter('offboard_rate_hz').value
        max_speed_m_s = self.get_parameter('max_speed_m_s').value
        ollama_url = self.get_parameter('ollama_url').value
        ollama_model = self.get_parameter('ollama_model').value
        self.enable_custom_shape_fallback = bool(
            self.get_parameter('enable_custom_shape_fallback').value
        )
        self.custom_shape_model = self.get_parameter('custom_shape_model').value

        # Initialize components
        self.llm = LLMClient(model=ollama_model, ollama_url=ollama_url)
        self.translator = CommandTranslator(max_speed_m_s=max_speed_m_s)
        self.custom_shape_generator = None

        # State
        self.current_command = ''
        self.latest_detections = '[]'
        self.last_nonempty_detections = []
        self.last_nonempty_detection_time = 0.0
        self.odometry = None
        self.gps = None
        self.vehicle_status = None
        self.battery = None
        self.last_llm_call = 0.0
        self.last_detection_classes = set()
        self.armed = False
        self.offboard_active = False
        self.llm_inflight = False
        self.llm_replan_pending = False
        self.pending_llm_result = None
        self.pending_llm_kind = None
        self.pending_llm_context = None
        self.pending_llm_error = None
        self.llm_request_id = 0
        self.llm_lock = threading.Lock()
        # Object to search for during orbit (set from LLM orbit target_class field)
        self.search_target: str | None = None
        self.visual_target_class: str | None = None
        self.visual_follow_active = False
        self.visual_search_active = False
        self.visual_target_speed_m_s = 0.3
        self.visual_last_seen_time = 0.0
        self.visual_last_update_time = 0.0
        self.visual_lost_timeout_sec = 1.5
        self.visual_reacquire_timeout_sec = 6.0
        self.visual_recent_detection_ttl_sec = 1.0
        self.visual_horizontal_fov_rad = math.radians(90.0)
        self.visual_target_cy = 0.52
        self.visual_deadband_x = 0.06
        self.visual_deadband_y = 0.10
        self.visual_bbox_alpha = 0.35
        self.visual_filtered_cx = 0.5
        self.visual_filtered_cy = 0.5
        self.visual_filtered_area = 0.0
        self.visual_min_lookahead_m = 0.25
        self.visual_max_lookahead_m = 0.8
        self.visual_climb_gain_m = 0.10
        self.visual_descend_gain_m = 0.30
        self.visual_max_climb_step_m = 0.02
        self.visual_max_descend_step_m = 0.08
        self.visual_bottom_slowdown_cy = 0.72
        self.visual_bottom_stop_cy = 0.86
        self.visual_forward_camera_standoff_scale = 1.35
        self.visual_search_center_x = 0.0
        self.visual_search_center_y = 0.0
        self.visual_search_radius_m = 1.6
        self.visual_search_radius_max_m = 4.0
        self.visual_search_expand_rate_m_s = 0.18
        self.visual_search_angle_rad = 0.0
        self.visual_search_direction = 1.0
        self.visual_search_speed_m_s = 0.35
        self.visual_search_target_z = -5.0
        self.visual_reacquire_hits = 0
        self.visual_reacquire_required_hits = 3
        self.visual_reacquire_last_hit_time = 0.0
        self.visual_reacquire_hit_window_sec = 0.6
        self.visual_lost_reported = False
        self.initial_hold_target_set = False
        self.mission_active = False
        self.mission_steps: list[dict] = []
        self.mission_step_index = -1
        self.mission_current_step: dict | None = None
        self.mission_step_started_at = 0.0
        self.mission_step_deadline = 0.0
        self.mission_step_success_since = 0.0
        self.mission_search_timeout_sec = 90.0
        self.mission_approach_timeout_sec = 300.0
        self.mission_approach_stall_timeout_sec = 45.0
        self.mission_path_timeout_sec = 180.0
        self.mission_last_target_class: str | None = None

        # QoS for PX4 topics (best-effort, volatile — matches PX4 uXRCE-DDS)
        px4_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # --- Subscribers ---

        # PX4 telemetry
        self.create_subscription(
            VehicleOdometry, '/fmu/out/vehicle_odometry',
            self._odom_cb, px4_qos,
        )
        self.create_subscription(
            SensorGps, '/fmu/out/vehicle_gps_position',
            self._gps_cb, px4_qos,
        )
        self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status',
            self._status_cb, px4_qos,
        )
        self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status_v1',
            self._status_cb, px4_qos,
        )
        self.create_subscription(
            BatteryStatus, '/fmu/out/battery_status',
            self._battery_cb, px4_qos,
        )

        # YOLO detections
        self.create_subscription(
            String, '/yolo/detections',
            self._yolo_cb, 10,
        )

        # User commands
        self.create_subscription(
            String, '/user_command',
            self._user_cmd_cb, 10,
        )

        # --- Publishers ---
        self.offboard_pub = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', 10,
        )
        self.setpoint_pub = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', 10,
        )
        self.command_pub = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', 10,
        )

        # --- Timers ---

        # 10Hz offboard control loop (MUST be continuous for PX4 OFFBOARD mode)
        period = 1.0 / offboard_rate
        self.create_timer(period, self._offboard_loop)

        # LLM decision timer
        self.create_timer(self.llm_interval, self._llm_decision_tick)
        self.create_timer(0.1, self._llm_result_tick)

        speed_cap_text = (
            f' Max translational speed cap: {max_speed_m_s:.2f} m/s.'
            if max_speed_m_s > 0.0 else
            ''
        )
        self.get_logger().info(
            f'Brain node started. Waiting for user command on /user_command ...{speed_cap_text}'
        )

    def _target_aliases(self) -> dict[str, tuple[str, ...]]:
        """Mapping from canonical YOLO classes to common prompt aliases."""
        return {
            'person': ('person', 'people', 'human'),
            'car': ('car', 'cars'),
            'truck': ('truck', 'trucks'),
            'bus': ('bus', 'buses'),
            'bicycle': ('bicycle', 'bike', 'bikes'),
            'motorcycle': ('motorcycle', 'motorbike'),
            'dog': ('dog', 'dogs'),
            'cat': ('cat', 'cats'),
            'boat': ('boat', 'boats'),
            'airplane': ('airplane', 'plane', 'planes'),
        }

    def _infer_target_class(self, command: str) -> str | None:
        """Infer a YOLO class from free text regardless of mission verb."""
        text = command.lower()
        for target_class, words in self._target_aliases().items():
            if any(word in text for word in words):
                return target_class
        return None

    def _canonicalize_target_class(self, value: str | None) -> str | None:
        """Normalize a planner- or user-provided class name to one canonical YOLO class."""
        if value is None:
            return None
        text = str(value).strip().lower()
        if not text:
            return None

        for target_class, words in self._target_aliases().items():
            if text == target_class or text in words:
                return target_class
        return None

    def _infer_search_target(self, command: str) -> str | None:
        """Infer a YOLO class from a natural-language search command."""
        text = command.lower()

        search_words = ('search', 'find', 'look for', 'scan')
        if not any(word in text for word in search_words):
            return None

        return self._infer_target_class(text)

    def _command_should_replan_on_detection(self, command: str) -> bool:
        """Whether generic YOLO class changes should trigger a new LLM call."""
        text = command.lower()
        if not text.strip():
            return False
        search_words = ('search', 'find', 'look for', 'scan')
        return any(word in text for word in search_words)

    def _extract_speed_m_s(self, command: str, default: float = 0.75) -> float:
        """Extract a requested speed from free text or map coarse words to values."""
        text = command.lower()
        match = re.search(
            r'(\d+(?:\.\d+)?)\s*(?:m/s|meter(?:s)? per second|metre(?:s)? per second)\b',
            text,
        )
        if match:
            return max(0.1, float(match.group(1)))
        if 'very slow' in text:
            return 0.15
        if 'slowly' in text or 'slow' in text:
            return 0.25
        if 'quickly' in text or 'fast' in text:
            return 1.2
        return default

    def _extract_heading_deg(self, command: str) -> float | None:
        """Extract a requested heading from free text."""
        text = command.lower()

        match = re.search(r'(\d+(?:\.\d+)?)\s*degrees?\b', text)
        if match:
            return float(match.group(1)) % 360.0

        cardinals = {
            'north': 0.0,
            'east': 90.0,
            'south': 180.0,
            'west': 270.0,
        }
        for word, heading_deg in cardinals.items():
            if re.search(rf'\b{word}\b', text):
                return heading_deg

        return None

    def _extract_duration_sec(self, command: str) -> float | None:
        """Extract a duration from free text, supporting seconds and minutes."""
        text = command.lower()
        minute_match = re.search(
            r'\bfor\s+(\d+(?:\.\d+)?)\s*(minutes?|mins?)\b',
            text,
        )
        if minute_match:
            return max(0.5, float(minute_match.group(1)) * 60.0)

        second_match = re.search(
            r'\bfor\s+(\d+(?:\.\d+)?)\s*(seconds?|secs?|s)\b',
            text,
        )
        if second_match:
            return max(0.5, float(second_match.group(1)))

        return None

    def _looks_like_custom_shape_request(self, command: str) -> bool:
        """Heuristic: whether the prompt is asking for a geometric custom path."""
        text = command.lower()
        if any(word in text for word in ('search', 'find', 'approach', 'follow', 'track')):
            return False
        shape_markers = (
            ' shape',
            'draw ',
            'outline ',
            'pattern ',
            'spiral',
            'star',
            'heart',
            'clover',
            'crescent',
            'diamond',
            'figure eight',
            'figure-eight',
            'zigzag',
            'triangle',
            'rectangle',
            'square',
            'polygon',
        )
        return any(marker in text for marker in shape_markers)

    def _custom_shape_size_m(self, cmd: dict) -> float:
        """Choose a target size for custom generated shapes."""
        for key in ('size_m', 'width', 'width_m', 'side', 'side_m'):
            value = cmd.get(key)
            if value is not None:
                return max(2.0, float(value))
        radius = cmd.get('radius', cmd.get('radius_m'))
        if radius is not None:
            return max(2.0, float(radius) * 2.0)

        match = re.search(
            r'(\d+(?:\.\d+)?)\s*(?:m|meter|meters|metre|metres)\b',
            self.current_command.lower(),
        )
        if match:
            return max(2.0, float(match.group(1)))
        return 12.0

    def _ensure_custom_shape_generator(self) -> FunctionGemmaPathGenerator:
        """Create the lazy FunctionGemma wrapper on first use."""
        if self.custom_shape_generator is None:
            ollama_url = self.get_parameter('ollama_url').value
            self.custom_shape_generator = FunctionGemmaPathGenerator(
                model_name=self.custom_shape_model,
                ollama_url=ollama_url,
            )
        return self.custom_shape_generator

    def _strip_duration_phrase(self, command: str) -> str:
        """Remove a trailing 'for N seconds/minutes' phrase from a clause."""
        stripped = re.sub(
            r'\bfor\s+\d+(?:\.\d+)?\s*(?:minutes?|mins?|seconds?|secs?|s)\b',
            '',
            command,
            flags=re.IGNORECASE,
        )
        return re.sub(r'\s+', ' ', stripped).strip(' ,')

    def _make_command_step(
        self,
        command_text: str,
        completion: dict,
        *,
        label: str | None = None,
        timeout_sec: float | None = None,
    ) -> dict:
        """Create one normalized queued mission step."""
        step = {
            'kind': 'command',
            'command': command_text.strip(),
            'completion': completion,
        }
        if label:
            step['label'] = label
        if timeout_sec is not None:
            step['timeout_sec'] = float(timeout_sec)
        return step

    def _looks_like_multi_step_mission(self, command: str) -> bool:
        """Detect explicit sequencing words that should trigger the mission queue parser."""
        return bool(re.search(r'\bthen\b|\bafter that\b|\bnext\b|\bfinally\b|[,;]', command.lower()))

    def _parse_mission_clause(self, clause: str, last_target_class: str | None) -> dict | None:
        """Parse one supported mission clause into a reusable command step."""
        text = clause.strip().lower()
        text = re.sub(r'^(?:and|then)\s+', '', text).strip()
        if not text:
            return None

        duration_sec = self._extract_duration_sec(text)
        bare_text = self._strip_duration_phrase(text) or text

        if re.fullmatch(r'(?:take off|takeoff)', bare_text):
            return self._make_command_step(
                'takeoff',
                {'type': 'airborne', 'min_altitude_m': 0.8},
                label='takeoff',
                timeout_sec=20.0,
            )

        hold_match = re.fullmatch(
            r'(?:hold|hover|wait)(?:\s+for)?\s+(\d+(?:\.\d+)?)\s*(?:seconds?|secs?|s)',
            text,
        )
        if hold_match:
            duration_sec = max(0.5, float(hold_match.group(1)))
            return self._make_command_step(
                'hold',
                {'type': 'duration', 'seconds': duration_sec},
                label=f'hold for {duration_sec:g} seconds',
                timeout_sec=duration_sec + 5.0,
            )

        altitude_m = self._extract_altitude_m(text, default=10.0)
        altitude_words = (
            'set altitude',
            'hover at',
            'take off',
            'takeoff',
            'climb to',
            'descend to',
            'go to',
            'rise to',
        )
        if any(word in text for word in altitude_words):
            tolerance_m = 0.5 if altitude_m <= 3.0 else 0.8
            return self._make_command_step(
                bare_text,
                {
                    'type': 'altitude_reached',
                    'target_m': altitude_m,
                    'tolerance_m': tolerance_m,
                },
                label=bare_text,
                timeout_sec=max(30.0, altitude_m * 6.0),
            )

        heading_words = ('set heading', 'face ', 'turn to', 'rotate to', 'point east', 'point west', 'point north', 'point south')
        heading_deg = self._extract_heading_deg(text)
        if heading_deg is not None and any(word in text for word in heading_words):
            return self._make_command_step(
                bare_text,
                {
                    'type': 'heading_reached',
                    'heading_deg': heading_deg,
                    'tolerance_deg': 8.0,
                },
                label=bare_text,
                timeout_sec=20.0,
            )

        move_words = ('fly ', 'go ', 'move ', 'move forward', 'move backward', 'move back')
        if any(word in text for word in move_words):
            distance_match = re.search(
                r'(\d+(?:\.\d+)?)\s*(?:m|meters?|metres?)\b',
                text,
            )
            if distance_match:
                return self._make_command_step(
                    bare_text,
                    {
                        'type': 'position_reached',
                        'tolerance_m': 1.0,
                    },
                    label=bare_text,
                    timeout_sec=max(30.0, float(distance_match.group(1)) * 6.0),
                )

        search_words = ('search', 'find', 'look for', 'scan')
        if any(word in text for word in search_words):
            target_class = self._infer_target_class(text)
            if target_class is None:
                return None
            return self._make_command_step(
                f'search for the {target_class}',
                {
                    'type': 'target_found',
                    'target_class': target_class,
                },
                label=f'search {target_class}',
                timeout_sec=self.mission_search_timeout_sec,
            )

        approach_words = (
            'approach',
            'move toward',
            'move towards',
            'toward',
            'towards',
            'follow',
            'track',
        )
        if any(word in text for word in approach_words):
            target_class = self._infer_target_class(text)
            if target_class is None and re.search(r'\b(it|them|him|her)\b', text):
                target_class = last_target_class
            if target_class is None:
                return None
            speed_m_s = self._extract_speed_m_s(text, default=0.75)
            return self._make_command_step(
                f'approach the {target_class} at {speed_m_s:g} metres per second',
                {
                    'type': 'approach_complete',
                    'target_class': target_class,
                },
                label=f'approach {target_class}',
                timeout_sec=self.mission_approach_timeout_sec,
            )

        pattern_words = ('rectangle', 'square', 'circle', 'orbit')
        custom_path_words = (
            'spiral',
            'triangle',
            'pentagon',
            'hexagon',
            'octagon',
            'star',
            'zigzag',
            'figure eight',
            'figure-eight',
            'lemniscate',
            'diamond',
            'heart',
            'wave',
            'arc',
        )
        if duration_sec is not None and any(word in text for word in pattern_words + custom_path_words):
            return self._make_command_step(
                bare_text,
                {'type': 'duration', 'seconds': duration_sec},
                label=f'{bare_text} for {duration_sec:g} seconds',
                timeout_sec=duration_sec + 10.0,
            )

        if any(word in text for word in custom_path_words):
            return self._make_command_step(
                bare_text,
                {'type': 'path_complete'},
                label=bare_text,
                timeout_sec=self.mission_path_timeout_sec,
            )

        return None

    def _parse_multi_step_mission(self, command: str) -> list[dict] | None:
        """Parse a supported multi-step mission prompt into a deterministic step queue."""
        if not self._looks_like_multi_step_mission(command):
            return None

        text = command.lower()
        text = re.sub(r'\band then\b', ' then ', text)
        text = re.sub(r'\bafter that\b', ' then ', text)
        text = re.sub(r'\bfinally\b', ' then ', text)
        text = re.sub(r'\bnext\b', ' then ', text)

        coarse_clauses = [clause.strip() for clause in re.split(r'\bthen\b|[,;]', text) if clause.strip()]
        clauses: list[str] = []
        for clause in coarse_clauses:
            clauses.extend(
                part.strip() for part in re.split(r'\band\b', clause) if part.strip()
            )

        if len(clauses) < 2:
            return None

        steps = []
        last_target_class = None
        for clause in clauses:
            step = self._parse_mission_clause(clause, last_target_class)
            if step is None:
                return None
            steps.append(step)
            if step.get('target_class') is not None:
                last_target_class = step['target_class']

        return steps if len(steps) >= 2 else None

    def _normalize_planned_steps(self, plan: dict) -> list[dict] | None:
        """Validate and normalize planner output into the internal mission-step schema."""
        if not isinstance(plan, dict):
            return None

        steps_raw = plan.get('steps')
        if not isinstance(steps_raw, list):
            return None

        steps: list[dict] = []
        last_target_class = None
        for step in steps_raw:
            if not isinstance(step, dict):
                return None

            command_text = step.get('command')
            completion = step.get('completion')
            if command_text is not None and isinstance(completion, dict):
                command_text_str = str(command_text).strip()
                completion_type = str(completion.get('type', '')).strip().lower()
                timeout_sec = step.get('timeout_sec')
                label = step.get('label')

                if completion_type == 'duration':
                    duration_value = completion.get('seconds', completion.get('duration_sec'))
                    try:
                        duration_sec = max(0.5, float(duration_value))
                    except (TypeError, ValueError):
                        return None
                    if float(duration_value) <= 0.0:
                        inferred_step = self._parse_mission_clause(command_text_str, last_target_class)
                        if inferred_step is not None and inferred_step.get('completion', {}).get('type') != 'duration':
                            steps.append(inferred_step)
                            inferred_target_class = inferred_step.get('completion', {}).get('target_class')
                            if inferred_target_class is not None:
                                last_target_class = inferred_target_class
                            continue
                    if re.match(r'^(hold|hover|wait)\b', command_text_str.lower()):
                        command_text_str = 'hold'
                    steps.append(self._make_command_step(
                        command_text_str,
                        {'type': 'duration', 'seconds': duration_sec},
                        label=label or (
                            str(command_text)
                            if re.search(r'\bfor\s+\d', str(command_text).lower())
                            else f'{command_text_str} for {duration_sec:g} seconds'
                        ),
                        timeout_sec=(
                            float(timeout_sec) if timeout_sec is not None else duration_sec + 10.0
                        ),
                    ))
                    continue

                if completion_type == 'altitude_reached':
                    try:
                        target_m = max(1.0, float(completion.get('target_m')))
                        tolerance_m = max(0.2, float(completion.get('tolerance_m', 0.7)))
                    except (TypeError, ValueError):
                        return None
                    steps.append(self._make_command_step(
                        str(command_text),
                        {
                            'type': 'altitude_reached',
                            'target_m': target_m,
                            'tolerance_m': tolerance_m,
                        },
                        label=label or str(command_text),
                        timeout_sec=(
                            float(timeout_sec) if timeout_sec is not None else max(30.0, target_m * 6.0)
                        ),
                    ))
                    continue

                if completion_type == 'airborne':
                    try:
                        min_altitude_m = max(0.3, float(completion.get('min_altitude_m', 0.8)))
                    except (TypeError, ValueError):
                        return None
                    steps.append(self._make_command_step(
                        str(command_text),
                        {
                            'type': 'airborne',
                            'min_altitude_m': min_altitude_m,
                        },
                        label=label or str(command_text),
                        timeout_sec=(
                            float(timeout_sec) if timeout_sec is not None else 20.0
                        ),
                    ))
                    continue

                if completion_type == 'heading_reached':
                    try:
                        heading_deg = float(completion.get('heading_deg', 0.0)) % 360.0
                        tolerance_deg = max(1.0, float(completion.get('tolerance_deg', 8.0)))
                    except (TypeError, ValueError):
                        return None
                    steps.append(self._make_command_step(
                        str(command_text),
                        {
                            'type': 'heading_reached',
                            'heading_deg': heading_deg,
                            'tolerance_deg': tolerance_deg,
                        },
                        label=label or str(command_text),
                        timeout_sec=(
                            float(timeout_sec) if timeout_sec is not None else 20.0
                        ),
                    ))
                    continue

                if completion_type == 'position_reached':
                    try:
                        tolerance_m = max(0.2, float(completion.get('tolerance_m', 1.0)))
                    except (TypeError, ValueError):
                        return None
                    steps.append(self._make_command_step(
                        str(command_text),
                        {
                            'type': 'position_reached',
                            'tolerance_m': tolerance_m,
                        },
                        label=label or str(command_text),
                        timeout_sec=(
                            float(timeout_sec) if timeout_sec is not None else 60.0
                        ),
                    ))
                    continue

                if completion_type == 'path_complete':
                    steps.append(self._make_command_step(
                        str(command_text),
                        {'type': 'path_complete'},
                        label=label or str(command_text),
                        timeout_sec=(
                            float(timeout_sec) if timeout_sec is not None else self.mission_path_timeout_sec
                        ),
                    ))
                    continue

                if completion_type in ('target_found', 'approach_complete'):
                    target_class = self._canonicalize_target_class(completion.get('target_class'))
                    if target_class is None:
                        target_class = last_target_class
                    if target_class is None:
                        return None
                    steps.append(self._make_command_step(
                        str(command_text),
                        {
                            'type': completion_type,
                            'target_class': target_class,
                        },
                        label=label or (
                            f'search {target_class}'
                            if completion_type == 'target_found'
                            else f'approach {target_class}'
                        ),
                        timeout_sec=(
                            float(timeout_sec) if timeout_sec is not None else (
                                self.mission_search_timeout_sec
                                if completion_type == 'target_found'
                                else self.mission_approach_timeout_sec
                            )
                        ),
                    ))
                    last_target_class = target_class
                    continue

                return None

            raw_type = str(step.get('type', '')).strip().lower()
            if raw_type in ('search', 'find', 'scan', 'look_for', 'look-for'):
                step_type = 'target_found'
            elif raw_type in (
                'approach',
                'follow',
                'track',
                'move_toward',
                'move_towards',
                'move-toward',
                'move-towards',
            ):
                step_type = 'approach_complete'
            elif raw_type in ('hold', 'hover', 'wait'):
                step_type = 'duration'
            else:
                return None

            if step_type == 'duration':
                duration_value = step.get('duration_sec', step.get('duration'))
                try:
                    duration_sec = max(0.5, float(duration_value))
                except (TypeError, ValueError):
                    return None
                steps.append(self._make_command_step(
                    'hold',
                    {'type': 'duration', 'seconds': duration_sec},
                    label=f'hold for {duration_sec:g} seconds',
                    timeout_sec=duration_sec + 5.0,
                ))
                continue

            target_class = self._canonicalize_target_class(step.get('target_class'))
            if target_class is None and step_type == 'approach_complete':
                target_class = last_target_class
            if target_class is None:
                return None

            default_speed = 1.0 if step_type == 'target_found' else 0.75
            speed_value = step.get('speed_m_s')
            try:
                speed_m_s = (
                    default_speed if speed_value is None else max(0.1, float(speed_value))
                )
            except (TypeError, ValueError):
                return None

            steps.append(self._make_command_step(
                (
                    f'search for the {target_class}'
                    if step_type == 'target_found'
                    else f'approach the {target_class} at {speed_m_s:g} metres per second'
                ),
                {
                    'type': step_type,
                    'target_class': target_class,
                },
                label=(
                    f'search {target_class}'
                    if step_type == 'target_found'
                    else f'approach {target_class}'
                ),
                timeout_sec=(
                    self.mission_search_timeout_sec
                    if step_type == 'target_found'
                    else self.mission_approach_timeout_sec
                ),
            ))
            last_target_class = target_class

        return steps if len(steps) >= 2 else None

    def _mission_step_label(self, step: dict) -> str:
        """Human-readable label for logs."""
        if step.get('label'):
            return str(step['label'])
        if step.get('command'):
            return str(step['command'])
        step_type = step.get('type', 'unknown')
        return str(step_type)

    def _clear_mission_queue(self):
        """Reset all queued-mission state."""
        self.mission_active = False
        self.mission_steps = []
        self.mission_step_index = -1
        self.mission_current_step = None
        self.mission_step_started_at = 0.0
        self.mission_step_deadline = 0.0
        self.mission_step_success_since = 0.0
        self.mission_last_target_class = None

    def _abort_mission(self, reason: str):
        """Abort the active queued mission and hold position safely."""
        if self.mission_current_step is not None:
            self.get_logger().warning(
                f'Aborting multi-step mission during "{self._mission_step_label(self.mission_current_step)}": {reason}'
            )
        else:
            self.get_logger().warning(f'Aborting multi-step mission: {reason}')
        self._freeze_at_current_pose('Mission aborted')
        self.current_command = ''
        self._clear_mission_queue()

    def _activate_mission_search_step(self, step: dict):
        """Run a deterministic local search orbit for one queued mission step."""
        target_class = step['target_class']
        speed_m_s = float(step.get('speed_m_s', 1.0))

        try:
            detections = json.loads(self.latest_detections)
        except json.JSONDecodeError:
            detections = []
        target_obj = self._best_detection(detections, target_class)
        if target_obj is None:
            age = time.time() - self.last_nonempty_detection_time
            if age <= self.visual_recent_detection_ttl_sec:
                detections = list(self.last_nonempty_detections)
                target_obj = self._best_detection(detections, target_class)

        if target_obj is not None:
            self.get_logger().info(
                f'Mission search step already sees "{target_class}" in frame; advancing immediately'
            )
            self._complete_current_mission_step(f'Found "{target_class}" immediately')
            return

        self._cancel_pending_llm()
        self.visual_follow_active = False
        self.visual_search_active = False
        self.visual_target_class = None
        self.visual_lost_reported = False
        self.translator.translation_speed_cap_override_m_s = 0.0
        self.visual_reacquire_hits = 0
        self.visual_reacquire_last_hit_time = 0.0

        if self.odometry is not None:
            center_x = float(self.odometry.position[0])
            center_y = float(self.odometry.position[1])
            current_z = float(self.odometry.position[2])
        else:
            center_x = float(self.translator.target_x)
            center_y = float(self.translator.target_y)
            current_z = float(self._current_target_altitude_z())

        search_alt_z = current_z if current_z <= -2.0 else -5.0
        search_speed = speed_m_s
        if self.translator.max_speed_m_s > 0.0:
            search_speed = min(search_speed, self.translator.max_speed_m_s)

        self.translator.orbiting = True
        self.translator.square_active = False
        self.translator.orbit_center_x = center_x
        self.translator.orbit_center_y = center_y
        self.translator.orbit_alt_z = search_alt_z
        self.translator.orbit_radius = 6.0
        self.translator.orbit_speed = max(0.2, search_speed)
        self.translator.orbit_angle = 0.0
        self.search_target = target_class
        self.current_command = ''
        self.get_logger().info(
            f'Mission search active: target="{target_class}", center=({center_x:.1f}, {center_y:.1f}), '
            f'altitude={abs(search_alt_z):.1f} m, radius=6.0 m, speed={self.translator.orbit_speed:.2f} m/s'
        )

    def _execute_mission_command_step(self, step: dict):
        """Execute one queued command step through the repo's normal dispatch path."""
        command_text = str(step.get('command', '')).strip()
        completion = step.get('completion', {})
        completion_type = completion.get('type')

        if not command_text:
            self._abort_mission('mission step is missing its command text')
            return

        if completion_type == 'duration' and re.match(r'^(hold|hover|wait)\b', command_text.lower()):
            duration_sec = float(completion.get('seconds', 0.0))
            self._cancel_pending_llm()
            self._freeze_at_current_pose(f'Mission hold for {duration_sec:g} seconds')
            self.current_command = ''
            self.mission_step_deadline = time.time() + duration_sec
            return

        self.current_command = command_text
        self.get_logger().info(f'Executing mission command step: "{command_text}"')
        self.mission_step_deadline = 0.0
        self._dispatch_user_command(
            command_text,
            allow_mission_planner=False,
            force_llm=True,
            include_history=False,
        )

        if completion_type == 'duration' and not self.llm_inflight:
            duration_sec = float(completion.get('seconds', 0.0))
            self.mission_step_started_at = time.time()
            self.mission_step_deadline = self.mission_step_started_at + duration_sec

    def _start_next_mission_step(self):
        """Advance to the next queued step and activate its controller."""
        self.mission_step_index += 1
        if self.mission_step_index >= len(self.mission_steps):
            self.get_logger().info('Multi-step mission complete.')
            self._freeze_at_current_pose('Multi-step mission complete')
            self.current_command = ''
            self._clear_mission_queue()
            return

        step = self.mission_steps[self.mission_step_index]
        self.mission_current_step = step
        self.mission_step_started_at = time.time()
        self.mission_step_deadline = 0.0
        self.mission_step_success_since = 0.0
        step['_retry_count'] = 0
        step['_last_progress_time'] = self.mission_step_started_at
        step['_best_progress_area'] = 0.0
        completion = step.get('completion', {})
        target_class = completion.get('target_class')
        if target_class is not None:
            self.mission_last_target_class = target_class

        self.get_logger().info(
            f'Mission step {self.mission_step_index + 1}/{len(self.mission_steps)}: '
            f'{self._mission_step_label(step)}'
        )
        self._execute_mission_command_step(step)

    def _complete_current_mission_step(self, reason: str):
        """Mark the active queued mission step complete and launch the next one."""
        if not self.mission_active or self.mission_current_step is None:
            return
        self.get_logger().info(
            f'Mission step {self.mission_step_index + 1}/{len(self.mission_steps)} complete: '
            f'{self._mission_step_label(self.mission_current_step)} ({reason})'
        )
        self.mission_current_step = None
        self.mission_step_started_at = 0.0
        self.mission_step_deadline = 0.0
        self.mission_step_success_since = 0.0
        self._start_next_mission_step()

    def _start_multi_step_mission(self, steps: list[dict]):
        """Start a deterministic multi-step mission queue."""
        self._cancel_pending_llm()
        self._clear_mission_queue()
        self.mission_active = True
        self.mission_steps = list(steps)
        self.current_command = ''
        self.get_logger().info(f'Starting multi-step mission with {len(steps)} steps:')
        for index, step in enumerate(steps, start=1):
            self.get_logger().info(f'  {index}. {self._mission_step_label(step)}')
        self._start_next_mission_step()

    def _mission_tick(self):
        """Monitor completion / failure conditions for an active queued mission."""
        if not self.mission_active or self.mission_current_step is None:
            return

        now = time.time()
        elapsed = now - self.mission_step_started_at
        step = self.mission_current_step
        completion = step.get('completion', {})
        completion_type = completion.get('type')
        timeout_sec = float(step.get('timeout_sec', 0.0))

        if completion_type == 'duration':
            if self.mission_step_deadline > 0.0 and now >= self.mission_step_deadline:
                self._complete_current_mission_step('duration elapsed')

        if completion_type not in ('approach_complete', 'duration') and timeout_sec > 0.0 and elapsed > timeout_sec:
            self._abort_mission(
                f'"{self._mission_step_label(step)}" timed out after {elapsed:.0f} seconds'
            )
            return

        if completion_type == 'duration':
            return

        if completion_type == 'airborne':
            min_altitude_m = float(completion.get('min_altitude_m', 0.8))
            if self.odometry is not None:
                current_alt_m = abs(float(self.odometry.position[2]))
            else:
                current_alt_m = abs(float(self._current_target_altitude_z()))
            if self.armed and current_alt_m >= min_altitude_m:
                if self.mission_step_success_since == 0.0:
                    self.mission_step_success_since = now
                elif now - self.mission_step_success_since >= 0.4:
                    self._complete_current_mission_step(
                        f'airborne above {min_altitude_m:.1f} m'
                    )
            else:
                self.mission_step_success_since = 0.0
            return

        if completion_type == 'altitude_reached':
            if self.odometry is not None:
                current_alt_m = abs(float(self.odometry.position[2]))
            else:
                current_alt_m = abs(float(self._current_target_altitude_z()))
            target_alt_m = float(completion.get('target_m', current_alt_m))
            tolerance_m = float(completion.get('tolerance_m', 0.7))
            if abs(current_alt_m - target_alt_m) <= tolerance_m:
                if self.mission_step_success_since == 0.0:
                    self.mission_step_success_since = now
                elif now - self.mission_step_success_since >= 0.7:
                    self._complete_current_mission_step(
                        f'altitude reached {target_alt_m:.1f} m'
                    )
            else:
                self.mission_step_success_since = 0.0
            return

        if completion_type == 'heading_reached':
            current_yaw = self._current_yaw_rad()
            target_yaw = self.translator.target_yaw
            if math.isnan(current_yaw) or math.isnan(target_yaw):
                self.mission_step_success_since = 0.0
                return

            tolerance_deg = float(completion.get('tolerance_deg', 8.0))
            angle_error_deg = abs(
                math.degrees(math.atan2(math.sin(current_yaw - target_yaw), math.cos(current_yaw - target_yaw)))
            )
            if angle_error_deg <= tolerance_deg:
                if self.mission_step_success_since == 0.0:
                    self.mission_step_success_since = now
                elif now - self.mission_step_success_since >= 0.4:
                    self._complete_current_mission_step(
                        f'heading reached within {tolerance_deg:.0f} degrees'
                    )
            else:
                self.mission_step_success_since = 0.0
            return

        if completion_type == 'position_reached':
            if self.odometry is None:
                return
            tolerance_m = float(completion.get('tolerance_m', 1.0))
            pos = self.odometry.position
            dx = float(pos[0]) - float(self.translator.target_x)
            dy = float(pos[1]) - float(self.translator.target_y)
            dz = float(pos[2]) - float(self.translator.target_z)
            distance_m = math.sqrt(dx * dx + dy * dy + dz * dz)
            if distance_m <= tolerance_m:
                if self.mission_step_success_since == 0.0:
                    self.mission_step_success_since = now
                elif now - self.mission_step_success_since >= 0.5:
                    self._complete_current_mission_step(
                        f'position reached within {tolerance_m:.1f} m'
                    )
            else:
                self.mission_step_success_since = 0.0
            return

        if completion_type == 'path_complete':
            path_sequence_id = int(step.get('_path_sequence_id', -1))
            if (
                path_sequence_id >= 0 and
                not self.translator.path_active and
                self.translator.path_completed_sequence_id == path_sequence_id
            ):
                if self.mission_step_success_since == 0.0:
                    self.mission_step_success_since = now
                elif now - self.mission_step_success_since >= 0.3:
                    self._complete_current_mission_step('path complete')
            else:
                self.mission_step_success_since = 0.0
            return

        if completion_type == 'target_found':
            return

        if completion_type != 'approach_complete':
            return

        target_class = completion.get('target_class')
        if not self.visual_follow_active or self.visual_target_class != target_class:
            self._abort_mission(f'visual follow for "{target_class}" is no longer active')
            return

        if self.visual_search_active:
            self.mission_step_success_since = 0.0
            return

        recent_visible = (now - self.visual_last_seen_time) <= self.visual_recent_detection_ttl_sec
        stop_area = self._visual_stop_area_for_class(target_class)
        if self.visual_filtered_cy >= self.visual_bottom_slowdown_cy:
            stop_area *= self.visual_forward_camera_standoff_scale

        best_progress_area = float(step.get('_best_progress_area', 0.0))
        progress_epsilon = max(0.0005, best_progress_area * 0.05)
        if recent_visible and self.visual_filtered_area >= best_progress_area + progress_epsilon:
            step['_best_progress_area'] = float(self.visual_filtered_area)
            step['_last_progress_time'] = now

        if recent_visible and self.visual_filtered_area >= stop_area:
            if self.mission_step_success_since == 0.0:
                self.mission_step_success_since = now
            elif now - self.mission_step_success_since >= 0.5:
                self._complete_current_mission_step(
                    f'{target_class} reached visual standoff'
                )
        else:
            self.mission_step_success_since = 0.0

        last_progress_time = float(step.get('_last_progress_time', self.mission_step_started_at))
        stall_timeout_sec = self.mission_approach_stall_timeout_sec
        if timeout_sec > 0.0 and elapsed > timeout_sec and (now - last_progress_time) > stall_timeout_sec:
            self._abort_mission(
                f'"{self._mission_step_label(step)}" stalled for {now - last_progress_time:.0f} '
                f'seconds after {elapsed:.0f} seconds of approach'
            )

    def _visual_stop_area_for_class(self, target_class: str) -> float:
        """Area threshold at which the target is considered close enough."""
        stop_areas = {
            'person': 0.015,
            'dog': 0.015,
            'cat': 0.015,
            'bicycle': 0.025,
            'motorcycle': 0.03,
            'car': 0.04,
            'truck': 0.05,
            'bus': 0.05,
            'boat': 0.05,
            'airplane': 0.05,
        }
        return stop_areas.get(target_class, 0.02)

    def _best_detection(self, detections: list, target_class: str) -> dict | None:
        """Pick the highest-confidence detection for a requested class."""
        candidates = [d for d in detections if d.get('class') == target_class]
        if not candidates:
            return None
        return max(candidates, key=lambda d: float(d.get('confidence', 0.0)))

    def _current_yaw_rad(self) -> float:
        """Extract current vehicle yaw from odometry quaternion."""
        if self.odometry is None:
            return float('nan')
        q = self.odometry.q
        return math.atan2(
            2.0 * (float(q[0]) * float(q[3]) + float(q[1]) * float(q[2])),
            1.0 - 2.0 * (float(q[2]) ** 2 + float(q[3]) ** 2),
        )

    def _activate_visual_follow(self, target_class: str, speed_m_s: float, detections: list):
        """Start deterministic visual approach/follow for the requested class."""
        self._cancel_pending_llm()
        self._freeze_at_current_pose(f'Visual approach start for "{target_class}"')
        self.visual_follow_active = True
        self.visual_search_active = False
        self.visual_target_class = target_class
        self.visual_target_speed_m_s = speed_m_s
        self.translator.translation_speed_cap_override_m_s = speed_m_s
        self.visual_last_seen_time = time.time()
        self.visual_last_update_time = 0.0
        self.visual_lost_reported = False
        self.visual_reacquire_hits = 0
        self.visual_reacquire_last_hit_time = 0.0
        self.current_command = ''
        target_obj = self._best_detection(detections, target_class)
        if target_obj:
            bbox_center = target_obj.get('bbox_center', [0.5, 0.5])
            self.visual_filtered_cx = float(bbox_center[0])
            self.visual_filtered_cy = float(bbox_center[1])
            self.visual_filtered_area = float(target_obj.get('bbox_area', 0.0))
            self._update_visual_follow()
        self.get_logger().info(
            f'Visual approach active: target="{target_class}", speed={speed_m_s:.2f} m/s'
        )

    def _activate_visual_search(self, target_class: str, speed_m_s: float, reason: str):
        """Start a deterministic local search around the last seen target area."""
        if self.odometry is None:
            return

        self._cancel_pending_llm()
        self.visual_follow_active = True
        self.visual_search_active = True
        self.visual_target_class = target_class
        self.visual_target_speed_m_s = speed_m_s
        self.visual_lost_reported = False
        self.visual_last_seen_time = max(self.last_nonempty_detection_time, time.time())
        self.visual_reacquire_hits = 0
        self.visual_reacquire_last_hit_time = 0.0

        cur_x = float(self.odometry.position[0])
        cur_y = float(self.odometry.position[1])
        cur_z = float(self.odometry.position[2])
        current_yaw = self._current_yaw_rad()
        if math.isnan(current_yaw):
            current_yaw = 0.0

        stop_area = self._visual_stop_area_for_class(target_class)
        area_ratio = min(1.0, self.visual_filtered_area / max(stop_area, 1e-4))
        center_dist = 1.5 + 3.0 * max(0.0, 1.0 - area_ratio)
        if self.visual_filtered_cy >= self.visual_bottom_slowdown_cy:
            center_dist *= 0.7
        center_dist = max(1.0, min(5.0, center_dist))

        bearing_offset = (self.visual_filtered_cx - 0.5) * self.visual_horizontal_fov_rad
        center_heading = current_yaw + bearing_offset
        self.visual_search_center_x = cur_x + center_dist * math.cos(center_heading)
        self.visual_search_center_y = cur_y + center_dist * math.sin(center_heading)
        self.visual_search_radius_m = max(1.2, min(3.0, 0.45 * center_dist + 0.7))
        self.visual_search_angle_rad = math.atan2(
            cur_y - self.visual_search_center_y,
            cur_x - self.visual_search_center_x,
        )
        self.visual_search_direction = -1.0 if self.visual_filtered_cx >= 0.5 else 1.0
        self.visual_search_speed_m_s = min(max(speed_m_s, 0.2), 0.45)

        if self.visual_filtered_cy >= self.visual_bottom_slowdown_cy:
            descend_bias = 0.25 + 0.9 * (self.visual_filtered_cy - self.visual_bottom_slowdown_cy)
            self.visual_search_target_z = min(cur_z + min(0.7, descend_bias), -2.0)
        else:
            self.visual_search_target_z = cur_z

        self.translator.translation_speed_cap_override_m_s = self.visual_search_speed_m_s
        self.visual_last_update_time = 0.0
        self.current_command = ''
        self.get_logger().info(
            f'{reason}: target="{target_class}", center='
            f'({self.visual_search_center_x:.1f}, {self.visual_search_center_y:.1f}), '
            f'radius={self.visual_search_radius_m:.1f}m, speed={self.visual_search_speed_m_s:.2f} m/s'
        )

    def _update_visual_follow(self):
        """Continuously move toward the visible target using filtered YOLO geometry."""
        if self.odometry is None:
            return
        current_yaw = self._current_yaw_rad()
        if math.isnan(current_yaw):
            return

        x_error = self.visual_filtered_cx - 0.5
        if abs(x_error) < self.visual_deadband_x:
            x_error = 0.0
        y_error = self.visual_filtered_cy - self.visual_target_cy
        if abs(y_error) < self.visual_deadband_y:
            y_error = 0.0

        bearing_offset = x_error * self.visual_horizontal_fov_rad
        target_yaw = current_yaw + bearing_offset
        stop_area = self._visual_stop_area_for_class(self.visual_target_class or '')

        cur_x = float(self.odometry.position[0])
        cur_y = float(self.odometry.position[1])
        cur_z = float(self.odometry.position[2])

        self.translator.orbiting = False
        self.translator.square_active = False
        self.translator.target_yaw = target_yaw
        if y_error >= 0.0:
            alt_step_m = min(
                self.visual_max_descend_step_m,
                y_error * self.visual_descend_gain_m,
            )
        else:
            alt_step_m = max(
                -self.visual_max_climb_step_m,
                y_error * self.visual_climb_gain_m,
            )
        self.translator.target_z = min(cur_z + alt_step_m, -2.0)

        stop_area_for_follow = stop_area
        if self.visual_filtered_cy >= self.visual_bottom_slowdown_cy:
            stop_area_for_follow *= self.visual_forward_camera_standoff_scale

        if self.visual_filtered_area >= stop_area_for_follow:
            self.translator.target_x = cur_x
            self.translator.target_y = cur_y
            return

        now = time.monotonic()
        if self.visual_last_update_time == 0.0:
            dt = 0.15
        else:
            dt = max(0.05, min(0.4, now - self.visual_last_update_time))
        self.visual_last_update_time = now

        speed_m_s = self.visual_target_speed_m_s
        if self.translator.max_speed_m_s > 0.0:
            speed_m_s = min(speed_m_s, self.translator.max_speed_m_s)
        self.translator.translation_speed_cap_override_m_s = speed_m_s

        area_ratio = min(1.0, self.visual_filtered_area / max(stop_area, 1e-4))
        forward_scale = max(0.15, 1.0 - area_ratio)
        # Move more cautiously when the target is far off-center so yaw can settle first.
        centering_scale = max(0.2, 1.0 - abs(x_error) / 0.3)
        if self.visual_filtered_cy >= self.visual_bottom_stop_cy:
            vertical_forward_scale = 0.0
        elif self.visual_filtered_cy <= self.visual_bottom_slowdown_cy:
            vertical_forward_scale = 1.0
        else:
            remaining = self.visual_bottom_stop_cy - self.visual_filtered_cy
            span = max(1e-3, self.visual_bottom_stop_cy - self.visual_bottom_slowdown_cy)
            vertical_forward_scale = max(0.0, min(1.0, remaining / span))
        lookahead_dist = min(
            self.visual_max_lookahead_m,
            max(
                self.visual_min_lookahead_m,
                0.20 + 0.8 * speed_m_s * forward_scale,
            ),
        )
        step_dist = lookahead_dist * centering_scale * vertical_forward_scale

        self.translator.target_x = cur_x + step_dist * math.cos(target_yaw)
        self.translator.target_y = cur_y + step_dist * math.sin(target_yaw)

    def _update_visual_search(self):
        """Run a small local orbit around the last seen target area until reacquired."""
        if self.odometry is None:
            return

        now = time.monotonic()
        if self.visual_last_update_time == 0.0:
            dt = 0.15
        else:
            dt = max(0.05, min(0.4, now - self.visual_last_update_time))
        self.visual_last_update_time = now

        self.visual_search_radius_m = min(
            self.visual_search_radius_max_m,
            self.visual_search_radius_m + self.visual_search_expand_rate_m_s * dt,
        )
        angular_speed = self.visual_search_speed_m_s / max(self.visual_search_radius_m, 0.5)
        self.visual_search_angle_rad += self.visual_search_direction * angular_speed * dt

        target_x = self.visual_search_center_x + self.visual_search_radius_m * math.cos(
            self.visual_search_angle_rad
        )
        target_y = self.visual_search_center_y + self.visual_search_radius_m * math.sin(
            self.visual_search_angle_rad
        )
        target_yaw = math.atan2(
            self.visual_search_center_y - target_y,
            self.visual_search_center_x - target_x,
        )

        self.translator.orbiting = False
        self.translator.square_active = False
        self.translator.translation_speed_cap_override_m_s = self.visual_search_speed_m_s
        self.translator.target_x = target_x
        self.translator.target_y = target_y
        self.translator.target_z = self.visual_search_target_z
        self.translator.target_yaw = target_yaw

    def _follow_target_tick(self):
        """Update the active local follow controller on every offboard cycle."""
        if not self.visual_follow_active or self.visual_target_class is None:
            return

        age = time.time() - self.visual_last_seen_time
        if age > self.visual_reacquire_timeout_sec:
            target_class = self.visual_target_class
            self._freeze_at_current_pose(
                f'Visual target "{target_class}" lost for too long; ending follow'
            )
            return

        if self.visual_search_active:
            self._update_visual_search()
            return

        if age > self.visual_lost_timeout_sec:
            self._activate_visual_search(
                self.visual_target_class,
                self.visual_target_speed_m_s,
                f'Visual target "{self.visual_target_class}" temporarily lost; starting local search',
            )
            self.visual_lost_reported = True
            self._update_visual_search()
            return

        self.visual_lost_reported = False
        self._update_visual_follow()

    def _handle_local_visual_target(self, command: str) -> bool:
        """Handle prompts like 'move slowly toward the person' without the LLM."""
        text = command.lower()
        approach_words = (
            'move toward',
            'move towards',
            'toward',
            'towards',
            'approach',
            'follow',
            'track',
        )
        if not any(word in text for word in approach_words):
            return False

        target_class = self._infer_target_class(command)
        if target_class is None:
            return False

        try:
            detections = json.loads(self.latest_detections)
        except json.JSONDecodeError:
            detections = []
        target_obj = self._best_detection(detections, target_class)
        if target_obj is None:
            age = time.time() - self.last_nonempty_detection_time
            if age <= self.visual_recent_detection_ttl_sec:
                detections = list(self.last_nonempty_detections)
                target_obj = self._best_detection(detections, target_class)

        speed_m_s = self._extract_speed_m_s(command, default=0.75)
        if target_obj is None:
            self._activate_visual_search(
                target_class,
                speed_m_s,
                f'Local visual search requested for "{target_class}"',
            )
            return True

        self._activate_visual_follow(target_class, speed_m_s, detections)
        return True

    # ─── Subscriber callbacks ──────────────────────────────────────────

    def _odom_cb(self, msg: VehicleOdometry):
        self.odometry = msg
        pos = msg.position
        self.translator.update_position(pos[0], pos[1], pos[2])
        if not self.initial_hold_target_set:
            self.translator.target_x = float(pos[0])
            self.translator.target_y = float(pos[1])
            self.translator.target_z = float(pos[2])
            self.initial_hold_target_set = True
            self.get_logger().info(
                f'Initial hold target primed from odometry: '
                f'NED=({self.translator.target_x:.1f}, '
                f'{self.translator.target_y:.1f}, '
                f'{self.translator.target_z:.1f})'
            )
        # Fallback: if vehicle_status is not arriving, infer "armed-ish"
        # from whether the vehicle is clearly airborne. This must reset back
        # to False after landing, otherwise later takeoff prompts are misread.
        if self.vehicle_status is None and abs(float(pos[2])) > 2.0:
            self.armed = True
        elif self.vehicle_status is None:
            self.armed = False

    def _gps_cb(self, msg: SensorGps):
        self.gps = msg
        # Set home position on first GPS fix
        if not self.translator.home_set and msg.fix_type >= 3:
            lat = msg.latitude_deg
            lon = msg.longitude_deg
            alt = msg.altitude_msl_m
            self.translator.set_home(lat, lon, alt)
            self.get_logger().info(
                f'Home set: lat={lat:.6f}, lon={lon:.6f}, alt={alt:.1f}m'
            )

    def _status_cb(self, msg: VehicleStatus):
        self.vehicle_status = msg
        self.armed = msg.arming_state == 2  # ARMED

    def _battery_cb(self, msg: BatteryStatus):
        self.battery = msg

    def _yolo_cb(self, msg: String):
        self.latest_detections = msg.data

        try:
            detections = json.loads(msg.data)
            new_classes = {d['class'] for d in detections}
        except (json.JSONDecodeError, KeyError):
            new_classes = set()
            detections = []

        if detections:
            self.last_nonempty_detections = detections
            self.last_nonempty_detection_time = time.time()

        if self.visual_follow_active and self.visual_target_class:
            now = time.time()
            target_obj = self._best_detection(detections, self.visual_target_class)
            if target_obj is not None:
                bbox_center = target_obj.get('bbox_center', [0.5, 0.5])
                bbox_cx = max(0.1, min(0.9, float(bbox_center[0])))
                bbox_cy = max(0.05, min(0.95, float(bbox_center[1])))
                bbox_area = max(0.0, float(target_obj.get('bbox_area', 0.0)))
                alpha = self.visual_bbox_alpha
                self.visual_filtered_cx = alpha * bbox_cx + (1.0 - alpha) * self.visual_filtered_cx
                self.visual_filtered_cy = alpha * bbox_cy + (1.0 - alpha) * self.visual_filtered_cy
                self.visual_filtered_area = (
                    alpha * bbox_area + (1.0 - alpha) * self.visual_filtered_area
                )
                self.visual_last_seen_time = now
                self.last_detection_classes = new_classes
                if self.visual_search_active:
                    if (
                        now - self.visual_reacquire_last_hit_time >
                        self.visual_reacquire_hit_window_sec
                    ):
                        self.visual_reacquire_hits = 0
                    self.visual_reacquire_hits += 1
                    self.visual_reacquire_last_hit_time = now
                    if self.visual_reacquire_hits >= self.visual_reacquire_required_hits:
                        self.visual_search_active = False
                        self.visual_lost_reported = False
                        self.visual_last_update_time = 0.0
                        self.visual_reacquire_hits = 0
                        self.visual_reacquire_last_hit_time = 0.0
                        self.get_logger().info(
                            f'Visual target "{self.visual_target_class}" confirmed for '
                            f'{self.visual_reacquire_required_hits} frames; resuming follow'
                        )
                else:
                    self.visual_reacquire_hits = 0
                    self.visual_reacquire_last_hit_time = 0.0
            elif self.visual_search_active:
                if now - self.visual_reacquire_last_hit_time > self.visual_reacquire_hit_window_sec:
                    self.visual_reacquire_hits = 0
            return

        # During an active target-search orbit, only react to the requested target.
        # Ignore unrelated class flicker (for example propeller self-detections).
        if self.translator.orbiting and self.search_target:
            if self.search_target in new_classes:
                self.last_detection_classes = new_classes
                self.get_logger().info(
                    f'"{self.search_target}" detected during orbit — transitioning to hover'
                )
                self._target_found(detections, self.search_target)
            return

        if self.mission_active:
            self.last_detection_classes = new_classes
            return

        # Otherwise only re-trigger the LLM for commands that are explicitly
        # about object discovery. Plain motion commands like "fly in a circle"
        # should not be starved by YOLO flicker.
        if (
            new_classes != self.last_detection_classes and
            self.current_command and
            self._command_should_replan_on_detection(self.current_command)
        ):
            self.last_detection_classes = new_classes
            if new_classes:
                self.get_logger().info(f'Detection change: {new_classes} — triggering LLM')
                self._call_llm()
            return

        self.last_detection_classes = new_classes

    def _target_found(self, detections: list, target_class: str):
        """Stop orbiting and descend toward a detected target object.

        Uses the YOLO bbox_center x-coordinate and the drone's current yaw
        to estimate the bearing toward the target, then flies a short distance
        in that direction and descends to 40% of current altitude (min 5 m).

        Args:
            detections: Parsed list of YOLO detection dicts.
            target_class: YOLO class name that was found (e.g. "person", "car").
        """
        self.translator.orbiting = False
        self.translator.square_active = False
        self.search_target = None  # search complete — clear the target

        # Current position from odometry or fall back to last known target
        if self.odometry:
            cur_x = float(self.odometry.position[0])
            cur_y = float(self.odometry.position[1])
            current_alt = abs(float(self.odometry.position[2]))
        else:
            cur_x = self.translator.target_x
            cur_y = self.translator.target_y
            current_alt = abs(self.translator.target_z)

        # Descend to 40% of current altitude, never below 5 m
        hover_alt_z = -max(5.0, current_alt * 0.4)
        target_x = cur_x
        target_y = cur_y
        target_yaw = float('nan')

        target_obj = next((d for d in detections if d.get('class') == target_class), None)
        if target_obj and self.odometry:
            # bbox_center[0]: 0 = left edge, 0.5 = centre, 1 = right edge
            bbox_cx = max(0.1, min(0.9, target_obj['bbox_center'][0]))

            # Extract yaw from VehicleOdometry quaternion [w, x, y, z]
            q = self.odometry.q
            yaw = math.atan2(
                2.0 * (float(q[0]) * float(q[3]) + float(q[1]) * float(q[2])),
                1.0 - 2.0 * (float(q[2]) ** 2 + float(q[3]) ** 2),
            )

            # Horizontal FOV ~90 deg; offset bearing left/right by pixel position
            bearing_offset = (bbox_cx - 0.5) * math.radians(90)
            target_yaw = yaw + bearing_offset

            # Fly a short distance toward the estimated target position
            move_dist = min(current_alt * 0.3, 8.0)
            target_x = cur_x + move_dist * math.cos(target_yaw)
            target_y = cur_y + move_dist * math.sin(target_yaw)

        self.translator.target_x = target_x
        self.translator.target_y = target_y
        self.translator.target_z = hover_alt_z
        self.translator.target_yaw = target_yaw

        self.get_logger().info(
            f'TARGET FOUND: "{target_class}"! Orbit stopped. '
            f'Descending to {abs(hover_alt_z):.1f} m AGL, '
            f'moving to NED=({target_x:.1f}, {target_y:.1f})'
        )
        if (
            self.mission_active and
            self.mission_current_step is not None and
            self.mission_current_step.get('completion', {}).get('type') == 'target_found' and
            self.mission_current_step.get('completion', {}).get('target_class') == target_class
        ):
            self._complete_current_mission_step(f'found "{target_class}"')

    def _dispatch_user_command(
        self,
        command: str,
        *,
        allow_mission_planner: bool = True,
        force_llm: bool = False,
        include_history: bool = True,
    ):
        """Route a new user command through planner, local handlers, or normal LLM control."""
        if allow_mission_planner and self._looks_like_multi_step_mission(command):
            self._freeze_at_current_pose('Planning multi-step mission')
            self._start_mission_planner(command)
            return

        mission_steps = self._parse_multi_step_mission(command)
        if mission_steps is not None:
            self._start_multi_step_mission(mission_steps)
            return

        if self._handle_local_takeoff(command):
            self.current_command = ''
            return

        if self._handle_local_visual_target(command):
            return

        self._freeze_at_current_pose('User command received')
        self._call_llm(force=force_llm, include_history=include_history)

    def _user_cmd_cb(self, msg: String):
        command = msg.data.strip()

        # Special commands handled locally — no LLM call needed
        if command.lower() in ('reset', 'reset memory', 'clear memory', 'new mission'):
            self._cancel_pending_llm()
            self.llm.reset_memory()
            self._clear_mission_queue()
            self.search_target = None
            self.visual_follow_active = False
            self.visual_search_active = False
            self.visual_target_class = None
            self.translator.translation_speed_cap_override_m_s = 0.0
            self.visual_reacquire_hits = 0
            self.visual_reacquire_last_hit_time = 0.0
            self.current_command = ''
            self.get_logger().info('Memory and search target cleared. Ready for new mission.')
            return

        if self.mission_active:
            self.get_logger().info('New user command overrides the active multi-step mission.')
            self._clear_mission_queue()

        self.current_command = command
        self.get_logger().info(f'New user command: "{self.current_command}"')
        self._dispatch_user_command(command)

    def _extract_altitude_m(self, command: str, default: float = 10.0) -> float:
        """Extract a requested altitude from free text, falling back to a sensible default."""
        match = re.search(r'(\d+(?:\.\d+)?)\s*(?:m|meter|meters|metre|metres)\b', command.lower())
        if not match:
            return default
        return max(1.0, float(match.group(1)))

    def _command_mentions_altitude(self, command: str) -> bool:
        """Return True when the user explicitly asked to change altitude."""
        text = command.lower()
        altitude_keywords = (
            'altitude',
            'height',
            'agl',
            'take off',
            'takeoff',
            'hover at',
            'climb',
            'ascend',
            'descend',
            'upward',
            'downward',
            'rise',
            'go up',
            'go down',
            'higher',
            'lower',
            'land',
        )
        if any(keyword in text for keyword in altitude_keywords):
            return True

        return bool(re.search(
            r'\b(?:at|to)\s+\d+(?:\.\d+)?\s*(?:m|meter|meters|metre|metres)\b',
            text,
        ))

    def _current_target_altitude_z(self) -> float:
        """Return the altitude target currently being tracked by the controller."""
        if self.translator.orbiting:
            return float(self.translator.orbit_alt_z)
        if self.translator.square_active:
            return float(self.translator.square_alt_z)
        if self.translator.path_active:
            return float(self.translator.target_z)
        return float(self.translator.target_z)

    def _preserve_altitude_if_unspecified(self, cmd: dict):
        """Keep the current altitude unless the user explicitly requested a change."""
        if self._command_mentions_altitude(self.current_command):
            return

        action = cmd.get('action')
        hold_alt_z = self._current_target_altitude_z()

        if action == 'position_ned':
            cmd['z'] = hold_alt_z
        elif action == 'orbit':
            cmd.pop('alt', None)
            cmd['alt_z'] = hold_alt_z
        elif action in ('square_survey', 'square'):
            cmd.pop('alt', None)
            cmd['alt_z'] = hold_alt_z
        elif action == 'shape_path':
            cmd['alt_z'] = hold_alt_z
            cmd['climb_m'] = 0.0
        elif action == 'path':
            points = cmd.get('points')
            if isinstance(points, list):
                adjusted_points = []
                for point in points:
                    if isinstance(point, dict):
                        adjusted = dict(point)
                        adjusted['z'] = hold_alt_z
                        adjusted_points.append(adjusted)
                    elif isinstance(point, (list, tuple)):
                        if len(point) >= 2:
                            adjusted_points.append([point[0], point[1], hold_alt_z])
                        else:
                            adjusted_points.append(point)
                    else:
                        adjusted_points.append(point)
                cmd['points'] = adjusted_points
            elif str(cmd.get('generator', '')).lower() == 'functiongemma':
                cmd['alt_z'] = hold_alt_z
            else:
                return
        elif action == 'goto':
            cmd['alt'] = abs(hold_alt_z)
        else:
            return

        self.get_logger().info(
            f'Altitude not requested in prompt; preserving current target altitude '
            f'at {abs(hold_alt_z):.1f} m AGL'
        )

    def _freeze_at_current_pose(self, reason: str, clear_follow: bool = True):
        """Stop active patterns and hold the current measured pose while replanning."""
        self.translator.orbiting = False
        self.translator.square_active = False
        self.translator.path_active = False
        self.search_target = None
        if clear_follow:
            self.visual_follow_active = False
            self.visual_search_active = False
            self.visual_target_class = None
            self.visual_lost_reported = False
            self.translator.translation_speed_cap_override_m_s = 0.0
            self.visual_reacquire_hits = 0
            self.visual_reacquire_last_hit_time = 0.0
        self.visual_last_update_time = 0.0
        if self.odometry:
            pos = self.odometry.position
            self.translator.target_x = float(pos[0])
            self.translator.target_y = float(pos[1])
            self.translator.target_z = float(pos[2])
            self.get_logger().info(
                f'{reason}: freezing at current pose '
                f'NED=({self.translator.target_x:.1f}, '
                f'{self.translator.target_y:.1f}, '
                f'{self.translator.target_z:.1f})'
            )
        else:
            self.get_logger().info(f'{reason}: stopping active patterns (odometry unavailable)')

    def _publish_vehicle_commands(self, extra_msgs: list):
        """Publish PX4 VehicleCommand messages with a consistent ROS-clock timestamp."""
        ts = int(self.get_clock().now().nanoseconds / 1000)
        for _, msg in extra_msgs:
            msg.timestamp = ts
            self.command_pub.publish(msg)

    def _cancel_pending_llm(self):
        """Invalidate any in-flight LLM request so stale replies are ignored."""
        with self.llm_lock:
            self.llm_request_id += 1
            self.llm_inflight = False
            self.llm_replan_pending = False
            self.pending_llm_result = None
            self.pending_llm_kind = None
            self.pending_llm_context = None
            self.pending_llm_error = None

    def _handle_local_takeoff(self, command: str) -> bool:
        """Handle simple vertical takeoff/altitude requests without depending on the LLM."""
        text = command.lower()
        takeoff_words = ('take off', 'takeoff', 'hover at', 'set altitude', 'climb to')
        if not any(word in text for word in takeoff_words):
            return False

        self._cancel_pending_llm()
        altitude_m = self._extract_altitude_m(command, default=10.0)
        if self.odometry:
            self.translator.target_x = float(self.odometry.position[0])
            self.translator.target_y = float(self.odometry.position[1])
        self.translator.target_z = -abs(altitude_m)
        self.translator.target_yaw = float('nan')
        self.translator.orbiting = False
        self.translator.square_active = False
        self.search_target = None
        self.visual_follow_active = False
        self.visual_search_active = False
        self.visual_target_class = None
        self.translator.translation_speed_cap_override_m_s = 0.0
        self.visual_reacquire_hits = 0
        self.visual_reacquire_last_hit_time = 0.0

        extra_msgs = []
        if not self.armed:
            extra_msgs = self.translator.process_command({'action': 'arm_offboard'})

        self._publish_vehicle_commands(extra_msgs)
        self.get_logger().info(
            f'Local takeoff handler: target altitude {altitude_m:.1f} m AGL '
            f'at current XY, arming_step={not self.armed}'
        )
        return True

    def _resolve_custom_shape_path_if_requested(self, cmd: dict) -> dict:
        """Expand unsupported custom shapes into explicit path waypoints via FunctionGemma."""
        if cmd.get('action') != 'path':
            return cmd
        if not self.enable_custom_shape_fallback:
            return cmd

        generator_name = str(cmd.get('generator', '')).lower()
        use_custom_generator = generator_name == 'functiongemma'
        if not use_custom_generator and self._looks_like_custom_shape_request(self.current_command):
            shape_name = str(cmd.get('shape', '')).strip().lower()
            use_custom_generator = not is_supported_shape_name(shape_name)

        if not use_custom_generator:
            return cmd

        prompt = str(cmd.get('shape_prompt') or self.current_command).strip()
        if not prompt:
            return cmd

        if self.odometry:
            origin_x = float(self.odometry.position[0])
            origin_y = float(self.odometry.position[1])
        else:
            origin_x = float(self.translator.target_x)
            origin_y = float(self.translator.target_y)
        origin_z = float(cmd.get('alt_z', self._current_target_altitude_z()))
        current_yaw = self._current_yaw_rad()
        rotation_rad = 0.0 if math.isnan(current_yaw) else current_yaw

        generator = self._ensure_custom_shape_generator()
        waypoints, shape_spec = generator.generate_waypoints(
            prompt,
            origin_x=origin_x,
            origin_y=origin_y,
            origin_z=origin_z,
            size_m=self._custom_shape_size_m(cmd),
            closed=bool(cmd.get('closed', True)),
            point_count=int(cmd.get('point_count', 120)),
            rotation_rad=rotation_rad,
        )

        resolved = dict(cmd)
        resolved['points'] = [[x, y, z] for x, y, z in waypoints]
        resolved['speed'] = float(cmd.get('speed', self.translator.target_speed))
        resolved['closed'] = bool(cmd.get('closed', True))
        resolved['loop'] = bool(cmd.get('loop', False))
        resolved['generator'] = 'functiongemma'
        resolved['_functiongemma_family'] = str(shape_spec.get('family', 'unknown'))
        resolved['_functiongemma_point_count'] = int(shape_spec.get('points', len(waypoints)))
        self.get_logger().info(
            f'FunctionGemma fallback selected family="{resolved["_functiongemma_family"]}" '
            f'and generated {len(waypoints)} waypoints '
            f'for unsupported custom shape prompt "{prompt}"'
        )
        return resolved

    # ─── Core loops ────────────────────────────────────────────────────

    def _offboard_loop(self):
        """10Hz loop: publish offboard heartbeat + trajectory setpoint."""
        # Use the ROS2 clock (synced to PX4 via uXRCE-DDS) for timestamps.
        # time.time() gives Unix epoch μs (~1.7×10^15) but PX4 expects μs
        # since boot (~10^6–10^9), so the messages would be silently dropped.
        ts = int(self.get_clock().now().nanoseconds / 1000)

        self._follow_target_tick()
        self._mission_tick()

        offboard_msg = self.translator.get_offboard_control_mode()
        offboard_msg.timestamp = ts
        self.offboard_pub.publish(offboard_msg)

        setpoint_msg = self.translator.get_setpoint_tick()
        setpoint_msg.timestamp = ts
        self.setpoint_pub.publish(setpoint_msg)

    def _llm_decision_tick(self):
        """Periodic LLM re-evaluation — disabled, LLM called on demand only.

        The LLM is called on new user commands or YOLO detection changes.
        """
        # if not self.current_command:
        #     return
        # self._call_llm()
        pass

    def _llm_worker(
        self,
        request_id: int,
        drone_state: str,
        user_command: str,
        yolo_detections: str,
        include_history: bool,
    ):
        """Run the blocking Ollama call off the ROS executor thread."""
        try:
            cmd = self.llm.ask(
                drone_state,
                user_command,
                yolo_detections,
                include_history=include_history,
            )
            with self.llm_lock:
                if request_id != self.llm_request_id:
                    return
                self.llm_inflight = False
                self.pending_llm_result = cmd
                self.pending_llm_kind = 'command'
                self.pending_llm_context = user_command
                self.pending_llm_error = None
        except Exception as e:
            with self.llm_lock:
                if request_id != self.llm_request_id:
                    return
                self.llm_inflight = False
                self.pending_llm_result = None
                self.pending_llm_kind = 'command'
                self.pending_llm_context = user_command
                self.pending_llm_error = str(e)

    def _mission_planner_worker(self, request_id: int, user_command: str):
        """Run the multi-step planner off the ROS executor thread."""
        try:
            plan = self.llm.plan_mission(user_command)
            with self.llm_lock:
                if request_id != self.llm_request_id:
                    return
                self.llm_inflight = False
                self.pending_llm_result = plan
                self.pending_llm_kind = 'plan'
                self.pending_llm_context = user_command
                self.pending_llm_error = None
        except Exception as e:
            with self.llm_lock:
                if request_id != self.llm_request_id:
                    return
                self.llm_inflight = False
                self.pending_llm_result = None
                self.pending_llm_kind = 'plan'
                self.pending_llm_context = user_command
                self.pending_llm_error = str(e)

    def _llm_result_tick(self):
        """Apply completed LLM responses on the ROS executor thread."""
        with self.llm_lock:
            cmd = self.pending_llm_result
            kind = self.pending_llm_kind
            context = self.pending_llm_context
            error = self.pending_llm_error
            replan_pending = self.llm_replan_pending
            if cmd is None and error is None:
                return
            self.pending_llm_result = None
            self.pending_llm_kind = None
            self.pending_llm_context = None
            self.pending_llm_error = None
            self.llm_replan_pending = False

        if kind == 'plan':
            if error is not None:
                self.get_logger().error(f'Mission planner failed: {error}')
                fallback_command = context or self.current_command
                if fallback_command:
                    self.get_logger().info(
                        'Falling back to deterministic parser / normal command flow'
                    )
                    self.current_command = fallback_command
                    self._dispatch_user_command(
                        fallback_command,
                        allow_mission_planner=False,
                        force_llm=True,
                    )
                return

            self._apply_llm_plan(context or self.current_command, cmd)
            return

        if replan_pending:
            self.get_logger().info('State changed while LLM was running; requesting fresh replan')
            self._call_llm(force=True)
            return

        if error is not None:
            if kind == 'command' and self.mission_active and self.mission_current_step is not None:
                step = self.mission_current_step
                step_command = str(step.get('command', '')).strip()
                if context == step_command:
                    retry_count = int(step.get('_retry_count', 0))
                    if 'timed out' in str(error).lower() and retry_count < 1:
                        step['_retry_count'] = retry_count + 1
                        self.get_logger().warning(
                            f'LLM timed out for mission step "{self._mission_step_label(step)}"; '
                            'retrying once'
                        )
                        self._execute_mission_command_step(step)
                        return
                    self._abort_mission(
                        f'LLM failed for mission step "{self._mission_step_label(step)}": {error}'
                    )
                    return
            self.get_logger().error(f'LLM call failed: {error}')
            return

        self._apply_llm_command(cmd)

    def _start_mission_planner(self, user_command: str):
        """Start an asynchronous mission-planning request for likely sequenced prompts."""
        self._cancel_pending_llm()
        with self.llm_lock:
            self.llm_request_id += 1
            request_id = self.llm_request_id
            self.llm_inflight = True

        self.get_logger().info('=' * 60)
        self.get_logger().info('>>> MISSION PLANNER INPUT <<<')
        self.get_logger().info(f'User command: {user_command}')
        self.get_logger().info('Calling Ollama mission planner...')

        thread = threading.Thread(
            target=self._mission_planner_worker,
            args=(request_id, user_command),
            daemon=True,
        )
        thread.start()

    def _apply_llm_plan(self, user_command: str, plan: dict):
        """Validate planner output and start a mission queue or fall back safely."""
        self.get_logger().info('>>> MISSION PLANNER OUTPUT <<<')
        self.get_logger().info(f'Planner result: {plan}')

        steps = self._normalize_planned_steps(plan)
        if steps is not None:
            self.current_command = ''
            self.get_logger().info(
                f'Mission planner produced a valid {len(steps)}-step queue'
            )
            self._start_multi_step_mission(steps)
            return

        fallback_steps = self._parse_multi_step_mission(user_command)
        if fallback_steps is not None:
            self.current_command = ''
        self.get_logger().warning(
            'Mission planner returned no usable structured plan; '
            'using deterministic parser fallback'
        )
        self._start_multi_step_mission(fallback_steps)
        return

        self.get_logger().info(
            'Mission planner returned no usable step list; '
            'continuing with the normal command flow'
        )
        self.current_command = user_command
        self._dispatch_user_command(
            user_command,
            allow_mission_planner=False,
            force_llm=True,
        )

    def _call_llm(self, force: bool = False, include_history: bool = True):
        """Start an asynchronous LLM request without blocking the offboard loop."""
        now = time.time()
        # Rate limit: don't call more than once per 3 seconds
        with self.llm_lock:
            if self.llm_inflight:
                self.llm_replan_pending = True
                self.get_logger().info('LLM already running; queued a fresh replan')
                return
            if not force and now - self.last_llm_call < 3.0:
                return
            self.llm_request_id += 1
            request_id = self.llm_request_id
            self.last_llm_call = now
            self.llm_inflight = True

        drone_state = self._format_drone_state()

        self.get_logger().info('=' * 60)
        self.get_logger().info('>>> LLM INPUT <<<')
        self.get_logger().info(f'User command: {self.current_command}')
        self.get_logger().info(f'Drone state:\n{drone_state}')
        self.get_logger().info(f'YOLO detections: {self.latest_detections}')
        self.get_logger().info('Calling Ollama LLM...')

        thread = threading.Thread(
            target=self._llm_worker,
            args=(
                request_id,
                drone_state,
                self.current_command,
                self.latest_detections,
                include_history,
            ),
            daemon=True,
        )
        thread.start()

    def _apply_llm_command(self, cmd: dict):
        """Apply an LLM command on the ROS executor thread."""
        try:
            self.get_logger().info('>>> LLM OUTPUT <<<')
            thought = cmd.pop('thought', None)
            if thought:
                self.get_logger().info(f'LLM reasoning: {thought}')
            self.get_logger().info(f'LLM action: {cmd}')

            action = cmd.get('action')
            self._preserve_altitude_if_unspecified(cmd)
            cmd = self._resolve_custom_shape_path_if_requested(cmd)
            action = cmd.get('action')

            # Orbit with a search goal: extract the YOLO class to watch for
            if action == 'orbit':
                if 'target_class' not in cmd:
                    inferred_target = self._infer_search_target(self.current_command)
                    if inferred_target:
                        cmd['target_class'] = inferred_target
                        self.get_logger().info(
                            f'Inferred search target from prompt: "{inferred_target}"'
                        )
                self.search_target = cmd.get('target_class', None)
                if self.search_target:
                    self.get_logger().info(
                        f'Search orbit: will stop when YOLO detects "{self.search_target}"'
                    )
                else:
                    self.get_logger().info('Pure loiter orbit (no search target)')
            elif action == 'path':
                self.search_target = None
                point_count = len(cmd.get('points', [])) if isinstance(cmd.get('points'), list) else 0
                if cmd.get('generator') == 'functiongemma':
                    self.get_logger().info(
                        f'FunctionGemma custom shape path: {point_count} points'
                    )
                else:
                    self.get_logger().info(
                        f'Custom waypoint path: {point_count} points'
                    )
            elif action == 'shape_path':
                self.search_target = None
                if 'reference_yaw_rad' not in cmd:
                    current_yaw = self._current_yaw_rad()
                    if not math.isnan(current_yaw):
                        cmd['reference_yaw_rad'] = current_yaw
                self.get_logger().info(
                    f'Deterministic shape path: shape={cmd.get("shape")} '
                    f'radius={cmd.get("radius", cmd.get("radius_m", "n/a"))} '
                    f'speed={cmd.get("speed", cmd.get("speed_m_s", "n/a"))}'
                )

            # Hold always freezes the current measured position.
            elif action == 'hold':
                self._freeze_at_current_pose('Hold command')

            extra_msgs = self.translator.process_command(cmd)
            self.get_logger().info(f'>>> COMMAND TRANSLATOR <<<')
            self.get_logger().info(
                f'Action: {cmd.get("action")} | '
                f'Target NED: x={self.translator.target_x:.1f}, '
                f'y={self.translator.target_y:.1f}, '
                f'z={self.translator.target_z:.1f} | '
                f'Orbiting: {self.translator.orbiting} | '
                f'Square: {self.translator.square_active} | '
                f'Path: {self.translator.path_active} | '
                f'Extra PX4 msgs: {len(extra_msgs)}'
            )
            self.get_logger().info('=' * 60)

            if (
                self.mission_active and
                self.mission_current_step is not None and
                self.mission_step_deadline == 0.0 and
                str(self.mission_current_step.get('command', '')).strip() == self.current_command.strip() and
                self.mission_current_step.get('completion', {}).get('type') == 'duration'
            ):
                duration_sec = float(
                    self.mission_current_step.get('completion', {}).get('seconds', 0.0)
                )
                if duration_sec > 0.0:
                    self.mission_step_started_at = time.time()
                    self.mission_step_deadline = self.mission_step_started_at + duration_sec
                    self.get_logger().info(
                        f'Mission duration step started: {duration_sec:g} second timer armed'
                    )

            if (
                self.mission_active and
                self.mission_current_step is not None and
                str(self.mission_current_step.get('command', '')).strip() == self.current_command.strip() and
                self.mission_current_step.get('completion', {}).get('type') == 'path_complete' and
                action in ('path', 'shape_path')
            ):
                self.mission_current_step['_path_sequence_id'] = self.translator.path_sequence_id

            self._publish_vehicle_commands(extra_msgs)

        except Exception as e:
            self.get_logger().error(f'LLM command apply failed: {e}')

    def _format_drone_state(self) -> str:
        """Format current telemetry into a human-readable string for the LLM."""
        lines = []

        if self.gps:
            lines.append(
                f'GPS: lat={self.gps.latitude_deg:.6f}, '
                f'lon={self.gps.longitude_deg:.6f}, '
                f'alt={self.gps.altitude_msl_m:.1f}m MSL, '
                f'fix={self.gps.fix_type}'
            )
        else:
            lines.append('GPS: No fix')

        if self.odometry:
            pos = self.odometry.position
            vel = self.odometry.velocity
            lines.append(
                f'Local position (NED): x={pos[0]:.1f}m, y={pos[1]:.1f}m, z={pos[2]:.1f}m'
            )
            lines.append(
                f'Velocity (NED): vx={vel[0]:.1f}, vy={vel[1]:.1f}, vz={vel[2]:.1f} m/s'
            )
        else:
            lines.append('Odometry: Not available')

        if self.battery:
            lines.append(
                f'Battery: {self.battery.remaining * 100:.0f}% '
                f'({self.battery.voltage_v:.1f}V)'
            )

        if self.vehicle_status:
            lines.append(f'Armed: {self.armed} (from vehicle_status arming_state={self.vehicle_status.arming_state})')
            lines.append(f'Nav state: {self.vehicle_status.nav_state}')
        else:
            lines.append(f'Armed: {self.armed} (vehicle_status not received — inferred from altitude)')

        if self.translator.home_set:
            lines.append(
                f'Home GPS (NED origin): lat={self.translator.home_lat:.6f}, '
                f'lon={self.translator.home_lon:.6f}, '
                f'alt={self.translator.home_alt:.1f}m MSL'
            )
        else:
            lines.append('Home GPS: not set yet')

        lines.append(f'Orbiting: {self.translator.orbiting}')
        if self.translator.max_speed_m_s > 0.0:
            lines.append(f'Max translational speed cap: {self.translator.max_speed_m_s:.2f} m/s')
        if self.search_target:
            lines.append(f'Search mode: actively scanning for "{self.search_target}"')
        if self.mission_active and self.mission_current_step is not None:
            lines.append(
                f'Multi-step mission: step {self.mission_step_index + 1}/{len(self.mission_steps)} '
                f'({self._mission_step_label(self.mission_current_step)})'
            )
        lines.append(
            f'Current target (NED): '
            f'x={self.translator.target_x:.1f}, '
            f'y={self.translator.target_y:.1f}, '
            f'z={self.translator.target_z:.1f}'
        )

        return '\n'.join(lines)


def main(args=None):
    rclpy.init(args=args)
    node = BrainNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
