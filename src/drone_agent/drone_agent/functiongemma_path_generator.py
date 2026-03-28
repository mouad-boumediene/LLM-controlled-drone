#!/usr/bin/env python3
"""Generate lightweight custom flight paths using Ollama FunctionGemma."""

from __future__ import annotations

import json
import math
import subprocess
import urllib.request


SYSTEM_PROMPT = """You must answer by calling emit_shape_family exactly once.

Map custom described shapes into one compact procedural family plus scalar parameters.

Preferred families:
- rose: clover, flower, petal, quatrefoil, trefoil
- crescent: crescent, moon
- butterfly: butterfly-like outline
- arrow: arrow, pointer
- diamond: diamond, rhombus, kite
- cloud: cloud, puff, blob

Examples:
- clover -> family=rose, petals=4, closed=true
- flower with 5 petals -> family=rose, petals=5, closed=true
- crescent moon -> family=crescent, closed=true
- butterfly outline -> family=butterfly, closed=true
- arrow -> family=arrow, closed=true
- diamond -> family=diamond, closed=true
- cloud outline -> family=cloud, lobes=4, closed=true

Use only the provided function, with short scalar arguments and no prose.
"""


class FunctionGemmaPathGenerator:
    """Lazy Ollama-backed custom-shape family generator."""

    def __init__(
        self,
        *,
        model_name: str = 'functiongemma',
        ollama_url: str = 'http://localhost:11434',
        request_timeout_sec: float = 60.0,
    ):
        self.model_name = str(model_name).strip() or 'functiongemma'
        self.ollama_url = ollama_url.rstrip('/')
        self.request_timeout_sec = float(request_timeout_sec)
        self._model_checked = False

    def ensure_model_available(self):
        """Ensure the local Ollama model is installed before first use."""
        if self._model_checked:
            return

        req = urllib.request.Request(
            f'{self.ollama_url}/api/tags',
            headers={'Content-Type': 'application/json'},
        )
        with urllib.request.urlopen(req, timeout=10.0) as resp:
            data = json.loads(resp.read().decode('utf-8'))

        available = {
            str(entry.get('name', '')).strip()
            for entry in data.get('models', [])
        }
        model_available = (
            self.model_name in available or
            any(name.startswith(f'{self.model_name}:') for name in available)
        )
        if not model_available:
            subprocess.run(
                ['ollama', 'pull', self.model_name],
                check=True,
            )

        self._model_checked = True

    def generate_shape_spec(
        self,
        prompt: str,
        *,
        point_count: int = 48,
        closed: bool = True,
    ) -> dict:
        """Generate a procedural custom-shape family spec."""
        self.ensure_model_available()

        tool_schema = {
            'type': 'function',
            'function': {
                'name': 'emit_shape_family',
                'description': 'Return a procedural custom flight-path family and its scalar parameters.',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'family': {'type': 'string'},
                        'closed': {'type': 'boolean'},
                        'petals': {'type': 'integer'},
                        'lobes': {'type': 'integer'},
                        'points': {'type': 'integer'},
                        'aspect_ratio': {'type': 'number'},
                        'thickness': {'type': 'number'},
                        'climb_m': {'type': 'number'},
                    },
                    'required': ['family', 'closed'],
                },
            },
        }
        user_prompt = (
            f'Create a custom flight shape for: {prompt}\n'
            f'Closed preferred: {"true" if closed else "false"}\n'
            f'Target smoothness: about {max(12, min(120, int(point_count)))} points\n'
            'Use emit_shape_family only.'
        )
        payload = json.dumps({
            'model': self.model_name,
            'messages': [
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': user_prompt},
            ],
            'stream': False,
            'tools': [tool_schema],
            'options': {
                'temperature': 0.0,
            },
        }).encode('utf-8')

        req = urllib.request.Request(
            f'{self.ollama_url}/api/chat',
            data=payload,
            headers={'Content-Type': 'application/json'},
        )
        with urllib.request.urlopen(req, timeout=self.request_timeout_sec) as resp:
            result = json.loads(resp.read().decode('utf-8'))

        arguments = self._extract_tool_arguments(result)
        return self._normalize_spec(
            arguments,
            closed=closed,
            point_count=point_count,
            prompt=prompt,
        )

    def generate_waypoints(
        self,
        prompt: str,
        *,
        origin_x: float,
        origin_y: float,
        origin_z: float,
        size_m: float = 12.0,
        closed: bool = True,
        point_count: int = 48,
        rotation_rad: float = 0.0,
    ) -> tuple[list[tuple[float, float, float]], dict]:
        """Generate absolute NED waypoints for an arbitrary custom shape prompt."""
        spec = self.generate_shape_spec(
            prompt,
            point_count=point_count,
            closed=closed,
        )
        local_points = self._family_to_local_points(spec, size_m=max(2.0, float(size_m)))
        if spec['family'] in {'rose', 'crescent', 'butterfly', 'cloud'}:
            local_points = self._chaikin_smooth(
                local_points,
                iterations=2,
                closed=bool(spec['closed']),
            )
        waypoints = self._rotate_translate_local_points(
            local_points,
            origin_x=origin_x,
            origin_y=origin_y,
            origin_z=origin_z,
            rotation_rad=rotation_rad,
            climb_m=float(spec.get('climb_m', 0.0)),
        )
        target_count = max(24, int(spec.get('points', point_count)) * 2)
        target_count = min(240, target_count)
        waypoints = self._resample_waypoints(waypoints, target_count=target_count)
        return waypoints, spec

    def _extract_tool_arguments(self, result: dict) -> dict:
        """Extract the first function-call argument object from an Ollama chat response."""
        message = result.get('message', {})
        tool_calls = message.get('tool_calls') or []
        if tool_calls:
            arguments = tool_calls[0].get('function', {}).get('arguments', {})
            if isinstance(arguments, dict):
                return arguments

        content = str(message.get('content', '')).strip()
        if content:
            start = content.find('{')
            end = content.rfind('}')
            if start >= 0 and end > start:
                content = content[start:end + 1]
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                return parsed

        raise ValueError('FunctionGemma returned no usable function-call arguments')

    def _normalize_spec(
        self,
        spec: dict,
        *,
        closed: bool,
        point_count: int,
        prompt: str,
    ) -> dict:
        """Normalize the raw model response into a safe procedural spec."""
        if not isinstance(spec, dict):
            raise ValueError('FunctionGemma did not return a JSON object')

        prompt_text = str(prompt).strip().lower()
        family = self._normalize_family_name(spec.get('family'))
        if not family:
            raise ValueError("FunctionGemma response missing 'family'")

        if 'clover' in prompt_text or 'quatrefoil' in prompt_text:
            family = 'rose'
        elif 'trefoil' in prompt_text:
            family = 'rose'
        elif 'crescent' in prompt_text or 'moon' in prompt_text:
            family = 'crescent'
        elif 'butterfly' in prompt_text:
            family = 'butterfly'
        elif 'arrow' in prompt_text or 'pointer' in prompt_text:
            family = 'arrow'
        elif 'diamond' in prompt_text or 'rhombus' in prompt_text or 'kite' in prompt_text:
            family = 'diamond'
        elif 'cloud' in prompt_text:
            family = 'cloud'

        try:
            petals = max(3, int(spec.get('petals', 4) or 4))
        except (TypeError, ValueError):
            petals = 4
        try:
            lobes = max(2, int(spec.get('lobes', 4) or 4))
        except (TypeError, ValueError):
            lobes = 4
        try:
            points = max(12, min(160, int(spec.get('points', point_count) or point_count)))
        except (TypeError, ValueError):
            points = max(12, min(160, int(point_count)))
        try:
            aspect_ratio = max(0.3, min(3.0, float(spec.get('aspect_ratio', 1.0) or 1.0)))
        except (TypeError, ValueError):
            aspect_ratio = 1.0
        try:
            thickness = max(0.1, min(1.0, float(spec.get('thickness', 0.45) or 0.45)))
        except (TypeError, ValueError):
            thickness = 0.45
        try:
            climb_m = max(0.0, min(40.0, float(spec.get('climb_m', 0.0) or 0.0)))
        except (TypeError, ValueError):
            climb_m = 0.0

        if 'clover' in prompt_text or 'quatrefoil' in prompt_text:
            petals = 4
        elif 'trefoil' in prompt_text:
            petals = 3
        elif 'flower' in prompt_text:
            flower_match = None
            for word, value in (
                ('three', 3),
                ('four', 4),
                ('five', 5),
                ('six', 6),
                ('seven', 7),
                ('eight', 8),
            ):
                if f'{word} petal' in prompt_text:
                    flower_match = value
                    break
            if flower_match is not None:
                petals = flower_match

        if not any(word in prompt_text for word in ('upward', 'rise', 'rising', 'climb', 'ascend', 'descending', 'descend', 'downward')):
            climb_m = 0.0
        if family == 'crescent' and thickness >= 0.9:
            thickness = 0.55
        if family == 'arrow' and thickness >= 0.8:
            thickness = 0.35

        normalized = {
            'family': family,
            'closed': bool(spec.get('closed', closed)),
            'petals': petals,
            'lobes': lobes,
            'points': points,
            'aspect_ratio': aspect_ratio,
            'thickness': thickness,
            'climb_m': climb_m,
        }
        return normalized

    def _normalize_family_name(self, family: object) -> str:
        """Map raw family names to implemented procedural families."""
        text = str(family or '').strip().lower().replace('-', '_').replace(' ', '_')
        aliases = {
            'rose': 'rose',
            'flower': 'rose',
            'petal': 'rose',
            'clover': 'rose',
            'quatrefoil': 'rose',
            'trefoil': 'rose',
            'crescent': 'crescent',
            'moon': 'crescent',
            'butterfly': 'butterfly',
            'arrow': 'arrow',
            'pointer': 'arrow',
            'diamond': 'diamond',
            'rhombus': 'diamond',
            'kite': 'diamond',
            'cloud': 'cloud',
            'blob': 'cloud',
            'puff': 'cloud',
        }
        return aliases.get(text, text)

    def _family_to_local_points(
        self,
        spec: dict,
        *,
        size_m: float,
    ) -> list[tuple[float, float]]:
        """Generate a local 2D point set for one procedural family."""
        family = spec['family']
        count = max(12, int(spec['points']))
        width = size_m
        height = size_m / max(0.3, float(spec['aspect_ratio']))

        if family == 'rose':
            petals = max(3, int(spec['petals']))
            k = petals if petals % 2 else max(1.0, petals / 2.0)
            points = []
            for idx in range(count):
                t = 2.0 * math.pi * idx / count
                r = 0.5 * math.cos(k * t)
                x = width * r * math.cos(t)
                y = height * r * math.sin(t)
                points.append((x, y))
            return self._close_points(points, spec['closed'])

        if family == 'crescent':
            outer_r = 0.5 * width
            inner_r = outer_r * max(0.25, min(0.85, float(spec['thickness'])))
            x_offset = outer_r * 0.35
            points = []
            half = max(8, count // 2)
            for idx in range(half):
                t = -0.9 * math.pi + 1.8 * math.pi * idx / max(half - 1, 1)
                points.append((outer_r * math.cos(t), height * 0.5 * math.sin(t)))
            for idx in range(half):
                t = 0.9 * math.pi - 1.8 * math.pi * idx / max(half - 1, 1)
                points.append((x_offset + inner_r * math.cos(t), height * 0.42 * math.sin(t)))
            return self._close_points(points, spec['closed'])

        if family == 'butterfly':
            points = []
            for idx in range(count):
                t = 12.0 * math.pi * idx / count
                r = math.exp(math.cos(t)) - 2.0 * math.cos(4.0 * t) + math.sin(t / 12.0) ** 5
                x = 0.12 * width * math.sin(t) * r
                y = 0.12 * height * math.cos(t) * r
                points.append((x, y))
            return self._close_points(points, spec['closed'])

        if family == 'arrow':
            tail_w = width * max(0.15, min(0.4, float(spec['thickness']) * 0.5))
            head_w = width * 0.6
            shaft_x = width * 0.15
            tip_x = width * 0.5
            back_x = -width * 0.5
            points = [
                (back_x, tail_w * 0.5),
                (shaft_x, tail_w * 0.5),
                (shaft_x, head_w * 0.5),
                (tip_x, 0.0),
                (shaft_x, -head_w * 0.5),
                (shaft_x, -tail_w * 0.5),
                (back_x, -tail_w * 0.5),
            ]
            return self._close_points(points, spec['closed'])

        if family == 'diamond':
            points = [
                (0.0, height * 0.5),
                (width * 0.5, 0.0),
                (0.0, -height * 0.5),
                (-width * 0.5, 0.0),
            ]
            return self._close_points(points, spec['closed'])

        if family == 'cloud':
            lobes = max(3, int(spec['lobes']))
            points = []
            for idx in range(count):
                t = 2.0 * math.pi * idx / count
                r = 0.38 + 0.12 * math.cos(lobes * t) + 0.04 * math.cos((lobes + 1) * t)
                x = width * r * math.cos(t)
                y = height * r * math.sin(t)
                points.append((x, y))
            return self._close_points(points, spec['closed'])

        raise ValueError(f'Unsupported FunctionGemma custom-shape family: {family}')

    def _rotate_translate_local_points(
        self,
        local_points: list[tuple[float, float]],
        *,
        origin_x: float,
        origin_y: float,
        origin_z: float,
        rotation_rad: float,
        climb_m: float,
    ) -> list[tuple[float, float, float]]:
        """Rotate and translate local XY points into absolute NED coordinates."""
        cos_yaw = math.cos(rotation_rad)
        sin_yaw = math.sin(rotation_rad)
        total = max(1, len(local_points) - 1)
        waypoints: list[tuple[float, float, float]] = []
        for idx, (lx, ly) in enumerate(local_points):
            gx = origin_x + lx * cos_yaw - ly * sin_yaw
            gy = origin_y + lx * sin_yaw + ly * cos_yaw
            gz = origin_z - climb_m * (idx / total)
            gz = max(-120.0, min(-1.0, gz))
            waypoints.append((gx, gy, gz))
        return waypoints

    def _close_points(
        self,
        points: list[tuple[float, float]],
        closed: bool,
    ) -> list[tuple[float, float]]:
        if not closed or not points:
            return points
        if self._dist2_xy(points[0], points[-1]) <= 1e-8:
            return points
        return points + [points[0]]

    def _chaikin_smooth(
        self,
        points: list[tuple[float, float]],
        *,
        iterations: int,
        closed: bool,
    ) -> list[tuple[float, float]]:
        """Apply Chaikin corner cutting to create a smoother dense curve."""
        if len(points) < 3 or iterations <= 0:
            return points

        result = list(points)
        if closed and self._dist2_xy(result[0], result[-1]) <= 1e-8:
            result = result[:-1]

        for _ in range(iterations):
            if len(result) < 3:
                break
            refined: list[tuple[float, float]] = []
            if closed:
                total = len(result)
                for idx in range(total):
                    p0 = result[idx]
                    p1 = result[(idx + 1) % total]
                    q = (
                        0.75 * p0[0] + 0.25 * p1[0],
                        0.75 * p0[1] + 0.25 * p1[1],
                    )
                    r = (
                        0.25 * p0[0] + 0.75 * p1[0],
                        0.25 * p0[1] + 0.75 * p1[1],
                    )
                    refined.extend((q, r))
            else:
                refined.append(result[0])
                for idx in range(len(result) - 1):
                    p0 = result[idx]
                    p1 = result[idx + 1]
                    q = (
                        0.75 * p0[0] + 0.25 * p1[0],
                        0.75 * p0[1] + 0.25 * p1[1],
                    )
                    r = (
                        0.25 * p0[0] + 0.75 * p1[0],
                        0.25 * p0[1] + 0.75 * p1[1],
                    )
                    refined.extend((q, r))
                refined.append(result[-1])
            result = refined

        return self._close_points(result, closed)

    def _resample_waypoints(
        self,
        waypoints: list[tuple[float, float, float]],
        *,
        target_count: int,
    ) -> list[tuple[float, float, float]]:
        """Redistribute waypoint spacing along the polyline for smoother flight."""
        if len(waypoints) < 3 or target_count <= len(waypoints):
            return waypoints

        cumulative = [0.0]
        for idx in range(1, len(waypoints)):
            prev = waypoints[idx - 1]
            curr = waypoints[idx]
            cumulative.append(
                cumulative[-1] + math.sqrt(
                    (curr[0] - prev[0]) ** 2 +
                    (curr[1] - prev[1]) ** 2 +
                    (curr[2] - prev[2]) ** 2
                )
            )

        total = cumulative[-1]
        if total <= 1e-6:
            return waypoints

        result = [waypoints[0]]
        for idx in range(1, target_count - 1):
            target_s = total * idx / (target_count - 1)
            seg_idx = 1
            while seg_idx < len(cumulative) and cumulative[seg_idx] < target_s:
                seg_idx += 1
            seg_idx = min(seg_idx, len(cumulative) - 1)
            s0 = cumulative[seg_idx - 1]
            s1 = cumulative[seg_idx]
            p0 = waypoints[seg_idx - 1]
            p1 = waypoints[seg_idx]
            alpha = 0.0 if s1 <= s0 else (target_s - s0) / (s1 - s0)
            result.append((
                p0[0] + alpha * (p1[0] - p0[0]),
                p0[1] + alpha * (p1[1] - p0[1]),
                p0[2] + alpha * (p1[2] - p0[2]),
            ))
        result.append(waypoints[-1])
        return result

    def _dist2_xy(
        self,
        a: tuple[float, float],
        b: tuple[float, float],
    ) -> float:
        dx = float(a[0]) - float(b[0])
        dy = float(a[1]) - float(b[1])
        return dx * dx + dy * dy
