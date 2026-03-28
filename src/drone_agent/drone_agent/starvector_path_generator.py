#!/usr/bin/env python3
"""Generate arbitrary SVG-like paths from text prompts using StarVector."""

from __future__ import annotations

import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable

import torch
from svgpathtools import svgstr2paths
from transformers import AutoModelForCausalLM


class StarVectorPathGenerator:
    """Lazy local wrapper around StarVector text-to-SVG generation."""

    def __init__(
        self,
        *,
        model_name: str = 'starvector/starvector-1b-im2svg',
        source_dir: str | None = None,
        hf_cache_dir: str | None = None,
    ):
        cache_root = Path.home() / '.cache' / 'drone_agent'
        self.model_name = model_name
        self.source_dir = Path(source_dir or (cache_root / 'starvector-src'))
        self.hf_cache_dir = Path(hf_cache_dir or (cache_root / 'huggingface'))
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dtype = torch.float16 if self.device == 'cuda' else torch.float32

    def ensure_runtime(self):
        """Ensure the upstream StarVector source tree is available on PYTHONPATH."""
        if not self.source_dir.exists():
            self.source_dir.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [
                    'git',
                    'clone',
                    '--depth',
                    '1',
                    'https://github.com/joanrod/star-vector.git',
                    str(self.source_dir),
                ],
                check=True,
            )

        source_path = str(self.source_dir)
        if source_path not in sys.path:
            sys.path.insert(0, source_path)

        os.environ.setdefault('HF_HUB_DISABLE_XET', '1')

    def _lazy_load_model(self):
        """Load the StarVector checkpoint once on first use."""
        if self.model is not None:
            return

        self.ensure_runtime()
        self.hf_cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=self.dtype,
            cache_dir=str(self.hf_cache_dir),
            low_cpu_mem_usage=True,
        )
        self.model = self.model.to(self.device)
        self.model.eval()

    def generate_svg(self, prompt: str, *, max_length: int = 1024) -> str:
        """Generate an SVG string from a natural-language shape prompt."""
        self._lazy_load_model()

        text_prompt = self._normalize_prompt(prompt)
        dummy_image = torch.zeros((1, 3, 224, 224), device=self.device, dtype=self.dtype)
        batch = {
            'caption': [text_prompt],
            'image': dummy_image,
        }
        with torch.inference_mode():
            token_ids = self.model.model.generate_text2svg(
                batch=batch,
                max_length=max_length,
                num_beams=2,
                use_nucleus_sampling=False,
                temperature=1.0,
            )
        svg = self.model.model.svg_transformer.tokenizer.batch_decode(
            token_ids,
            skip_special_tokens=True,
        )[0].strip()
        return self._normalize_svg_text(svg)

    def generate_waypoints(
        self,
        prompt: str,
        *,
        origin_x: float,
        origin_y: float,
        origin_z: float,
        size_m: float = 12.0,
        closed: bool = True,
        point_count: int = 120,
        rotation_rad: float = 0.0,
    ) -> tuple[list[tuple[float, float, float]], str]:
        """Generate absolute NED waypoints for an arbitrary prompt-described shape."""
        svg_text = self.generate_svg(prompt)
        polylines = self._svg_to_polylines(svg_text, point_count=max(24, point_count))
        points = self._normalize_polylines_to_ned(
            polylines,
            origin_x=origin_x,
            origin_y=origin_y,
            origin_z=origin_z,
            size_m=size_m,
            closed=closed,
            rotation_rad=rotation_rad,
        )
        return points, svg_text

    def _normalize_prompt(self, prompt: str) -> str:
        """Bias StarVector toward clean centered vector outlines."""
        text = ' '.join(str(prompt).strip().split())
        if not text:
            text = 'simple geometric shape'
        return (
            'A clean centered single-outline SVG icon of '
            f'{text}. Use a white background and simple vector paths.'
        )

    def _normalize_svg_text(self, svg: str) -> str:
        """Ensure the model output is wrapped as an SVG document."""
        text = svg.strip()
        if '<svg' in text.lower():
            return text
        return (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512">'
            f'{text}'
            '</svg>'
        )

    def _svg_to_polylines(
        self,
        svg_text: str,
        *,
        point_count: int,
    ) -> list[list[tuple[float, float]]]:
        """Convert an SVG string into sampled 2D polylines."""
        try:
            paths, _, _ = svgstr2paths(svg_text)
        except Exception:
            # Some parsers behave better on files than raw strings.
            with tempfile.NamedTemporaryFile('w', suffix='.svg', delete=False) as tmp:
                tmp.write(svg_text)
                tmp_path = tmp.name
            try:
                from svgpathtools import svg2paths2
                paths, _, _ = svg2paths2(tmp_path)
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

        if not paths:
            raise ValueError('StarVector returned no SVG paths')

        lengths = [max(path.length(error=1e-4), 1e-6) for path in paths]
        total_length = sum(lengths)
        polylines = []
        for path, length in zip(paths, lengths):
            samples = max(8, int(round(point_count * (length / total_length))))
            polyline = []
            for idx in range(samples):
                t = idx / max(samples - 1, 1)
                point = path.point(t)
                polyline.append((float(point.real), float(point.imag)))
            polylines.append(polyline)

        return self._reorder_polylines(polylines)

    def _reorder_polylines(
        self,
        polylines: list[list[tuple[float, float]]],
    ) -> list[list[tuple[float, float]]]:
        """Greedily connect multiple SVG strokes into one sensible flight order."""
        if len(polylines) <= 1:
            return polylines

        remaining = [list(poly) for poly in polylines if poly]
        ordered = [remaining.pop(0)]

        while remaining:
            last_point = ordered[-1][-1]
            best_idx = 0
            best_reverse = False
            best_dist = float('inf')
            for idx, poly in enumerate(remaining):
                start_dist = self._dist2(last_point, poly[0])
                end_dist = self._dist2(last_point, poly[-1])
                if start_dist < best_dist:
                    best_idx = idx
                    best_reverse = False
                    best_dist = start_dist
                if end_dist < best_dist:
                    best_idx = idx
                    best_reverse = True
                    best_dist = end_dist

            next_poly = remaining.pop(best_idx)
            if best_reverse:
                next_poly.reverse()
            ordered.append(next_poly)

        return ordered

    def _normalize_polylines_to_ned(
        self,
        polylines: list[list[tuple[float, float]]],
        *,
        origin_x: float,
        origin_y: float,
        origin_z: float,
        size_m: float,
        closed: bool,
        rotation_rad: float,
    ) -> list[tuple[float, float, float]]:
        """Center, scale, rotate, and translate SVG points into NED metres."""
        all_points = [pt for poly in polylines for pt in poly]
        if not all_points:
            raise ValueError('No SVG points available for waypoint conversion')

        min_x = min(pt[0] for pt in all_points)
        max_x = max(pt[0] for pt in all_points)
        min_y = min(pt[1] for pt in all_points)
        max_y = max(pt[1] for pt in all_points)
        span_x = max(max_x - min_x, 1e-6)
        span_y = max(max_y - min_y, 1e-6)
        span = max(span_x, span_y)
        scale = max(0.5, float(size_m)) / span
        center_x = 0.5 * (min_x + max_x)
        center_y = 0.5 * (min_y + max_y)

        cos_yaw = math.cos(rotation_rad)
        sin_yaw = math.sin(rotation_rad)

        waypoints: list[tuple[float, float, float]] = []
        for poly in polylines:
            for px, py in poly:
                local_x = (px - center_x) * scale
                # SVG Y points down; convert to a conventional upright local frame.
                local_y = -(py - center_y) * scale
                ned_x = origin_x + local_x * cos_yaw - local_y * sin_yaw
                ned_y = origin_y + local_x * sin_yaw + local_y * cos_yaw
                waypoints.append((ned_x, ned_y, origin_z))

        if closed and waypoints and self._dist2(waypoints[0][:2], waypoints[-1][:2]) > 1e-8:
            waypoints.append((waypoints[0][0], waypoints[0][1], origin_z))

        return self._resample_waypoints(waypoints, target_count=120 if len(waypoints) > 2 else len(waypoints))

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

    def _dist2(self, a: Iterable[float], b: Iterable[float]) -> float:
        ax, ay = a
        bx, by = b
        dx = ax - bx
        dy = ay - by
        return dx * dx + dy * dy
