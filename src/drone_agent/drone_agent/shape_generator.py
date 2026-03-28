#!/usr/bin/env python3
"""Deterministic geometry for named flight shapes."""

from __future__ import annotations

import math

SUPPORTED_SHAPES = {
    "circle",
    "square",
    "rectangle",
    "triangle",
    "polygon",
    "star",
    "figure_eight",
    "zigzag",
    "spiral",
    "heart",
}


def clamp(value: float, low: float, high: float) -> float:
    """Clamp a floating-point value to a closed interval."""
    return max(low, min(high, value))


def _shape_alias(name: str) -> str:
    """Normalize shape aliases to one canonical name."""
    text = str(name).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "circle": "circle",
        "orbit": "circle",
        "square": "square",
        "box": "square",
        "rectangle": "rectangle",
        "rect": "rectangle",
        "triangle": "triangle",
        "spiral": "spiral",
        "helix": "spiral",
        "corkscrew": "spiral",
        "heart": "heart",
        "zigzag": "zigzag",
        "figure_eight": "figure_eight",
        "figure8": "figure_eight",
        "lemniscate": "figure_eight",
        "infinity": "figure_eight",
        "star": "star",
        "polygon": "polygon",
        "pentagon": "polygon",
        "hexagon": "polygon",
        "octagon": "polygon",
    }
    return aliases.get(text, text)


def is_supported_shape_name(name: str | None) -> bool:
    """Return True when the provided name maps to a deterministic built-in shape."""
    if name is None:
        return False
    return _shape_alias(name) in SUPPORTED_SHAPES


def _shape_default_point_count(shape: str) -> int:
    if shape in {"spiral", "circle", "figure_eight", "heart"}:
        return 48
    if shape in {"zigzag", "star"}:
        return 10
    return 5


def _shape_default_radius(shape: str) -> float:
    if shape == "spiral":
        return 12.0
    if shape in {"circle", "figure_eight", "triangle", "polygon", "star", "heart"}:
        return 10.0
    return 12.0


def _shape_default_width(shape: str) -> float:
    return 18.0 if shape == "rectangle" else 16.0


def _shape_default_height(shape: str) -> float:
    return 10.0


def _shape_default_side(shape: str) -> float:
    return 12.0


def normalize_shape_spec(spec: dict) -> dict:
    """Normalize a shape spec into a safe deterministic parameter set."""
    normalized = dict(spec)
    shape = _shape_alias(spec.get("shape", ""))
    if not shape:
        raise ValueError("shape spec missing 'shape'")

    normalized["shape"] = shape
    normalized["radius"] = max(0.5, float(spec.get("radius", spec.get("radius_m", _shape_default_radius(shape))) or _shape_default_radius(shape)))
    normalized["width"] = max(0.5, float(spec.get("width", spec.get("width_m", _shape_default_width(shape))) or _shape_default_width(shape)))
    normalized["height"] = max(0.5, float(spec.get("height", spec.get("height_m", _shape_default_height(shape))) or _shape_default_height(shape)))
    normalized["side"] = max(0.5, float(spec.get("side", spec.get("side_m", _shape_default_side(shape))) or _shape_default_side(shape)))
    normalized["sides"] = max(3, int(spec.get("sides", 5) or 5))
    normalized["turns"] = max(0.25, float(spec.get("turns", 2.0) or 2.0))
    normalized["climb_m"] = float(spec.get("climb_m", 0.0) or 0.0)
    normalized["forward_m"] = max(1.0, float(spec.get("forward_m", 24.0) or 24.0))
    normalized["amplitude_m"] = max(0.2, float(spec.get("amplitude_m", 4.0) or 4.0))
    normalized["point_count"] = max(4, int(spec.get("point_count", _shape_default_point_count(shape)) or _shape_default_point_count(shape)))
    normalized["clockwise"] = bool(spec.get("clockwise", False))
    normalized["closed"] = bool(spec.get("closed", True))
    normalized["loop"] = bool(spec.get("loop", False))
    normalized["speed"] = max(0.1, float(spec.get("speed", spec.get("speed_m_s", 2.0)) or 2.0))
    normalized["rotation_rad"] = float(spec.get("rotation_rad", spec.get("reference_yaw_rad", 0.0)) or 0.0)

    if shape in {"spiral", "circle", "figure_eight", "heart"}:
        normalized["point_count"] = int(clamp(normalized["point_count"], 24, 120))
    elif shape in {"zigzag", "star"}:
        normalized["point_count"] = int(clamp(normalized["point_count"], 6, 32))
    else:
        normalized["point_count"] = int(clamp(normalized["point_count"], 4, 24))

    if shape == "polygon":
        polygon_sides = spec.get("polygon_sides")
        if polygon_sides is not None:
            normalized["sides"] = max(3, int(polygon_sides))
        elif "pentagon" in str(spec.get("shape", "")).lower():
            normalized["sides"] = 5
        elif "hexagon" in str(spec.get("shape", "")).lower():
            normalized["sides"] = 6
        elif "octagon" in str(spec.get("shape", "")).lower():
            normalized["sides"] = 8

    return normalized


def _close_points(points: list[tuple[float, float, float]]) -> list[tuple[float, float, float]]:
    if not points or points[0] == points[-1]:
        return points
    return points + [points[0]]


def _rotate_translate(
    local_points: list[tuple[float, float, float]],
    origin_x: float,
    origin_y: float,
    origin_z: float,
    rotation_rad: float,
) -> list[tuple[float, float, float]]:
    cos_yaw = math.cos(rotation_rad)
    sin_yaw = math.sin(rotation_rad)
    points = []
    for lx, ly, z in local_points:
        gx = origin_x + lx * cos_yaw - ly * sin_yaw
        gy = origin_y + lx * sin_yaw + ly * cos_yaw
        gz = clamp(z, -120.0, -1.0)
        points.append((gx, gy, gz))
    return points


def _regular_polygon_points(
    sides: int,
    radius: float,
    z_value: float,
    start_angle_rad: float = 0.0,
) -> list[tuple[float, float, float]]:
    return [
        (
            radius * math.cos(start_angle_rad + 2.0 * math.pi * i / sides),
            radius * math.sin(start_angle_rad + 2.0 * math.pi * i / sides),
            z_value,
        )
        for i in range(sides)
    ]


def generate_shape_waypoints(
    spec: dict,
    origin_x: float,
    origin_y: float,
    origin_z: float,
) -> tuple[list[tuple[float, float, float]], dict]:
    """Generate absolute NED waypoints for a named shape."""
    norm = normalize_shape_spec(spec)
    shape = norm["shape"]
    z0 = clamp(float(spec.get("alt_z", origin_z)), -120.0, -1.0)
    cw = -1.0 if norm["clockwise"] else 1.0

    if shape == "circle":
        local = []
        for i in range(norm["point_count"]):
            theta = cw * (2.0 * math.pi * i / norm["point_count"])
            local.append((norm["radius"] * math.cos(theta), norm["radius"] * math.sin(theta), z0))
        if norm["closed"]:
            local = _close_points(local)
        return _rotate_translate(local, origin_x, origin_y, z0, norm["rotation_rad"]), norm

    if shape == "square":
        half = norm["side"] / 2.0
        local = [(half, half, z0), (half, -half, z0), (-half, -half, z0), (-half, half, z0)]
        if norm["closed"]:
            local = _close_points(local)
        return _rotate_translate(local, origin_x, origin_y, z0, norm["rotation_rad"]), norm

    if shape == "rectangle":
        half_w = norm["width"] / 2.0
        half_h = norm["height"] / 2.0
        local = [
            (half_w, half_h, z0),
            (half_w, -half_h, z0),
            (-half_w, -half_h, z0),
            (-half_w, half_h, z0),
        ]
        if norm["closed"]:
            local = _close_points(local)
        return _rotate_translate(local, origin_x, origin_y, z0, norm["rotation_rad"]), norm

    if shape == "triangle":
        local = _regular_polygon_points(3, norm["radius"], z0, start_angle_rad=-math.pi / 2.0)
        if norm["closed"]:
            local = _close_points(local)
        return _rotate_translate(local, origin_x, origin_y, z0, norm["rotation_rad"]), norm

    if shape == "polygon":
        local = _regular_polygon_points(norm["sides"], norm["radius"], z0, start_angle_rad=-math.pi / 2.0)
        if norm["closed"]:
            local = _close_points(local)
        return _rotate_translate(local, origin_x, origin_y, z0, norm["rotation_rad"]), norm

    if shape == "zigzag":
        segments = max(4, norm["point_count"])
        local = []
        for i in range(segments):
            progress = i / max(segments - 1, 1)
            x = progress * norm["forward_m"]
            y = 0.0 if i == 0 else (norm["amplitude_m"] if i % 2 else -norm["amplitude_m"])
            local.append((x, y, z0))
        return _rotate_translate(local, origin_x, origin_y, z0, norm["rotation_rad"]), norm

    if shape == "figure_eight":
        local = []
        for i in range(norm["point_count"]):
            t = cw * (2.0 * math.pi * i / norm["point_count"])
            x = norm["radius"] * math.sin(t)
            y = 0.5 * norm["radius"] * math.sin(2.0 * t)
            local.append((x, y, z0))
        if norm["closed"]:
            local = _close_points(local)
        return _rotate_translate(local, origin_x, origin_y, z0, norm["rotation_rad"]), norm

    if shape == "heart":
        local = []
        scale = norm["radius"] / 17.0
        for i in range(norm["point_count"]):
            t = cw * (2.0 * math.pi * i / norm["point_count"])
            # Classic parametric heart, normalized and rotated into XY.
            x = 16.0 * math.sin(t) ** 3
            y = (
                13.0 * math.cos(t)
                - 5.0 * math.cos(2.0 * t)
                - 2.0 * math.cos(3.0 * t)
                - math.cos(4.0 * t)
            )
            local.append((scale * x, scale * y, z0))
        if norm["closed"]:
            local = _close_points(local)
        return _rotate_translate(local, origin_x, origin_y, z0, norm["rotation_rad"]), norm

    if shape == "star":
        outer = norm["radius"]
        inner = max(outer * 0.45, 0.5)
        local = []
        vertex_count = max(5, norm["sides"])
        for i in range(vertex_count * 2):
            angle = cw * (math.pi * i / vertex_count) - math.pi / 2.0
            radius = outer if i % 2 == 0 else inner
            local.append((radius * math.cos(angle), radius * math.sin(angle), z0))
        if norm["closed"]:
            local = _close_points(local)
        return _rotate_translate(local, origin_x, origin_y, z0, norm["rotation_rad"]), norm

    if shape == "spiral":
        local = []
        z1 = clamp(z0 - norm["climb_m"], -120.0, -1.0)
        for i in range(norm["point_count"]):
            progress = i / max(norm["point_count"] - 1, 1)
            theta = cw * (2.0 * math.pi * norm["turns"] * progress)
            radius = norm["radius"] * progress
            z = z0 + (z1 - z0) * progress
            local.append((radius * math.cos(theta), radius * math.sin(theta), z))
        return _rotate_translate(local, origin_x, origin_y, z0, norm["rotation_rad"]), norm

    raise ValueError(f"Unsupported shape: {shape}")
