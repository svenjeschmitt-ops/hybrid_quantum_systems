"""Validated neighboring-level specification for the effective atomic model.

Offsets are MHz relative to r0 (branch 0) or r1 (branch 1) at zero field.
This is an effective basis specification, not an atomic-structure calculation.
"""
import json
import math

DEFAULT_MANIFOLD = json.dumps([dict(label="wrong_upper", branch=1,
    offset_MHz=1320.0, stark_MHz_per_Vcm2=-12.0, lifetime_us=100.0,
    optical_relative=1.0, cavity_from={"wrong": 1.0})])


def parse_manifold(value):
    """Accept a JSON string or list; reject ambiguous or unphysical entries."""
    try:
        levels = json.loads(value) if isinstance(value, str) else value
    except (ValueError, TypeError) as exc:
        raise ValueError("RYDBERG_MANIFOLD must be a JSON array of level objects.") from exc
    if not isinstance(levels, list):
        raise ValueError("RYDBERG_MANIFOLD must be a JSON array.")
    labels = {"h0", "h1", "r0", "r1", "lost", "wrong"}
    result = []
    required = {"label", "branch", "offset_MHz", "stark_MHz_per_Vcm2",
                "lifetime_us", "optical_relative", "cavity_from"}
    for raw in levels:
        if not isinstance(raw, dict) or set(raw) != required:
            raise ValueError("Each neighboring level requires exactly: " + ", ".join(sorted(required)))
        level = dict(raw)
        label = level["label"]
        if not isinstance(label, str) or not label.isidentifier() or label in labels:
            raise ValueError("Neighboring-level labels must be unique identifiers, distinct from built-in levels.")
        labels.add(label)
        if isinstance(level["branch"], bool) or level["branch"] not in (0, 1):
            raise ValueError("Atomic branch must be 0 (r0-like) or 1 (r1-like).")
        level["branch"] = int(level["branch"])
        for key in ("offset_MHz", "stark_MHz_per_Vcm2", "lifetime_us", "optical_relative"):
            try:
                level[key] = float(level[key])
            except (ValueError, TypeError):
                raise ValueError(f"{label}: {key} must be finite.") from None
            if not math.isfinite(level[key]):
                raise ValueError(f"{label}: {key} must be finite.")
        if level["lifetime_us"] <= 0:
            raise ValueError(f"{label}: lifetime_us must be positive.")
        links = level["cavity_from"]
        if not isinstance(links, dict) or (level["branch"] == 0 and links):
            raise ValueError("cavity_from must be a mapping on upper-branch levels only.")
        level["cavity_from"] = dict(links)
        for source, amplitude in links.items():
            try:
                amplitude = float(amplitude)
            except (ValueError, TypeError):
                raise ValueError("Cavity relative amplitudes must be finite real numbers.") from None
            if not math.isfinite(amplitude):
                raise ValueError("Cavity relative amplitudes must be finite real numbers.")
            level["cavity_from"][source] = amplitude
        result.append(level)
    lower = {"r0", "wrong"} | {x["label"] for x in result if x["branch"] == 0}
    for level in result:
        if set(level["cavity_from"]) - lower:
            raise ValueError("cavity_from must reference a defined lower-branch Rydberg level.")
    return result
