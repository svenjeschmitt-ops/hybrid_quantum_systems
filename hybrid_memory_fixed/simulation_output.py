"""Run-specific numerical reports for the panel assistant, independent of plot export."""
import json
from pathlib import Path

import numpy as np


def real_probability(value, tolerance=1e-7):
    """Accept roundoff imaginary parts, reject nonphysical expectation values."""
    scalar=complex(value)
    if not (np.isfinite(scalar.real) and np.isfinite(scalar.imag)):
        raise ValueError('Probability expectation is nonfinite.')
    if abs(scalar.imag)>tolerance:
        raise ValueError('Probability expectation has a significant imaginary component; check solver convergence.')
    if not -tolerance<=scalar.real<=1+tolerance:
        raise ValueError('Probability expectation is outside the physical range; check solver convergence.')
    return float(scalar.real)


def trace(time_us, **series):
    times = np.asarray(time_us, dtype=float)
    if times.ndim != 1 or not len(times) or not np.isfinite(times).all():
        raise ValueError("Trace needs a finite, nonempty time axis.")
    if np.any(np.diff(times) < 0):
        raise ValueError("Trace times must be ordered.")
    values = {name: np.asarray(value, dtype=float) for name, value in series.items()}
    if any(value.shape != times.shape or not np.isfinite(value).all() for value in values.values()):
        raise ValueError("Trace series must be finite and match the time axis.")
    return {"time_us": times.tolist(), "series": {key: value.tolist() for key, value in values.items()}}


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_report(directory, direction, settings, metadata, summary, traces, basis):
    """Atomically publish complete numerical data before the figure show blocks."""
    if direction not in ("write", "read", "roundtrip"):
        raise ValueError("Unsupported report direction")
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder/(direction+".json")
    report = {"schema_version": 1, "direction": direction, "settings": settings,
              "metadata": metadata, "summary": summary, "traces": traces,
              "basis_outputs": basis}
    encoded = json.dumps(report, default=_json_value, allow_nan=False)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(path)
    print(f"Assistant numerical report ready: {path}", flush=True)
    return path


def trace_preview(data):
    """Keep a bounded preview plus exact extrema; full-resolution data stays in the file."""
    times = np.asarray(data["time_us"])
    indices = np.unique(np.linspace(0, len(times)-1, min(33, len(times)), dtype=int))
    stats = {}
    for name, values in data["series"].items():
        values = np.asarray(values)
        low, high = int(np.argmin(values)), int(np.argmax(values))
        stats[name] = {"initial": float(values[0]), "final": float(values[-1]),
                       "minimum": float(values[low]), "minimum_time_us": float(times[low]),
                       "maximum": float(values[high]), "maximum_time_us": float(times[high])}
    return {"full_point_count": len(times), "sampled_time_us": times[indices].tolist(),
            "sampled_series": {name: np.asarray(values)[indices].tolist() for name, values in data["series"].items()},
            "exact_sampled_grid_statistics": stats,
            "sampling_note": "Uniform index preview only; narrow features may be omitted. "
                             "Extrema use the full exported grid, not continuous-time extrema. Read the full file for detail."}


def load_reports(directory, settings):
    """Read only this run's known report files and reject mismatched provenance."""
    reports, errors = [], []
    if not directory:
        return reports, errors
    for direction in ("write", "read", "roundtrip"):
        path = Path(directory)/(direction+".json")
        if not path.exists():
            continue
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
            if report["schema_version"] != 1 or report["settings"] != settings or report["direction"] != direction:
                raise ValueError("Report settings/schema/direction do not match this run")
            reports.append({"direction": direction, "full_resolution_file": str(path.resolve()),
                            "metadata": report["metadata"], "summary": report["summary"],
                            "basis_outputs": report["basis_outputs"],
                            "trace_previews": {name: trace_preview(data) for name, data in report["traces"].items()}})
        except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
            errors.append(f"Could not use {direction} numerical report: {exc}")
    return reports, errors
