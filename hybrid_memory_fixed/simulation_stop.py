"""Cooperative cancellation and honest plots of calculated trajectory samples."""
import os
from pathlib import Path

STOP_EXIT_CODE = 3
_label = "Simulation"
_trajectory_label = "Simulation"
_times = []
_states = []


class CalculationStopped(Exception):
    pass


def requested():
    path = os.environ.get("HYBRID_STOP_FILE")
    return bool(path and Path(path).is_file())


def check():
    if requested():
        raise CalculationStopped()


def stage(label):
    global _label
    _label = label
    check()


def begin():
    global _times, _states, _trajectory_label
    if os.environ.get("HYBRID_STOP_FILE"):
        check()
        _times, _states = [], []
        _trajectory_label = _label


def sample(t, state):
    if os.environ.get("HYBRID_STOP_FILE"):
        _times.append(float(t))
        _states.append(state.copy())
        check()
    return 0.0


def solve(solver, *args, **kwargs):
    """Keep output-grid samples without changing solver accuracy or results."""
    capture = kwargs.pop("_capture", True)
    if not os.environ.get("HYBRID_STOP_FILE"):
        return solver(*args, **kwargs)
    if capture:
        begin()
    def checkpoint(t, state):
        if capture:
            return sample(t, state)
        check()
        return 0.0
    ops = kwargs.get("e_ops")
    if isinstance(ops, dict):
        existing = dict(ops)
    else:
        items = [] if ops is None else list(ops) if isinstance(ops, (list, tuple)) else [ops]
        existing = dict(enumerate(items))
    checkpoint_key = object()
    kwargs["e_ops"] = {**existing, checkpoint_key: checkpoint}
    # Adding an expectation callback must not disable QuTiP's default storage.
    options = dict(kwargs.get("options") or {})
    if options.get("store_states") is None and not existing:
        options["store_states"] = True
    kwargs["options"] = options
    result = solver(*args, **kwargs)
    del result.e_data[checkpoint_key]
    del result.e_ops[checkpoint_key]
    return result


def show_available(backend, output_directory=None):
    """Keep completed figures and add the latest physical trajectory."""
    import matplotlib.pyplot as plt
    import qutip as qt
    import json

    if _states and len(_states[0].dims[0]) == 3:
        operators = (backend.Pe, backend.n, backend.Ph0, backend.Ph1,
                     backend.Pr0, backend.Pr1, backend.Plost, backend.Pwrong)
        labels = ("Transmon e", "Cavity mean photons", "Atom h0", "Atom h1",
                  "Atom r0", "Atom r1", "Atomic loss", "Unwanted atomic levels")
        fig, ax = plt.subplots(figsize=(11, 6), layout="constrained")
        series = {}
        for op, label in zip(operators, labels):
            values = qt.expect(op, _states).real
            ax.plot(_times, values, label=label)
            series[label] = values.tolist()
        ax.set(xlabel="Trajectory time (us)", ylabel="Population / mean photon number",
               title=f"Stopped — {_trajectory_label}\nCalculated samples through {_times[-1]:.6g} us")
        ax.legend()
        if output_directory:
            folder = Path(output_directory)
            folder.mkdir(parents=True, exist_ok=True)
            fig.savefig(folder / "stopped_trajectory.png", dpi=150)
            (folder / "stopped_trajectory.json").write_text(json.dumps({
                "status": "stopped", "stage": _trajectory_label, "time_us": _times,
                "series": series}, indent=2))
    if output_directory:
        folder = Path(output_directory)
        sweep = folder / "parameter_sweep.json"
        if sweep.is_file():
            from sweep_figure import SweepFigure
            report = json.loads(sweep.read_text())
            if report["results"]:
                for direction in ("write", "read"):
                    if direction + "_fidelity" in report["results"][0]:
                        SweepFigure(backend, report, folder, direction)
        # Sweep workers cannot open windows; recover their saved partial plot.
        partial = folder / "stopped_trajectory.png"
        if not _states and partial.is_file():
            fig, ax = plt.subplots(figsize=(11, 6), layout="constrained")
            ax.imshow(plt.imread(partial))
            ax.set_axis_off()
    count = len(plt.get_fignums())
    print(f"Calculation stopped. Showing {count} available figure(s)." if count else
          "Calculation stopped before any trajectory samples or figures were available.", flush=True)
