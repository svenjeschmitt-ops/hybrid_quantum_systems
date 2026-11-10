"""Deterministic, static-error grid scan with a fixed nominal phase reference."""
from contextlib import contextmanager
from datetime import datetime
from itertools import product
import json
from pathlib import Path
import uuid

import numpy as np
import simulation_progress as progress

from hybrid_settings import validate

AXES = (
    ("FC", "FC_TOL", "MHz"),
    ("FT", "FT_TOL", "MHz"),
    ("FA", "FA_TOL", "MHz"),
    ("FLUX", "FLUX_TOL", "Phi/Phi0"),
    ("EFIELD", "EFIELD_TOL", "V/cm"),
)


def scenarios(config):
    """Nominal first, then every distinct -bound/zero/+bound combination."""
    c = validate(config)
    zero = {axis: 0.0 for axis, _, _ in AXES}
    yield zero
    levels = [(-c[key], 0.0, c[key]) if c[key] else (0.0,)
              for _, key, _ in AXES]
    for values in product(*levels):
        if any(values):
            yield dict(zip(zero, values))


@contextmanager
def fixed_offsets(backend, offsets):
    """Do not let the cached parked Lindblad generator cross realizations."""
    previous = dict(backend.TOLERANCE_OFFSETS)
    backend.parked_storage_solver.cache_clear()
    backend.TOLERANCE_OFFSETS.update(offsets)
    try:
        yield
    finally:
        backend.TOLERANCE_OFFSETS.update(previous)
        backend.parked_storage_solver.cache_clear()


def evaluate(backend, alpha, beta, phases):
    """Selected-input joint overlaps, including storage in both directions.

    Readout starts from an independently prepared atomic input, just like the
    normal read figure. No phase is fitted to the perturbed device.
    """
    b = backend
    target = (alpha * b.g + beta * b.e).unit()
    _, states = b.simulate(alpha, beta, noisy=True, include_storage=False)
    held = b.hold_memory(states[-1], noisy=True)
    write = float(np.real(b.qt.expect(b.atomic_storage_effect(target, phases[0]), held)))
    initial = b.stored_retrieval_initial(alpha, beta, noisy=True)
    _, states, _, _, _ = b.evolve_read(initial, noisy=True)
    read = float(b.read_metrics(states[-1:], target, phases[1])[2][-1])
    return {"write_fidelity": write, "read_fidelity": read}


def summarize(results):
    return {
        direction: {
            "nominal": results[0],
            "best": max(results, key=lambda row: row[direction + "_fidelity"]),
            "worst": min(results, key=lambda row: row[direction + "_fidelity"]),
        }
        for direction in ("write", "read")
    }


def run(backend, config, save=False, output_directory=None):
    import matplotlib.pyplot as plt

    b = backend
    c = validate(config)
    points = list(scenarios(c))
    theta, phi = np.deg2rad([c["INPUT_THETA"], c["INPUT_PHI"]])
    alpha, beta = np.cos(theta / 2), np.exp(1j * phi) * np.sin(theta / 2)
    print(f"Tolerance scan: {len(points)} deterministic points; selected input "
          f"theta={c['INPUT_THETA']:g}, phi={c['INPUT_PHI']:g} degrees.\n"
          "Offsets are constant throughout each run. Best/worst are sampled extrema, "
          "not guaranteed bounds. Ordinary transfer plots also show sampled tolerance bands.", flush=True)
    with fixed_offsets(b, {axis: 0.0 for axis, _, _ in AXES}):
        print("Computing fixed nominal phase references...", flush=True)
        progress.update(0, "Calibrating tolerance scan")
        phases = (b.calibrated_phase(), b.read_phase())
        progress.update(.10, "Tolerance scan calibrated")
    results = []
    for index, offsets in enumerate(points):
        print(f"Tolerance point {index + 1}/{len(points)}: {offsets}", flush=True)
        with fixed_offsets(b, offsets):
            values = evaluate(b, alpha, beta, phases)
        results.append({"offsets": offsets, **values})
        progress.update(.10+.85*(index+1)/len(points), f"Tolerance scenario {index+1}/{len(points)} complete")
        print(f"  write+storage={values['write_fidelity']:.8f}; "
              f"storage+read={values['read_fidelity']:.8f}", flush=True)
    progress.update(.95, "Building tolerance summary")
    summary = summarize(results)
    for direction, values in summary.items():
        print(f"\n{direction.upper()}: nominal "
              f"{100 * values['nominal'][direction + '_fidelity']:.6f}%", flush=True)
        for label in ("best", "worst"):
            row = values[label]
            print(f"  {label} sampled: {100 * row[direction + '_fidelity']:.6f}% "
                  f"at offsets {row['offsets']}", flush=True)
    report = {
        "settings": c, "offset_units": {axis: unit for axis, _, unit in AXES},
        "phase_corrections_rad": phases, "results": results, "summary": summary,
        "method": "Cartesian grid of -tolerance, zero, +tolerance per nonzero axis.",
        "metric": "Selected-input unconditional joint fidelity with configured loss and storage; independent write/read inputs.",
        "limitation": "Sampled extrema only; interior or off-grid extrema may be missed. Not statistical confidence bounds.",
        "calibration": "Nominal pulse timing, couplings, optical carriers and phase references; no per-point retuning.",
    }
    folder = (Path(output_directory) if output_directory else
              Path(b.__file__).resolve().parent.parent / "output" / "tolerance_runs" /
              (datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8]))
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "tolerance_scan.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Full tolerance results: {path}", flush=True)

    fig, panels = plt.subplots(2, 2, figsize=(13, 8),
                              gridspec_kw={'height_ratios': [3, 1]})
    axes = panels[0]
    for ax, (direction, values) in zip(axes, summary.items()):
        fidelity = np.asarray([row[direction + '_fidelity'] for row in results]) * 100
        ax.plot(np.arange(len(results)), fidelity, '.', color='tab:blue')
        for label, color, marker in (("best", "tab:green", '^'), ("worst", "tab:red", 'v')):
            index = results.index(values[label])
            ax.scatter(index, fidelity[index], color=color, marker=marker, s=90,
                       label=f"{label} sampled: {fidelity[index]:.4f}%")
        ax.axhline(fidelity[0], color='gray', linestyle='--',
                   label=f"nominal: {fidelity[0]:.4f}%")
        ax.set(xlabel='Deterministic grid point (0 = nominal)',
               ylabel='Joint fidelity (%)',
               title='Write + storage' if direction == 'write' else 'Storage + read')
        ax.legend(fontsize=9)
        ax.grid(alpha=.2)
    for ax, (direction, values) in zip(panels[1], summary.items()):
        ax.axis('off')
        rows = [[f"{values[label]['offsets'][axis]:+.5g}" for axis, _, _ in AXES]
                for label in ('best', 'worst')]
        table = ax.table(cellText=rows, rowLabels=['Best sampled', 'Worst sampled'],
                         colLabels=['Cavity\nMHz', 'Transmon\nMHz', 'Atom\nMHz',
                                    'Flux\nPhi/Phi0', 'Field\nV/cm'],
                         loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.8)
        ax.set_title('Fixed offsets producing these results', fontsize=10)
    fig.suptitle(f"Fixed tolerance scan — {len(results)} points\n"
                 f"Selected input: theta={c['INPUT_THETA']:g}°, phi={c['INPUT_PHI']:g}°")
    fig.text(.5, .02, "Best/worst among tested points; off-grid extrema may be missed.\n"
             "All sampled offsets and fidelities are saved in tolerance_scan.json.",
             ha='center', fontsize=9)
    fig.tight_layout(rect=(.02, .08, 1, .92), h_pad=2.5)
    if save:
        fig.savefig(folder / 'tolerance_scan.png', dpi=170)
    return report
