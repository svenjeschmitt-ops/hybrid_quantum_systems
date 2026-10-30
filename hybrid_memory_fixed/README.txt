Keep backend.py, hybrid_panel.py and hybrid_settings.py in the same folder.
Start the parameter panel: python backend.py
Direct plots without the panel: python backend.py --run

The default output is two compact windows: write/store and retrieval.
Both figures include the same linear preparation-pulse scan (0.02–3 ns), showing
final desired- and wrong-Rydberg probabilities for isolated equal-area π pulses.
Both retain the original 3D basis-output columns: input state, joint output
state, and probability.
The olive target columns show joint basis-state fidelity; gold columns show
other outputs. "Other" includes wrong Rydberg, residual excitation, and loss.
Write columns are evaluated at the end of mapping. Retrieval columns use
the complete state after writing and storage. The selected
superposition's phase-corrected overlap is reported below each figure.

Requires a Python installation with Tk, plus:
python -m pip install "qutip>=5,<6" numpy matplotlib

The panel starts the backend path supplied by backend.py, so it does not
depend on the old codetest.py filename. Plot exports remain optional.
