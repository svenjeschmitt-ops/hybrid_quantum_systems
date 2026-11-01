Keep backend.py, hybrid_panel.py and hybrid_settings.py in the same folder.
Start the parameter panel: python backend.py
Direct plots without the panel: python backend.py --run

The default output is two compact windows: write/store and retrieval.
Both figures include the same linear preparation-pulse scan (0.02–3 ns), showing
final desired- and wrong-Rydberg probabilities for isolated equal-area π pulses.
Both retain the original 3D basis-output columns: input state, joint output
state, and probability.
Each input/output plot includes the two logical basis inputs and the coherent
equal superposition "+". Olive columns show intended logical outputs; gold
columns show other outputs. "Other" includes wrong Rydberg, residual excitation,
and loss. Phase-corrected joint fidelities for all three inputs are printed
below each figure; populations alone do not establish superposition fidelity.
Write columns are evaluated at the end of mapping. Retrieval columns use
the complete state after writing and storage. The selected
superposition's phase-corrected overlap is reported below each figure.

Requires a Python installation with Tk, plus:
python -m pip install "qutip>=5,<6" numpy matplotlib

The panel starts the backend path supplied by backend.py, so it does not
depend on the old codetest.py filename. Plot exports remain optional.

Couplings are now calculated rather than entered directly. The Coupling
parameters tab takes beta (capacitive participation), V_ZPF_UV (local cavity
zero-point voltage in microvolts), N_01 (dimensionless charge matrix element
|<0|n|1>|), D_01_EA0 (transition dipole magnitude in elementary charge times
Bohr radius), E_ZPF_V_PER_M (local cavity zero-point electric field in V/m),
and DIPOLE_ALIGNMENT (cosine of the dipole/field angle).

The formulas are g_TC = 2 e beta V_zpf |n_01| / hbar and
g_AC = |d_01| E_zpf |cos(theta)| / hbar. The panel displays g/(2*pi) in MHz;
the Hamiltonian uses g in rad/us. Swap hold times use these same calculated
couplings. Defaults give approximately 19.34 MHz and 0.5118 MHz, respectively.
Voltage and field inputs are local vacuum RMS amplitudes; the static electrode
field used for Stark tuning is separate. No cavity geometry or atomic wave
functions are solved. Supply n_01 at the operating point and the selected
transition dipole; both couplings remain constant during flux/voltage ramps.
Zero coupling (including perpendicular alignment) is rejected because the
transfer schedule requires finite swap times. Defaults remain illustrative.

Settings JSON stores the physical inputs. Older files containing G_TC/G_AC
must be updated to the six physical inputs above; old coupling values are
rejected with an explanation rather than used as overrides.

Verify the coupling calculation and backend integration:
python -m unittest discover -s hybrid_memory_fixed -p 'test_*.py'
(Run from the repository root.)
