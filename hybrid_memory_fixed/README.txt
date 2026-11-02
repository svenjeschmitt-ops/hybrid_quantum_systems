Keep backend.py, hybrid_panel.py and hybrid_settings.py in the same folder.
Start the parameter panel: python backend.py
Direct plots without the panel: python backend.py --run

The default output is two independent state-transfer windows:
transmon to atom and atom to transmon. Each fidelity panel shows three
time traces, with final values in the legend: logical ground, excited, and
the coherent equal superposition +. All curves include configured noise.
For writing, the input is (alpha|g> + beta|e>) tensor |0,h0>, and the
target is |g,0> tensor (alpha|h0> + beta|h1>).
For reading, the input is |g,0> tensor (alpha|h0> + beta|h1>), prepared
independently of writing. The target is (alpha|g> + beta|e>) tensor |0,h0>.
Fidelity is <target|rho|target>, including the required empty source and
cavity vacuum. Residual excitation, loss and decoherence count as errors.
The + target tests coherence as well as population. A single noiseless phase
calibration for each direction is shared across its three input states.
No conditioning on successful transfer or postselection is applied.
The default figures measure transfer only; STORAGE does not add a hold.
With --save, hybrid_transfer_write_fidelity.csv and
hybrid_transfer_read_fidelity.csv contain time and the three fidelities.

Use --roundtrip explicitly for the separate write/store/read diagnostic.
That mode retains the hold-time sweep, atomic storage curves, transmon
return and direct-transmon reference. A high return overlap can result
from an excitation that never entered the atom. Its six-state atomic mean
approaches 1/3 for an unchanged-transmon bypass. With --save, that sweep
is exported to hybrid_storage_fidelity.csv.
Run python backend.py --leakage-scan for the separate preparation diagnostic.
Both retain the original 3D basis-output columns: input state, joint output
state, and probability.
Each input/output plot includes the two logical basis inputs and the coherent
equal superposition "+". Olive columns show intended logical outputs; gold
columns show other outputs. "Other" includes wrong Rydberg, residual excitation,
and loss. Phase-corrected joint fidelities for all three inputs are printed
below each figure; populations alone do not establish superposition fidelity.
Write columns are evaluated at the end of mapping. Round-trip columns use
the complete state after writing and storage, and are labeled by the ORIGINAL
TRANSMON input (g/e/+), not an independently prepared memory input.
Retrieval-only columns use an independently prepared atomic input. The selected
superposition's phase-corrected overlap is reported below each figure.

Requires a Python installation with Tk, plus:
python -m pip install "qutip>=5,<6" numpy matplotlib

The panel starts the backend path supplied by backend.py, so it does not
depend on the old codetest.py filename. Plot exports remain optional.

Couplings are now calculated rather than entered directly. The Coupling
parameters tab takes beta (capacitive participation), V_ZPF_UV (local cavity
zero-point voltage in microvolts), D_01_EA0 (transition dipole magnitude in elementary charge times
Bohr radius), E_ZPF_V_PER_M (local cavity zero-point electric field in V/m),
and DIPOLE_ALIGNMENT (cosine of the dipole/field angle).

The approximate transmon formula is n_01 = [EJ(Phi_swap)/(32 EC)]^(1/4),
where EJ(Phi) = EJ_MAX * sqrt(cos(pi Phi/Phi0)^2 + ASYMMETRY^2 sin(pi Phi/Phi0)^2).
EC and EJ are both entered in frequency units. The applied swap flux
sets the effective EJ and the resulting n_01.
The formulas are g_TC = 2 e beta V_zpf [EJ(Phi_swap)/(32 EC)]^(1/4) / hbar and
g_AC = |d_01| E_zpf |cos(theta)| / hbar. The calculated couplings are g/(2*pi) in MHz;
the Hamiltonian uses g in rad/us. Swap hold times use these same calculated
couplings. Defaults give approximately 2.770 MHz and 1.024 MHz, respectively.
Voltage and field inputs are local vacuum RMS amplitudes; the static electrode
field used for Stark tuning is separate. No cavity geometry or atomic wave
functions are solved. Supply the selected atomic transition dipole. The
transmon coupling is evaluated at the swap point and remains constant during
flux ramps; the atomic coupling also remains constant during a run.
Zero coupling (including perpendicular alignment) is rejected because the
transfer schedule requires finite swap times. Defaults remain illustrative.

Settings JSON stores the physical inputs. Older files containing G_TC/G_AC
must be updated to the physical inputs above; old coupling values are
rejected with an explanation rather than used as overrides. Saved settings
containing N_01 must remove that key; it is now derived from the device inputs.

Verify the coupling calculation and backend integration:
python -m unittest discover -s hybrid_memory_fixed -p 'test_*.py'
(Run from the repository root.)

Swap frequencies are calculated from the couplings: f_swap = 2*g/(2*pi).
The coupling tab displays both TC and AC swap frequencies and nominal holds,
updated when parameters change. There is no editable swap-frequency input.
The applied interaction flux PHI_SWAP (Phi/Phi0) is an independent input on
0 <= PHI_SWAP <= 0.5. The transmon resonance is calculated from it:
EJ(phi) = EJ_MAX*sqrt(cos(pi*phi)^2 + d^2*sin(pi*phi)^2),
f_T(phi) = sqrt(8*EC*EJ(phi)) - EC, with EC and EJ in MHz.
The default flux preserves the previous 5000 MHz interaction resonance.
The parked flux PHI_PARK is also an input on 0 <= PHI_PARK <= 0.5;
its resonance frequency is calculated with the same forward relation.
The default parked flux preserves the previous 5800 MHz parked resonance.
The interaction frequency is displayed live and can be detuned from the cavity.
The coupling is evaluated at this transmon operating point and determines
the nominal resonant hold time pi/(2*g_TC). Read and write use the same plateau.
The atomic swap voltage V_SWAP remains an independent input. Its field is
E_swap = V_SWAP/GAP_CM + STRAY_FIELD; the atomic transition can be detuned.
At FC=4900 MHz with other defaults unchanged (calculated f_T=5000 MHz),
both transmon-cavity and atom-cavity detunings are +100 MHz. Changing FC alone
does not retune either transition or change the nominal swap holds.
The panel reports both detunings. Detuning reduces excitation transfer.
The Cavity and transmon tab also displays the interaction and parked
detunings live, and identifies whether the interaction is resonant or detuned.
Old F_SWAP_TARGET JSON entries must be removed. Replace F_T_INTERACTION
with PHI_SWAP, and F_T_PARK with PHI_PARK, in saved settings; frequency
overrides are rejected explicitly.

Parking applies PHI_PARK and compensates the specified stray field at
E_PARK; optical carrier phases remain calibrated to the parked energies.
Swap durations remain nominal pi/(2g) holds, recomputed from the couplings;
phase calibration corrects a phase only and cannot repair failed transfer.

Each parameter tab includes typeset equations describing its implemented
model and units. The larger panel shows inputs and equations side by side
without scrollbars, with the initial window sized to the screen. The numerical tab documents the
master equation, state-space truncation, and integration-step setting.

High-fidelity default operating point:
BETA=0.01, E_ZPF_V_PER_M=0.08, OMEGA=20 MHz, RAMP=0.0005 us.
The previous BETA=0.1 gave g_TC/(2*pi)=27.70 MHz. At the 800 MHz park
detuning its spectator interaction shifted the cavity by roughly 0.96 MHz,
comparable to or larger than the old g_AC/(2*pi)=0.512 MHz. Bare atomic
resonance therefore did not produce a complete swap. Smaller participation
reduces this shift to about 0.0096 MHz; the larger local atomic vacuum field
and shorter optical pulses reduce time exposed to loss. Coherence times,
unwanted-level coupling, the full Hamiltonian, and joint fidelity definitions
are retained. These are illustrative model defaults, not a calibrated device.
Finite losses and residual coherent errors prevent exactly unit fidelity.
Restart the panel and choose Reset defaults; loaded older settings retain
their own parameter values.
Validation: round-trip return six-state mean joint fidelity over the 41-point 0-10 us storage
sweep is 0.999408-0.999555 without noise and 0.987787-0.987893 with noise.
At STORAGE=5 us these are 0.999483 and 0.987841, using one fixed phase
calibration, the full joint state, and no postselection.

Detuning diagnostic: inspect the atomic storage curves and the excited-input
write probability, as well as retrieval from independently prepared atomic h1.
The red excited-input transfer curve shows the joint probability of
|g,0,h1> after writing and holding; it approaches zero when transfer fails.
The round-trip figure and console also report the selected input's atomic
joint fidelity BEFORE retrieval, ahead of the transmon return overlaps.
Round-trip return alone does not prove transfer: an excitation can remain on
the transmon throughout write/store/read and return a large overlap.
The atomic target excludes that excited-transmon bypass. A six-state average
need not vanish when writing fails: an unchanged |g,0,h0> component still
overlaps some intended targets. For the exact unchanged-input bypass the joint
atomic six-state mean is 1/3, while excited-input write fidelity is zero and
round-trip return fidelity is one. No values are forced to zero or normalized
by transfer success. These curves characterize the configured write-and-hold
protocol; failed writing cannot establish the intrinsic atomic memory lifetime.
