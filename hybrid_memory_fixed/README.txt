Keep backend.py, hybrid_panel.py and hybrid_settings.py in the same folder.
Start the parameter panel: python backend.py
Direct plots without the panel: python backend.py --run

Flux and electric-field pulses now make a rounded excursion with no plateau:
w(t) = sin^2(pi*(t-start)/T) inside each swap interval, zero outside.
The applied flux is PHI_PARK + (PHI_SWAP-PHI_PARK)*w(t); the total electric
field is E_PARK + (E_SWAP-E_PARK)*w(t). Stray-field compensation is retained.
Both controls reach their interaction value at the midpoint and return to park
with zero endpoint slopes. Readout uses the same shape in the reverse order.
Optical pulses retain their existing sin-squared envelopes.
These finite pulses have a nonzero spectral width, not exactly one frequency.
Stage durations are T_TC = TC_PULSE_SCALE*(pi/(2*g_TC) + 2*RAMP) and
T_AC = AC_PULSE_SCALE*(pi/(2*g_AC) + 2*RAMP). The multipliers independently
stretch the rounded pulses; RAMP contributes to the reference time before scaling.
The resonant hold time is only a reference: the pulse peaks pass beyond bare
resonance, producing two coherent passages through resonance per pulse.
The couplings, peak controls and durations have been calibrated using the full,
always-on-coupling Hamiltonian for writing and independent readout. The rounded
shape, 5 us storage, lifetimes and dephasing rates are retained. This is a local
numerical calibration of an illustrative model, not an experimental calibration
or proof of a global optimum. Residual coherent errors and finite dissipation
prevent exactly 100% fidelity. Changing device inputs requires recalibration.
Restart the panel and select Reset defaults to use the new values. Loading old
settings retains saved values; absent pulse multipliers take the new defaults.
For the former unscaled rounded pulses, explicitly set both multipliers to 1.

Rounded-pulse calibration (default parameters in hybrid_settings.py):
PHI_SWAP=0.26588630919611617, V_SWAP=0.3923668021130249 V,
BETA=0.04, E_ZPF_V_PER_M=0.8 V/m, OMEGA/(2*pi)=100 MHz,
TC_PULSE_SCALE=9.664304506350172, AC_PULSE_SCALE=3.3585450169623496.
The TC pulse lasts 228.834 ns, the AC pulse 85.383 ns, and each optical
pulse 10 ns. Peak detunings are -53.7114 MHz (TC) and -20.0464 MHz (AC).
The optical envelopes remain sin-squared; faster optical pulses reduce exposure
to loss and unwanted exchange while mapping. All loss rates are unchanged.

Independent six-state validation after 5 us storage:
Input              Noiseless write/read      Noisy write          Noisy read
0                  99.9897%                  99.6355%             99.6355%
1                  99.9167%                  99.3258%             99.3258%
Each of +,-,+i,-i   99.9532%                  99.4642%             99.4639%
Fidelities are unconditional joint overlaps, with one fixed noiseless phase
reference per direction shared by all six inputs. The tests check these six
states in both directions, solver-step convergence, and round trips at multiple
storage durations. Unscaled rounded pulses with the former coupling values
gave only 0.145% noiseless excited-state transfer; they were not calibrated.

The default output is two independent state-transfer windows:
transmon to atom and atom to transmon. The write figure summarizes total write
time, rounded swap pulse durations, final joint fidelities for g/e/+,
and selected-input coherence and residual populations in a table. The selected
input is identified by its Bloch angles below the figure. The read figure has
a matching table of storage hold, readout and combined duration, rounded pulse durations, final joint fidelities for
the two logical basis inputs and +, selected-input transmon coherence, cavity
photons, atom reset h0 population, and residual atomic populations. Metrics include
configured noise; fidelity uses a fixed phase correction without postselection.
For writing, the input is (alpha|g> + beta|e>) tensor |0,h0>, and the
target is |g,0> tensor (alpha|h0> + beta|h1>).
For reading, the input is |g,0> tensor (alpha|h0> + beta|h1>), prepared
independently of writing, then evolved at the parked operating point for
STORAGE before retrieval. The target is (alpha|g> + beta|e>) tensor |0,h0>.
All read fidelities, output columns, residual populations and CSV traces include
this hold, without resetting or renormalizing the stored state. The plotted
retrieval time starts at zero AFTER the hold. Noiseless phase calibration
includes the same hold and is shared across inputs. Set STORAGE=0 for readout only.
Fidelity is <target|rho|target>, including the required empty source and
cavity vacuum. Residual excitation, loss and decoherence count as errors.
The + target tests coherence as well as population. A single noiseless phase
calibration for each direction is shared across its three input states.
No conditioning on successful transfer or postselection is applied.
The write summary and output columns include writing followed by STORAGE.
It separately reports the selected input's fidelity immediately after writing
and after storage. The write dynamics plot still shows only the short transfer
sequence. All other write-table state metrics are evaluated after storage.
The read figure measures storage plus readout from an independently prepared atom.
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
Write columns are evaluated after mapping and the STORAGE hold. Round-trip columns use
the complete state after writing and storage, and are labeled by the ORIGINAL
TRANSMON input (g/e/+), not an independently prepared memory input.
Read columns use an independently prepared atomic input after the STORAGE hold. The selected
superposition's phase-corrected overlap is reported below each figure.

Requires a Python installation with Tk, plus:
python -m pip install "qutip>=5,<6" numpy matplotlib

The panel starts the backend path supplied by backend.py, so it does not
depend on the old codetest.py filename. Plot exports remain optional.

Couplings default to calculation from physical inputs. Direct information for
coupling enables optional manual input. The Coupling
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
couplings. The current defaults give approximately 11.0 MHz and 10.2 MHz, respectively.
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
The default peak flux passes beyond the 5000 MHz interaction resonance.
The parked flux PHI_PARK is also an input on 0 <= PHI_PARK <= 0.5;
its resonance frequency is calculated with the same forward relation.
The default parked flux preserves the previous 5800 MHz parked resonance.
The interaction frequency is displayed live and can be detuned from the cavity.
The coupling is evaluated at this transmon operating point and determines
the nominal resonant reference time pi/(2*g_TC). Read and write use the same rounded shape.
The atomic swap voltage V_SWAP remains an independent input. Its field is
E_swap = V_SWAP/GAP_CM + STRAY_FIELD; the atomic transition can be detuned.
Reducing FC by 100 MHz increases both peak detunings by +100 MHz. Changing FC alone
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

Historical plateau-pulse benchmark (does not apply to the rounded pulses):
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

Codex assistant:
Install the Codex CLI if it is not already on PATH, then run codex login in a
terminal and sign in with ChatGPT. Restart the panel from that terminal. In the
Assistant tab click Check Codex, then ask your question. Each request
uses a separate codex exec session and the CLI's built-in default model.
No MCP server, Python dependency, or API key is required.
For macOS GUI launches with a limited PATH, the panel also searches common CLI
installation directories and the architecture-matching Codex executable bundled
with the OpenAI VS Code/VS Code Insiders extension. PATH installations take
precedence; the newest executable extension version is selected as a fallback.
An existing signed-in extension installation does not need a separate CLI install.

Each question includes validated current inputs, units, calculated quantities,
and the latest simulation's original settings, mode, status and last 3,000
characters of console output, separately from current edits. New panel runs
also automatically export full numerical reports to unique folders under
output/assistant_runs, even when Also save plots/data is unchecked. Reports
include selected-input joint-fidelity and coherence traces, all eight population/
photon-number traces, flux and voltage controls, detunings, stage timing, exact
summary values, displayed tables and logical basis-output probabilities/fidelities.
Write dynamics end before storage; after-storage values are reported separately.
Read dynamics start after the storage hold, from an independently prepared atom.
Every basis-output column is labeled, including wrong Rydberg and remaining other.

Each assistant question includes a preview of up to 33 points per trace, full-grid
sampled minima/maxima with their times, and paths to full-resolution JSON files
which Codex may read for closer inspection. Previews can miss narrow features;
extrema describe the exported grid rather than continuous time. The assistant
does not receive screenshot images. Reports are published atomically as each
direction finishes computing, before the blocking plot windows close. The panel
shows when reports are available; ask again after another direction finishes.
New runs use new folders, and report settings must match the original run.
Stopped/failed runs may have partial reports, which retain that run's status.
Old runs made before this feature have no numerical reports; run a new simulation.
Reports remain on disk after closing the panel for inspection.

Codex may read
relevant project source and documentation. It cannot see the screen or automatically
inherit this chat. Context and files read by Codex are sent to OpenAI; internet
access is required and your ChatGPT plan's Codex usage limits apply. This is not
free offline inference. API-key authentication is rejected, inherited OpenAI API
key variables are removed, and ChatGPT authentication is required for execution.

The CLI runs with a read-only sandbox and no interactive approvals. It is
instructed not to edit files or run simulations. User CLI configuration is not
loaded for these requests (authentication is retained), so personal MCP servers
and custom model settings are not inherited. The final answer appears when Codex
finishes; a three-minute timeout and Cancel answer terminate the request's
process group. Existing Codex sessions are not stopped. Connection checks and answers
run in background workers so the panel remains usable.

Check the assistant without making network requests:
  python -m unittest discover -s hybrid_memory_fixed -p 'test_*assistant.py'
  python -m unittest discover -s hybrid_memory_fixed -p 'test_simulation_output.py'

Fixed experimental tolerances:
The Experimental tolerances tab accepts absolute +/- bounds FC_TOL, FT_TOL,
FA_TOL (MHz), FLUX_TOL (Phi/Phi0), and EFIELD_TOL (V/cm). Defaults are zero.
Nonzero tolerances automatically add shaded sampled ranges to the ordinary
Both transfer directions, Transmon to atom and Atom to transmon figures.
The separate 'Tolerance scan: best and worst sampled' output compares final results.
CLI: python hybrid_memory_fixed/backend.py --tolerance-scan --config settings.json
Use --no-show for unattended runs; --save also exports the summary plot.

The scan tests all combinations of -bound, zero and +bound for each nonzero
axis: 3^N points, up to 243. Each error remains fixed throughout a scenario.
Flux and total-field errors apply at park and during both transfer pulses;
field error is an uncompensated residual, not a changed voltage command.
The flux relation remains periodic and is not clipped at the nominal 0-0.5
input range. FT_TOL and FA_TOL add residual transition-frequency shifts on top
of flux/Stark shifts; avoid double-counting one physical source of error.
FC_TOL shifts cavity photon energy in the fixed nominal rotating frame.
Optical carriers, couplings, pulse timings and noiseless phase corrections
retain their nominal calibration. There is no random sampling or per-point
retuning. Dissipation remains enabled and separate from these static errors.

The chart and console show nominal, best sampled and worst sampled joint
fidelities for the selected Bloch-angle input, independently for write+storage
and storage+read. Read starts from an independently prepared atomic input.
These are NOT guaranteed global extrema or statistical confidence bounds:
interior/off-grid extrema can be missed. Different directions may have their
best/worst at different offsets. All offsets and results are saved automatically
to a unique output/tolerance_runs folder (or the panel's run report folder),
in tolerance_scan.json.

Ordinary transfer figures show nominal population/control lines with shaded
sampled pointwise min/max ranges. Summary tables have Metric, Nominal and
With tolerances columns. Outcome metrics have sampled min/max ranges, including
the three logical-input fidelities, selected-input coherence and residual
populations. Write end-state metrics include storage; read includes storage
before retrieval. All input fidelities use the fixed nominal phase reference.
Pulse timings remain fixed, so their tolerance cells show a dash.
Combined populations are summed within each realization before taking extrema.
The nominal comparison retains configured loss and dephasing. Basis-output
columns in the 3D plot remain nominal. Exported write fidelity
traces end before storage; the summary uses the final state after the
storage hold. Read trajectories include the hold before retrieval time zero.
The detuning bands subtract the ACTUAL cavity frequency, including FC error.
Electrode voltage is the commanded voltage and does not change with residual
field error; that error affects field, Stark detunings and quantum dynamics.
A pointwise envelope can switch between different fixed-error realizations;
it is not one physical trajectory, a confidence interval or a guaranteed bound.
Panel numerical reports include full-resolution selected_input_tolerance_ranges
traces and the final best/worst offsets under summary.tolerance_ranges.
With zero tolerances, no extra simulations are run and outcome intervals
collapse to their nominal values; timing cells still show a dash.


Simultaneous shared-cavity resonance:
Choose "Simultaneous cavity resonance: transmon to atom" in the Output menu.
The Simultaneous resonance tab sets the hold multiplier (default 1) and
sin-squared ramp duration (default 0.0005 us per ramp). The backend prepares
r0, ramps the transmon and r0-r1 transition onto the same cavity resonance,
holds both there, returns both controls to park, and maps r0/r1 to h0/h1.
The configured STORAGE hold follows mapping. Both cavity couplings remain
active throughout, with the transmon coupling evaluated at the calculated
resonant flux. The cavity is explicitly simulated; there is no direct
transmon-atom exchange term. PHI_SWAP, V_SWAP and sequential pulse scales
are not used by this mode. Unreachable resonance settings raise an error.

The ideal flat-hold reference time is pi/sqrt(g_TC^2+g_AC^2), with angular
g in rad/us. Its multiplier can be adjusted to account for finite ramps.
Even at exact resonance, unequal coupling magnitudes limit ideal exchange
to 4*g_TC^2*g_AC^2/(g_TC^2+g_AC^2)^2; full exchange requires equal magnitudes.
The reported limit excludes ramps, dissipation, optical errors and storage.
Fidelities use the full joint target |g,0,atomic target>, no postselection,
and one fixed noise-free phase calibration for every logical input.
The figure uses the same structure as Transmon to atom: controls and detuning
on top, full-width write dynamics in the middle, and a write + storage summary
and 3D joint-output probabilities below. Dynamics end at optical mapping;
the summary and g/e/+ output probabilities are evaluated after storage. Reported results are
nominal: experimental tolerance bounds are not scanned in this mode.
Assistant numerical reports identify the simultaneous protocol under write.json.
Also save plots/data exports hybrid_simultaneous.png and hybrid_simultaneous.json.
CLI: python hybrid_memory_fixed/backend.py --run --simultaneous
Unattended: add --no-show; numerical exports: add --save.
Checks: python -m unittest discover -s hybrid_memory_fixed -p 'test_simultaneous.py'

Parameter sweeps:
In Numerics, find Optional and set Parameter sweep to On. Select a parameter
(PHI_SWAP by default) and set start, stop and step. Choose Both transfer
directions, Transmon to atom, or Atom to transmon in Output, then click
Run simulation. Set Parameter sweep to Off for ordinary single runs. The
sweep controls are hidden while off; Reset defaults also turns sweeping off.
Simultaneous resonance and the separate tolerance scan output are available
while sweeping is off. The current
inputs in all other tabs form the baseline. PHI_SWAP sweeps the pulse peak;
PHI_PARK sweeps the parked flux. A negative step runs a descending sweep.
Points follow start+n*step; stop is included only when reached. All points
are validated before starting; the limit is 501 points. Stop calculation
terminates the sweep and its active solver process on macOS/Linux.

Each point uses a freshly loaded model, including recalculated frequencies,
couplings and derived pulse durations, and a separate noiseless phase
calibration. No pulse optimization is performed. Configured experimental tolerances are
scanned at each sweep point, holding that point's nominal calibration fixed. The summary
metric is selected-input unconditional joint fidelity with configured noise.
Write includes subsequent storage; read uses an independently prepared atomic
input with storage before retrieval. Write traces end before storage; final
write fidelity includes storage. Read trace time starts after storage.

CSV/JSON summaries and per-point JSON dynamics (fidelities and populations)
are always saved to a unique output folder. Completed points survive a stopped
run; check status and completed count against total_points before use.
Sweep figures use the ordinary transfer layout, including controls, detunings,
populations, summary table and 3D basis outputs. An in-figure slider and
Previous/Next buttons select the precomputed sweep point and update every
panel together. Each direction has its own figure. Also save plots/data adds
a PNG of the initial point for each direction; use the figure toolbar to save
other selected points. Per-point files include the full figure report. Panel output is saved under
output/assistant_runs; CLI output defaults to output/sweep_runs.

CLI example:
python hybrid_memory_fixed/backend.py --parameter-sweep --sweep-parameter PHI_SWAP --sweep-start 0.26 --sweep-stop 0.27 --sweep-step 0.001 --sweep-direction both --no-show --save

Sweep tolerances:
Nonzero bounds in Experimental tolerances also apply during parameter sweeps.
Each point tests the Cartesian -bound/zero/+bound grid around its own nominal
model, using fixed nominal couplings, timing and phase within that point.
The sweep slider updates shaded flux/detuning/population ranges and the
With tolerances summary column. 3D output columns remain nominal. These are
sampled extrema, not guaranteed continuous bounds or statistical confidence
intervals. All-zero bounds retain nominal output without extra simulations.
CSV and summary JSON also include final fidelity low/high columns per direction.
The cost multiplies by 3^N scenarios per point per direction for N nonzero bounds.

Direct coupling inputs:
In Coupling parameters, enable Direct information for coupling to enter
transmon–cavity and atom–cavity coupling magnitudes directly as g/(2*pi) in
MHz. Both must be finite and positive. Physical coupling fields are disabled
while manual mode is selected. Turn the button off to restore calculated
couplings. The equations and live coupling/timing display follow the mode.
The mode and values are saved with settings and numerical reports.
CLI/config keys: DIRECT_COUPLING=1, G_TC_MHZ and G_AC_MHZ; the default
DIRECT_COUPLING=0 retains the physical-input formulas. Older settings without
these keys continue to use calculated couplings. Ambiguous legacy G_TC/G_AC
keys remain rejected. Manual values apply to sequential and simultaneous
Hamiltonians, swap timing, tolerance scans and parameter sweeps. Flux and
voltage continue to set detuning; changing coupling does not retune resonance
or optimize pulses. Manual-coupling sweeps require manual mode enabled.

Enable Simultaneous resonance in Coupling parameters to reveal its hold
multiplier and ramp settings in the same tab. The equations, live resonance
summary and Output selection follow the toggle. There is no separate
Simultaneous resonance tab. Calculated/manual coupling selection and manual
strengths are retained. Turning the toggle off selects sequential writing.
If parameter sweeping was on, enabling simultaneous resonance turns it off
and reports this in the status line. Click Run simulation when ready.

Simulation progress:
The panel shows a 0–100% bar based on completed calculation stages, including
logical inputs, tolerance scenarios, transfer directions and sweep points.
Stages are weighted work units, not a wall-clock estimate; a percentage can
pause during a long solver call. The bar reaches 100% when results and figures
are ready, before blocking plot windows close. New runs reset it to zero.
Stopping or failing a calculation preserves its last percentage.
CLI --progress emits structured HYBRID_PROGRESS events for panel consumption.
