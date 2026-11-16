# Configurable master equation

The effective Hamiltonian is unchanged. The density matrix now supports

\[
\dot\rho=-i[H(t),\rho]+\sum_k\mathcal D[c_k(t)]\rho,
\qquad \mathcal D[c]\rho=c\rho c^\dagger-\tfrac12\{c^\dagger c,\rho\}.
\]

Here H is already H_eff/hbar, in rad/us. Every dissipative rate is in **1/us**,
with no factor of 2 pi. Each independent physical channel has its own collapse
operator. Nonnegative rates give a trace-preserving, completely positive
Lindblad generator, up to numerical integration error.

## Usage

`DISSIPATION` is a JSON object in a settings file, or a JSON string in the
parameter panel's **Loss and coherence → Additional Lindblad channels** field.
It is saved in run reports along with the other settings. Missing keys use the
values below. `{}` reproduces the previous model exactly; additions are opt-in.

From the repository root:

```sh
.venv/bin/python hybrid_memory_fixed/backend.py --config hybrid_memory_fixed/dissipation_example.json --write-only --no-show
.venv/bin/python hybrid_memory_fixed/backend.py --config hybrid_memory_fixed/dissipation_example.json --retrieve --no-show
```

**The example contains arbitrary demonstration rates and branching fractions.**
Replace these with measured or independently calculated values before interpreting
results as a device prediction. Existing pulse calibrations may need retuning.

## Channels and conventions

| JSON key | Default | Meaning |
| --- | --- | --- |
| `cavity_nbar` | 0 | Mean bath occupation: c_down = sqrt((nbar+1)/T1_C) a, c_up = sqrt(nbar/T1_C) a†. T1_C is the bare photon-number damping time. |
| `transmon_nbar` | 0 | Effective bath occupation for transmon heating; downward/upward rates carry nbar+1/nbar. One constant occupation approximates a bath across the ladder bandwidth. It is not a frequency-dependent Gibbs bath. |
| `transmon_down_rates_per_us` | [] | Exactly NT−1 bare rates ordered 1→0, 2→1, etc. Nonempty selects independent jumps sqrt(gamma_j(nbar+1)) ket(j−1)bra(j) and their thermal reverses. There is **no additional j or sqrt(j)** factor. Empty preserves the original collective oscillator jump b/sqrt(T1_T), with optional thermal reverse. |
| `atomic_branching` | {} | Mapping source → {destination: fraction}. Replaces that source's lifetime-to-loss jump. Fractions must sum to one; include `lost` explicitly. Total lifetime decay remains 1/tau_source. Unspecified sources decay entirely to `lost`. |
| `atomic_transitions` | [] | Additional independent channels `{source, target, rate_per_us}`. Can represent measured blackbody redistribution or other incoherent transitions. Rates **add** to lifetime decay. Specify both directions when appropriate; detailed balance is not inferred. |
| `atomic_projector_dephasing_per_us` | {} | Mapping state → gamma for sqrt(gamma) P_state. A coherence between this state and an otherwise unaffected state decays at gamma/2. Adds to existing TPHI channels. |
| `flux_white_psd` | 0 | White flux noise strength S_phi in (Phi/Phi0)^2 us, as defined below. |
| `field_white_psd` | 0 | White electric-field noise strength S_E in (V/cm)^2 us, as defined below. |
| `optical_scattering` | [] | Channels `{source, target, arm, rate_per_us}`. The rate is at peak effective Rabi frequency; arm is 0 or 1. Equal source/target models elastic scattering; unequal states model population transfer or loss. |

Atomic names are `h0`, `h1`, `r0`, `r1`, `wrong`, `lost`, and labels from
`RYDBERG_MANIFOLD`. `lost` is absorbing. Each neighboring Rydberg level's
`lifetime_us` also supports branching. Transmon measured rates that already
include thermal enhancement should be converted to bare rates before also
setting a nonzero nbar; at nbar=0 the supplied downward rates are used directly.

The transmon's independent-jump option makes a secular approximation: resolved
ladder transitions couple to separate dissipative channels. Matching gamma_j=j/T1_T
matches population decay of the oscillator model, but does not preserve all
higher-level coherence transfer terms of the collective b jump.

Lifetime branching and additional blackbody transitions have deliberately distinct
semantics. If a measured lifetime already includes blackbody redistribution,
subtract that contribution from its base loss rate before adding those channels.
Do not count the same measured process in TPHI and an added dephasing rate.

## Control noise

The precise convention is

\[
\langle\delta x(t)\delta x(t')\rangle=S_x\delta(t-t'),\qquad
c_x(t)=\sqrt{S_x}\,\partial_x(H/\hbar).
\]

Thus an instantaneous coherence with noise sensitivities v_i and v_j has added
rate S_x(v_i−v_j)^2/2. Convert laboratory one-sided spectra and Hz/angular-frequency
units to this convention before entering values.

Flux noise uses the analytic derivative of the existing transmon frequency model:
`c_phi = sqrt(S_phi) * 2*pi * df_T/dphi * Nt`.
Field noise uses a **single correlated diagonal operator**
`c_E = sqrt(S_E) * 2*pi * 2*E(t) * sum_i K_i P_i`, including r0, r1,
wrong and all extra atomic levels. Using independent projectors for each level
would incorrectly lose the common-field correlations. Static tolerance offsets
remain separate and affect the sensitivities through the actual controls.

This is first-order white classical control noise in the existing effective
Hamiltonian. It does not include colored/1/f spectra, second-order sweet-spot
noise, flux dependence of coupling or anharmonicity, or non-Markovian baths.
The existing tolerance scan remains a bounded quasi-static sensitivity analysis,
not a probability-weighted stochastic ensemble or a fidelity confidence interval.

## Optical scattering and protocol consistency

For each configured channel,
`c_sc(t) = sqrt(rate_per_us) * abs(Omega_arm(t)/OMEGA_peak) * |target><source|`.
The resulting rate follows the **square** of the sin-squared pulse envelope.
The peak is the effective optical Rabi frequency already used by this model.

Arm 0 is active during preparation and h0/r0 mapping (or reverse mapping and
cleanup). Arm 1 is active only during h1/r1 mapping or reverse mapping.
Sequential read, write, and simultaneous read/write bind these channels to their
own stage times. Storage uses constant parked Hamiltonian and collapse operators,
with all optical drives/scattering off. White control dephasing remains active
at park where its sensitivity is nonzero. Read adjoint effects use the same
read channels as forward evolution. `noisy=False` removes every collapse channel.
The direct-transmon six-state comparison also includes the configured ladder
rates, thermal heating, higher-level leakage, and nominal parked flux noise.

Scattering is an effective, independently supplied rate model. It does not resolve
intermediate optical states, infer laser detunings or branching ratios, or derive
spontaneous scattering from the effective two-photon Rabi frequency alone.

Bath occupation describes the environment, not initial preparation. Transfer still
starts with the existing prepared inputs and a vacuum cavity; the bath then heats
the joint state during evolution. A thermally equilibrated initial cavity is not
automatically assumed.

Thermal baths can populate the truncation boundary. Increase NC and NT and compare
results with fixed controls before claiming convergence, particularly when nbar
is appreciable. This remains a local bare-basis Lindblad approximation, not a
global dressed-state bath calculation.

## Implementation and checks

`dissipation.py` validates channel data and builds static/QobjEvo collapse operators.
`backend.py` supplies each protocol's controls and freezes channels for parked
propagation. `simultaneous_coupling.py` supplies its own optical and control timing.
The implementation follows [QuTiP's time-dependent collapse-operator interface](https://qutip.readthedocs.io/en/stable/guide/dynamics/dynamics-time.html).

Run the independent physics tests:

```sh
MPLBACKEND=Agg .venv/bin/python -m unittest discover -s hybrid_memory_fixed -p 'test_dissipation.py'
```

Checks cover the legacy generator limit, resolved transition rates and coherence
normalization, finite-truncation thermal equilibrium, branching conservation,
control derivatives, optical timing, parked versus integrated evolution, forward
versus adjoint readout, and the isolated thermal-transmon reference.

With ATOMIC_SPECTRUM enabled, field noise uses the interpolated energy derivatives
and field-dependent exchange derivatives. OPTICAL_SCHEME derives its own coherent
scattering operators (or explicit intermediate decay); avoid counting those same
processes again in optical_scattering. See PHYSICAL_HAMILTONIAN.md.
