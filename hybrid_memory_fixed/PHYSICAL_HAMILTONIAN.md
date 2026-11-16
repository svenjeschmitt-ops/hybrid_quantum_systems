# Atomic and optical Hamiltonian inputs

The backend now supports field-dependent atomic energies and cavity dipoles,
consistent two-photon AC Stark shifts and spontaneous scattering, and an optional
explicit intermediate state for each optical arm. Writing, reading, simultaneous
transfer, parked storage, phase calibration, and field-noise derivatives use the
same model. The cavity and transmon couplings remain present throughout.

Existing settings remain valid: `ATOMIC_SPECTRUM={}` and `OPTICAL_SCHEME={}` select
the previous quadratic Stark shifts, constant dipoles and ideal effective drives.
The new inputs appear as JSON fields in the parameter panel and are included in
saved settings and numerical reports.

## Run the example

From the repository root:

```sh
.venv/bin/python hybrid_memory_fixed/backend.py --config hybrid_memory_fixed/physical_hamiltonian_example.json --write-only
.venv/bin/python hybrid_memory_fixed/backend.py --config hybrid_memory_fixed/physical_hamiltonian_example.json --retrieve
.venv/bin/python hybrid_memory_fixed/backend.py --config hybrid_memory_fixed/physical_hamiltonian_example.json --simultaneous --simultaneous-direction both
```

**The supplied curves and laser parameters are synthetic demonstrations.** They
are not an identified Rb level scheme or a calibrated device. Replace them with
measurements or calculations for the chosen states, magnetic field, polarizations,
mode geometry and optical lasers. Existing pulse multipliers are reference inputs;
new atomic data or light shifts generally require pulse recalibration.

## Atomic data

`ATOMIC_SPECTRUM` accepts the following object or its JSON serialization:

```json
{
  "provenance": "State labels, magnetic field, polarization, calculation/measurement source",
  "field_V_per_cm": [0, 2, 5],
  "energies_MHz": {
    "r0": [0, -4, -10],
    "r1": [5150, 5066, 4940],
    "wrong": [1320, 1316, 1310]
  },
  "dipoles_ea0": {"r1<-r0": [1000, 800, 500]}
}
```

The grid must be finite and strictly increasing. Every array must match it.
Shape-preserving piecewise cubic interpolation (PCHIP) supplies energies, signed
couplings, derivatives and resonance roots. Extrapolation is disabled, apart from
clamping numerical roundoff within 1e-12 of the domain scale. The grid must cover
park, the entered interaction field and all configured field tolerance offsets.
This check also applies to parameter sweeps and search candidates.

Energy values are **ordinary MHz relative to the zero-field r0 reference**, before
subtracting the cavity rotating frame. Thus r1 includes the actual r0/r1 transition
frequency, not just its Stark shift. Upper-branch energies have FC subtracted inside
the Hamiltonian, and FA tolerance shifts affect upper branches as before. Energies
must include both r0 and r1 when overriding either. Missing neighboring energies
retain their existing quadratic approximation; a partially supplied table therefore
remains a mixture of physical data and illustrative inputs. No accuracy claim is
inferred from the presence of a provenance string.

Dipoles are **real, signed projections onto the cavity polarization**, in e*a0.
They already contain polarization and angular matrix elements: DIPOLE_ALIGNMENT
is not multiplied into tabulated values. Field amplitudes use E_ZPF_V_PER_M, and
G_AC(E) = 2*pi * d_projected(E)*E_zpf/h in the backend's rad/us units.
`DIRECT_COUPLING=0` is required when providing dipole tables. Unspecified links use
the original scalar coupling and the manifold's cavity_from relative strengths.
Allowed keys are `r1<-r0` and links declared by RYDBERG_MANIFOLD, such as
`wrong_upper<-wrong`. Tabulated links can change sign; Hermitian-conjugate exchange
terms are always supplied. The reference swap duration uses the magnitude of the
primary coupling at the entered interaction field. During dynamics, every link
uses its own value at the actual time-dependent field.

Resonance is found from interpolated E_r1(E)-E_r0(E)=FC. If multiple roots exist,
the root closest to the entered swap field is selected. Simultaneous transfer uses
the coupling evaluated at that selected field for its reference hold time. Plots,
tolerance traces and assistant diagnostics also use the same atomic transition.
Searches with physical inputs vary applied controls directly instead of inverting
the quadratic Stark approximation.

For white electric-field noise, the collapse operator uses
sqrt(S_E) * d(H/hbar)/dE. This includes both interpolated energy derivatives and
**exchange derivatives from the field-dependent dipoles**, with their shared-field
correlations. It reduces to the original diagonal Stark derivative with scalar
energies and constant dipoles.

These tables describe **smoothly tracked adiabatic branches in a consistent real
phase convention**. They do not calculate atomic structure or nonadiabatic basis
connections. Energy and dipole tables alone do not describe Landau-Zener dynamics
across strongly mixed avoided crossings. Such a regime requires a coupled bare-state
Stark Hamiltonian or explicit basis-connection terms, plus corresponding transformed
decay/optical operators. Do not feed eigenstates independently sorted by energy into
these tables when their state character or dipole phase switches between samples.

[ARC's StarkMap documentation](https://arc-alkali-rydberg-calculator.readthedocs.io/en/latest/calculations_atom_single.StarkMap.html)
provides one source for atomic calculations; this implementation adds no ARC dependency
and does not select quantum numbers automatically.

## Optical model

`OPTICAL_SCHEME` requires `mode`, `provenance`, and two `arms`, keyed `"0"` and `"1"`.
Each arm supplies:

- `detuning_MHz`: signed intermediate detuning, ordinary MHz.
- `lower_rabi_MHz`, `upper_rabi_MHz`: signed peak single-photon Rabi frequencies
  divided by 2*pi, ordinary MHz. A negative sign represents a pi laser phase shift.
- `linewidth_per_us`: total intermediate population decay Gamma, in 1/us.
  For a natural linewidth quoted as Gamma/(2*pi) in MHz, multiply by 2*pi.
- `branching`: destination atomic labels and nonnegative fractions summing to one.
  `lost` collects destinations outside the modeled states.

For example, one arm is

```json
{
  "detuning_MHz": 2000,
  "lower_rabi_MHz": 63.245553203367585,
  "upper_rabi_MHz": 63.245553203367585,
  "linewidth_per_us": 1,
  "branching": {"h0": 0.7, "lost": 0.3}
}
```

Arm 0 connects h0 to r0 and the configured lower-branch neighbors; arm 1 connects
h1 to r1 and upper-branch neighbors. Neighbor optical_relative factors apply to the
upper single-photon leg. Each arm has a distinct intermediate and independent
spontaneous-emission channels; this does not assume shared-intermediate interference
between arms. The model uses real matrix elements with the specified carrier phase;
arbitrary complex laser/polarization matrices are not provided by this schema.

The single-photon envelopes are sine pulses, so their product yields the previous
sin-squared effective pulse. The peak effective Rabi magnitude is derived as

    Omega_eff = |Delta * Omega_lower * Omega_upper / (2*(Delta^2+(Gamma/2)^2))|

where all quantities on the right are angular frequencies/rates in rad/us or 1/us.
OMEGA is then derived and the shared reference duration is T_OPT=2*pi/Omega_eff.
Both arms must have equal effective peak magnitudes. The configured scalar OMEGA is
inactive and cannot be swept or optimized with a physical optical scheme enabled.
In explicit mode this duration is a reference derived from the eliminated model,
not a guarantee of a pi pulse outside the far-detuned regime.

For each arm, let W(t) map its ground and coupled Rydberg states into its
intermediate state with matrix elements Omega_lower(t)/2 and Omega_upper(t)/2.
The lower leg includes the carrier phase. The convention is

    H_p/hbar = (Delta + nominal_Rydberg_carrier) P_p
    H_opt,explicit/hbar = H_p/hbar + W(t) + W(t).dag()
    L_f,explicit = sqrt(Gamma * branching_f) |f><p|

With **Two-photon effective model**, intermediates are eliminated consistently:

    H_opt,effective/hbar = -Delta/(Delta^2+(Gamma/2)^2) * W(t).dag()*W(t)
    L_f,effective(t) = sqrt(Gamma * branching_f)/(Delta-i*Gamma/2) * |f><p| W(t)

The reduced operators are constructed directly on the original atomic space;
intermediate population is absent. This retains differential light shifts,
Rydberg-neighbor coherent terms, and interference between virtual scattering paths.
Replacing these with independent population-loss rates would miss dark-state
interference. The derivation follows the
[effective-operator formalism](https://arxiv.org/abs/1112.2806).

Effective mode requires detuning at least ten times the Rabi/linewidth scales,
retained optical-neighbor energy offsets at the laser-on fields, and reference pulse
bandwidth. This guard is a regime check, not an accuracy certificate. Compare with
explicit mode and vary detuning before trusting an effective calculation.

With **Explicit intermediate states**, two states `optical_p0` and `optical_p1` are added. Their
populations appear in plots and numerical diagnostics. Spontaneous decay remains
active after lasers turn off, including during storage. The solver resolves the
intermediate detuning adaptively; large detunings can make explicit simulations
substantially slower. Effective scattering is not added in this mode.

Both modes retain light shifts without automatically chirping the lasers to
compensate them. Carriers are fixed to nominal atomic energies and remain fixed
through tolerance scenarios. Single-photon detunings and optical dipoles are treated
as fixed inputs over the laser-on field range. No intermediate Stark curve or
additional far-off-resonant optical levels are inferred.

Existing DISSIPATION optical_scattering entries remain independent extra channels.
Avoid double-counting spontaneous scattering already represented by OPTICAL_SCHEME.
`noisy=False` removes jumps while keeping the same coherent Hamiltonian. Storage
turns off driven effective scattering; explicit intermediate decay remains on.
The legacy isolated three-level leakage and paper-summary scans require ideal optics
(OPTICAL_SCHEME={}); the physical optical model is supported in the main write,
read, round-trip and simultaneous figures.

## Verification

Run:

```sh
MPLBACKEND=Agg .venv/bin/python -m unittest discover -s hybrid_memory_fixed -p 'test_physical_hamiltonian.py'
```

Tests independently check table validation and boundaries, signed dipoles and units,
reference times, resonance and diagnostics, the full field-noise derivative, light
shifts, coherent scattering/dark states, pulse area, intermediate decay during
storage, and convergence of effective dynamics toward explicit dynamics.
