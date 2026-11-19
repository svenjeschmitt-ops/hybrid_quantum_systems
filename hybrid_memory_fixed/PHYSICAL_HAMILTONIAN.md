# Atomic and optical Hamiltonian inputs

Atomic energies use quadratic Stark shifts and atom–cavity couplings remain
constant during each run. The optical model uses independent effective
h0/r0 and h1/r1 Rabi frequencies (OMEGA and OMEGA_1, in MHz) with separate
carrier frequencies and sin-squared pulse durations 1/OMEGA and 1/OMEGA_1
in microseconds. Both mapping pulses start together and the stage ends after
the longer pulse.

For each arm, enter both OPTICAL_TRANSITION_0/1 and OPTICAL_MAPPING_0/1 in
MHz or leave both as auto. The transition value is the measured effective
h-to-Rydberg frequency at zero electric field; the mapping value is the fixed
effective two-photon frequency. At field E the nominal transition is
f_transition(0) + K_Rj E^2, assuming the hyperfine Stark shift is negligible.
The mapping frequency stays fixed when the electric field changes. With auto
settings the effective carrier follows the nominal transition at the laser-on
field. Legacy nonzero offsets are incorporated into an explicit mapping
frequency; auto-mode legacy offsets require the user to enter frequencies.

The same effective mapping carrier applies to preparation, write and read. The
solver removes absolute carrier frequencies using a rotating frame. Entered
frequencies do not identify the atom, calculate the two individual lasers, or
include hyperfine field shifts, intermediate-state light shifts or spontaneous
optical scattering. The existing Rydberg splitting and energy model remain
separate inputs and should be checked for consistency with the chosen states.

Saved settings with only OMEGA assign that value to both arms. Empty legacy
OPTICAL_SCHEME and ATOMIC_SPECTRUM settings are accepted; nonempty settings
are rejected because those advanced models have been removed.
