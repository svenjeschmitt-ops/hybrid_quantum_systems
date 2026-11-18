# Atomic and optical Hamiltonian inputs

Atomic energies use quadratic Stark shifts and atom–cavity couplings remain
constant during each run. The optical model uses independent effective
h0/r0 and h1/r1 Rabi frequencies (OMEGA and OMEGA_1, in MHz) with separate
carrier frequencies and sin-squared pulse durations 1/OMEGA and 1/OMEGA_1
in microseconds. Both mapping pulses start together and the stage ends after
the longer pulse. The model does not calculate intermediate-state light shifts
or spontaneous optical scattering.

Saved settings with only OMEGA assign that value to both arms. Empty legacy
OPTICAL_SCHEME and ATOMIC_SPECTRUM settings are accepted; nonempty settings
are rejected because those advanced models have been removed.
