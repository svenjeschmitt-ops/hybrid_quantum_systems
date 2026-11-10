"""Shared parameter definitions and mathematical checks for the hybrid model."""
import json
import math
from pathlib import Path

# SI constants (e and h are exact; a0 is the Bohr radius in metres).
ELEMENTARY_CHARGE = 1.602176634e-19
PLANCK = 6.62607015e-34
BOHR_RADIUS = 5.29177210544e-11

# Group, key, label, units, default, positive/nonnegative/free/integer.
FIELDS = [
("Cavity and transmon","FC","Cavity frequency","MHz",5000.0,"positive"),
("Cavity and transmon","PHI_SWAP","Transmon flux at pulse peak","Phi/Phi0",0.26588630919611617,"nonnegative"),
("Cavity and transmon","EC","Charging energy E_C / h","MHz",250.0,"positive"),
("Cavity and transmon","EJ_MAX","Maximum Josephson energy / h","MHz",20000.0,"positive"),
("Cavity and transmon","ASYMMETRY","SQUID junction asymmetry","dimensionless",0.10,"nonnegative"),
("Cavity and transmon","PHI_PARK","Parked transmon flux","Phi/Phi0",0.13284821090062188,"nonnegative"),
("Coupling parameters","BETA","Capacitive participation beta","dimensionless",0.04,"positive"),
("Coupling parameters","V_ZPF_UV","Cavity zero-point voltage at transmon","uV",0.5,"positive"),
("Coupling parameters","D_01_EA0","Atomic transition dipole magnitude |d_01|","e a0",1000.0,"positive"),
("Coupling parameters","E_ZPF_V_PER_M","Cavity zero-point field at atom","V/m",0.8,"positive"),
("Coupling parameters","DIPOLE_ALIGNMENT","Dipole-field alignment cos(theta)","dimensionless",1.0,"free"),
("Coupling parameters","DIRECT_COUPLING","Use direct coupling inputs","0=calculated, 1=manual",0,"boolean"),
("Coupling parameters","G_TC_MHZ","Transmon-cavity coupling g_TC / (2 pi)","MHz",11.023791568183459,"positive"),
("Coupling parameters","G_AC_MHZ","Atom-cavity coupling g_AC / (2 pi)","MHz",10.236358418015183,"positive"),
("Atom and electrodes","F_A_ZERO","Zero-field Rydberg transition","MHz",5150.0,"positive"),
("Atom and electrodes","K_R0","Lower Rydberg Stark coefficient","MHz/(V/cm)^2",-2.0,"free"),
("Atom and electrodes","K_R1","Upper Rydberg Stark coefficient","MHz/(V/cm)^2",-12.0,"free"),
("Atom and electrodes","K_WRONG","Unwanted-level Stark coefficient","MHz/(V/cm)^2",-2.0,"free"),
("Atom and electrodes","GAP_CM","Electrode separation","cm",0.10,"positive"),
("Atom and electrodes","STRAY_FIELD","Static stray electric field","V/cm",0.20,"free"),
("Atom and electrodes","V_SWAP","Applied voltage at pulse peak","V",0.3923668021130249,"free"),
("Atom and electrodes","E_PARK","Parked total electric field","V/cm",0.0,"free"),
("Atom and electrodes","F_WRONG_SPLIT","Unwanted-level frequency separation","MHz",1320.0,"free"),
("Atom and electrodes","REL_WRONG_DIPOLE","Relative unwanted optical amplitude","dimensionless",1.0,"nonnegative"),
("Loss and coherence","T1_T","Transmon relaxation time","us",100.0,"positive"),
("Loss and coherence","TPHI_T","Transmon PURE dephasing time","us",150.0,"positive"),
("Loss and coherence","T1_C","Cavity photon-number lifetime","us",200.0,"positive"),
("Loss and coherence","TAU_R0","Lower Rydberg lifetime","us",100.0,"positive"),
("Loss and coherence","TAU_R1","Upper Rydberg lifetime","us",100.0,"positive"),
("Loss and coherence","TAU_WRONG","Unwanted Rydberg lifetime","us",100.0,"positive"),
("Loss and coherence","TPHI_R","Rydberg-pair pure dephasing time","us",100.0,"positive"),
("Loss and coherence","TPHI_H","Hyperfine pure dephasing time","us",100000.0,"positive"),
("Pulses and input state","OMEGA","Peak effective optical Rabi frequency / 2pi","MHz",100.0,"positive"),
("Pulses and input state","RAMP","Extra duration per pulse half","us",0.0005,"positive"),
("Pulses and input state","TC_PULSE_SCALE","Transmon pulse duration multiplier","dimensionless",9.664304506350172,"positive"),
("Pulses and input state","AC_PULSE_SCALE","Atomic pulse duration multiplier","dimensionless",3.3585450169623496,"positive"),
("Pulses and input state","GAP","Delay between operations","us",0.01,"nonnegative"),
("Pulses and input state","STORAGE","Memory storage duration","us",5.0,"nonnegative"),
("Pulses and input state","INPUT_THETA","Input Bloch polar angle","degrees",90.0,"free"),
("Pulses and input state","INPUT_PHI","Input relative phase","degrees",60.0,"free"),
("Experimental tolerances","FC_TOL","Cavity frequency tolerance (+/-)","MHz",0.0,"nonnegative"),
("Experimental tolerances","FT_TOL","Transmon frequency tolerance (+/-)","MHz",0.0,"nonnegative"),
("Experimental tolerances","FA_TOL","Atomic transition frequency tolerance (+/-)","MHz",0.0,"nonnegative"),
("Experimental tolerances","FLUX_TOL","Applied flux tolerance (+/-)","Phi/Phi0",0.0,"nonnegative"),
("Experimental tolerances","EFIELD_TOL","Total electric field tolerance (+/-)","V/cm",0.0,"nonnegative"),
("Simultaneous resonance","SIM_HOLD_SCALE","Resonant hold time multiplier","dimensionless",1.0,"positive"),
("Simultaneous resonance","SIM_RAMP","Simultaneous tuning ramp duration","us",0.0005,"positive"),
("Numerics","MAX_STEP","Maximum solver time step","us",0.0005,"positive"),
("Numerics","NC","Number of cavity Fock states","integer",3,"integer"),
]
DEFAULTS = {key: default for _,key,_,_,default,_ in FIELDS}


def transmon_operating_point(config):
    """Approximate n_01 from the effective EJ/h at the applied interaction flux."""
    phi = config["PHI_SWAP"]
    ej = config["EJ_MAX"] * math.sqrt(math.cos(math.pi*phi)**2
                                    + config["ASYMMETRY"]**2 * math.sin(math.pi*phi)**2)
    return {
        "transmon_swap_EJ_MHz": ej,
        "transmon_interaction_frequency_MHz": math.sqrt(8*config["EC"]*ej) - config["EC"],
        "transmon_charge_matrix_element": (ej/(32*config["EC"]))**0.25,
    }


def coupling_strengths(config):
    """Return g/(2*pi) in MHz from validated physical inputs.

    g_TC = 2*e*beta*V_zpf*(EJ_swap/(32*EC))**(1/4)/hbar and
    g_AC = |d_01 dot E_zpf|/hbar. Dividing by 2*pi replaces
    hbar with h; dividing Hz by 1e6 gives MHz.
    V_zpf and E_zpf are local vacuum RMS amplitudes, not peak fields.
    """
    if config.get("DIRECT_COUPLING", 0):
        return {"transmon_coupling_MHz": config["G_TC_MHZ"],
                "atom_coupling_MHz": config["G_AC_MHZ"]}
    n_01 = transmon_operating_point(config)["transmon_charge_matrix_element"]
    return {
        "transmon_coupling_MHz": (2 * ELEMENTARY_CHARGE / PLANCK
                                  * config["BETA"] * (config["V_ZPF_UV"] * 1e-6)
                                  * n_01 / 1e6),
        "atom_coupling_MHz": (ELEMENTARY_CHARGE * BOHR_RADIUS / PLANCK
                              * config["D_01_EA0"] * config["E_ZPF_V_PER_M"]
                              * abs(config["DIPOLE_ALIGNMENT"]) / 1e6),
    }


def calculated_transmon_frequency(config, phi):
    """Return the transmon transition frequency in MHz at applied flux Phi/Phi0."""
    ej = config["EJ_MAX"] * math.hypot(math.cos(math.pi*phi),
                                    config["ASYMMETRY"]*math.sin(math.pi*phi))
    return math.sqrt(8*config["EC"]*ej) - config["EC"]


def calculated_swap_field(config):
    """Total field produced by the applied swap voltage and stray field."""
    return config["V_SWAP"] / config["GAP_CM"] + config["STRAY_FIELD"]


def validate(values):
    if not isinstance(values, dict):
        raise ValueError("Settings must be a JSON object.")
    if "F_SWAP_TARGET" in values:
        raise ValueError("F_SWAP_TARGET has been removed. Remove it from saved settings; "
                         "use PHI_SWAP for the transmon interaction flux. Swap rates come from the couplings.")
    if "F_T_INTERACTION" in values:
        raise ValueError("F_T_INTERACTION is now calculated from PHI_SWAP. "
                         "Replace F_T_INTERACTION with the applied interaction flux PHI_SWAP in saved settings.")
    if "F_T_PARK" in values:
        raise ValueError("F_T_PARK is now calculated from PHI_PARK. "
                         "Replace F_T_PARK with the applied parked flux PHI_PARK in saved settings.")
    if {"G_TC", "G_AC"} & set(values):
        raise ValueError("G_TC/G_AC have ambiguous units. For manual coupling use DIRECT_COUPLING=1 "
                         "and G_TC_MHZ/G_AC_MHZ in g/(2*pi) MHz, or use the physical coupling inputs.")
    if "N_01" in values:
        raise ValueError("N_01 is now calculated from the effective Josephson energy at "
                         "the swap flux and EC. Remove N_01 from saved settings.")
    unknown = set(values) - set(DEFAULTS)
    if unknown:
        raise ValueError("Unknown settings: " + ", ".join(sorted(unknown)))
    config = dict(DEFAULTS)
    config.update(values)
    for _,key,label,_,_,rule in FIELDS:
        try:
            value = float(config[key])
        except (TypeError,ValueError):
            raise ValueError(label + " must be a number.") from None
        if not math.isfinite(value):
            raise ValueError(label + " must be finite.")
        if rule == "positive" and value <= 0:
            raise ValueError(label + " must be greater than zero.")
        if rule == "nonnegative" and value < 0:
            raise ValueError(label + " cannot be negative.")
        if rule == "boolean":
            if value not in (0, 1):
                raise ValueError(label + " must be 0 (calculated) or 1 (manual).")
            value = int(value)
        if rule == "integer":
            if value != int(value) or value < 3:
                raise ValueError("Use at least 3 cavity Fock states (an integer).")
            value = int(value)
        config[key] = value
    if not 0 <= config["ASYMMETRY"] < 1:
        raise ValueError("SQUID asymmetry must be between 0 (inclusive) and 1 (exclusive).")
    if config["BETA"] > 1:
        raise ValueError("Capacitive participation beta must be at most 1.")
    if not -1 <= config["DIPOLE_ALIGNMENT"] <= 1:
        raise ValueError("Dipole-field alignment cos(theta) must be between -1 and 1.")
    if not 0 <= config["INPUT_THETA"] <= 180:
        raise ValueError("Input polar angle must be between 0 and 180 degrees.")
    for key, label in (("PHI_SWAP", "Interaction"), ("PHI_PARK", "Parked")):
        if not 0 <= config[key] <= 0.5:
            raise ValueError(label + " transmon flux must be between 0 and 0.5 Phi/Phi0.")
        if calculated_transmon_frequency(config, config[key]) <= 0:
            raise ValueError(label + " flux gives a nonpositive transmon frequency; use a transmon operating point with EJ >> EC.")
    for label, value in coupling_strengths(config).items():
        if not math.isfinite(value) or value <= 0:
            raise ValueError(label + " must be finite and greater than zero for swap timing.")
    return config


def derived(config):
    c=validate(config)
    couplings=coupling_strengths(c)
    operating_point=transmon_operating_point(c)
    differential_stark = c["K_R1"] - c["K_R0"]
    swap_field = calculated_swap_field(c)
    swap_voltage = c["V_SWAP"]
    swap_flux = c["PHI_SWAP"]
    return {
        **couplings,
        **operating_point,
        "transmon_park_frequency_MHz": calculated_transmon_frequency(c, c["PHI_PARK"]),
        "swap_field_V_per_cm": swap_field,
        "swap_voltage_V": swap_voltage,
        "swap_flux": swap_flux,
        "transmon_swap_detuning_MHz": (
            operating_point["transmon_interaction_frequency_MHz"] - c["FC"]),
        "atom_swap_detuning_MHz": c["F_A_ZERO"]+differential_stark*swap_field**2-c["FC"],
        "park_voltage_V": c["GAP_CM"]*(c["E_PARK"]-c["STRAY_FIELD"]),
        "optical_pi_pulse_us": 1/c["OMEGA"],
        "transmon_swap_frequency_MHz": 2*couplings["transmon_coupling_MHz"],
        "atom_swap_frequency_MHz": 2*couplings["atom_coupling_MHz"],
        "transmon_swap_hold_us": 1/(4*couplings["transmon_coupling_MHz"]),
        "atom_swap_hold_us": 1/(4*couplings["atom_coupling_MHz"]),
        "transmon_pulse_duration_us": c["TC_PULSE_SCALE"] * (2*c["RAMP"] + 1/(4*couplings["transmon_coupling_MHz"])),
        "atom_pulse_duration_us": c["AC_PULSE_SCALE"] * (2*c["RAMP"] + 1/(4*couplings["atom_coupling_MHz"])),
        "cavity_Q": 2*math.pi*c["FC"]*c["T1_C"],
    }


def load(path):
    return validate(json.loads(Path(path).read_text()))
