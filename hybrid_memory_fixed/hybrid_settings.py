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
("Cavity and transmon","EC","Charging energy E_C / h","MHz",250.0,"positive"),
("Cavity and transmon","EJ_MAX","Maximum Josephson energy / h","MHz",20000.0,"positive"),
("Cavity and transmon","ASYMMETRY","SQUID junction asymmetry","dimensionless",0.10,"nonnegative"),
("Cavity and transmon","F_T_PARK","Parked transmon frequency","MHz",5800.0,"positive"),
("Cavity and transmon","PHI_SWAP","Swap flux / flux quantum","dimensionless",0.2597137967167845,"nonnegative"),
("Coupling parameters","BETA","Capacitive participation beta","dimensionless",0.10,"positive"),
("Coupling parameters","V_ZPF_UV","Cavity zero-point voltage at transmon","uV",0.5,"positive"),
("Coupling parameters","N_01","Charge matrix element |<0|n|1>|","dimensionless",0.8,"positive"),
("Coupling parameters","D_01_EA0","Atomic transition dipole magnitude |d_01|","e a0",1000.0,"positive"),
("Coupling parameters","E_ZPF_V_PER_M","Cavity zero-point field at atom","V/m",0.04,"positive"),
("Coupling parameters","DIPOLE_ALIGNMENT","Dipole-field alignment cos(theta)","dimensionless",1.0,"free"),
("Atom and electrodes","F_A_ZERO","Zero-field Rydberg transition","MHz",5150.0,"positive"),
("Atom and electrodes","K_R0","Lower Rydberg Stark coefficient","MHz/(V/cm)^2",-2.0,"free"),
("Atom and electrodes","K_R1","Upper Rydberg Stark coefficient","MHz/(V/cm)^2",-12.0,"free"),
("Atom and electrodes","K_WRONG","Unwanted-level Stark coefficient","MHz/(V/cm)^2",-2.0,"free"),
("Atom and electrodes","GAP_CM","Electrode separation","cm",0.10,"positive"),
("Atom and electrodes","STRAY_FIELD","Static stray electric field","V/cm",0.20,"free"),
("Atom and electrodes","E_PARK","Parked total electric field","V/cm",0.0,"free"),
("Atom and electrodes","V_SWAP","Swap electrode voltage","V",0.3672983346207417,"free"),
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
("Pulses and input state","OMEGA","Peak effective optical Rabi frequency / 2pi","MHz",10.0,"positive"),
("Pulses and input state","RAMP","Flux/voltage ramp duration","us",0.002,"positive"),
("Pulses and input state","GAP","Delay between operations","us",0.01,"nonnegative"),
("Pulses and input state","STORAGE","Memory storage duration","us",5.0,"nonnegative"),
("Pulses and input state","INPUT_THETA","Input Bloch polar angle","degrees",90.0,"free"),
("Pulses and input state","INPUT_PHI","Input relative phase","degrees",60.0,"free"),
("Numerics","MAX_STEP","Maximum solver time step","us",0.0005,"positive"),
("Numerics","NC","Number of cavity Fock states","integer",3,"integer"),
]
DEFAULTS = {key: default for _,key,_,_,default,_ in FIELDS}


def coupling_strengths(config):
    """Return g/(2*pi) in MHz from validated physical inputs.

    g_TC = 2*e*beta*V_zpf*|n_01|/hbar and
    g_AC = |d_01 dot E_zpf|/hbar. Dividing by 2*pi replaces
    hbar with h; dividing Hz by 1e6 gives MHz.
    V_zpf and E_zpf are local vacuum RMS amplitudes, not peak fields.
    """
    return {
        "transmon_coupling_MHz": (2 * ELEMENTARY_CHARGE / PLANCK
                                  * config["BETA"] * (config["V_ZPF_UV"] * 1e-6)
                                  * config["N_01"] / 1e6),
        "atom_coupling_MHz": (ELEMENTARY_CHARGE * BOHR_RADIUS / PLANCK
                              * config["D_01_EA0"] * config["E_ZPF_V_PER_M"]
                              * abs(config["DIPOLE_ALIGNMENT"]) / 1e6),
    }


def validate(values):
    if not isinstance(values, dict):
        raise ValueError("Settings must be a JSON object.")
    if {"G_TC", "G_AC"} & set(values):
        raise ValueError("G_TC and G_AC are now calculated. Replace coupling inputs with "
                         "BETA, V_ZPF_UV, N_01, D_01_EA0, E_ZPF_V_PER_M, and DIPOLE_ALIGNMENT.")
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
    for label, value in coupling_strengths(config).items():
        if not math.isfinite(value) or value <= 0:
            raise ValueError(label + " must be finite and greater than zero for swap timing.")
    if not 0 <= config["INPUT_THETA"] <= 180:
        raise ValueError("Input polar angle must be between 0 and 180 degrees.")
    if config["PHI_SWAP"] > 0.5:
        raise ValueError("Swap flux / flux quantum must be between 0 and 0.5.")
    ratio = (config["F_T_PARK"]+config["EC"])**2 / (8*config["EC"]*config["EJ_MAX"])
    if not config["ASYMMETRY"] <= ratio <= 1:
        raise ValueError("The parked transmon frequency is outside the SQUID tuning range.")
    return config


def derived(config):
    c=validate(config)
    couplings=coupling_strengths(c)
    # Resonance is a diagnostic only; unreachable crossings remain valid runs.
    differential_stark = c["K_R1"] - c["K_R0"]
    field_squared = ((c["FC"]-c["F_A_ZERO"])/differential_stark
                     if differential_stark else None)
    field = math.sqrt(field_squared) if field_squared is not None and field_squared >= 0 else None
    swap_field = c["V_SWAP"]/c["GAP_CM"] + c["STRAY_FIELD"]
    ej = c["EJ_MAX"] * math.sqrt(math.cos(math.pi*c["PHI_SWAP"])**2
                               + c["ASYMMETRY"]**2 * math.sin(math.pi*c["PHI_SWAP"])**2)
    transmon_frequency = math.sqrt(8*c["EC"]*ej) - c["EC"]
    return {
        **couplings,
        "resonance_field_V_per_cm": field,
        "resonance_voltage_V": (c["GAP_CM"]*(field-c["STRAY_FIELD"])
                                if field is not None else None),
        "transmon_swap_detuning_MHz": transmon_frequency-c["FC"],
        "atom_swap_detuning_MHz": c["F_A_ZERO"]+differential_stark*swap_field**2-c["FC"],
        "park_voltage_V": c["GAP_CM"]*(c["E_PARK"]-c["STRAY_FIELD"]),
        "optical_pi_pulse_us": 1/c["OMEGA"],
        "transmon_swap_hold_us": 1/(4*couplings["transmon_coupling_MHz"]),
        "atom_swap_hold_us": 1/(4*couplings["atom_coupling_MHz"]),
        "cavity_Q": 2*math.pi*c["FC"]*c["T1_C"],
    }


def load(path):
    return validate(json.loads(Path(path).read_text()))
