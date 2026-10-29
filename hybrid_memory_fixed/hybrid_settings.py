"""Shared parameter definitions and mathematical checks for the hybrid model."""
import json
import math
from pathlib import Path

# Group, key, label, units, default, positive/nonnegative/free/integer.
FIELDS = [
("Cavity and transmon","FC","Cavity frequency","MHz",5000.0,"positive"),
("Cavity and transmon","EC","Charging energy E_C / h","MHz",250.0,"positive"),
("Cavity and transmon","EJ_MAX","Maximum Josephson energy / h","MHz",20000.0,"positive"),
("Cavity and transmon","ASYMMETRY","SQUID junction asymmetry","dimensionless",0.10,"nonnegative"),
("Cavity and transmon","F_T_PARK","Parked transmon frequency","MHz",5800.0,"positive"),
("Cavity and transmon","G_TC","Transmon-cavity coupling g / 2pi","MHz",20.0,"positive"),
("Cavity and transmon","G_AC","Atom-cavity coupling g / 2pi","MHz",0.5,"positive"),
("Atom and electrodes","F_A_ZERO","Zero-field Rydberg transition","MHz",5150.0,"positive"),
("Atom and electrodes","K_R0","Lower Rydberg Stark coefficient","MHz/(V/cm)^2",-2.0,"free"),
("Atom and electrodes","K_R1","Upper Rydberg Stark coefficient","MHz/(V/cm)^2",-12.0,"free"),
("Atom and electrodes","K_WRONG","Unwanted-level Stark coefficient","MHz/(V/cm)^2",-2.0,"free"),
("Atom and electrodes","GAP_CM","Electrode separation","cm",0.10,"positive"),
("Atom and electrodes","STRAY_FIELD","Static stray electric field","V/cm",0.20,"free"),
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


def validate(values):
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
    if not 0 <= config["INPUT_THETA"] <= 180:
        raise ValueError("Input polar angle must be between 0 and 180 degrees.")
    if config["K_R1"] == config["K_R0"]:
        raise ValueError("Different Rydberg Stark coefficients are needed for voltage tuning.")
    field_squared = ((config["FC"]-config["F_A_ZERO"]) /
                     (config["K_R1"]-config["K_R0"]))
    if field_squared < 0:
        raise ValueError("These Stark coefficients cannot tune the atomic transition to this cavity.")
    for key in ("FC","F_T_PARK"):
        ratio = (config[key]+config["EC"])**2 / (8*config["EC"]*config["EJ_MAX"])
        if not config["ASYMMETRY"] <= ratio <= 1:
            raise ValueError("The cavity or parked transmon frequency is outside the SQUID tuning range.")
    return config


def derived(config):
    c=validate(config)
    field=math.sqrt((c["FC"]-c["F_A_ZERO"])/(c["K_R1"]-c["K_R0"]))
    return {
        "resonance_field_V_per_cm": field,
        "resonance_voltage_V": c["GAP_CM"]*(field-c["STRAY_FIELD"]),
        "park_voltage_V": c["GAP_CM"]*(c["E_PARK"]-c["STRAY_FIELD"]),
        "optical_pi_pulse_us": 1/c["OMEGA"],
        "transmon_swap_hold_us": 1/(4*c["G_TC"]),
        "atom_swap_hold_us": 1/(4*c["G_AC"]),
        "cavity_Q": 2*math.pi*c["FC"]*c["T1_C"],
    }


def load(path):
    return validate(json.loads(Path(path).read_text()))
