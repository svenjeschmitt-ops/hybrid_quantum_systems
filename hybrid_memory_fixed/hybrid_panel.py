"""Native startup parameter panel. Keep beside backend.py and hybrid_settings.py."""
import base64
from copy import deepcopy
import io
import json
import math
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import tempfile
import threading


def configure_tk_libraries():
    """Find Tcl/Tk bundled with the base interpreter when running in a venv.

    Relocatable Python builds can retain a build-time Tcl search path. Use
    their own runtime scripts rather than scripts from another Python install.
    Explicit TCL_LIBRARY/TK_LIBRARY settings remain authoritative.
    """
    library_root = Path(sys.base_prefix) / "lib"
    for variable, pattern, script in (("TCL_LIBRARY", "tcl*", "init.tcl"),
                                       ("TK_LIBRARY", "tk*", "tk.tcl")):
        if variable not in os.environ:
            for directory in sorted(library_root.glob(pattern)):
                if (directory / script).is_file():
                    os.environ[variable] = str(directory)
                    break


configure_tk_libraries()

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
from hybrid_settings import FIELDS, DEFAULTS, PULSE_DELAY_KEYS, validate, derived
from model_editors import TITLES as MODEL_TITLES, add_model_control
from parameter_sweep import SWEEP_FIELDS, sweep_configs
from simulation_progress import parse_line as parse_progress_line
from assistant_proposals import parse_proposal, prepare_application
from assistant_calculations import ComparisonCancelled, reference_only, run_comparison, checked_answer
from assistant_search import requested_target, run_target_search, search_answer, constant_field_requested, validate_search_result
from codex_assistant import answer as codex_answer, build_messages, check_codex

MODES = [
("Both transfer directions", []),
("Transmon to atom", ["--write-only"]),
("Atom to transmon", ["--retrieve"]),
]


# Symbols use the same notation as the equations beneath each tab.
PARAMETER_SYMBOLS = {
    "CONSTANT_FLUX": r"\phi(t)=\phi_{\mathrm{swap}}",
    "CONSTANT_FIELD": r"E(t)=V_{\mathrm{swap}}/\ell+E_{\mathrm{stray}}",
    "WRITE_PULSE_DELAY_NS": r"\tau_{\mathrm{write}}",
    "READ_PULSE_DELAY_NS": r"\tau_{\mathrm{read}}",
    "SIM_HOLD_SCALE": r"s_{\mathrm{sim}}",
    "SIM_RAMP": r"T_{\mathrm{ramp,sim}}",
    "FC": r"f_c",
    "PHI_SWAP": r"\phi_{\mathrm{swap}}",
    "V_SWAP": r"V_{\mathrm{swap}}",
    "EC": r"\epsilon_C=E_C/h",
    "EJ_MAX": r"\epsilon_{J,\max}=E_{J,\max}/h",
    "ASYMMETRY": r"d",
    "PHI_PARK": r"\phi_{\mathrm{park}}",
    "G_TC_MHZ": r"g_{TC}/(2\pi)",
    "G_AC_MHZ": r"g_{AC}/(2\pi)",
    "BETA": r"\beta",
    "V_ZPF_UV": r"V_{\mathrm{zpf}}",
    "D_01_EA0": r"|d_{01}|",
    "E_ZPF_V_PER_M": r"E_{\mathrm{zpf}}",
    "DIPOLE_ALIGNMENT": r"\cos\theta",
    "F_A_ZERO": r"f_{A,0}",
    "K_R0": r"K_{r0}",
    "K_R1": r"K_{r1}",
    "K_WRONG": r"K_w",
    "GAP_CM": r"\ell",
    "STRAY_FIELD": r"E_{\mathrm{stray}}",
    "E_PARK": r"E_{\mathrm{park}}",
    "F_WRONG_SPLIT": r"f_{\mathrm{wrong,split}}",
    "REL_WRONG_DIPOLE": r"\eta",
    "T1_T": r"T_{1,T}",
    "TPHI_T": r"T_{\phi,T}",
    "T1_C": r"T_{1,c}",
    "TAU_R0": r"\tau_{r0}",
    "TAU_R1": r"\tau_{r1}",
    "TAU_WRONG": r"\tau_w",
    "TPHI_R": r"T_{\phi,R}",
    "TPHI_H": r"T_{\phi,H}",
    "OMEGA": r"\Omega_{0}/(2\pi)",
    "OMEGA_1": r"\Omega_{1}/(2\pi)",
    "RAMP": r"T_{\mathrm{ramp}}",
    "TC_PULSE_SCALE": r"s_{TC}",
    "AC_PULSE_SCALE": r"s_{AC}",
    "GAP": r"T_{\mathrm{gap}}",
    "STORAGE": r"T_{\mathrm{storage}}",
    "INPUT_THETA": r"\vartheta",
    "INPUT_PHI": r"\varphi",
    "MAX_STEP": r"\Delta t_{\max}",
    "NC": r"N_c",
    "NT": r"N_T",
    "RYDBERG_MANIFOLD": r"\mathcal{R}_{extra}",
    "FC_TOL": r"\epsilon_c",
    "FT_TOL": r"\epsilon_T",
    "FA_TOL": r"\epsilon_A",
    "FLUX_TOL": r"\epsilon_\phi",
    "EFIELD_TOL": r"\epsilon_E",
}

# Keep prose labels free of duplicate plain-text versions of typeset symbols.
PARAMETER_CAPTIONS = {
    "EC": "Charging energy",
    "EJ_MAX": "Maximum Josephson energy",
    "BETA": "Capacitive participation",
    "D_01_EA0": "Atomic transition dipole magnitude",
    "DIPOLE_ALIGNMENT": "Dipole-field alignment",
    "OMEGA": "h0 ↔ r0 peak effective optical Rabi frequency",
    "OMEGA_1": "h1 ↔ r1 peak effective optical Rabi frequency",
}


def parameter_symbol_png(key, dpi=120):
    """Render one inline symbol with real Greek letters and subscripts."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure=Figure(figsize=(2,.4),dpi=dpi)
    FigureCanvasAgg(figure)
    figure.text(0,.5,"$"+PARAMETER_SYMBOLS[key]+"$",va="center",
                fontsize=14,math_fontfamily="stix",color="#242424")
    output=io.BytesIO()
    figure.savefig(output,format="png",transparent=True,bbox_inches="tight",pad_inches=.025)
    return output.getvalue()


# Equations describe the implemented model; epsilon denotes energy divided by h.
SECTION_EQUATIONS = {
    "Simultaneous resonance": (
        (
            r"$\mathrm{pulsed\ targets}:\quad f_T(\phi_{\mathrm{res}})=f_A(E_{\mathrm{res}})=f_c$",
            r"$G=\sqrt{g_{TC}^2+g_{AC}^2},\qquad T_{\mathrm{sim}}=s_{\mathrm{sim}}(2T_{r,\mathrm{eff}}+\pi/G)$",
            r"$P_{\max}=4g_{TC}^2g_{AC}^2/(g_{TC}^2+g_{AC}^2)^2$",
        ),
        'Pulsed controls reach their resonant targets at the shared midpoint and return to park. Constant controls use the entered values and may be detuned. T_r,eff = 0 when both controls are constant; otherwise it is SIM_RAMP. Couplings are reference magnitudes at the interaction point. The ideal P_max assumes constant resonance, one excitation, no loss and only the desired three-state chain; rounded pulses and extra levels change transfer. Storage follows writing and precedes retrieval.',
    ),
    "Cavity and transmon": (
        (
            r"$\epsilon_C=E_C/h,\quad \epsilon_J(\phi)=\epsilon_{J,\max}\sqrt{\cos^2(\pi\phi)+d^2\sin^2(\pi\phi)},\quad \phi=\Phi/\Phi_0$",
            r"$f_T(\phi)\approx\sqrt{8\epsilon_C\epsilon_J(\phi)}-\epsilon_C,\qquad \Delta_T(t)=2\pi[f_T(\phi(t))-f_c]$",
            r"$f_{T,\mathrm{int}}=f_T(\phi_{\mathrm{swap}}),\qquad f_{T,\mathrm{park}}=f_T(\phi_{\mathrm{park}})$",
            r"$\Delta_{T,\mathrm{int}}=2\pi(f_{T,\mathrm{int}}-f_c),\qquad \mathrm{resonance}:\ f_{T,\mathrm{int}}=f_c$",
            r"$\phi(t)=\phi_{\mathrm{park}}+[\phi_{\mathrm{swap}}-\phi_{\mathrm{park}}]w_{TC}(t)$",
            r"$\Delta_{T,\mathrm{park}}=2\pi(f_{T,\mathrm{park}}-f_c),\qquad H_T(t)/\hbar=\Delta_T(t)\hat n_T-\pi\epsilon_C\hat n_T(\hat n_T-1)$",
        ),
        'Energies divided by h: MHz. d: SQUID asymmetry. n̂_T = b†b is the excitation operator; NT is the retained ladder size. Equations use nominal controls. Hold flux constant uses the entered pulse-peak flux throughout; interpret φ_park = φ_swap in these equations. Otherwise w_TC is a sin² excursion, zero outside its pulse and one at its midpoint. Entered interaction flux sets f_T independently of f_c. Coupling remains active at park. See Tolerances for static errors.',
    ),
    "Coupling parameters": (
        (
            r"$g_{\mathrm{TC}}\approx\frac{2e\beta V_{\mathrm{zpf}}}{\hbar}\left(\frac{E_J(\Phi_{\mathrm{swap}})}{32E_C}\right)^{1/4}$",
            r"$g_{\mathrm{AC}}=\frac{|\mathbf{d}_{01}\cdot\mathbf{E}_{\mathrm{zpf}}|}{\hbar}=\frac{|d_{01}|E_{\mathrm{zpf}}|\cos\theta|}{\hbar}$",
            r"$f_{\mathrm{swap},TC}=2g_{TC}/(2\pi),\qquad f_{\mathrm{swap},AC}=2g_{AC}/(2\pi),\qquad t_{\mathrm{ref}}=1/(2f_{\mathrm{swap}})$",
            r"$\beta=\frac{C_g}{C_\Sigma},\qquad n_{01}\approx\left(\frac{E_J(\Phi_{\mathrm{swap}})}{32E_C}\right)^{1/4}$",
        ),
        'C_g: coupling capacitance; C_Σ: total transmon capacitance. The charge matrix element is evaluated at the interaction flux and held fixed during a run. These formulas apply without manual overrides. Swap frequencies are ideal single-excitation references; rounded pulses and detuning change transfer.',
    ),
    "Direct coupling": (
        (
            r"$g_{TC}=2\pi\,g_{TC,\mathrm{input}},\qquad g_{AC}=2\pi\,g_{AC,\mathrm{input}}$",
            r"$f_{\mathrm{swap},j}=2g_{j,\mathrm{input}},\qquad t_{\mathrm{ref},j}=1/(4g_{j,\mathrm{input}})$",
            r"$T_j=s_j(2T_{\mathrm{ramp}}+t_{\mathrm{ref},j}),\qquad j\in\{TC,AC\}$",
        ),
        'Inputs are g/(2π) in MHz; Hamiltonian couplings are in rad/μs. Manual couplings remain constant during each run. Changing couplings does not retune sequential controls or optimize pulse multipliers.',
    ),
    "Atom and electrodes": (
        (
            r"$E(t)=V(t)/\ell+E_{\mathrm{stray}},\quad E_{\mathrm{swap}}=V_{\mathrm{swap}}/\ell+E_{\mathrm{stray}}$",
            r"$\varepsilon_i(E)=\varepsilon_i(0)+K_iE^2,\quad f_A(E)=\varepsilon_{r1}(E)-\varepsilon_{r0}(E)$",
            r"$V_{\mathrm{park}}=\ell(E_{\mathrm{park}}-E_{\mathrm{stray}}),\quad \Delta_A(t)=2\pi[f_A(E(t))-f_c]$",
            r"$\Delta_{w,r0}(E)=2\pi[\varepsilon_w(E)-\varepsilon_{r0}(E)],\qquad \Omega_w(t)=\eta\,\Omega(t)$",
        ),
        'ε_i = energy/h relative to zero-field r0 (MHz). Quadratic defaults: ε_r0(0) = 0, ε_r1(0) = f_A,0, ε_w(0) = f_wrong,split. Extra offsets are relative to their branch.\n\nHold field constant uses the entered voltage throughout; interpret E_park = E_swap. ℓ: cm; E: V/cm; K: MHz/(V/cm)², including −α/(2h). w denotes one unwanted level (r0_wrong), with relative optical amplitude η. Equations use nominal controls.',
    ),
    "Loss and coherence": (
        (
            r"$L_T=b/\sqrt{T_{1,T}},\quad L_C=a/\sqrt{T_{1,c}},\quad L_j=|k_j\rangle\langle j|/\sqrt{\tau_j}$",
            r"$L_{\phi,T}=\frac{Z_T}{\sqrt{2T_{\phi,T}}},\qquad L_{\phi,R}=\frac{Z_R}{\sqrt{2T_{\phi,R}}},\qquad L_{\phi,H}=\frac{Z_H}{\sqrt{2T_{\phi,H}}}$",
            r"$Z_T=2b^\dagger b-I,\quad Z_R=P_{r1}-P_{r0}+\sum_{j\in\mathrm{extra}}s_jP_j,\quad Z_H=P_{h1}-P_{h0}$",
            r"$\frac{1}{T_{2,T}}=\frac{1}{2T_{1,T}}+\frac{1}{T_{\phi,T}},\qquad Q_c=2\pi f_cT_{1,c}$",
        ),
        'Times: μs. Rydberg r1 decays to r0; other Rydberg levels decay to lost. Extra-level s_j = −1/+1 for lower/upper branch; built-in wrong has zero weight in Z_R. The T2 relation assumes standard relaxation and no optical drive.',
    ),
    "Pulses and input state": (
        (
            r"$\Omega_j(t)=\Omega_{j,\max}\sin^2\!\left(\frac{\pi(t-t_0)}{T_j}\right),\quad \int\Omega_j(t)\,dt=\pi,\quad T_j=\frac{2\pi}{\Omega_{j,\max}},\quad j=0,1$",
            r"$t_{\mathrm{ref},j}=\frac{\pi}{2g_j},\qquad T_j=s_j(2T_{\mathrm{ramp}}+t_{\mathrm{ref},j}),\quad j=TC,AC$",
            r"$w(u)=\sin^2(\pi u/T_{\mathrm{swap}}),\quad 0\leq u\leq T_{\mathrm{swap}};\qquad w=0\ \mathrm{outside}$",
            r"$|\psi_{\mathrm{in}}\rangle=\cos(\vartheta/2)|0_L\rangle+e^{i\varphi}\sin(\vartheta/2)|1_L\rangle,\quad t_{\mathrm{end,write}}=t_{\mathrm{map,end}}+T_{\mathrm{storage}}$",
        ),
        'Times: μs; delays: ns; angles: degrees. Ω_max is angular frequency. The π area is a reference, not guaranteed transfer. Each transition has its own effective peak Ω and sin² pulse duration; the two mapping pulses start together and the stage ends when both finish.\n\nLogical states: g/e for writing, h0/h1 for retrieval. Storage follows writing and precedes retrieval. GAP separates optical and exchange stages. Delay = voltage peak − flux peak; auto uses flux first for writing, voltage first for reading. Pulse multipliers include RAMP.',
    ),
    "Experimental tolerances": (
        (
            r"$\delta x\in\{-\epsilon_x,0,+\epsilon_x\},\qquad x\in\{f_c,f_T,f_A,\phi,E\}$",
            r"$\phi_{\mathrm{actual}}(t)=\phi_{\mathrm{nominal}}(t)+\delta\phi$",
            r"$E_{\mathrm{actual}}(t)=E_{\mathrm{nominal}}(t)+\delta E$",
            r"$f_{T,\mathrm{actual}}(t)=f_T(\phi_{\mathrm{actual}}(t))+\delta f_T$",
            r"$f_{A,\mathrm{actual}}(t)=f_A(E_{\mathrm{actual}}(t))+\delta f_A,\quad f_{c,\mathrm{actual}}=f_c+\delta f_c$",
        ),
        "Enter absolute ± bounds; 0 disables a tolerance.\n"
        "Transfer plots and sweeps show shaded ranges and a With tolerances column.\n"
        "Detunings use actual frequencies: Δ_j = 2π(f_j,actual − f_c,actual). f_A(E) includes energy tables when supplied. "
        "Offsets are static; nominal pulse timings, optical carriers and coupling calibration stay fixed. "
        "Tabulated dipoles are evaluated at the actual field. Ranges are sampled, not guaranteed limits. Details: README.txt.",
    ),
    "Numerics": (
        (
            r"$\mathcal{H}=H/\hbar,\quad \dot\rho=-i[\mathcal{H},\rho]+\sum_k\left(L_k\rho L_k^\dagger-\frac{1}{2}\{L_k^\dagger L_k,\rho\}\right)$",
            r"$i\,\frac{d|\psi\rangle}{dt}=\mathcal{H}|\psi\rangle\quad\mathrm{(noiseless)},\qquad \Delta t_{\mathrm{internal}}\leq\Delta t_{\max}=\mathrm{MAX\_STEP}$",
            r"$D=N_T N_c(6+N_{extra}+N_p),\qquad N_T\geq2,\quad N_c\geq3$",
        ),
        "NT sets the weakly anharmonic transmon ladder size. Use Rydberg levels to edit additional atomic levels and cavity links. "
        "Defaults include one illustrative upper neighbor. N_p = 2 for the explicit laser model, otherwise 0. "
        "The Hamiltonian ℋ is in rad/μs; collapse rates are in 1/μs. "
        "Compare increasing bases with fixed controls before claiming convergence.",
    ),
}


def section_equations_png(group, dpi=120, max_width=None):
    """Typeset a tab's equations without an external LaTeX installation."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    equations, _ = SECTION_EQUATIONS[group]
    height = .66 * len(equations)
    figure=Figure(figsize=(8.4,height),dpi=dpi)
    FigureCanvasAgg(figure)
    for index,equation in enumerate(equations):
        figure.text(0.01,1-(index+.5)/len(equations),equation,va="center",
                    fontsize=16,math_fontfamily="stix",color="#242424")
    if max_width is not None:
        figure.canvas.draw()
        renderer=figure.canvas.get_renderer()
        text_width=max(text.get_window_extent(renderer).width for text in figure.texts)
        scale=min(1.0,max(1,max_width-.2*dpi)/text_width)
        for text in figure.texts:
            text.set_fontsize(text.get_fontsize()*scale)
        figure.set_figheight(height*scale)
    output=io.BytesIO()
    figure.savefig(output,format="png",transparent=True,bbox_inches="tight",pad_inches=0.10)
    return output.getvalue()


def coupling_equations_png(dpi=120):
    """Compatibility wrapper for the coupling-equation renderer."""
    return section_equations_png("Coupling parameters", dpi=dpi)


class ParameterTab(ttk.Frame):
    """Allocate fixed panes so content cannot displace the bottom controls."""
    def __init__(self, parent):
        super().__init__(parent)
        self.body=ttk.Frame(self)
        self.body.place(x=0,y=0,relwidth=1,relheight=1)



class ParameterPanel:
    def __init__(self, root, backend_path=None):
        self.root=root
        if backend_path is None:
            folder=Path(__file__).resolve().parent
            backend_path=folder/("backend.py" if (folder/"backend.py").exists() else "codetest.py")
        self.backend_path=Path(backend_path).resolve()
        self.process=None
        self.plot_processes=[]
        self.events=queue.Queue()
        self.temp_path=None
        self.stop_path=None
        self.last_run=None
        self.assistant_context_run=None
        self.assistant_history=[]
        self.assistant_pending=None
        self.assistant_proposal=None
        self.assistant_preview=None
        self._updating_assistant_values=False
        self.assistant_request=0
        self.assistant_cancel=threading.Event()
        root.title("Hybrid quantum memory — simulation parameters")
        width=min(1800,int(root.winfo_screenwidth()*.96))
        height=min(1100,int(root.winfo_screenheight()*.90))
        root.geometry(f"{width}x{height}")
        root.minsize(min(width,1100),min(height,700))
        self.entries={}
        self.input_widgets={}
        self.equation_images={}
        self.symbol_images={}
        self.parameter_tabs={}
        outer=ttk.Frame(root,padding=14)
        outer.pack(fill="both",expand=True)
        outer.columnconfigure(0,weight=1)
        outer.rowconfigure(2,weight=1)
        ttk.Label(outer,text="Hybrid quantum memory",font=("",18,"bold")).grid(row=0,column=0,sticky="w")
        ttk.Label(outer,text="Edit settings to preview pulses. Run simulation for full results.").grid(
            row=1,column=0,sticky="w",pady=(3,10))
        self.tabs=ttk.Notebook(outer)
        self.tabs.grid(row=2,column=0,sticky="nsew")
        groups=list(dict.fromkeys(row[0] for row in FIELDS if row[0]!="Simultaneous resonance"))
        for group in groups:
            tab=ParameterTab(self.tabs)
            caption={"Cavity and transmon":"Cavity & transmon", "Coupling parameters":"Couplings",
                     "Atom and electrodes":"Atom & electrodes", "Loss and coherence":"Loss & coherence",
                     "Pulses and input state":"Pulses & input", "Experimental tolerances":"Tolerances"}.get(group,group)
            self.tabs.add(tab,text=caption)
            self.parameter_tabs[group]=tab
            tab.body.columnconfigure(0,weight=3,uniform="panes")
            tab.body.columnconfigure(1,weight=2,uniform="panes")
            tab.body.rowconfigure(0,weight=1)
            frame=ttk.Frame(tab.body,padding=(8,6))
            frame.grid(row=0,column=0,sticky="nsew")
            frame.grid_propagate(False)
            frame.columnconfigure(1,weight=1)
            equations=ttk.Frame(tab.body,padding=(8,8))
            equations.grid(row=0,column=1,sticky="nsew")
            equations.grid_propagate(False)
            equations.columnconfigure(0,weight=1)
            if group == "Pulses and input state":
                equations.grid(sticky="nsew")
                equations.rowconfigure(0,weight=1)
                self.pulse_preview_tabs=ttk.Notebook(equations)
                self.pulse_preview_tabs.grid(row=0,column=0,sticky="nsew")
                self.pulse_preview_frame=ttk.Frame(self.pulse_preview_tabs,padding=4)
                reference=ttk.Frame(self.pulse_preview_tabs,padding=8)
                self.pulse_preview_tabs.add(self.pulse_preview_frame,text="Live preview")
                self.pulse_preview_tabs.add(reference,text="Equations and notes")
                equations=reference
                equations.columnconfigure(0,weight=1)
            if group == "Numerics":
                equations.rowconfigure(0,weight=1)
                self.numerics_tabs=ttk.Notebook(equations)
                self.numerics_tabs.grid(row=0,column=0,sticky="nsew")
                reference=ttk.Frame(self.numerics_tabs,padding=8)
                sweep=ttk.Frame(self.numerics_tabs,padding=8)
                sweep.columnconfigure(0,weight=1)
                self.numerics_tabs.add(reference,text="Equations and notes")
                self.numerics_tabs.add(sweep,text="Parameter sweep")
                self.build_sweep_controls(sweep,0,width)
                self.sweep_tab=sweep
                equations=reference
                equations.columnconfigure(0,weight=1)
            for row,(_,key,label,units,default,rule) in enumerate(f for f in FIELDS if f[0]==group and f[1] not in ("DIRECT_COUPLING","G_TC_MHZ","G_AC_MHZ")):
                label_cell=ttk.Frame(frame)
                label_cell.grid(row=row,column=0,sticky="w",padx=(0,8),pady=3)
                ttk.Label(label_cell,text=PARAMETER_CAPTIONS.get(key,label),wraplength=230,justify="left").pack(side="left")
                self.symbol_images[key]=tk.PhotoImage(
                    master=root,data=base64.b64encode(
                        parameter_symbol_png(key,dpi=root.winfo_fpixels("1i"))).decode("ascii"))
                ttk.Label(label_cell,image=self.symbol_images[key]).pack(side="left",padx=(8,0))
                value=tk.StringVar(value=str(default))
                self.entries[key]=value
                if key in MODEL_TITLES:
                    entry=add_model_control(frame,row,key,value,self.entries,validate)
                else:
                    entry=(ttk.Checkbutton(frame,variable=value,onvalue="1",offvalue="0")
                           if rule == "boolean" else ttk.Entry(frame,textvariable=value,width=14))
                    entry.grid(row=row,column=1,sticky="ew",pady=3)
                    ttk.Label(frame,text=units,wraplength=95).grid(row=row,column=2,sticky="w",padx=6)
                self.input_widgets[key]=entry
            if group == "Cavity and transmon":
                self.transmon_tuning=tk.StringVar()
                ttk.Label(frame,textvariable=self.transmon_tuning,wraplength=int(width*.4)).grid(
                    row=row+1,column=0,columnspan=3,sticky="w",pady=(12,0))
            if group == "Coupling parameters":
                self.build_direct_coupling_controls(frame,row+1,width)
                self.swap_rates=tk.StringVar()
                ttk.Label(frame,textvariable=self.swap_rates,wraplength=int(width*.4)).grid(
                    row=row+2,column=0,columnspan=3,sticky="w",pady=(12,0))
            self.equation_images[group]=tk.PhotoImage(
                master=root,data=base64.b64encode(section_equations_png(
                    group,dpi=root.winfo_fpixels("1i"),max_width=int(width*.36))).decode("ascii"))
            equation_label=ttk.Label(equations,image=self.equation_images[group])
            equation_label.grid(row=0,column=0,sticky="w",pady=(0,8))
            note=ttk.Label(equations,text=SECTION_EQUATIONS[group][1],
                           wraplength=int(width*.35))
            note.grid(row=1,column=0,sticky="ew")
            if group == "Coupling parameters":
                self.coupling_equation_label=equation_label
                self.coupling_note=note
            equations.bind("<Configure>",lambda event,g=group,label=equation_label,note=note:
                           self.resize_reference(g,label,note,event.width))
        self.build_assistant_tab()
        controls=ttk.Frame(outer)
        controls.grid(row=3,column=0,sticky="ew")
        ttk.Label(controls,text="Output:").pack(side="left")
        self.mode=tk.StringVar(value="Both transfer directions")
        self.output_selector=ttk.Combobox(controls,textvariable=self.mode,values=[m[0] for m in MODES],
                                         state="readonly",width=56)
        self.output_selector.pack(side="left",padx=8)
        self.save=tk.BooleanVar(value=False)
        ttk.Checkbutton(controls,text="Also save plots/data",variable=self.save).pack(side="left")
        tools=ttk.Frame(outer)
        tools.grid(row=4,column=0,sticky="ew",pady=8)
        ttk.Button(tools,text="Reset defaults",command=self.reset).pack(side="left")
        ttk.Button(tools,text="Load settings",command=self.load).pack(side="left",padx=6)
        ttk.Button(tools,text="Save settings",command=self.save_settings).pack(side="left")
        ttk.Button(tools,text="Check parameters",command=self.check).pack(side="left",padx=6)
        self.run_button=ttk.Button(tools,text="Run simulation",command=self.run)
        self.run_button.pack(side="right")
        self.stop_button=ttk.Button(tools,text="Stop calculation",command=self.stop,state="disabled")
        self.stop_button.pack(side="right",padx=6)
        self.status=tk.StringVar(value="Ready. Default parameters are illustrative, not an experimentally validated design.")
        status_label=ttk.Label(outer,textvariable=self.status,wraplength=940)
        status_label.grid(row=5,column=0,sticky="ew",pady=4)
        outer.bind("<Configure>",lambda event: status_label.configure(wraplength=max(200,event.width-28)))
        progress_row=ttk.Frame(outer)
        progress_row.grid(row=6,column=0,sticky="ew",pady=(3,6))
        self.progress=tk.DoubleVar(value=0)
        self.progress_label=tk.StringVar(value="0%")
        self.progress_bar=ttk.Progressbar(progress_row,mode="determinate",maximum=100,variable=self.progress)
        self.progress_bar.pack(side="left",fill="x",expand=True)
        ttk.Label(progress_row,textvariable=self.progress_label,width=6,anchor="e").pack(side="right",padx=(8,0))
        from pulse_preview_panel import PulsePreview
        self.pulse_preview=PulsePreview(
            self.pulse_preview_frame,self.root,self.config,
            lambda: (self.tabs.select()==str(self.parameter_tabs["Pulses and input state"])
                     and self.pulse_preview_tabs.select()==str(self.pulse_preview_frame)))
        self.tabs.bind("<<NotebookTabChanged>>",self.pulse_preview.visibility_changed,add="+")
        self.pulse_preview_tabs.bind("<<NotebookTabChanged>>",self.pulse_preview.visibility_changed,add="+")
        self.mode.trace_add("write",lambda *_: self.pulse_preview.sync_output(self.mode.get()))
        for value in self.entries.values():
            value.trace_add("write",self.update_swap_rates)
            value.trace_add("write",self.pulse_preview.schedule)
        for key in ("CONSTANT_FLUX", "CONSTANT_FIELD"):
            self.entries[key].trace_add("write",self.update_coupling_mode)
        self.mode.trace_add("write",self.update_swap_rates)
        self.update_coupling_mode()
        self.update_sweep_state()
        self.update_swap_rates()
        self.root.protocol("WM_DELETE_WINDOW",self.close)
        self.root.after(100,self.poll)

    def resize_reference(self,group,label,note,width):
        note.configure(wraplength=max(160,width-16))
        if group == "Coupling parameters" and self.entries.get("DIRECT_COUPLING"):
            group=("Direct coupling" if self.entries["DIRECT_COUPLING"].get() in ("1","1.0")
                   else "Coupling parameters")
        target=max(160,(width-16)//16*16)
        if getattr(label,"_reference_size",None)==(group,target):
            return
        label._reference_size=(group,target)
        picture=tk.PhotoImage(master=self.root,data=base64.b64encode(section_equations_png(
            group,dpi=self.root.winfo_fpixels("1i"),max_width=target)).decode("ascii"))
        label.configure(image=picture)
        label._reference_image=picture

    def build_direct_coupling_controls(self,parent,row,width):
        box=ttk.Frame(parent)
        box.grid(row=row,column=0,columnspan=3,sticky="ew",pady=(14,0))
        self.entries["DIRECT_COUPLING"]=tk.StringVar(value=str(DEFAULTS["DIRECT_COUPLING"]))
        self.direct_coupling_button=ttk.Checkbutton(box,text="Direct information for coupling",
            variable=self.entries["DIRECT_COUPLING"],onvalue="1",offvalue="0")
        self.direct_coupling_button.grid(row=0,column=0,sticky="w")
        self.direct_coupling_frame=ttk.Frame(box,padding=(0,10,0,0))
        self.direct_coupling_frame.grid(row=1,column=0,sticky="ew")
        self.direct_coupling_frame.columnconfigure(1,weight=1)
        for index,(key,label) in enumerate((("G_TC_MHZ","Transmon–cavity g_TC / (2π)"),
                                            ("G_AC_MHZ","Atom–cavity g_AC / (2π)"))):
            self.entries[key]=tk.StringVar(value=str(DEFAULTS[key]))
            ttk.Label(self.direct_coupling_frame,text=label).grid(row=index,column=0,sticky="w",padx=(0,12),pady=5)
            entry=ttk.Entry(self.direct_coupling_frame,textvariable=self.entries[key],width=16)
            entry.grid(row=index,column=1,sticky="ew",pady=5)
            ttk.Label(self.direct_coupling_frame,text="MHz").grid(row=index,column=2,padx=8)
            self.input_widgets[key]=entry
        self.direct_coupling_frame.grid_remove()
        self.entries["DIRECT_COUPLING"].trace_add("write",self.update_coupling_mode)
        # Preserve legacy CLI-only settings when loading/saving existing files.
        for _,key,_,_,default,_ in (f for f in FIELDS if f[0]=="Simultaneous resonance"):
            self.entries[key]=tk.StringVar(value=str(default))


    def update_coupling_mode(self,*_):
        if getattr(self,"_updating_assistant_values",False):
            return
        manual=self.entries["DIRECT_COUPLING"].get() in ("1","1.0")
        if manual:
            self.direct_coupling_frame.grid()
        else:
            self.direct_coupling_frame.grid_remove()
        for key in ("BETA","V_ZPF_UV","D_01_EA0","E_ZPF_V_PER_M","DIPOLE_ALIGNMENT"):
            self.input_widgets[key].configure(state="disabled" if manual else "normal")
        constant_flux=self.entries['CONSTANT_FLUX'].get() in ('1','1.0')
        constant_field=self.entries['CONSTANT_FIELD'].get() in ('1','1.0')
        for key in ('PHI_PARK', 'E_PARK'):
            constant=constant_flux if key=='PHI_PARK' else constant_field
            self.input_widgets[key].configure(state='disabled' if constant else 'normal')
        group="Direct coupling" if manual else "Coupling parameters"
        self.equation_images["Coupling parameters"]=tk.PhotoImage(master=self.root,
            data=base64.b64encode(section_equations_png(group,dpi=self.root.winfo_fpixels("1i"),
                 max_width=int(self.root.winfo_screenwidth()*.42))).decode("ascii"))
        self.coupling_note.configure(text=SECTION_EQUATIONS[group][1])
        self.resize_reference(group,self.coupling_equation_label,self.coupling_note,
                              self.coupling_note.master.winfo_width())

    def build_sweep_controls(self,parent,row,width):
        optional=ttk.LabelFrame(parent,text="Optional",padding=12)
        optional.grid(row=row,column=0,columnspan=3,sticky="ew",pady=(22,0))
        optional.columnconfigure(1,weight=1)
        ttk.Label(optional,text="Parameter sweep").grid(row=0,column=0,sticky="w",padx=(0,16))
        self.sweep_enabled=tk.StringVar(value="Off")
        ttk.Combobox(optional,textvariable=self.sweep_enabled,values=("Off","On"),
                     state="readonly",width=9).grid(row=0,column=1,sticky="w")
        self.sweep_controls=ttk.Frame(optional)
        self.sweep_controls.grid(row=1,column=0,columnspan=2,sticky="ew",pady=(12,0))
        self.sweep_controls.columnconfigure(1,weight=1)
        self.sweep_labels={f"{key} ({unit})":key for _,key,_,unit,_,_ in SWEEP_FIELDS}
        default=next(label for label,key in self.sweep_labels.items() if key=="PHI_SWAP")
        self.sweep_parameter=tk.StringVar(value=default)
        self.sweep_start=tk.StringVar(value="0.26")
        self.sweep_stop=tk.StringVar(value="0.27")
        self.sweep_step=tk.StringVar(value="0.001")
        ttk.Label(self.sweep_controls,text="Parameter").grid(row=0,column=0,sticky="w",padx=(0,16),pady=5)
        ttk.Combobox(self.sweep_controls,textvariable=self.sweep_parameter,values=list(self.sweep_labels),
                     state="readonly",width=28).grid(row=0,column=1,sticky="ew")
        for index,(label,value) in enumerate((("Start",self.sweep_start),("Stop",self.sweep_stop),
                                             ("Step",self.sweep_step)),start=1):
            ttk.Label(self.sweep_controls,text=label).grid(row=index,column=0,sticky="w",pady=5)
            ttk.Entry(self.sweep_controls,textvariable=value,width=16).grid(row=index,column=1,sticky="ew")
        self.sweep_preview=tk.StringVar()
        ttk.Label(optional,textvariable=self.sweep_preview,wraplength=int(width*.36)).grid(
            row=2,column=0,columnspan=2,sticky="w",pady=(10,0))
        ttk.Label(self.sweep_controls,text=("Uses the selected Output and Run simulation button. "
            "The figure slider steps through results, including configured tolerances. "
            "Negative steps sweep downward; up to 501 points."),
            wraplength=int(width*.36),justify="left").grid(row=4,column=0,columnspan=2,sticky="w",pady=(8,0))
        self.sweep_controls.grid_remove()
        for value in (self.sweep_parameter,self.sweep_start,self.sweep_stop,self.sweep_step):
            value.trace_add("write",self.update_sweep_preview)
        self.sweep_enabled.trace_add("write",self.update_sweep_state)

    def update_sweep_state(self,*_):
        enabled=self.sweep_enabled.get()=="On"
        if enabled:
            self.sweep_controls.grid()
        else:
            self.sweep_controls.grid_remove()
        modes=MODES[:3] if enabled else MODES
        self.output_selector.configure(values=[name for name,_ in modes])
        if self.mode.get() not in dict(modes):
            self.mode.set("Both transfer directions")
        self.update_sweep_preview()

    def simulation_mode_args(self):
        if self.sweep_enabled.get()!="On":
            return list(dict(MODES)[self.mode.get()])
        directions={"Both transfer directions":"both", "Transmon to atom":"write", "Atom to transmon":"read"}
        if self.mode.get() not in directions:
            raise ValueError("Choose a transfer output for sweeping, or turn Parameter sweep Off in Numerics.")
        return ["--parameter-sweep","--sweep-parameter",self.sweep_labels[self.sweep_parameter.get()],
                "--sweep-start",self.sweep_start.get(),"--sweep-stop",self.sweep_stop.get(),
                "--sweep-step",self.sweep_step.get(),"--sweep-direction",directions[self.mode.get()]]

    def sweep_points(self):
        return sweep_configs(self.config(),self.sweep_labels[self.sweep_parameter.get()],
                             self.sweep_start.get(),self.sweep_stop.get(),self.sweep_step.get())

    def update_sweep_preview(self,*_):
        if self.sweep_enabled.get()!="On":
            self.sweep_preview.set("Sweep is off. Turn on to simulate a parameter range.")
            return
        try:
            points=self.sweep_points()
            key=self.sweep_labels[self.sweep_parameter.get()]
            self.sweep_preview.set(f"{len(points)} simulations: {key} from {points[0][key]:.10g} "
                                   f"to {points[-1][key]:.10g}. "
                                   f"{3**sum(points[0][name]>0 for name in ('FC_TOL','FT_TOL','FA_TOL','FLUX_TOL','EFIELD_TOL'))} "
                                   "tolerance scenarios per point per direction. Each point may take some time.")
        except ValueError as exc:
            self.sweep_preview.set(str(exc))

    def build_assistant_tab(self):
        tab=ttk.Frame(self.tabs,padding=12)
        self.tabs.add(tab,text="Assistant")
        tab.columnconfigure(0,weight=1)
        tab.rowconfigure(3,weight=1)
        header=ttk.Frame(tab)
        header.grid(row=0,column=0,sticky="ew",pady=(0,6))
        ttk.Label(header,text="Simulation assistant",font=("",14,"bold")).pack(side="left")
        ttk.Button(header,text="Details",command=self.show_assistant_details).pack(side="right")
        self.clear_assistant_button=ttk.Button(header,text="Clear chat",command=self.clear_assistant)
        self.clear_assistant_button.pack(side="right",padx=8)
        self.check_codex_button=ttk.Button(header,text="Check Codex",command=self.check_codex_connection)
        self.check_codex_button.pack(side="right",padx=8)
        ttk.Label(tab,text="Review values before applying · ChatGPT sign-in · Settings and relevant files are sent to OpenAI.",
                  wraplength=1100).grid(row=1,column=0,sticky="w",pady=(0,6))
        status=ttk.Frame(tab)
        status.grid(row=2,column=0,sticky="ew",pady=(0,6))
        self.assistant_status=tk.StringVar(value="Ready for a question.")
        ttk.Label(status,textvariable=self.assistant_status,wraplength=1100).pack(anchor="w")
        self.numerical_status=tk.StringVar(value="Run a simulation to include numerical results.")
        ttk.Label(status,textvariable=self.numerical_status,wraplength=1100).pack(anchor="w")
        self.assistant_output=ScrolledText(tab,wrap="word",height=24,state="disabled")
        self.assistant_output.grid(row=3,column=0,sticky="nsew")
        ttk.Label(tab,text="Your question (up to 1,000 characters):").grid(row=4,column=0,sticky="w",pady=(8,3))
        self.assistant_question=tk.Text(tab,height=3,wrap="word")
        self.assistant_question.grid(row=5,column=0,sticky="ew")
        self.assistant_question.insert("1.0","Explain my current detunings and pulse durations. What should I check before running?")
        buttons=ttk.Frame(tab)
        buttons.grid(row=6,column=0,sticky="ew",pady=(8,0))
        self.ask_button=ttk.Button(buttons,text="Ask assistant",command=self.ask_assistant)
        self.ask_button.pack(side="left")
        self.cancel_answer_button=ttk.Button(buttons,text="Cancel answer",command=self.cancel_answer,state="disabled")
        self.cancel_answer_button.pack(side="left",padx=8)
        self.vary_cavity=tk.BooleanVar(value=True)
        ttk.Checkbutton(buttons,text="Vary cavity frequency",variable=self.vary_cavity).pack(side="left",padx=8)
        self.test_suggestions=tk.BooleanVar(value=True)
        ttk.Checkbutton(buttons,text="Test trial values",variable=self.test_suggestions).pack(side="left",padx=8)
        self.apply_values_button=ttk.Button(buttons,text="Apply suggested values",
                                           command=self.preview_assistant_values,state="disabled")
        self.apply_values_button.pack(side="right")

    def show_assistant_details(self):
        text=("Questions include current settings and the latest run's numerical results. "
              "The last six exchanges are retained for follow-ups. Clear chat cancels pending assistant work, removes history and suggestions, and excludes old run data until a new simulation is started. Current panel values and saved files are preserved. Suggested values update the panel only after you preview and apply them. Write a fidelity target in Your question, for example: Find settings reaching at least 80% fidelity. Enabled constant-field and constant-flux controls remain exactly fixed across assistant trials. Explicit fidelity targets start a bounded local multi-parameter search. Test trial values compares individual proposals for other questions. Exact bare-resonance references need only formula calculations. The main simulation is started separately.\n\n"
              "Codex uses ChatGPT sign-in and your plan's usage limits. Settings, output and relevant project files "
              "are sent to OpenAI. Project access is read-only; API-key billing is disabled.")
        if self.last_run and self.last_run.get("output_directory"):
            if getattr(self,"assistant_context_run",self.last_run) is None:
                text+="\n\nPrevious run data is not attached to assistant questions."
            text+="\n\nNumerical report folder:\n"+self.last_run["output_directory"]
        messagebox.showinfo("Assistant details",text,parent=self.root)

    def clear_assistant(self):
        """Start a fresh assistant session without resetting the simulation panel."""
        self.assistant_cancel.set()
        self.assistant_request+=1
        self.assistant_history=[]
        self.assistant_pending=None
        self.assistant_proposal=None
        self.assistant_context_run=None
        if self.assistant_preview is not None:
            self.assistant_preview.destroy()
            self.assistant_preview=None
        self.assistant_output.configure(state="normal")
        self.assistant_output.delete("1.0","end")
        self.assistant_output.configure(state="disabled")
        self.assistant_question.delete("1.0","end")
        self.ask_button.configure(state="normal")
        self.cancel_answer_button.configure(state="disabled")
        self.apply_values_button.configure(state="disabled")
        self.assistant_status.set("Chat cleared. Using current panel settings.")
        self.numerical_status.set("Previous run data excluded. Start a new simulation to attach fresh results.")

    def assistant_append(self,text):
        self.assistant_output.configure(state="normal")
        self.assistant_output.insert("end",text)
        self.assistant_output.see("end")
        self.assistant_output.configure(state="disabled")

    def check_codex_connection(self):
        self.check_codex_button.configure(state="disabled")
        self.assistant_status.set("Checking Codex sign-in...")
        def worker():
            try:
                check_codex()
                self.events.put(("codex_ready","Codex is ready with ChatGPT sign-in."))
            except Exception as exc:
                self.events.put(("codex_error",str(exc)))
        threading.Thread(target=worker,daemon=True).start()

    def ask_assistant(self):
        if self.ask_button.instate(["disabled"]):
            return
        question=self.assistant_question.get("1.0","end").strip()
        if not question or len(question)>1000:
            self.assistant_status.set("Enter a question between 1 and 1,000 characters.")
            return
        try:
            target=requested_target(question)
            settings=self.config()
            if target is not None and constant_field_requested(question) and not settings['CONSTANT_FIELD']:
                self.assistant_status.set("Enable Hold electric field constant at entered value in the atom tab, then search again. Your field settings were not changed.")
                return
            mode=self.mode.get()
            last_run=deepcopy(getattr(self,"assistant_context_run",self.last_run))
            history=deepcopy(self.assistant_history)
            test_values=self.test_suggestions.get()
            vary_cavity=self.vary_cavity.get() if hasattr(self,"vary_cavity") else False
            can_test=self.process is None or self.process.poll() is not None
        except ValueError as exc:
            messagebox.showerror("Check parameters",str(exc))
            return
        if target is not None and not can_test:
            self.assistant_status.set("Finish or stop the active calculation before starting a target search.")
            return
        self.assistant_proposal=None
        self.apply_values_button.configure(state="disabled")
        if self.assistant_preview is not None:
            self.assistant_preview.destroy()
            self.assistant_preview=None
        self.assistant_pending={"question":question,"settings":settings,"mode":mode,"vary_cavity":vary_cavity,"fidelity_target":target}
        self.assistant_request+=1
        request_id=self.assistant_request
        self.assistant_cancel=threading.Event()
        cancelled=self.assistant_cancel
        project_root=self.backend_path.parent.parent
        self.ask_button.configure(state="disabled")
        self.cancel_answer_button.configure(state="normal")
        self.assistant_status.set("Starting multi-parameter search..." if target is not None else "Asking Codex...")
        label="numerical search" if target is not None else "Codex"
        self.assistant_append(f"\nYou: {question}\nAssistant ({label}):\n")
        def worker():
            try:
                if target is not None:
                    result=run_target_search(settings,mode,target,cancelled,project_root,vary_cavity=vary_cavity,
                        on_progress=lambda message:self.events.put(("assistant_progress",(request_id,message))))
                    self.events.put(("assistant_search_result",(request_id,result)))
                    self.events.put(("assistant_text",(request_id,search_answer(result,settings,mode,target))))
                    self.events.put(("assistant_done",(request_id,"Target search finished.")))
                    return
                messages=build_messages(question,settings,mode,last_run,history,vary_cavity=vary_cavity)
                response=codex_answer(messages,cancelled,project_root)
                for text in response:
                    if cancelled.is_set():
                        return
                    display,proposal,error=parse_proposal(text,settings,mode)
                    if test_values and proposal and not reference_only(proposal):
                        if not can_test:
                            text=display+"\n\nNumerical testing is unavailable while the main calculation is active. Finish or stop it, then ask again. No tested proposal is available."
                        else:
                            comparison=run_comparison(proposal,cancelled,project_root,vary_cavity=vary_cavity,
                                on_progress=lambda message:self.events.put(("assistant_progress",(request_id,message))))
                            self.events.put(("assistant_calculation",(request_id,comparison)))
                            text=checked_answer(proposal,comparison)
                    self.events.put(("assistant_text",(request_id,text)))
                self.events.put(("assistant_done",(request_id,"Answer finished.")))
            except ComparisonCancelled:
                return
            except Exception as exc:
                if not cancelled.is_set():
                    self.events.put(("assistant_error",(request_id,f"Assistant request failed: {exc}")))
        threading.Thread(target=worker,daemon=True).start()

    def receive_assistant_answer(self,text):
        if self.assistant_pending is None:
            self.assistant_append(text)
            return
        captured=self.assistant_pending
        display,proposal,error=parse_proposal(text,captured["settings"],captured["mode"])
        if proposal and 'FC' in proposal['changes'] and not captured.get('vary_cavity',False):
            proposal=None
            error="Cavity frequency is fixed for this request. Enable Vary cavity frequency and ask again."
        if proposal and captured.get('fidelity_target') is not None:
            try:
                evidence = captured.get('target_search')
                if evidence is None:
                    raise ValueError('No numerical target-search evidence is available.')
                verified = validate_search_result(evidence, captured['settings'], captured['mode'],
                                                captured['fidelity_target'], captured.get('vary_cavity', False))
                if not verified['target_reached']:
                    raise ValueError('The numerical search did not reach the requested fidelity.')
                if proposal['changes'] != verified['changes'] or proposal['evidence'] != 'measured simulation result':
                    raise ValueError('Suggested values do not match the validated target-search result.')
            except (ValueError, TypeError, KeyError) as exc:
                proposal = None
                error = str(exc)
        self.assistant_proposal=proposal
        self.apply_values_button.configure(state="normal" if proposal else "disabled")
        self.assistant_history.append(dict(captured,answer=display,
                                          suggested_values=proposal["changes"] if proposal else None))
        self.assistant_history=self.assistant_history[-6:]
        self.assistant_pending=None
        self.assistant_append(display)
        if proposal:
            self.assistant_append("\n[Suggested values are available for preview.]")
        elif error:
            self.assistant_append("\n[Suggested values rejected: "+error+"]")

    def checked_assistant_values(self,proposal):
        if proposal is not self.assistant_proposal:
            raise ValueError("This proposal has been replaced. Ask for a fresh proposal.")
        if self.process is not None and self.process.poll() is None:
            raise ValueError("Wait for the active calculation to finish or stop before applying values.")
        if 'FC' in proposal['changes'] and not (hasattr(self,'vary_cavity') and self.vary_cavity.get()):
            raise ValueError("Cavity-frequency changes are disabled. Enable Vary cavity frequency and ask again.")
        return prepare_application(proposal,self.config(),self.mode.get())

    def preview_assistant_values(self):
        proposal=self.assistant_proposal
        if proposal is None:
            return
        try:
            changes=self.checked_assistant_values(proposal)
        except ValueError as exc:
            self.assistant_status.set(str(exc))
            return
        if self.assistant_preview is not None:
            self.assistant_preview.destroy()
        preview=tk.Toplevel(self.root)
        self.assistant_preview=preview
        preview.title("Review suggested values")
        preview.transient(self.root)
        frame=ttk.Frame(preview,padding=12)
        frame.pack(fill="both",expand=True)
        ttk.Label(frame,text="Assistant evidence: "+proposal['evidence'],font=("",12,"bold")).pack(anchor="w")
        ttk.Label(frame,text=proposal['reason'],wraplength=700).pack(anchor="w",pady=(4,10))
        table_frame=ttk.Frame(frame)
        table_frame.pack(fill="both",expand=True)
        table=ttk.Treeview(table_frame,columns=("parameter","current","proposed","units"),
                           show="headings",height=min(10,len(changes)))
        for column,title,width in (("parameter","Parameter",200),("current","Current",150),
                                   ("proposed","Proposed",150),("units","Units",140)):
            table.heading(column,text=title)
            table.column(column,width=width)
        units={key:unit for _,key,_,unit,_,_ in FIELDS}
        for key,value in changes.items():
            table.insert("","end",values=(key,proposal['base_settings'][key],value,units[key]))
        table.pack(side="left",fill="both",expand=True)
        scrollbar=ttk.Scrollbar(table_frame,orient="vertical",command=table.yview)
        scrollbar.pack(side="right",fill="y")
        table.configure(yscrollcommand=scrollbar.set)
        ttk.Label(frame,text="Applying changes updates the inputs. Start the simulation separately.",
                  wraplength=700).pack(anchor="w",pady=10)
        buttons=ttk.Frame(frame)
        buttons.pack(fill="x")
        def close():
            preview.destroy()
            if self.assistant_preview is preview:
                self.assistant_preview=None
        def apply():
            try:
                self.apply_assistant_values(proposal)
            except ValueError as exc:
                messagebox.showerror("Cannot apply values",str(exc),parent=preview)
                return
            close()
        ttk.Button(buttons,text="Apply values",command=apply).pack(side="right")
        ttk.Button(buttons,text="Cancel",command=close).pack(side="right",padx=8)
        preview.protocol("WM_DELETE_WINDOW",close)

    def apply_assistant_values(self,proposal):
        changes=self.checked_assistant_values(proposal)
        previous={key:self.entries[key].get() for key in changes}
        self._updating_assistant_values=True
        try:
            for key,value in changes.items():
                self.entries[key].set("auto" if key in PULSE_DELAY_KEYS and value is None else str(value))
        except Exception as exc:
            for key,value in previous.items():
                self.entries[key].set(value)
            raise ValueError("Could not update panel values: "+str(exc)) from exc
        finally:
            self._updating_assistant_values=False
        self.update_coupling_mode()
        self.update_swap_rates()
        self.assistant_proposal=None
        self.apply_values_button.configure(state="disabled")
        self.assistant_status.set("Suggested values applied. Run a simulation to evaluate them.")
        self.status.set("Assistant values applied. Ready to run.")
        self.assistant_append("\n[Applied to panel: "+", ".join(f"{key}={value}" for key,value in changes.items())+"]\n")

    def cancel_answer(self):
        self.assistant_pending=None
        self.assistant_cancel.set()
        self.assistant_request+=1  # Ignore queued tokens from the cancelled worker.
        self.ask_button.configure(state="normal")
        self.cancel_answer_button.configure(state="disabled")
        self.assistant_status.set("Answer cancelled.")
        self.assistant_append("\n[Answer cancelled]\n")

    def update_swap_rates(self,*_):
        if getattr(self,"_updating_assistant_values",False):
            return
        self.update_sweep_preview()
        try:
            config=self.config()
            d=derived(config)
        except ValueError:
            self.transmon_tuning.set("Enter valid model parameters to calculate detuning.")
            self.swap_rates.set("Calculated swap rates: enter valid model parameters.")
            return
        detuning=d['transmon_swap_detuning_MHz']
        condition="resonant" if abs(detuning)<1e-8 else "detuned"
        self.transmon_tuning.set(
            f"Transmon {d['transmon_interaction_frequency_MHz']:.6g} MHz; cavity {config['FC']:.6g} MHz.\n"
            f"Interaction detuning (f_T - f_c): {detuning:+.6g} MHz ({condition}).\n"
            f"Parked transmon {d['transmon_park_frequency_MHz']:.6g} MHz; "
            f"detuning: {d['transmon_park_frequency_MHz']-config['FC']:+.6g} MHz.\n"
            "Tune the interaction flux to set the transmon resonance frequency.")
        if config['CONSTANT_FLUX']:
            self.transmon_tuning.set(
                f"Constant entered flux: {d['swap_flux']:.6g}.\n"
                f"Transmon frequency: {d['transmon_interaction_frequency_MHz']:.6g} MHz; detuning: {detuning:+.6g} MHz.\n"
                "Parked flux is unused; the entered pulse-peak flux stays constant.")
        self.swap_rates.set(
            f"{'Manual' if config['DIRECT_COUPLING'] else 'Calculated'} couplings g/(2π): TC {d['transmon_coupling_MHz']:.5g}; AC {d['atom_coupling_MHz']:.5g} MHz.\n"
            f"Nominal resonant swap frequencies: TC {d['transmon_swap_frequency_MHz']:.5g} MHz; "
            f"AC {d['atom_swap_frequency_MHz']:.5g} MHz.\n"
            f"Rounded pulse durations: TC {d['transmon_pulse_duration_us']:.5g} us; "
            f"AC {d['atom_pulse_duration_us']:.5g} us.\n"
            f"Voltage minus flux peak delay: write {d['write_pulse_delay_ns']:+.5g} ns; "
            f"read {d['read_pulse_delay_ns']:+.5g} ns.")

    def config(self):
        return validate({key:value.get() for key,value in self.entries.items()})

    def append(self,text):
        if self.last_run is not None:
            self.last_run["output"]=(self.last_run["output"]+text)[-3000:]

    def reset(self):
        self.sweep_enabled.set("Off")
        self.mode.set("Both transfer directions")
        for key,value in DEFAULTS.items():
            self.entries[key].set("auto" if key in PULSE_DELAY_KEYS and value is None else str(value))
        self.status.set("Default model parameters restored.")

    def load(self):
        path=filedialog.askopenfilename(filetypes=[("Settings JSON","*.json")])
        if not path:
            return
        try:
            config=validate(json.loads(Path(path).read_text()))
            for key,value in config.items():
                self.entries[key].set("auto" if key in PULSE_DELAY_KEYS and value is None else str(value))
            self.status.set("Settings loaded.")
        except Exception as exc:
            messagebox.showerror("Invalid settings",str(exc))

    def save_settings(self):
        try:
            config=self.config()
        except ValueError as exc:
            messagebox.showerror("Check parameters",str(exc))
            return
        path=filedialog.asksaveasfilename(defaultextension=".json",filetypes=[("Settings JSON","*.json")])
        if path:
            Path(path).write_text(json.dumps(config,indent=2))
            self.status.set("Settings saved.")

    def check(self):
        try:
            config=self.config()
            d=derived(config)
            if "--parameter-sweep" in self.simulation_mode_args():
                points=self.sweep_points()
                self.status.set(f"Sweep checks passed: {len(points)} points.")
                return True
            self.status.set(
                f"Model checks passed. Swap detuning TC/AC: "
                f"{d['transmon_swap_detuning_MHz']:.5g}/{d['atom_swap_detuning_MHz']:.5g} MHz; "
                f"optical pulses h0/r0: {d['optical_pi_pulse_us']:.5g} us, h1/r1: {d['optical_pi_pulse_1_us']:.5g} us; "
                f"nominal atom swap hold: {d['atom_swap_hold_us']:.5g} us; "
                f"cavity Q: {d['cavity_Q']:.4g}.")
            return True
        except ValueError as exc:
            messagebox.showerror("Check parameters",str(exc))
            return False

    def run(self):
        if self.process and self.process.poll() is None:
            return
        if not self.check():
            return
        run_config=self.config()
        with tempfile.NamedTemporaryFile(mode="w",suffix=".json",prefix="hybrid-settings-",delete=False) as f:
            json.dump(run_config,f)
            self.temp_path=Path(f.name)
        mode_args=self.simulation_mode_args()
        reports_root=self.backend_path.parent.parent/"output"/"assistant_runs"
        try:
            reports_root.mkdir(parents=True,exist_ok=True)
            report_directory=tempfile.mkdtemp(prefix="run-",dir=reports_root)
        except OSError as exc:
            self.temp_path.unlink(missing_ok=True)
            messagebox.showerror("Could not prepare simulation output",str(exc))
            return
        self.stop_path=Path(report_directory)/"stop_requested"
        # An explicit --run bypasses the startup panel in the child.
        command=[sys.executable,"-u",str(self.backend_path),
                 "--run","--progress","--stop-file",str(self.stop_path),"--config",str(self.temp_path),"--assistant-output",report_directory,*mode_args]
        if self.save.get():
            command.append("--save")
        try:
            self.process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                          text=True,bufsize=1,start_new_session=(os.name=="posix"))
        except Exception as exc:
            self.temp_path.unlink(missing_ok=True)
            messagebox.showerror("Could not start simulation",str(exc))
            return
        self.close_previous_plots()
        self.progress.set(0)
        self.progress_label.set("0%")
        self.run_button.configure(state="disabled")
        self.last_run={"settings":run_config,"mode":self.mode.get(),"status":"running","output":"",
                       "output_directory":report_directory}
        self.assistant_context_run=self.last_run
        if "--parameter-sweep" in mode_args:
            self.last_run["sweep"]={"parameter":self.sweep_labels[self.sweep_parameter.get()],
                                    "start":self.sweep_start.get(),"stop":self.sweep_stop.get(),
                                    "step":self.sweep_step.get(),"direction":mode_args[mode_args.index("--sweep-direction")+1]}
        self.numerical_status.set("Calculating numerical results...")
        self.stop_button.configure(state="normal")
        self.status.set("Calculating. Starting another run will replace these plots.")
        self.append("\n--- New simulation ---\n")
        threading.Thread(target=self.read_output,args=(self.process,),daemon=True).start()

    def read_output(self,process):
        for line in process.stdout:
            self.events.put(("line",(process,line)))
        self.events.put(("done",(process,process.wait())))

    def close_previous_plots(self):
        """Each previous result window belongs to its completed child process."""
        for process in self.plot_processes:
            if process.poll() is None:
                try:
                    if os.name=="posix":
                        os.killpg(process.pid,signal.SIGTERM)
                    else:
                        process.terminate()
                except ProcessLookupError:
                    pass
        self.plot_processes=[]

    def release_calculation(self,status,message):
        """Release the Run button while the child continues displaying figures."""
        self.plot_processes=[p for p in self.plot_processes if p.poll() is None]
        if self.process is not None:
            self.plot_processes.append(self.process)
            self.process=None
        if self.last_run is not None:
            self.last_run["status"]=status
        self.run_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self.status.set(message)
        if self.temp_path:
            self.temp_path.unlink(missing_ok=True)
            self.temp_path=None

    def handle_progress(self,line):
        event=parse_progress_line(line)
        if event is None:
            return False
        percent=max(self.progress.get(),event["percent"])
        self.progress.set(percent)
        self.progress_label.set(f"{int(percent)}%")
        if self.last_run is not None:
            self.last_run["progress_percent"]=percent
        if percent>=100:
            self.release_calculation("finished","Simulation ready. Starting a new run will close these plots.")
        else:
            self.status.set("Calculating: "+event["label"])
        return True

    def poll(self):
        try:
            while True:
                kind,value=self.events.get_nowait()
                if kind in ("line","done"):
                    process,value=value
                    if process is not self.process:
                        # Plot-only children may exit after a newer run starts.
                        continue
                if kind=="line":
                    if self.handle_progress(value):
                        continue
                    self.append(value)
                    if value.startswith("Calculation stopped."):
                        self.release_calculation("stopped","Calculation stopped. Starting a new run will close these plots.")
                    if self.last_run is not None and getattr(self,"assistant_context_run",self.last_run) is self.last_run:
                        if value.startswith("Full parameter sweep results:"):
                            self.numerical_status.set("Parameter sweep results available.")
                        elif value.startswith("Full tolerance results:"):
                            self.numerical_status.set("Tolerance results available.")
                        elif value.startswith("Assistant numerical report ready:"):
                            self.numerical_status.set("Numerical plot data available.")
                elif kind in ("codex_ready","codex_error"):
                    self.check_codex_button.configure(state="normal")
                    self.assistant_status.set(value)
                elif kind.startswith("assistant_"):
                    request_id,text=value
                    if request_id!=self.assistant_request:
                        continue
                    if kind=="assistant_progress":
                        self.assistant_status.set(text)
                    elif kind=="assistant_calculation":
                        if self.assistant_pending is not None:
                            self.assistant_pending['numerical_comparison']=text
                    elif kind=="assistant_search_result":
                        if self.assistant_pending is not None:
                            self.assistant_pending['target_search']=text
                    elif kind=="assistant_text":
                        self.receive_assistant_answer(text)
                    else:
                        self.assistant_pending=None
                        self.ask_button.configure(state="normal")
                        self.cancel_answer_button.configure(state="disabled")
                        if kind=="assistant_error":
                            self.assistant_append("\n"+text+"\n")
                        else:
                            self.assistant_append("\n")
                        self.assistant_status.set(text)
                elif kind=="done":
                    if value==0:
                        self.progress.set(100)
                        self.progress_label.set("100%")
                    if self.last_run is not None:
                        self.last_run["status"]="finished" if value==0 else "stopped" if value==3 else "failed"
                        folder=Path(self.last_run["output_directory"])
                        available=[name for name in ("write","read","roundtrip","tolerance_scan","parameter_sweep","stopped_trajectory") if (folder/(name+".json")).is_file()]
                        if getattr(self,"assistant_context_run",self.last_run) is self.last_run:
                            self.numerical_status.set("Numerical reports: "+", ".join(available)
                                                      if available else "No numerical reports were produced for this run.")
                    self.status.set("Run finished. Edit parameters and run again." if value==0
                                    else "Calculation stopped. Available figures were shown." if value==3
                                    else "Run failed.")
                    self.run_button.configure(state="normal")
                    self.stop_button.configure(state="disabled")
                    if self.temp_path:
                        self.temp_path.unlink(missing_ok=True)
                        self.temp_path=None
        except queue.Empty:
            pass
        self.root.after(100,self.poll)

    def stop(self):
        if self.process and self.process.poll() is None:
            self.stop_path.touch()
            if self.last_run is not None:
                self.last_run["status"]="stopping"
            self.stop_button.configure(state="disabled")
            self.status.set("Stopping calculation and preparing available figures...")

    def close(self):
        processes=[p for p in [self.process,*self.plot_processes]
                   if p is not None and p.poll() is None]
        if processes:
            if not messagebox.askyesno("Close simulation panel","Close the active simulation and its plot windows?"):
                return
            for process in processes:
                if os.name=="posix":
                    try:
                        os.killpg(process.pid,signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                else:
                    process.terminate()
        self.assistant_cancel.set()
        if self.temp_path:
            self.temp_path.unlink(missing_ok=True)
        if getattr(self,"pulse_preview",None) is not None:
            self.pulse_preview.close()
        self.root.destroy()


def launch(backend_path=None):
    root=tk.Tk()
    ParameterPanel(root, backend_path=backend_path)
    root.mainloop()


if __name__=="__main__":
    launch()
