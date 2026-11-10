"""Native startup parameter panel. Keep beside backend.py and hybrid_settings.py."""
import base64
import io
import json
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
from hybrid_settings import FIELDS, DEFAULTS, validate, derived
from parameter_sweep import SWEEP_FIELDS, sweep_configs
from codex_assistant import answer as codex_answer, build_messages, check_codex

MODES = [
("Parameter sweep", ["--parameter-sweep"]),
("Both transfer directions", []),
("Transmon to atom", ["--write-only"]),
("Atom to transmon", ["--retrieve"]),
("Simultaneous cavity resonance: transmon to atom", ["--simultaneous"]),
("Tolerance scan: best and worst sampled", ["--tolerance-scan"]),
]


# Symbols use the same notation as the equations beneath each tab.
PARAMETER_SYMBOLS = {
    "SIM_HOLD_SCALE": r"s_{\mathrm{sim}}",
    "SIM_RAMP": r"T_{\mathrm{ramp,sim}}",
    "FC": r"f_c",
    "PHI_SWAP": r"\phi_{\mathrm{swap}}",
    "V_SWAP": r"V_{\mathrm{swap}}",
    "EC": r"\epsilon_C=E_C/h",
    "EJ_MAX": r"\epsilon_{J,\max}=E_{J,\max}/h",
    "ASYMMETRY": r"d",
    "PHI_PARK": r"\phi_{\mathrm{park}}",
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
    "OMEGA": r"\Omega_{\max}/(2\pi)",
    "RAMP": r"T_{\mathrm{ramp}}",
    "TC_PULSE_SCALE": r"s_{TC}",
    "AC_PULSE_SCALE": r"s_{AC}",
    "GAP": r"T_{\mathrm{gap}}",
    "STORAGE": r"T_{\mathrm{storage}}",
    "INPUT_THETA": r"\vartheta",
    "INPUT_PHI": r"\varphi",
    "MAX_STEP": r"\Delta t_{\max}",
    "NC": r"N_c",
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
    "OMEGA": "Peak effective optical Rabi frequency",
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
            r"$f_T(\phi_{\mathrm{res}})=f_A(E_{\mathrm{res}})=f_c$",
            r"$G=\sqrt{g_{TC}^2+g_{AC}^2},\qquad T_{\mathrm{hold}}=s_{\mathrm{sim}}\pi/G$",
            r"$P_{\max}=4g_{TC}^2g_{AC}^2/(g_{TC}^2+g_{AC}^2)^2$",
        ),
        "Select Simultaneous cavity resonance in Output. Both controls use sin-squared ramps "
        "onto a shared flat resonant hold, then return to park before optical mapping. "
        "Resonant flux and voltage are calculated from the cavity frequency; PHI_SWAP, V_SWAP "
        "and the sequential pulse multipliers apply only to the sequential modes. "
        "Coupling is evaluated at the resonant transmon flux. The hold multiplier starts at "
        "the first ideal transfer maximum; ramps, losses and spectator interactions can shift it. "
        "Unequal couplings limit ideal transfer; unity requires equal coupling magnitudes. "
        "This output writes to atomic memory and includes the configured storage hold. "
        "Tolerance bounds are not scanned in this mode; the output explicitly reports nominal results.",
    ),
    "Cavity and transmon": (
        (
            r"$\epsilon_C=E_C/h,\quad \epsilon_J(\phi)=\epsilon_{J,\max}\sqrt{\cos^2(\pi\phi)+d^2\sin^2(\pi\phi)},\quad \phi=\Phi/\Phi_0$",
            r"$f_T(\phi)\approx\sqrt{8\epsilon_C\epsilon_J(\phi)}-\epsilon_C,\qquad \Delta_T(t)=2\pi[f_T(\phi(t))-f_c]$",
            r"$f_{T,\mathrm{int}}=f_T(\phi_{\mathrm{swap}}),\qquad f_{T,\mathrm{park}}=f_T(\phi_{\mathrm{park}})$",
            r"$\Delta_{T,\mathrm{int}}=2\pi(f_{T,\mathrm{int}}-f_c),\qquad \mathrm{resonance}:\ f_{T,\mathrm{int}}=f_c$",
            r"$\phi(t)=\phi_{\mathrm{park}}+[\phi_{\mathrm{swap}}-\phi_{\mathrm{park}}]w_{TC}(t)$",
            r"$\Delta_{T,\mathrm{park}}=2\pi(f_{T,\mathrm{park}}-f_c),\qquad H_T(t)/\hbar=\Delta_T(t)\,\sigma_+\sigma_-$",
        ),
        "Energies divided by h are entered in MHz. d is SQUID junction asymmetry. "
        "Interaction and parked flux inputs use 0 ≤ flux / flux quantum ≤ 0.5. The applied interaction flux sets the calculated transmon frequency independently of f_c. Resonance requires f_T,int = f_c; coupling sets the nominal swap hold. "
        "w_TC(t) is a sin-squared excursion, zero outside the pulse and one only at its midpoint. The parked flux sets idle detuning to suppress unwanted exchange while coupling remains active.",
    ),
    "Coupling parameters": (
        (
            r"$g_{\mathrm{TC}}\approx\frac{2e\beta V_{\mathrm{zpf}}}{\hbar}\left(\frac{E_J(\Phi_{\mathrm{swap}})}{32E_C}\right)^{1/4}$",
            r"$g_{\mathrm{AC}}=\frac{|\mathbf{d}_{01}\cdot\mathbf{E}_{\mathrm{zpf}}|}{\hbar}=\frac{|d_{01}|E_{\mathrm{zpf}}|\cos\theta|}{\hbar}$",
            r"$f_{\mathrm{swap},TC}=2g_{TC}/(2\pi),\qquad f_{\mathrm{swap},AC}=2g_{AC}/(2\pi),\qquad t_{\mathrm{hold}}=1/(2f_{\mathrm{swap}})$",
            r"$\beta=\frac{C_g}{C_\Sigma},\qquad n_{01}\approx\left(\frac{E_J(\Phi_{\mathrm{swap}})}{32E_C}\right)^{1/4}$",
        ),
        "C_g is the coupling capacitance; "
        "C_Σ is total transmon capacitance. The charge matrix element is calculated at the swap point. "
        "Swap frequencies shown are nominal resonant population oscillation rates; detuning reduces transfer.",
    ),
    "Atom and electrodes": (
        (
            r"$E(t)=\frac{V(t)}{\ell}+E_{\mathrm{stray}},\qquad f_A(E)=f_{A,0}+(K_{r1}-K_{r0})E^2$",
            r"$E_{\mathrm{swap}}=\frac{V_{\mathrm{swap}}}{\ell}+E_{\mathrm{stray}},\qquad f_{A,\mathrm{swap}}=f_{A,0}+(K_{r1}-K_{r0})E_{\mathrm{swap}}^2$",
            r"$V_{\mathrm{park}}=\ell(E_{\mathrm{park}}-E_{\mathrm{stray}}),\qquad \Delta_A(t)=2\pi[f_A(E(t))-f_c]$",
            r"$\Delta_{w,r0}(E)=2\pi[f_{\mathrm{wrong,split}}+(K_w-K_{r0})E^2],\qquad \Omega_w(t)=\eta\,\Omega(t)$",
        ),
        "ℓ is electrode separation in cm; E is in V/cm and K is in MHz/(V/cm)². "
        "η is relative unwanted optical amplitude.",
    ),
    "Loss and coherence": (
        (
            r"$L_T=\frac{\sigma_-}{\sqrt{T_{1,T}}},\qquad L_c=\frac{a}{\sqrt{T_{1,c}}},\qquad L_{rj}=\frac{|\mathrm{loss}\rangle\langle r_j|}{\sqrt{\tau_{rj}}}$",
            r"$L_{\phi,T}=\frac{Z_T}{\sqrt{2T_{\phi,T}}},\qquad L_{\phi,R}=\frac{Z_R}{\sqrt{2T_{\phi,R}}},\qquad L_{\phi,H}=\frac{Z_H}{\sqrt{2T_{\phi,H}}}$",
            r"$Z_T=P_e-P_g,\qquad Z_R=P_{r1}-P_{r0},\qquad Z_H=P_{h1}-P_{h0}$",
            r"$\frac{1}{T_{2,T}}=\frac{1}{2T_{1,T}}+\frac{1}{T_{\phi,T}},\qquad Q_c=2\pi f_cT_{1,c}$",
        ),
        "All lifetimes are in μs; T_phi denotes pure dephasing, not total T2. "
        "Rydberg decay includes r0, r1, and the unwanted level. "
        "The T2 relation shown is for the transmon; cavity T1 is photon-number lifetime.",
    ),
    "Pulses and input state": (
        (
            r"$\Omega(t)=\Omega_{\max}\sin^2\!\left(\frac{\pi(t-t_0)}{T_{\mathrm{opt}}}\right),\quad \int_{t_0}^{t_0+T_{\mathrm{opt}}}\Omega(t)\,dt=\pi,\quad T_{\mathrm{opt}}=\frac{2\pi}{\Omega_{\max}}$",
            r"$t_{\mathrm{ref},j}=\frac{\pi}{2g_j},\qquad T_j=s_j(2T_{\mathrm{ramp}}+t_{\mathrm{ref},j}),\quad j=TC,AC$",
            r"$w(u)=\sin^2(\pi u/T_{\mathrm{swap}}),\quad 0\leq u\leq T_{\mathrm{swap}};\qquad w=0\ \mathrm{outside}$",
            r"$|\psi_T\rangle=\cos(\vartheta/2)|g\rangle+e^{i\varphi}\sin(\vartheta/2)|e\rangle,\qquad t_{\mathrm{end}}=t_{\mathrm{map,end}}+T_{\mathrm{storage}}$",
        ),
        "Times are in μs. Optical envelopes vanish outside "
        "their pulse intervals. Flux and field use a rounded sin-squared pulse with zero endpoint slopes and no flat hold. "
        "GAP separates operations. Input Bloch angles are entered in degrees. "
        "The nominal resonant time is only a reference. Separate TC and AC multipliers stretch each rounded pulse. RAMP contributes to the reference duration before scaling. Finite pulses still have spectral width. Peak controls can cross resonance twice for coherent transfer; changing device inputs requires recalibration.",
    ),
    "Experimental tolerances": (
        (
            r"$\delta x\in\{-\epsilon_x,0,+\epsilon_x\},\qquad x\in\{f_c,f_T,f_A,\phi,E\}$",
            r"$\phi_{\mathrm{actual}}(t)=\phi_{\mathrm{nominal}}(t)+\delta\phi$",
            r"$E_{\mathrm{actual}}(t)=E_{\mathrm{nominal}}(t)+\delta E$",
            r"$f_{T,\mathrm{actual}}(t)=f_T(\phi_{\mathrm{actual}}(t))+\delta f_T$",
            r"$f_{A,\mathrm{actual}}(t)=f_{A,0}+(K_{r1}-K_{r0})E_{\mathrm{actual}}^2(t)+\delta f_A$",
        ),
        "Nonzero tolerances automatically add shaded sampled ranges to Both transfer directions, "
        "Transmon to atom, and Atom to transmon outputs. Each tolerance is a fixed absolute +/- bound, "
        "not a percentage or random noise. The scan tests all combinations of negative bound, "
        "zero and positive bound (3^N runs for N nonzero tolerances; up to 243). "
        "Offsets remain constant through writing, storage and readout. "
        "Pulse timing, couplings, optical carriers and phase correction keep their nominal calibration. "
        "Frequency offsets are additional to flux/Stark shifts; avoid counting the same uncertainty twice. "
        "Summary columns compare nominal values with sampled ranges for all metrics, including logical-input fidelities, coherence and residual populations. "
        "Best and worst are among sampled points, not guaranteed extrema within the bounds. "
        "Lines and 3D basis-output columns are nominal; shaded dynamics and With tolerances summary columns "
        "show sampled error ranges. The separate Tolerance scan output compares final outcomes. "
        "Flux errors can cross a SQUID sweet spot; the periodic flux relation is used without clipping.",
    ),
    "Numerics": (
        (
            r"$\dot\rho=-i[H,\rho]+\sum_k\left(L_k\rho L_k^\dagger-\frac{1}{2}\{L_k^\dagger L_k,\rho\}\right)$",
            r"$i\,\frac{d|\psi\rangle}{dt}=H|\psi\rangle\quad\mathrm{(noiseless)},\qquad \Delta t_{\mathrm{internal}}\leq\Delta t_{\max}=\mathrm{MAX\_STEP}$",
            r"$D=2\,N_c\,6=12N_c,\qquad n_{\mathrm{cavity}}\in\{0,\ldots,N_c-1\},\qquad N_c\geq3$",
        ),
        "",
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


class ParameterPanel:
    def __init__(self, root, backend_path=None):
        self.root=root
        if backend_path is None:
            folder=Path(__file__).resolve().parent
            backend_path=folder/("backend.py" if (folder/"backend.py").exists() else "codetest.py")
        self.backend_path=Path(backend_path).resolve()
        self.process=None
        self.events=queue.Queue()
        self.temp_path=None
        self.last_run=None
        self.assistant_request=0
        self.assistant_cancel=threading.Event()
        root.title("Hybrid quantum memory — simulation parameters")
        width=min(1800,int(root.winfo_screenwidth()*.96))
        height=min(1100,int(root.winfo_screenheight()*.90))
        root.geometry(f"{width}x{height}")
        root.minsize(min(width,1100),min(height,700))
        self.entries={}
        self.equation_images={}
        self.symbol_images={}
        outer=ttk.Frame(root,padding=14)
        outer.pack(fill="both",expand=True)
        ttk.Label(outer,text="Hybrid quantum memory",font=("",18,"bold")).pack(anchor="w")
        ttk.Label(outer,text="Set model parameters, choose output, then run. Plots open in closable windows.").pack(anchor="w",pady=(3,10))
        self.tabs=ttk.Notebook(outer)
        self.tabs.pack(fill="both",expand=True)
        groups=list(dict.fromkeys(row[0] for row in FIELDS))
        for group in groups:
            tab=ttk.Frame(self.tabs)
            self.tabs.add(tab,text=group)
            tab.columnconfigure(0,weight=1)
            tab.columnconfigure(1,weight=1)
            tab.rowconfigure(0,weight=1)
            frame=ttk.Frame(tab,padding=12)
            frame.grid(row=0,column=0,sticky="new")
            frame.columnconfigure(1,weight=1)
            equations=ttk.Frame(tab,padding=(12,16,12,12))
            equations.grid(row=0,column=1,sticky="new")
            equations.columnconfigure(0,weight=1)
            for row,(_,key,label,units,default,rule) in enumerate(f for f in FIELDS if f[0]==group):
                label_cell=ttk.Frame(frame)
                label_cell.grid(row=row,column=0,sticky="w",padx=(0,12),pady=7)
                ttk.Label(label_cell,text=PARAMETER_CAPTIONS.get(key,label)).pack(side="left")
                self.symbol_images[key]=tk.PhotoImage(
                    master=root,data=base64.b64encode(
                        parameter_symbol_png(key,dpi=root.winfo_fpixels("1i"))).decode("ascii"))
                ttk.Label(label_cell,image=self.symbol_images[key]).pack(side="left",padx=(8,0))
                value=tk.StringVar(value=str(default))
                entry=ttk.Entry(frame,textvariable=value,width=18)
                entry.grid(row=row,column=1,sticky="ew",pady=7)
                ttk.Label(frame,text=units).grid(row=row,column=2,sticky="w",padx=10)
                self.entries[key]=value
            if group == "Cavity and transmon":
                self.transmon_tuning=tk.StringVar()
                ttk.Label(frame,textvariable=self.transmon_tuning,wraplength=int(width*.4)).grid(
                    row=row+1,column=0,columnspan=3,sticky="w",pady=(12,0))
            if group == "Coupling parameters":
                self.swap_rates=tk.StringVar()
                ttk.Label(frame,textvariable=self.swap_rates,wraplength=int(width*.4)).grid(
                    row=row+1,column=0,columnspan=3,sticky="w",pady=(12,0))
            self.equation_images[group]=tk.PhotoImage(
                master=root,data=base64.b64encode(section_equations_png(
                    group,dpi=root.winfo_fpixels("1i"),max_width=int(width*.45))).decode("ascii"))
            ttk.Label(equations,image=self.equation_images[group]).grid(
                row=0,column=0,sticky="w",pady=(0,12))
            note=ttk.Label(equations,text=SECTION_EQUATIONS[group][1],
                           wraplength=int(width*.43))
            note.grid(row=1,column=0,sticky="ew")
            equations.bind("<Configure>",lambda event,label=note:
                           label.configure(wraplength=max(200,event.width-24)))
        self.build_sweep_tab()
        self.build_assistant_tab()
        controls=ttk.Frame(outer)
        controls.pack(fill="x")
        ttk.Label(controls,text="Output:").pack(side="left")
        self.mode=tk.StringVar(value="Both transfer directions")
        ttk.Combobox(controls,textvariable=self.mode,values=[m[0] for m in MODES],
                     state="readonly",width=56).pack(side="left",padx=8)
        self.save=tk.BooleanVar(value=False)
        ttk.Checkbutton(controls,text="Also save plots/data",variable=self.save).pack(side="left")
        tools=ttk.Frame(outer)
        tools.pack(fill="x",pady=8)
        ttk.Button(tools,text="Reset defaults",command=self.reset).pack(side="left")
        ttk.Button(tools,text="Load settings",command=self.load).pack(side="left",padx=6)
        ttk.Button(tools,text="Save settings",command=self.save_settings).pack(side="left")
        ttk.Button(tools,text="Check parameters",command=self.check).pack(side="left",padx=6)
        self.run_button=ttk.Button(tools,text="Run simulation",command=self.run)
        self.run_button.pack(side="right")
        self.stop_button=ttk.Button(tools,text="Stop calculation",command=self.stop,state="disabled")
        self.stop_button.pack(side="right",padx=6)
        self.status=tk.StringVar(value="Ready. Default parameters are illustrative, not an experimentally validated design.")
        ttk.Label(outer,textvariable=self.status,wraplength=940).pack(anchor="w",pady=4)
        self.log=tk.Text(outer,height=4,wrap="word",state="disabled")
        self.log.pack(fill="x")
        for value in self.entries.values():
            value.trace_add("write",self.update_swap_rates)
        self.update_swap_rates()
        self.root.protocol("WM_DELETE_WINDOW",self.close)
        self.root.after(100,self.poll)

    def build_sweep_tab(self):
        tab=ttk.Frame(self.tabs,padding=18)
        self.tabs.add(tab,text="Parameter sweep")
        ttk.Label(tab,text="Step a model parameter and compare simulation outputs",
                  font=("",15,"bold")).grid(row=0,column=0,columnspan=3,sticky="w",pady=(0,14))
        self.sweep_labels={f"{label} [{key}] ({unit})":key
                           for _,key,label,unit,_,_ in SWEEP_FIELDS}
        default=next(label for label,key in self.sweep_labels.items() if key=="PHI_SWAP")
        self.sweep_parameter=tk.StringVar(value=default)
        self.sweep_start=tk.StringVar(value="0.26")
        self.sweep_stop=tk.StringVar(value="0.27")
        self.sweep_step=tk.StringVar(value="0.001")
        self.sweep_direction=tk.StringVar(value="both")
        ttk.Label(tab,text="Parameter").grid(row=1,column=0,sticky="w",pady=8)
        ttk.Combobox(tab,textvariable=self.sweep_parameter,values=list(self.sweep_labels),
                     state="readonly",width=72).grid(row=1,column=1,columnspan=2,sticky="w")
        for row,(label,value) in enumerate((("Start",self.sweep_start),("Stop",self.sweep_stop),
                                           ("Step",self.sweep_step)),start=2):
            ttk.Label(tab,text=label).grid(row=row,column=0,sticky="w",padx=(0,20),pady=8)
            ttk.Entry(tab,textvariable=value,width=22).grid(row=row,column=1,sticky="w")
        ttk.Label(tab,text="Transfer directions").grid(row=5,column=0,sticky="w",pady=8)
        ttk.Combobox(tab,textvariable=self.sweep_direction,values=("both","write","read"),
                     state="readonly",width=20).grid(row=5,column=1,sticky="w")
        self.sweep_preview=tk.StringVar()
        ttk.Label(tab,textvariable=self.sweep_preview,wraplength=850).grid(
            row=6,column=0,columnspan=3,sticky="w",pady=12)
        ttk.Button(tab,text="Run sweep",command=self.run_sweep).grid(row=7,column=0,sticky="w",pady=10)
        ttk.Label(tab,text=("Uses the current inputs from the other tabs. PHI_SWAP changes the flux pulse peak; "
            "PHI_PARK changes the parked flux. Values use the selected parameter's units. "
            "Points follow start + n × step; stop is included only if reached by that grid. "
            "Use a negative step for descending sweeps. Up to 501 points.\n\n"
            "Each point reloads the model, recomputes derived frequencies, couplings and pulse durations, "
            "and independently calibrates its noiseless phase reference. No pulse optimization is performed. "
            "Configured experimental tolerances are scanned at every sweep point and displayed as shaded ranges and summary columns. Write includes storage; read starts from an independently "
            "prepared atomic state, held for the configured storage duration.\n\n"
            "The result uses the full transfer figure layout with a sweep slider inside each figure. Drag the slider or use Previous / Next to update controls, detuning, populations, summary and 3D outputs. CSV/JSON summaries and per-point "
            "fidelity/population traces are saved automatically in the run output folder. "
            "Also save plots/data adds a PNG of the initial point for each direction. Stop calculation stops the sweep."),
            wraplength=1000,justify="left").grid(row=8,column=0,columnspan=3,sticky="w",pady=12)
        for value in (self.sweep_parameter,self.sweep_start,self.sweep_stop,self.sweep_step):
            value.trace_add("write",self.update_sweep_preview)
        self.update_sweep_preview()

    def sweep_points(self):
        return sweep_configs(self.config(),self.sweep_labels[self.sweep_parameter.get()],
                             self.sweep_start.get(),self.sweep_stop.get(),self.sweep_step.get())

    def update_sweep_preview(self,*_):
        try:
            points=self.sweep_points()
            key=self.sweep_labels[self.sweep_parameter.get()]
            self.sweep_preview.set(f"{len(points)} simulations: {key} from {points[0][key]:.10g} "
                                   f"to {points[-1][key]:.10g}. "
                                   f"{3**sum(points[0][name]>0 for name in ('FC_TOL','FT_TOL','FA_TOL','FLUX_TOL','EFIELD_TOL'))} "
                                   "tolerance scenarios per point per direction. Each point may take some time.")
        except ValueError as exc:
            self.sweep_preview.set(str(exc))

    def run_sweep(self):
        if self.process and self.process.poll() is None:
            return
        self.mode.set("Parameter sweep")
        self.run()

    def build_assistant_tab(self):
        tab=ttk.Frame(self.tabs,padding=12)
        self.tabs.add(tab,text="Assistant")
        ttk.Label(tab,text="Ask about your settings and simulation output",
                  font=("",14,"bold")).pack(anchor="w")
        ttk.Label(tab,text="Each question includes current settings, calculated quantities, "
                  "and the latest run's numerical plot data and console output.\nQuestions are independent. Answers cannot change parameters "
                  "or run simulations.",wraplength=900).pack(anchor="w",pady=8)
        ttk.Label(tab,text="Codex uses ChatGPT sign-in and your plan's usage limits. Settings, output and relevant "
                  "project files it reads are sent to OpenAI. Read-only project access; API-key billing is disabled.",
                  wraplength=900).pack(anchor="w",pady=(0,8))
        self.check_codex_button=ttk.Button(tab,text="Check Codex",command=self.check_codex_connection)
        self.check_codex_button.pack(anchor="w")
        self.assistant_status=tk.StringVar(value="Click Check Codex or ask a question. The answer appears when Codex finishes.")
        ttk.Label(tab,textvariable=self.assistant_status,wraplength=900).pack(anchor="w",pady=8)
        self.numerical_status=tk.StringVar(value="Run a simulation in this panel to provide numerical plot data.")
        ttk.Label(tab,textvariable=self.numerical_status,wraplength=900).pack(anchor="w",pady=(0,8))
        self.assistant_output=ScrolledText(tab,wrap="word",height=12,state="disabled")
        self.assistant_output.pack(fill="both",expand=True)
        ttk.Label(tab,text="Your question (up to 1,000 characters):").pack(anchor="w",pady=(8,3))
        self.assistant_question=tk.Text(tab,height=3,wrap="word")
        self.assistant_question.pack(fill="x")
        self.assistant_question.insert("1.0","Explain my current detunings and pulse durations. What should I check before running?")
        buttons=ttk.Frame(tab)
        buttons.pack(fill="x",pady=8)
        self.ask_button=ttk.Button(buttons,text="Ask assistant",command=self.ask_assistant)
        self.ask_button.pack(side="left")
        self.cancel_answer_button=ttk.Button(buttons,text="Cancel answer",command=self.cancel_answer,state="disabled")
        self.cancel_answer_button.pack(side="left",padx=8)

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
        question=self.assistant_question.get("1.0","end").strip()
        if not question or len(question)>1000:
            self.assistant_status.set("Enter a question between 1 and 1,000 characters.")
            return
        try:
            messages=build_messages(question,self.config(),self.mode.get(),self.last_run)
        except ValueError as exc:
            messagebox.showerror("Check parameters",str(exc))
            return
        self.assistant_request+=1
        request_id=self.assistant_request
        self.assistant_cancel=threading.Event()
        cancelled=self.assistant_cancel
        project_root=self.backend_path.parent.parent
        self.ask_button.configure(state="disabled")
        self.cancel_answer_button.configure(state="normal")
        self.assistant_status.set("Asking Codex... settings captured when you clicked Ask. The answer appears when Codex finishes.")
        self.assistant_append(f"\nYou: {question}\nAssistant (Codex):\n")
        def worker():
            try:
                response=codex_answer(messages,cancelled,project_root)
                for text in response:
                    self.events.put(("assistant_text",(request_id,text)))
                self.events.put(("assistant_done",(request_id,"Answer finished. Check suggested explanations against simulation results.")))
            except Exception as exc:
                self.events.put(("assistant_error",(request_id,f"Codex error: {exc}")))
        threading.Thread(target=worker,daemon=True).start()

    def cancel_answer(self):
        self.assistant_cancel.set()
        self.assistant_request+=1  # Ignore queued tokens from the cancelled worker.
        self.ask_button.configure(state="normal")
        self.cancel_answer_button.configure(state="disabled")
        self.assistant_status.set("Answer cancelled.")
        self.assistant_append("\n[Answer cancelled]\n")

    def update_swap_rates(self,*_):
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
        self.swap_rates.set(
            f"Nominal resonant swap frequencies: TC {d['transmon_swap_frequency_MHz']:.5g} MHz; "
            f"AC {d['atom_swap_frequency_MHz']:.5g} MHz.\n"
            f"Rounded pulse durations: TC {d['transmon_pulse_duration_us']:.5g} us; "
            f"AC {d['atom_pulse_duration_us']:.5g} us.")

    def config(self):
        return validate({key:value.get() for key,value in self.entries.items()})

    def append(self,text):
        self.log.configure(state="normal")
        self.log.insert("end",text)
        self.log.see("end")
        self.log.configure(state="disabled")

    def reset(self):
        for key,value in DEFAULTS.items():
            self.entries[key].set(str(value))
        self.status.set("Default model parameters restored.")

    def load(self):
        path=filedialog.askopenfilename(filetypes=[("Settings JSON","*.json")])
        if not path:
            return
        try:
            config=validate(json.loads(Path(path).read_text()))
            for key,value in config.items():
                self.entries[key].set(str(value))
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
            if "--parameter-sweep" in dict(MODES)[self.mode.get()]:
                points=self.sweep_points()
                self.status.set(f"Sweep checks passed: {len(points)} points.")
                return True
            if "--simultaneous" in dict(MODES)[self.mode.get()]:
                from simultaneous_coupling import resonance_settings
                resonance=resonance_settings(config)
                self.status.set(
                    f"Simultaneous resonance: flux {resonance['resonant_flux']:.6g}; "
                    f"voltage {resonance['resonant_voltage_V']:.6g} V; "
                    f"hold {config['SIM_HOLD_SCALE']*resonance['reference_hold_us']:.6g} us; "
                    f"ideal exchange limit {resonance['ideal_transfer_limit']:.6f}. Nominal results.")
                return True
            self.status.set(
                f"Model checks passed. Swap detuning TC/AC: "
                f"{d['transmon_swap_detuning_MHz']:.5g}/{d['atom_swap_detuning_MHz']:.5g} MHz; "
                f"optical pulse: {d['optical_pi_pulse_us']:.5g} us; "
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
        mode_args=list(dict(MODES)[self.mode.get()])
        if "--parameter-sweep" in mode_args:
            mode_args.extend(["--sweep-parameter",self.sweep_labels[self.sweep_parameter.get()],
                              "--sweep-start",self.sweep_start.get(),"--sweep-stop",self.sweep_stop.get(),
                              "--sweep-step",self.sweep_step.get(),"--sweep-direction",self.sweep_direction.get()])
        reports_root=self.backend_path.parent.parent/"output"/"assistant_runs"
        try:
            reports_root.mkdir(parents=True,exist_ok=True)
            report_directory=tempfile.mkdtemp(prefix="run-",dir=reports_root)
        except OSError as exc:
            self.temp_path.unlink(missing_ok=True)
            messagebox.showerror("Could not prepare simulation output",str(exc))
            return
        # An explicit --run bypasses the startup panel in the child.
        command=[sys.executable,"-u",str(self.backend_path),
                 "--run","--config",str(self.temp_path),"--assistant-output",report_directory,*mode_args]
        if self.save.get():
            command.append("--save")
        try:
            self.process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                          text=True,bufsize=1,start_new_session=(os.name=="posix"))
        except Exception as exc:
            self.temp_path.unlink(missing_ok=True)
            messagebox.showerror("Could not start simulation",str(exc))
            return
        self.run_button.configure(state="disabled")
        self.last_run={"settings":run_config,"mode":self.mode.get(),"status":"running","output":"",
                       "output_directory":report_directory}
        if "--parameter-sweep" in mode_args:
            self.last_run["sweep"]={"parameter":self.sweep_labels[self.sweep_parameter.get()],
                                    "start":self.sweep_start.get(),"stop":self.sweep_stop.get(),
                                    "step":self.sweep_step.get(),"direction":self.sweep_direction.get()}
        self.numerical_status.set("Numerical reports are being calculated for this run. Ask after a figure is ready.")
        self.stop_button.configure(state="normal")
        self.status.set("Calculating. Selected plot windows will open when ready; close them to finish the run.")
        self.append("\n--- New simulation ---\n")
        threading.Thread(target=self.read_output,args=(self.process,),daemon=True).start()

    def read_output(self,process):
        for line in process.stdout:
            self.events.put(("line",line))
        self.events.put(("done",process.wait()))

    def poll(self):
        try:
            while True:
                kind,value=self.events.get_nowait()
                if kind=="line":
                    self.append(value)
                    if self.last_run is not None:
                        self.last_run["output"]=(self.last_run["output"]+value)[-3000:]
                        if value.startswith("Full parameter sweep results:"):
                            self.numerical_status.set("Parameter sweep results available. Files: "+self.last_run["output_directory"])
                        elif value.startswith("Full tolerance results:"):
                            self.numerical_status.set("Tolerance scan results available. Files: "+self.last_run["output_directory"])
                        elif value.startswith("Assistant numerical report ready:"):
                            self.numerical_status.set("Numerical plot data is available. Ask now, even while plot windows remain open. "
                                                      "Files: "+self.last_run["output_directory"])
                elif kind in ("codex_ready","codex_error"):
                    self.check_codex_button.configure(state="normal")
                    self.assistant_status.set(value)
                elif kind.startswith("assistant_"):
                    request_id,text=value
                    if request_id!=self.assistant_request:
                        continue
                    if kind=="assistant_text":
                        self.assistant_append(text)
                    else:
                        self.ask_button.configure(state="normal")
                        self.cancel_answer_button.configure(state="disabled")
                        if kind=="assistant_error":
                            self.assistant_append("\n"+text+"\n")
                        else:
                            self.assistant_append("\n")
                        self.assistant_status.set(text)
                elif kind=="done":
                    if self.last_run is not None:
                        self.last_run["status"]="finished" if value==0 else "stopped or failed"
                        folder=Path(self.last_run["output_directory"])
                        available=[name for name in ("write","read","roundtrip","tolerance_scan","parameter_sweep") if (folder/(name+".json")).is_file()]
                        self.numerical_status.set("Numerical reports available: "+", ".join(available)+". Files: "+str(folder)
                                                  if available else "No numerical reports were produced for this run. See the calculation log.")
                    self.status.set("Run finished. Edit parameters and run again." if value==0
                                    else "Run stopped or failed. See the calculation log.")
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
            if os.name=="posix":
                try:
                    os.killpg(self.process.pid,signal.SIGTERM)
                except ProcessLookupError:
                    pass
            else:
                self.process.terminate()
            self.status.set("Stopping calculation...")

    def close(self):
        if self.process and self.process.poll() is None:
            if not messagebox.askyesno("Close simulation panel","Stop the active run and close the panel?"):
                return
            if os.name=="posix":
                try:
                    os.killpg(self.process.pid,signal.SIGTERM)
                except ProcessLookupError:
                    pass
            else:
                self.process.terminate()
        self.assistant_cancel.set()
        if self.temp_path:
            self.temp_path.unlink(missing_ok=True)
        self.root.destroy()


def launch(backend_path=None):
    root=tk.Tk()
    ParameterPanel(root, backend_path=backend_path)
    root.mainloop()


if __name__=="__main__":
    launch()
