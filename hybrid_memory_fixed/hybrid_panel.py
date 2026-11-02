"""Native startup parameter panel. Keep beside backend.py and hybrid_settings.py."""
import base64
import io
import json
import os
from pathlib import Path
import queue
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
from hybrid_settings import FIELDS, DEFAULTS, validate, derived

MODES = [
("Both transfer directions", []),
("Transmon to atom", ["--write-only"]),
("Atom to transmon", ["--retrieve"]),
]


# Symbols use the same notation as the equations beneath each tab.
PARAMETER_SYMBOLS = {
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
    "GAP": r"T_{\mathrm{gap}}",
    "STORAGE": r"T_{\mathrm{storage}}",
    "INPUT_THETA": r"\vartheta",
    "INPUT_PHI": r"\varphi",
    "MAX_STEP": r"\Delta t_{\max}",
    "NC": r"N_c",
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
        "w_TC(t) is zero outside the interaction pulse and one during its flat hold, with raised-cosine ramps. The parked flux sets idle detuning to suppress unwanted exchange while coupling remains active.",
    ),
    "Coupling parameters": (
        (
            r"$g_{\mathrm{TC}}\approx\frac{2e\beta V_{\mathrm{zpf}}}{\hbar}\left(\frac{E_J(\Phi_{\mathrm{swap}})}{32E_C}\right)^{1/4}$",
            r"$g_{\mathrm{AC}}=\frac{|\mathbf{d}_{01}\cdot\mathbf{E}_{\mathrm{zpf}}|}{\hbar}=\frac{|d_{01}|E_{\mathrm{zpf}}|\cos\theta|}{\hbar}$",
            r"$f_{\mathrm{swap},TC}=2g_{TC}/(2\pi),\qquad f_{\mathrm{swap},AC}=2g_{AC}/(2\pi),\qquad t_{\mathrm{hold}}=1/(2f_{\mathrm{swap}})$",
            r"$\beta=\frac{C_g}{C_\Sigma},\qquad n_{01}\approx\left(\frac{E_J(\Phi_{\mathrm{swap}})}{32E_C}\right)^{1/4}$",
        ),
        "Use local vacuum RMS voltage and field amplitudes. C_g is the coupling capacitance; "
        "C_Σ is total transmon capacitance. The charge matrix element is calculated at the swap point. "
        "Couplings stay constant during a run. Swap frequencies shown are nominal resonant population oscillation rates; detuning reduces transfer.",
    ),
    "Atom and electrodes": (
        (
            r"$E(t)=\frac{V(t)}{\ell}+E_{\mathrm{stray}},\qquad f_A(E)=f_{A,0}+(K_{r1}-K_{r0})E^2$",
            r"$E_{\mathrm{swap}}=\frac{V_{\mathrm{swap}}}{\ell}+E_{\mathrm{stray}},\qquad f_{A,\mathrm{swap}}=f_{A,0}+(K_{r1}-K_{r0})E_{\mathrm{swap}}^2$",
            r"$V_{\mathrm{park}}=\ell(E_{\mathrm{park}}-E_{\mathrm{stray}}),\qquad \Delta_A(t)=2\pi[f_A(E(t))-f_c]$",
            r"$\Delta_{w,r0}(E)=2\pi[f_{\mathrm{wrong,split}}+(K_w-K_{r0})E^2],\qquad \Omega_w(t)=\eta\,\Omega(t)$",
        ),
        "ℓ is electrode separation in cm; E is in V/cm and K is in MHz/(V/cm)². "
        "Stark coefficients include their sign and −α/(2h). "
        "η is relative unwanted optical amplitude. The applied swap voltage is independent of f_c. "
        "Changing the cavity frequency does not retune the voltage.",
    ),
    "Loss and coherence": (
        (
            r"$L_T=\frac{\sigma_-}{\sqrt{T_{1,T}}},\qquad L_c=\frac{a}{\sqrt{T_{1,c}}},\qquad L_{rj}=\frac{|\mathrm{loss}\rangle\langle r_j|}{\sqrt{\tau_{rj}}}$",
            r"$L_{\phi,T}=\frac{Z_T}{\sqrt{2T_{\phi,T}}},\qquad L_{\phi,R}=\frac{Z_R}{\sqrt{2T_{\phi,R}}},\qquad L_{\phi,H}=\frac{Z_H}{\sqrt{2T_{\phi,H}}}$",
            r"$Z_T=P_e-P_g,\qquad Z_R=P_{r1}-P_{r0},\qquad Z_H=P_{h1}-P_{h0}$",
            r"$\frac{1}{T_{2,T}}=\frac{1}{2T_{1,T}}+\frac{1}{T_{\phi,T}},\qquad Q_c=2\pi f_cT_{1,c}$",
        ),
        "All lifetimes are in μs; T_phi denotes pure dephasing, not total T2. "
        "Rydberg decay includes r0, r1, and the unwanted level. P_s = |s⟩⟨s|. "
        "The T2 relation shown is for the transmon; cavity T1 is photon-number lifetime.",
    ),
    "Pulses and input state": (
        (
            r"$\Omega(t)=\Omega_{\max}\sin^2\!\left(\frac{\pi(t-t_0)}{T_{\mathrm{opt}}}\right),\quad \int_{t_0}^{t_0+T_{\mathrm{opt}}}\Omega(t)\,dt=\pi,\quad T_{\mathrm{opt}}=\frac{2\pi}{\Omega_{\max}}$",
            r"$t_{\mathrm{hold},TC}=\frac{\pi}{2g_{TC}},\qquad t_{\mathrm{hold},AC}=\frac{\pi}{2g_{AC}},\qquad T_{\mathrm{swap}}=2T_{\mathrm{ramp}}+t_{\mathrm{hold}}$",
            r"$w_{\mathrm{rise}}(u)=\frac{1-\cos(\pi u/T_{\mathrm{ramp}})}{2},\qquad 0\leq u\leq T_{\mathrm{ramp}}$",
            r"$|\psi_T\rangle=\cos(\vartheta/2)|g\rangle+e^{i\varphi}\sin(\vartheta/2)|e\rangle,\qquad t_{\mathrm{end}}=t_{\mathrm{map,end}}+T_{\mathrm{storage}}$",
        ),
        "The OMEGA input is Ω_max / (2π) in MHz; times are in μs. Optical envelopes vanish outside "
        "their pulse intervals. Ramps use a raised cosine, a flat hold, and a reversed falling ramp. "
        "GAP separates operations. Input Bloch angles are entered in degrees. "
        "Swap holds are nominal isolated, resonant values.",
    ),
    "Numerics": (
        (
            r"$\dot\rho=-i[H,\rho]+\sum_k\left(L_k\rho L_k^\dagger-\frac{1}{2}\{L_k^\dagger L_k,\rho\}\right)$",
            r"$i\,\frac{d|\psi\rangle}{dt}=H|\psi\rangle\quad\mathrm{(noiseless)},\qquad \Delta t_{\mathrm{internal}}\leq\Delta t_{\max}=\mathrm{MAX\_STEP}$",
            r"$D=2\,N_c\,6=12N_c,\qquad n_{\mathrm{cavity}}\in\{0,\ldots,N_c-1\},\qquad N_c\geq3$",
        ),
        "H is expressed in angular-frequency units (rad/μs). MAX_STEP limits internal integration steps, "
        "not plot sampling. The transfer solver uses DOP853, absolute tolerance 10⁻¹¹ and relative "
        "tolerance 10⁻⁹. N_c is the cavity truncation; transmon and atom use 2 and 6 states. "
        "Check convergence by reducing MAX_STEP and increasing N_c.",
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
        controls=ttk.Frame(outer)
        controls.pack(fill="x")
        ttk.Label(controls,text="Output:").pack(side="left")
        self.mode=tk.StringVar(value=MODES[0][0])
        ttk.Combobox(controls,textvariable=self.mode,values=[m[0] for m in MODES],
                     state="readonly",width=40).pack(side="left",padx=8)
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

    def update_swap_rates(self,*_):
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
            f"Nominal resonant holds: TC {d['transmon_swap_hold_us']:.5g} us; "
            f"AC {d['atom_swap_hold_us']:.5g} us.\n"
            f"Plateau detuning TC/AC: {d['transmon_swap_detuning_MHz']:.5g} / "
            f"{d['atom_swap_detuning_MHz']:.5g} MHz.")

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
            d=derived(self.config())
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
        with tempfile.NamedTemporaryFile(mode="w",suffix=".json",prefix="hybrid-settings-",delete=False) as f:
            json.dump(self.config(),f)
            self.temp_path=Path(f.name)
        mode_args=dict(MODES)[self.mode.get()]
        # An explicit --run bypasses the startup panel in the child.
        command=[sys.executable,"-u",str(self.backend_path),
                 "--run","--config",str(self.temp_path),*mode_args]
        if self.save.get():
            command.append("--save")
        try:
            self.process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                          text=True,bufsize=1)
        except Exception as exc:
            self.temp_path.unlink(missing_ok=True)
            messagebox.showerror("Could not start simulation",str(exc))
            return
        self.run_button.configure(state="disabled")
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
                else:
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
            self.process.terminate()
            self.status.set("Stopping calculation...")

    def close(self):
        if self.process and self.process.poll() is None:
            if not messagebox.askyesno("Close simulation panel","Stop the active run and close the panel?"):
                return
            self.process.terminate()
        if self.temp_path:
            self.temp_path.unlink(missing_ok=True)
        self.root.destroy()


def launch(backend_path=None):
    root=tk.Tk()
    ParameterPanel(root, backend_path=backend_path)
    root.mainloop()


if __name__=="__main__":
    launch()
