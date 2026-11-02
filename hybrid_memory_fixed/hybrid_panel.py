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
("Atom to transmon", ["--roundtrip"]),
]


def coupling_equations_png(dpi=120):
    """Typeset the coupling equations without a LaTeX installation or GUI backend."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure=Figure(figsize=(5.6,1.25),dpi=dpi)
    FigureCanvasAgg(figure)
    equations=(
        r"$g_{\mathrm{TC}} = \frac{2e\beta V_{\mathrm{zpf}}}{\hbar}"
        r"\left|\langle 0|\hat{n}|1\rangle\right|$",
        r"$g_{\mathrm{AC}} = \frac{\left|\mathbf{d}_{01}\cdot"
        r"\mathbf{E}_{\mathrm{zpf}}\right|}{\hbar}"
        r"= \frac{|d_{01}|E_{\mathrm{zpf}}|\cos\theta|}{\hbar}$",
    )
    for y,equation in zip((0.75,0.20),equations):
        figure.text(0.02,y,equation,fontsize=18,math_fontfamily="stix",color="#242424")
    output=io.BytesIO()
    figure.savefig(output,format="png",transparent=True,bbox_inches="tight",pad_inches=0.08)
    return output.getvalue()


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
        root.geometry("1000x820")
        root.minsize(850,650)
        self.entries={}
        self.coupling_summary=tk.StringVar()
        outer=ttk.Frame(root,padding=14)
        outer.pack(fill="both",expand=True)
        ttk.Label(outer,text="Hybrid quantum memory",font=("",18,"bold")).pack(anchor="w")
        ttk.Label(outer,text="Set model parameters, choose output, then run. Plots open in closable windows.").pack(anchor="w",pady=(3,10))
        self.tabs=ttk.Notebook(outer)
        self.tabs.pack(fill="both",expand=True)
        groups=list(dict.fromkeys(row[0] for row in FIELDS))
        for group in groups:
            frame=ttk.Frame(self.tabs,padding=12)
            self.tabs.add(frame,text=group)
            frame.columnconfigure(1,weight=1)
            for row,(_,key,label,units,default,rule) in enumerate(f for f in FIELDS if f[0]==group):
                ttk.Label(frame,text=label).grid(row=row,column=0,sticky="w",padx=(0,12),pady=7)
                value=tk.StringVar(value=str(default))
                entry=ttk.Entry(frame,textvariable=value,width=18)
                entry.grid(row=row,column=1,sticky="ew",pady=7)
                ttk.Label(frame,text=units).grid(row=row,column=2,sticky="w",padx=10)
                self.entries[key]=value
                value.trace_add("write",self.update_couplings)
            if group == "Coupling parameters":
                self.coupling_equations=tk.PhotoImage(
                    master=root,data=base64.b64encode(
                        coupling_equations_png(dpi=root.winfo_fpixels("1i"))).decode("ascii"))
                ttk.Label(frame,image=self.coupling_equations).grid(
                    row=row+1,column=0,columnspan=3,sticky="w",pady=(12,6))
                ttk.Label(frame,text="Use local vacuum RMS voltage and field amplitudes.\n"
                                     "Supply the charge matrix element at the operating point. "
                                     "Couplings stay constant during a run.",
                          wraplength=800).grid(row=row+2,column=0,columnspan=3,sticky="w",pady=(0,8))
        ttk.Label(outer,textvariable=self.coupling_summary).pack(anchor="w",pady=(8,0))
        self.update_couplings()
        ttk.Label(outer,text="Couplings are calculated from device and atomic inputs; cavity geometry is not solved.\n"
                            "Swap flux and voltage are independent inputs; device changes do not retune them.\n"
                            "Stark coefficients include their sign and the factor -alpha/(2h). Pure dephasing times are not total T2.",
                  wraplength=940).pack(anchor="w",pady=(10,6))
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
        self.log=tk.Text(outer,height=8,wrap="word",state="disabled")
        self.log.pack(fill="x")
        self.root.protocol("WM_DELETE_WINDOW",self.close)
        self.root.after(100,self.poll)

    def config(self):
        return validate({key:value.get() for key,value in self.entries.items()})

    def update_couplings(self,*_):
        # Traces can fire while entries are still being constructed.
        if len(self.entries) != len(FIELDS):
            return
        try:
            d=derived(self.config())
            self.coupling_summary.set(
                f"Calculated g(TC) / 2π: {d['transmon_coupling_MHz']:.6g} MHz; "
                f"g(AC) / 2π: {d['atom_coupling_MHz']:.6g} MHz\n"
                f"Swap detuning TC: {d['transmon_swap_detuning_MHz']:.6g} MHz; "
                f"AC: {d['atom_swap_detuning_MHz']:.6g} MHz")
        except ValueError:
            self.coupling_summary.set("Calculated couplings unavailable: check parameter values.")

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
