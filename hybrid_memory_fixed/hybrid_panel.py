"""Native startup parameter panel. Keep beside codetest.py and hybrid_settings.py."""
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from hybrid_settings import FIELDS, DEFAULTS, validate, derived

MODES = [
("All seven figures", []),
("Write and store", ["--run"]),
("Transmon-to-atom paper-style figure", ["--write-paper-figure"]),
("Six-panel scientific summary", ["--paper-figure"]),
("Transmon-to-atom paper-style figure", ["--write-paper-figure"]),
("Preparation leakage", ["--leakage-scan"]),
("Hyperfine to transmon", ["--retrieve"]),
("Rydberg to transmon", ["--retrieve","--source","rydberg"]),
("Write-store-read round trip", ["--roundtrip"]),
]


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
        ttk.Label(outer,text="This model takes a cavity frequency and couplings as inputs; it does not solve cavity geometry.\n"
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
                f"Model checks passed. Resonance voltage: {d['resonance_voltage_V']:.5g} V; "
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
        if self.mode.get()=="Write and store":
            mode_args=["--write-only"]
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
