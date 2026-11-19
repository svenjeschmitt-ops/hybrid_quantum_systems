"""Visual editor for reproducible multi-peak pulse shapes."""
import json
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pulse_shapes import CHANNELS, parse_shapes, pulse_window

LABELS = ('Optical h0 ↔ r0', 'Optical h1 ↔ r1', 'Transmon flux', 'Atomic voltage')


class PulseShapeEditor:
    def __init__(self, parent, variable):
        self.variable = variable
        self.window = tk.Toplevel(parent)
        self.window.title('Pulse shapes')
        self.window.geometry('900x760')
        self.window.transient(parent)
        self.window.columnconfigure(0, weight=1)
        self.window.rowconfigure(1, weight=1)
        ttk.Label(self.window, text='Build a signal using a sin² baseline, multiple Gaussian peaks and repeatable noise. The sin² window tapers all components to zero at the pulse boundaries.', wraplength=850).grid(row=0, column=0, sticky='ew', padx=12, pady=8)
        tabs = ttk.Notebook(self.window)
        tabs.grid(row=1, column=0, sticky='nsew', padx=12)
        spec = parse_shapes(variable.get())
        self.pages = {}
        for channel, title in zip(CHANNELS, LABELS):
            page = ttk.Frame(tabs, padding=10)
            tabs.add(page, text=title)
            page.columnconfigure(1, weight=1)
            current = spec.get(channel, {})
            variables = {}
            for row, (key, label, default) in enumerate((('base', 'Sin² baseline multiplier', 1), ('noise', 'Noise amplitude (relative)', 0), ('seed', 'Noise seed', 0))):
                ttk.Label(page, text=label).grid(row=row, column=0, sticky='w', pady=3)
                var = tk.StringVar(value=str(current.get(key, default)))
                ttk.Entry(page, textvariable=var).grid(row=row, column=1, sticky='ew', pady=3)
                variables[key] = var
            ttk.Label(page, text='Gaussian peaks (amplitude before taper; center and width are fractions of duration)', wraplength=800).grid(row=3, column=0, columnspan=2, sticky='w', pady=(7, 2))
            table = ttk.Treeview(page, columns=('amplitude', 'center', 'width'), show='headings', height=3, selectmode='browse')
            for key, title2 in (('amplitude', 'Amplitude'), ('center', 'Center (0–1)'), ('width', 'Width (0.01–1)')):
                table.heading(key, text=title2)
                table.column(key, width=150, stretch=True)
            table.grid(row=4, column=0, columnspan=2, sticky='ew')
            for peak in current.get('gaussians', []):
                table.insert('', 'end', values=peak)
            inputs = ttk.Frame(page)
            inputs.grid(row=5, column=0, columnspan=2, sticky='ew', pady=5)
            peak_vars = {}
            for key, text, default in (('amplitude', 'Amplitude', '1'), ('center', 'Center', '0.35'), ('width', 'Width', '0.08')):
                ttk.Label(inputs, text=text).pack(side='left', padx=(0, 3))
                var = tk.StringVar(value=default)
                ttk.Entry(inputs, textvariable=var, width=8).pack(side='left', padx=(0, 10))
                peak_vars[key] = var
            controls = ttk.Frame(page)
            controls.grid(row=6, column=0, columnspan=2, sticky='w', pady=4)
            ttk.Button(controls, text='Add peak', command=lambda ch=channel: self.add_peak(ch)).pack(side='left', padx=(0, 5))
            ttk.Button(controls, text='Update selected', command=lambda ch=channel: self.update_peak(ch)).pack(side='left', padx=5)
            ttk.Button(controls, text='Remove selected', command=lambda ch=channel: self.remove_peak(ch)).pack(side='left', padx=5)
            ttk.Button(controls, text='Load two-peak example', command=lambda ch=channel: self.example(ch)).pack(side='left', padx=5)
            figure = Figure(figsize=(6, 2.4), dpi=90, layout='constrained')
            ax = figure.subplots()
            canvas = FigureCanvasTkAgg(figure, master=page)
            canvas.get_tk_widget().grid(row=8, column=0, columnspan=2, sticky='nsew')
            page.rowconfigure(8, weight=1)
            status = tk.StringVar(value='')
            ttk.Label(page, textvariable=status, foreground='#a03030', wraplength=800).grid(row=7, column=0, columnspan=2, sticky='w')
            self.pages[channel] = dict(variables=variables, table=table, peak_vars=peak_vars, ax=ax, canvas=canvas, status=status)
            table.bind('<<TreeviewSelect>>', lambda event, ch=channel: self.select_peak(ch))
            for var in variables.values():
                var.trace_add('write', lambda *_, ch=channel: self.preview(ch))
            self.preview(channel)
        buttons = ttk.Frame(self.window)
        buttons.grid(row=2, column=0, sticky='e', padx=12, pady=10)
        ttk.Button(buttons, text='Reset all', command=self.reset).pack(side='left', padx=4)
        ttk.Button(buttons, text='Apply', command=self.apply).pack(side='left', padx=4)
        ttk.Button(buttons, text='Cancel', command=self.window.destroy).pack(side='left', padx=4)

    def data(self, channel):
        page = self.pages[channel]
        variables, table = page['variables'], page['table']
        rows = [list(table.item(item, 'values')) for item in table.get_children()]
        return dict(base=variables['base'].get(), noise=variables['noise'].get(), seed=variables['seed'].get(), gaussians=rows)

    def peak_values(self, channel):
        page = self.pages[channel]
        row = [page['peak_vars'][key].get() for key in ('amplitude', 'center', 'width')]
        return parse_shapes({channel: {'gaussians': [row]}})[channel]['gaussians'][0]

    def select_peak(self, channel):
        page = self.pages[channel]
        selection = page['table'].selection()
        if selection:
            for key, value in zip(('amplitude', 'center', 'width'), page['table'].item(selection[0], 'values')):
                page['peak_vars'][key].set(value)

    def add_peak(self, channel):
        page = self.pages[channel]
        if len(page['table'].get_children()) >= 20:
            messagebox.showerror('Check pulse shape', 'Use at most 20 Gaussian peaks.', parent=self.window)
            return
        try:
            row = self.peak_values(channel)
        except ValueError as exc:
            messagebox.showerror('Check pulse shape', str(exc), parent=self.window)
            return
        item = page['table'].insert('', 'end', values=row)
        page['table'].selection_set(item)
        self.preview(channel)

    def update_peak(self, channel):
        page = self.pages[channel]
        selected = page['table'].selection()
        if not selected:
            messagebox.showinfo('Select a peak', 'Select a peak to update.', parent=self.window)
            return
        try:
            page['table'].item(selected[0], values=self.peak_values(channel))
        except ValueError as exc:
            messagebox.showerror('Check pulse shape', str(exc), parent=self.window)
            return
        self.preview(channel)

    def remove_peak(self, channel):
        page = self.pages[channel]
        for item in page['table'].selection():
            page['table'].delete(item)
        self.preview(channel)

    def example(self, channel):
        page = self.pages[channel]
        page['variables']['base'].set('0')
        page['variables']['noise'].set('0.05')
        page['variables']['seed'].set('42')
        table = page['table']
        table.delete(*table.get_children())
        for peak in ((1.5, 0.35, 0.08), (0.7, 0.65, 0.08)):
            table.insert('', 'end', values=peak)
        self.preview(channel)

    def preview(self, channel):
        page = self.pages[channel]
        try:
            spec = parse_shapes({channel: self.data(channel)})
        except ValueError as exc:
            page['status'].set(str(exc))
            return
        page['status'].set('')
        u = np.linspace(0, 1, 501)
        ax, canvas = page['ax'], page['canvas']
        ax.clear()
        ax.plot(u, pulse_window(u, 0., 1., spec, channel))
        ax.axhline(0, color='gray', linewidth=.6)
        ax.set(xlabel='Fraction of pulse duration', ylabel='Relative amplitude')
        ax.grid(alpha=.2)
        canvas.draw_idle()

    def reset(self):
        for channel, page in self.pages.items():
            for key, value in (('base', 1), ('noise', 0), ('seed', 0)):
                page['variables'][key].set(str(value))
            table = page['table']
            table.delete(*table.get_children())
            self.preview(channel)

    def apply(self):
        try:
            spec = parse_shapes({channel: self.data(channel) for channel in CHANNELS})
        except ValueError as exc:
            messagebox.showerror('Check pulse shape', str(exc), parent=self.window)
            return
        self.variable.set(json.dumps(spec, sort_keys=True))
        self.window.destroy()
