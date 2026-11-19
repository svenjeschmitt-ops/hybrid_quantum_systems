"""Embedded interactive preview of pulse timing and detuning."""
import numpy as np
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from pulse_preview import control_preview


class PulsePreview:
    def __init__(self, parent, root, get_settings, is_visible):
        self.root, self.get_settings = root, get_settings
        self.is_visible = is_visible
        self.closed = False
        self.controls = None
        self.refresh_after = None
        self.direction = tk.StringVar(value='Write')
        self.status = tk.StringVar(value='Select the pulse tab to inspect controls and detunings.')
        self.timing = tk.StringVar()
        self.readout = tk.StringVar(value='Move the time cursor to inspect the trajectory.')
        self.cursor_ns = tk.DoubleVar(value=0)
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(2, weight=1)
        controls = ttk.Frame(parent)
        controls.grid(row=0, column=0, sticky='ew')
        ttk.Label(controls, text='Preview:').pack(side='left')
        ttk.Combobox(controls, textvariable=self.direction, values=('Write', 'Read'),
                     state='readonly', width=7).pack(side='left', padx=5)
        ttk.Button(controls, text='Refresh', command=self.refresh_now).pack(side='right')
        ttk.Label(parent, textvariable=self.timing).grid(row=1, column=0, sticky='w', pady=(4, 0))
        self.figure = Figure(figsize=(5, 3.4), dpi=100, layout='constrained')
        self.figure.get_layout_engine().set(h_pad=0.06, hspace=0.08)
        self.axes = self.figure.subplots(3, 1, sharex=True)
        self.voltage_axis = self.axes[0].twinx()
        self.canvas = FigureCanvasTkAgg(self.figure, master=parent)
        self.canvas.get_tk_widget().grid(row=2, column=0, sticky='nsew')
        toolbar_frame = ttk.Frame(parent)
        toolbar_frame.grid(row=3, column=0, sticky='ew')
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame, pack_toolbar=False)
        self.toolbar.pack(side='left', fill='x')
        cursor_row = ttk.Frame(parent)
        cursor_row.grid(row=4, column=0, sticky='ew')
        cursor_row.columnconfigure(1, weight=1)
        ttk.Label(cursor_row, text='Time (ns)').grid(row=0, column=0, padx=(0, 6))
        self.time_slider = ttk.Scale(cursor_row, from_=0, to=1, variable=self.cursor_ns,
                                     command=self.inspect_time)
        self.time_slider.grid(row=0, column=1, sticky='ew')
        ttk.Label(parent, textvariable=self.readout, justify='left', wraplength=800).grid(
            row=5, column=0, sticky='ew', pady=4)
        ttk.Label(parent, textvariable=self.status, justify='left', wraplength=800).grid(
            row=6, column=0, sticky='ew')
        parent.bind('<Configure>', self.resize, add='+')
        self.canvas.mpl_connect('button_press_event', self.click_time)
        self.direction.trace_add('write', self.schedule)
        self.draw_empty('Select this tab to preview controls')
        self.root.bind('<Destroy>', lambda event: self.close() if event.widget is self.root else None, add='+')
        self.schedule()

    def selected_direction(self):
        return self.direction.get().lower()

    def resize(self, event):
        for child in event.widget.winfo_children():
            if isinstance(child, ttk.Label):
                child.configure(wraplength=max(250, event.width-16))

    def sync_output(self, mode):
        if mode == 'Transmon to atom':
            self.direction.set('Write')
        elif mode == 'Atom to transmon':
            self.direction.set('Read')

    def visibility_changed(self, *_):
        if self.is_visible():
            self.schedule()

    def schedule(self, *_):
        if self.closed:
            return
        if self.refresh_after is not None:
            self.root.after_cancel(self.refresh_after)
        self.readout.set('Parameters changed; refreshing the preview…')
        self.status.set('Updating control pulses and detunings…')
        self.refresh_after = self.root.after(70, self.refresh_controls)

    def refresh_controls(self):
        self.refresh_after = None
        try:
            self.controls = control_preview(self.get_settings(), self.selected_direction())
        except (ValueError, TypeError, KeyError) as exc:
            self.controls = None
            self.draw_empty('Enter valid model parameters')
            self.status.set(str(exc))
            self.readout.set('Preview unavailable while a parameter is invalid.')
            return
        self.draw()
        self.status.set('Nominal controls. Drag the time cursor or click a plot to inspect values.')

    def refresh_now(self):
        if self.closed:
            return
        if self.refresh_after is not None:
            self.root.after_cancel(self.refresh_after)
        self.refresh_controls()

    def set_axis_labels(self):
        """Keep labels compact enough to fit beside each short subplot."""
        for axis, text in zip(self.axes, ('Flux\nΦ/Φ₀', 'Detuning\nMHz', 'Optical\nMHz')):
            axis.set_ylabel(text, fontsize=8, rotation=0, ha='right', va='center', labelpad=5)
        self.voltage_axis.set_ylabel('Voltage\nV', fontsize=8, rotation=0, ha='left', va='center', labelpad=5)
        self.voltage_axis.yaxis.set_label_position('right')
        self.voltage_axis.yaxis.tick_right()

    def draw_empty(self, message):
        for axis in (*self.axes, self.voltage_axis):
            axis.clear()
        self.set_axis_labels()
        self.axes[2].set_xlabel('Time (ns)')
        self.axes[1].text(.5, .5, message, transform=self.axes[1].transAxes, ha='center')
        self.canvas.draw_idle()

    def draw(self):
        data = self.controls
        times = np.asarray(data['time_us'])*1000
        for axis in (*self.axes, self.voltage_axis):
            axis.clear()
        flux_line, = self.axes[0].plot(times, data['flux'], color='tab:blue', label='Flux')
        voltage_line, = self.voltage_axis.plot(times, data['voltage'], '--', color='tab:orange', label='Voltage')
        self.set_axis_labels()
        self.axes[0].legend(handles=[flux_line, voltage_line], loc='upper right', fontsize=7, ncol=2)
        for name in ('preparation', 'mapping'):
            start, end = data['stages'][name]
            self.axes[0].axvspan(start*1000, end*1000, alpha=.09, color='tab:green')
            self.axes[0].text((start+end)*500, 1.02, 'Prep' if name=='preparation' else 'Map',
                              transform=self.axes[0].get_xaxis_transform(), ha='center', fontsize=7)
        self.axes[1].plot(times, data['transmon_detuning'], label='Transmon', color='tab:blue')
        self.axes[1].plot(times, data['atom_detuning'], '--', label='Atom', color='tab:orange')
        self.axes[1].axhline(0, color='gray', lw=.6)
        self.axes[1].legend(loc='upper right', ncol=2, fontsize=7)
        self.axes[2].plot(times, data['optical0'], label='h0 ↔ r0', color='tab:green')
        self.axes[2].plot(times, data['optical1'], '--', label='h1 ↔ r1', color='tab:purple')
        self.axes[2].legend(loc='upper right', ncol=2, fontsize=7)
        self.axes[2].set_xlabel('Time (ns)', fontsize=8)
        self.cursors = []
        for axis in self.axes:
            axis.grid(alpha=.18)
            axis.tick_params(labelsize=8)
            self.cursors.append(axis.axvline(self.cursor_ns.get(), color='black', alpha=.45, lw=.8))
        finish_ns = max(float(times[-1]), 1e-6)
        self.axes[1].set_xlim(0, finish_ns)
        self.time_slider.configure(to=finish_ns)
        self.cursor_ns.set(min(max(0, self.cursor_ns.get()), finish_ns))
        value = data['settings'][self.selected_direction().upper()+'_PULSE_DELAY_NS']
        mode = 'auto' if value is None else 'manual'
        self.timing.set(f"{self.direction.get()} delay {data['delay_ns']:+.3f} ns ({mode})   •   "
                        f"Pulse overlap {data['overlap_ns']:.3f} ns   •   End {finish_ns:.3f} ns")
        self.toolbar.update()
        self.inspect_time()

    def click_time(self, event):
        if event.inaxes in (*self.axes, self.voltage_axis) and event.xdata is not None and not self.toolbar.mode:
            self.cursor_ns.set(max(0., min(float(event.xdata), float(self.time_slider['to']))))
            self.inspect_time()

    def inspect_time(self, *_):
        if self.controls is None:
            return
        times = np.asarray(self.controls['time_us'])*1000
        index = int(np.argmin(abs(times-self.cursor_ns.get())))
        t = times[index]
        for cursor in getattr(self, 'cursors', []):
            cursor.set_xdata([t, t])
        data = self.controls
        text = (f"t={t:.3f} ns   Φ/Φ₀={data['flux'][index]:.5f}   V={data['voltage'][index]:.5f} V   "
                f"ΔT={data['transmon_detuning'][index]:+.2f} MHz   ΔA={data['atom_detuning'][index]:+.2f} MHz")
        self.readout.set(text)
        self.canvas.draw_idle()

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.refresh_after is not None:
            self.root.after_cancel(self.refresh_after)
