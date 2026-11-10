"""Ordinary transfer figure layout with a discrete sweep slider inside it."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider


class SweepFigure:
    def __init__(self, renderer, report, folder, direction):
        self.renderer = renderer
        self.report = report
        self.folder = Path(folder)
        self.direction = direction
        self.index = 0
        self.figure = plt.figure(figsize=(16, 11))
        # Keep widgets alive for as long as the figure is open.
        self.figure._sweep_viewer = self
        grid = self.figure.add_gridspec(3, 2, left=.07, right=.96, bottom=.23,
                                        top=.91, hspace=.55, wspace=.32,
                                        height_ratios=(1, 1, 1.15))
        self.control_axis = self.figure.add_subplot(grid[0, 0])
        self.voltage_axis = self.control_axis.twinx()
        self.detuning_axis = self.figure.add_subplot(grid[0, 1])
        self.population_axis = self.figure.add_subplot(grid[1, :])
        self.summary_axis = self.figure.add_subplot(grid[2, 0])
        self.basis_axis = self.figure.add_subplot(grid[2, 1], projection='3d')
        title = 'Transmon to atom' if direction == 'write' else 'Atom to transmon'
        self.figure.canvas.manager.set_window_title(title+' — parameter sweep')
        self.heading = self.figure.suptitle('', fontsize=16, y=.98)
        self.input_label = self.figure.text(.5, .175, '', ha='center', fontsize=10)
        self.footer = self.figure.text(.5, .02, 'Drag the sweep slider or use Previous / Next to select a simulated point.\n'
                         'Noise and storage included; independent noiseless phase calibration per point. '
                         'Configured tolerance ranges are included when enabled.', ha='center', fontsize=9)
        slider_axis = self.figure.add_axes((.27, .10, .40, .025))
        slider_axis.text(0, 1.6, report['parameter'], transform=slider_axis.transAxes, fontsize=10)
        self.slider = Slider(slider_axis, '', 0, len(report['results'])-1,
                             valinit=0, valstep=1, valfmt='%0.0f')
        self.previous_button = Button(self.figure.add_axes((.07, .09, .10, .045)), 'Previous')
        self.next_button = Button(self.figure.add_axes((.87, .09, .07, .045)), 'Next')
        self.slider.on_changed(self.select)
        self.previous_button.on_clicked(lambda event: self.move(-1))
        self.next_button.on_clicked(lambda event: self.move(1))
        self.draw_point(0)

    def move(self, amount):
        self.slider.set_val(max(0, min(len(self.report['results'])-1, self.index+amount)))

    def select(self, value):
        index = int(value)
        if index != self.index:
            self.draw_point(index)
        else:
            row = self.report['results'][index]
            self.slider.valtext.set_text(f'{row[self.report["parameter"]]:.10g} {self.report["unit"]}')

    def draw_point(self, index):
        row = self.report['results'][index]
        point = json.loads((self.folder/row['data_file']).read_text(encoding='utf-8'))
        payload = point['directions'][self.direction]['figure_report']
        controls = payload['traces']['controls']
        dynamics = payload['traces']['selected_input_dynamics']
        metadata = payload['metadata']
        basis = payload['basis_outputs']
        for ax in (self.control_axis, self.voltage_axis, self.detuning_axis,
                   self.population_axis, self.summary_axis, self.basis_axis):
            ax.clear()
        x, series = controls['time_us'], controls['series']
        flux, = self.control_axis.plot(x, series['flux_Phi_over_Phi0'], color='tab:blue', label='Flux (left axis)')
        voltage, = self.voltage_axis.plot(x, series['electrode_voltage_V'], '--', color='tab:orange', label='Voltage (right axis)')
        self.control_axis.set(title='Flux and voltage control', ylabel='Flux / flux quantum')
        self.voltage_axis.set_ylabel('Electrode voltage (V)', color='tab:orange')
        self.voltage_axis.yaxis.set_label_position('right')
        self.voltage_axis.yaxis.tick_right()
        self.control_axis.yaxis.label.set_color('tab:blue')
        self.control_axis.tick_params(axis='y', colors='tab:blue')
        self.voltage_axis.tick_params(axis='y', colors='tab:orange')
        self.voltage_axis.legend(handles=[flux, voltage], loc='upper center', ncol=2, fontsize=8)
        self.detuning_axis.plot(x, series['transmon_detuning_MHz'], label='Transmon', color='tab:blue')
        self.detuning_axis.plot(x, series['atom_detuning_MHz'], '--', label='Atom', color='tab:orange')
        self.detuning_axis.axhline(0, color='gray', alpha=.5, linewidth=.7)
        self.detuning_axis.set(title='Detuning from cavity', ylabel='Detuning (MHz)')
        self.detuning_axis.legend(fontsize=8)
        for key, label in (('transmon_e', 'Transmon e'), ('cavity_mean_photons', 'Cavity photons'),
                           ('atom_h0', 'h0'), ('atom_h1', 'h1'), ('atom_r0', 'r0'),
                           ('atom_r1', 'r1'), ('atom_loss', 'Loss'), ('atom_wrong_rydberg', 'Wrong Rydberg')):
            self.population_axis.plot(dynamics['time_us'], dynamics['series'][key], label=label)
        self.population_axis.set(title=('Write' if self.direction=='write' else 'Read')+' transfer dynamics',
                                 ylabel='Population / photon number')
        self.population_axis.legend(ncol=2, fontsize=8)
        for ax in (self.control_axis, self.detuning_axis, self.population_axis):
            ax.set(xlabel='Time (us)' if self.direction=='write' else 'Retrieval time (us)',
                   xlim=(x[0], x[-1]))
            for start, end in metadata['stages_us'].values():
                ax.axvline(start, color='gray', alpha=.15)
                ax.axvline(end, color='gray', alpha=.15)
        self.summary_axis.set_axis_off()
        self.summary_axis.set_title('Write + storage summary' if self.direction=='write' else 'Storage + readout summary')
        import numpy as np
        tolerance_summary = payload['summary'].get('tolerance_ranges')
        tolerance_trace = payload['traces'].get('selected_input_tolerance_ranges')
        if tolerance_summary and tolerance_trace:
            import tolerance_bands
            values = tolerance_trace['series']
            bounds = {side: {
                'flux': np.asarray(values['flux_'+side]),
                'transmon_detuning': np.asarray(values['transmon_detuning_'+side]),
                'atom_detuning': np.asarray(values['atom_detuning_'+side]),
                'populations': np.asarray([values[name+'_'+side] for name in tolerance_bands.POPULATION_NAMES]),
            } for side in ('low', 'high')}
            # Viewer population order is the same in both transfer directions.
            tolerance_bands.draw({'times': np.asarray(tolerance_trace['time_us']), **bounds},
                                 self.control_axis, self.detuning_axis, self.population_axis)
        rows = [row[:3] if tolerance_summary else row[:2]
                for row in payload['summary']['displayed_metrics']]
        headers = ['Metric', 'Nominal', 'With tolerances'] if tolerance_summary else ['Metric', 'Value']
        widths = [.58, .16, .26] if tolerance_summary else [.77, .23]
        table = self.renderer.results_table(self.summary_axis, headers, rows,
                                            widths=widths, bbox=(0, -.10, 1, 1.05), fontsize=7.5)
        for number in range(1, len(rows)+1):
            table[number, 0].set_text_props(ha='left')
        self.renderer.plot_basis_columns(self.basis_axis, basis['input_labels'],
                                          np.asarray(basis['probabilities']),
                                          ('Write outputs after storage' if self.direction=='write' else 'Outputs after storage + readout')
                                          + (' (nominal)' if tolerance_summary else ''),
                                          mode=self.direction)
        self.basis_axis.set_box_aspect((4, 4, 3), zoom=.92)
        parameter = self.report['parameter']
        title = 'Transmon to atom' if self.direction=='write' else 'Atom to transmon'
        self.heading.set_text(f'{title} — {parameter} = {row[parameter]:.10g} {self.report["unit"]}\n'
                              f'Sweep point {index+1} / {len(self.report["results"])}'
                              + (' — shaded sampled tolerance ranges' if tolerance_summary else ''))
        self.input_label.set_text(f'Selected input: θ={metadata["input_theta_degrees"]:g}°, '
                                  f'φ={metadata["input_phi_degrees"]:g}°. '
                                  f'Storage: {metadata["storage_duration_us"]:g} µs.')
        tolerance_note = (f"{tolerance_summary['point_count']} fixed-error scenarios at this sweep point; "
                          "shading shows sampled extrema, not guaranteed bounds. 3D columns are nominal."
                          if tolerance_summary else 'All configured tolerance bounds are zero; nominal results shown.')
        self.footer.set_text('Drag the sweep slider or use Previous / Next to select a simulated point.\n'
                             + tolerance_note)
        self.slider.valtext.set_text(f'{row[parameter]:.10g} {self.report["unit"]}')
        self.index = index
        self.figure.canvas.draw_idle()
