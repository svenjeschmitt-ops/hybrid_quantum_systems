"""Structured Tk editors for optional model settings; JSON stays at the boundary."""
import json
import math
import tkinter as tk
from tkinter import ttk, messagebox

TITLES = {
    'RYDBERG_MANIFOLD': 'Additional Rydberg levels',
}


BUTTON_LABELS = {
    'RYDBERG_MANIFOLD': 'Rydberg levels…',
}


def example_model(key, config):
    """Editable illustrative values; called with validated current panel settings."""
    from multilevel import DEFAULT_MANIFOLD
    if key == 'RYDBERG_MANIFOLD':
        return json.loads(DEFAULT_MANIFOLD)
    raise KeyError(key)


def number(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError('Enter finite numbers.')
    return result


def unique_mapping(rows, key_columns, value_column):
    result = {}
    for row in rows:
        key = tuple(row[i] for i in key_columns)
        if key in result:
            raise ValueError('Duplicate entry: ' + ', '.join(map(str, key)))
        result[key] = row[value_column]
    return result


class RowTable:
    """Add/edit/remove rows using labeled entry fields, with optional TSV paste."""
    def __init__(self, parent, title, columns, rows=(), hint=''):
        self.columns = columns  # (caption, converter, choices)
        box = ttk.LabelFrame(parent, text=title, padding=8)
        box.pack(fill='both', expand=True, pady=6)
        if hint:
            ttk.Label(box, text=hint, wraplength=800).pack(anchor='w', pady=(0, 5))
        holder = ttk.Frame(box)
        holder.pack(fill='both', expand=True)
        self.tree = ttk.Treeview(holder, columns=list(range(len(columns))), show='headings', height=5)
        for i, (caption, _, _) in enumerate(columns):
            self.tree.heading(i, text=caption)
            self.tree.column(i, width=145, minwidth=90, stretch=True)
        self.tree.grid(row=0, column=0, sticky='nsew')
        ybar = ttk.Scrollbar(holder, orient='vertical', command=self.tree.yview)
        ybar.grid(row=0, column=1, sticky='ns')
        xbar = ttk.Scrollbar(holder, orient='horizontal', command=self.tree.xview)
        xbar.grid(row=1, column=0, sticky='ew')
        self.tree.configure(yscrollcommand=ybar.set, xscrollcommand=xbar.set)
        holder.columnconfigure(0, weight=1)
        holder.rowconfigure(0, weight=1)
        buttons = ttk.Frame(box)
        buttons.pack(fill='x', pady=(5, 0))
        for caption, command in [('Add', lambda: self.edit()), ('Edit', self.edit_selected),
                                 ('Remove', self.remove), ('Paste rows', self.paste)]:
            ttk.Button(buttons, text=caption, command=command).pack(side='left', padx=(0, 6))
        self.tree.bind('<Double-1>', lambda _: self.edit_selected())
        for row in rows:
            self.tree.insert('', 'end', values=row)

    def converted(self, values):
        result = []
        if len(values) != len(self.columns):
            raise ValueError(f'Expected {len(self.columns)} columns, separated by tabs.')
        for value, (caption, convert, choices) in zip(values, self.columns):
            try:
                value = convert(str(value).strip())
                if choices and value not in choices:
                    raise ValueError('Choose ' + ', '.join(map(str, choices)))
                if value == '':
                    raise ValueError('A value is required.')
            except (ValueError, TypeError) as exc:
                raise ValueError(f'{caption}: {exc}') from None
            result.append(value)
        return result

    def rows(self):
        return [self.converted(self.tree.item(item, 'values')) for item in self.tree.get_children()]

    def remove(self):
        for item in self.tree.selection():
            self.tree.delete(item)

    def paste(self):
        try:
            text = self.tree.clipboard_get()
            rows = [self.converted(line.split('\t')) for line in text.splitlines() if line.strip()]
            for row in rows:
                self.tree.insert('', 'end', values=row)
        except (ValueError, tk.TclError) as exc:
            messagebox.showerror('Could not paste rows', str(exc), parent=self.tree)

    def edit_selected(self):
        selected = self.tree.selection()
        if selected:
            self.edit(selected[0])

    def edit(self, item=None):
        dialog = tk.Toplevel(self.tree)
        dialog.title('Edit row' if item else 'Add row')
        dialog.transient(self.tree.winfo_toplevel())
        body = ttk.Frame(dialog, padding=14)
        body.pack(fill='both', expand=True)
        values = self.tree.item(item, 'values') if item else [''] * len(self.columns)
        variables = []
        for i, ((caption, _, choices), value) in enumerate(zip(self.columns, values)):
            ttk.Label(body, text=caption).grid(row=i, column=0, sticky='w', padx=(0, 12), pady=5)
            var = tk.StringVar(value=value)
            variables.append(var)
            widget = (ttk.Combobox(body, textvariable=var, values=choices, state='readonly')
                      if choices else ttk.Entry(body, textvariable=var, width=30))
            widget.grid(row=i, column=1, sticky='ew', pady=5)
        def save():
            try:
                row = self.converted([var.get() for var in variables])
                if item:
                    self.tree.item(item, values=row)
                else:
                    self.tree.insert('', 'end', values=row)
                dialog.destroy()
            except ValueError as exc:
                messagebox.showerror('Check row', str(exc), parent=dialog)
        actions = ttk.Frame(body)
        actions.grid(row=len(variables), column=0, columnspan=2, sticky='e', pady=(12, 0))
        ttk.Button(actions, text='Cancel', command=dialog.destroy).pack(side='left', padx=6)
        ttk.Button(actions, text='Save row', command=save).pack(side='left')
        dialog.protocol('WM_DELETE_WINDOW', dialog.destroy)
        dialog.grab_set()
        dialog.wait_window()
        if self.tree.winfo_exists():
            self.tree.winfo_toplevel().grab_set()


def col(caption, convert=str, choices=()):
    return caption, convert, choices


class ModelEditor:
    def __init__(self, parent, key, value, apply, defaults=None):
        self.key, self.apply = key, apply
        self.window = tk.Toplevel(parent)
        self.window.title(TITLES[key])
        self.window.transient(parent)
        self.window.geometry('980x720')
        self.window.minsize(650, 450)
        outer = ttk.Frame(self.window, padding=12)
        outer.pack(fill='both', expand=True)
        self.enabled = tk.BooleanVar(value=bool(value))
        if key != 'RYDBERG_MANIFOLD':
            ttk.Checkbutton(outer, text='Enable ' + TITLES[key].lower(), variable=self.enabled,
                            command=self.toggle).pack(anchor='w', pady=(0, 8))
        self.tabs = ttk.Notebook(outer)
        self.tabs.pack(fill='both', expand=True)
        self.defaults = defaults
        if defaults is not None:
            ttk.Label(outer, text='Example values are illustrative. Review before enabling or applying.',
                      wraplength=900).pack(fill='x', pady=4)
        getattr(self, 'build_' + key.lower())(value if value or key == 'RYDBERG_MANIFOLD' else (defaults if defaults is not None else value))
        self.error = tk.StringVar()
        ttk.Label(outer, textvariable=self.error, wraplength=900, foreground='#a02020').pack(fill='x', pady=6)
        actions = ttk.Frame(outer)
        actions.pack(fill='x')
        if defaults is not None:
            ttk.Button(actions, text='Load example values', command=self.load_defaults).pack(side='left')
        ttk.Button(actions, text='Cancel', command=self.window.destroy).pack(side='right', padx=6)
        ttk.Button(actions, text='Apply', command=self.save).pack(side='right')
        self.toggle()
        self.window.grab_set()

    def load_defaults(self):
        for tab in self.tabs.tabs():
            self.tabs.nametowidget(tab).destroy()
        getattr(self, 'build_' + self.key.lower())(self.defaults)
        self.error.set('')
        self.toggle()

    def page(self, title, hint=''):
        page = ttk.Frame(self.tabs, padding=10)
        self.tabs.add(page, text=title)
        if hint:
            ttk.Label(page, text=hint, wraplength=850).pack(anchor='w', pady=(0, 8))
        return page

    def field(self, parent, title, value='', choices=()):
        row = ttk.Frame(parent)
        row.pack(fill='x', pady=4)
        ttk.Label(row, text=title, width=40).pack(side='left')
        var = tk.StringVar(value=value)
        widget = (ttk.Combobox(row, textvariable=var, values=choices, state='readonly')
                  if choices else ttk.Entry(row, textvariable=var))
        widget.pack(side='left', fill='x', expand=True)
        return var

    def toggle(self):
        active = self.key == 'RYDBERG_MANIFOLD' or self.enabled.get()
        for tab in self.tabs.tabs():
            self.tabs.tab(tab, state='normal' if active else 'disabled')

    def save(self):
        try:
            value = self.collect() if self.key == 'RYDBERG_MANIFOLD' or self.enabled.get() else {}
            self.apply(value)
        except (ValueError, TypeError, KeyError) as exc:
            self.error.set(str(exc))
            return
        self.window.destroy()

    def build_rydberg_manifold(self, value):
        page = self.page('Levels', 'Branch 0: r0-like; branch 1: r1-like. Offsets are relative to that branch at zero field. '
                         'Remove all rows to use only the built-in levels. Paste rows in the displayed column order.')
        keys = ('label', 'branch', 'offset_MHz', 'stark_MHz_per_Vcm2', 'lifetime_us', 'optical_relative')
        levels = RowTable(page, 'Extra atomic levels', [col('Label'), col('Branch', int, (0, 1)),
                          col('Offset (MHz)', number), col('Stark (MHz/(V/cm)²)', number),
                          col('Lifetime (μs)', number), col('Optical amplitude', number)],
                          [[level[k] for k in keys] for level in value])
        page = self.page('Cavity links', 'Only upper-branch levels have cavity links. Lower states may be r0, wrong, or a branch-0 extra level.')
        links = RowTable(page, 'Relative cavity amplitudes', [col('Upper level'), col('Lower level'), col('Signed amplitude', number)],
                         [(x['label'], source, amp) for x in value for source, amp in x['cavity_from'].items()])
        def collect():
            result = [dict(zip(keys, row), cavity_from={}) for row in levels.rows()]
            by_name = {x['label']: x for x in result}
            for (upper, lower), amplitude in unique_mapping(links.rows(), (0, 1), 2).items():
                if upper not in by_name:
                    raise ValueError(f'Cavity link references undefined extra level: {upper}')
                by_name[upper]['cavity_from'][lower] = amplitude
            return result
        self.collect = collect


def add_model_control(parent, row, key, variable, entries, validate):
    """Keep serialized values compatible with existing load/save and preview traces."""
    cell = ttk.Frame(parent)
    cell.grid(row=row, column=1, columnspan=2, sticky='ew', pady=3)
    summary = tk.StringVar()
    def refresh(*_):
        try:
            value = json.loads(variable.get())
            if key == 'RYDBERG_MANIFOLD':
                summary.set(f'Active: {len(value)} extra level(s)' if value else 'Built-in levels only')
            else:
                summary.set('Active: additional channels' if any(value.values()) else 'Off: standard lifetimes')
        except (ValueError, KeyError, TypeError):
            summary.set('Check configuration')
    def apply(value):
        candidate = {name: var.get() for name, var in entries.items()}
        candidate[key] = json.dumps(value)
        checked = validate(candidate)
        variable.set(checked[key])
    def open_editor():
        try:
            value = json.loads(variable.get())
            config = validate({name: var.get() for name, var in entries.items()})
            ModelEditor(parent.winfo_toplevel(), key, value, apply, example_model(key, config))
        except ValueError as exc:
            messagebox.showerror('Check configuration', str(exc), parent=parent)
    button = ttk.Button(cell, text=BUTTON_LABELS[key], command=open_editor)
    button.pack(side='left')
    ttk.Label(cell, textvariable=summary, wraplength=180).pack(side='left', padx=8)
    variable.trace_add('write', refresh)
    refresh()
    return button
