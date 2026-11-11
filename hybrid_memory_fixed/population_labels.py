"""Labels for individually resolved level populations, including saved sweep data."""


def resolved_population_labels(series):
    labels = {}
    for key in series:
        if key.startswith("transmon_level_"):
            index = int(key.removeprefix("transmon_level_"))
            if index == 1:  # Already displayed as Transmon e |1>.
                continue
            name = {0: "g", 2: "f", 3: "h"}.get(index)
            labels[key] = f"Transmon {name} |{index}>" if name else f"Transmon |{index}>"
        elif key.startswith("atom_neighbor_"):
            state = key.removeprefix("atom_neighbor_")
            labels[key] = {"wrong": "r0_wrong", "wrong_upper": "r1_wrong"}.get(state, state)
    return labels


def population_legend(axis):
    """One row of colour keys above the data and beneath the panel title."""
    # Keep the internal curve slot for tolerance-series indexing, but display
    # only the individual unwanted levels.
    for line in axis.lines:
        if line.get_label() == "Unwanted Rydberg total":
            line.set_visible(False)
            line.set_label("_nolegend_")
    handles, labels = axis.get_legend_handles_labels()
    if "r0_wrong" in labels and "r1_wrong" in labels:
        upper_index = labels.index("r1_wrong")
        upper_handle = handles.pop(upper_index)
        labels.pop(upper_index)
        position = labels.index("r0_wrong") + 1
        handles.insert(position, upper_handle)
        labels.insert(position, "r1_wrong")
    axis.set_title(axis.get_title(), pad=32)
    return axis.legend(handles, labels, ncol=max(1, len(labels)), fontsize=6.5,
                       loc="lower left", bbox_to_anchor=(0, 1.015, 1, .08),
                       mode="expand", borderaxespad=0, frameon=False,
                       handlelength=1.5, handletextpad=.4, columnspacing=.7)
