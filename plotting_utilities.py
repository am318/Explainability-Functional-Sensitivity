import numpy as np

def _slug(text):
    return text.replace(" ", "_").replace("/", "_")


def parameter_location_metadata(model):
    labels = []
    spans = []
    cursor = 0
    for name, p in model.named_parameters():
        n = p.numel()
        labels.extend([name] * n)
        spans.append((cursor, cursor + n, name))
        cursor += n
    labels = np.asarray(labels, dtype=object)
    unique_labels = list(dict.fromkeys(labels.tolist()))
    # High-contrast qualitative palette (perceptually distinct even for many layers)
    _CONTRAST_PALETTE = [
        "#E63946",  # vivid red
        "#2196F3",  # vivid blue
        "#FF9800",  # vivid orange
        "#4CAF50",  # vivid green
        "#9C27B0",  # vivid purple
        "#00BCD4",  # vivid cyan
        "#FF5722",  # deep orange
        "#3F51B5",  # indigo
        "#CDDC39",  # lime
        "#F06292",  # pink
        "#26A69A",  # teal
        "#FFC107",  # amber
        "#5C6BC0",  # medium indigo
        "#66BB6A",  # medium green
        "#EF5350",  # medium red
        "#29B6F6",  # light blue
        "#AB47BC",  # medium purple
        "#FF7043",  # deep orange variant
        "#26C6DA",  # cyan variant
        "#D4E157",  # yellow-green
    ]
    colour_map = {lab: _CONTRAST_PALETTE[i % len(_CONTRAST_PALETTE)] for i, lab in enumerate(unique_labels)}
    colours = np.asarray([colour_map[lab] for lab in labels], dtype=object)
    return labels, colours, unique_labels, colour_map, spans


def add_parameter_location_legend(ax, unique_labels, colour_map, *, loc="best", max_labels=16):
    from matplotlib.patches import Patch
    shown = unique_labels[:max_labels]
    handles = [
        Patch(facecolor=colour_map[label], edgecolor="none", label=label)
        for label in shown
    ]
    if len(unique_labels) > max_labels:
        from matplotlib.lines import Line2D
        handles.append(Line2D([0], [0], linestyle="None", label=f"+{len(unique_labels) - max_labels} more"))
    ax.legend(handles=handles, frameon=True, framealpha=0.85, edgecolor="none",
              loc=loc, fontsize=7, ncol=max(1, len(shown) // 10))


def add_parameter_location_boundaries(ax, spans, *, axis="y", color="white", alpha=0.55):
    for _, stop, _ in spans[:-1]:
        boundary = stop - 0.5
        if axis == "y":
            ax.axhline(boundary, color=color, linewidth=0.45, alpha=alpha)
        else:
            ax.axvline(boundary, color=color, linewidth=0.45, alpha=alpha)