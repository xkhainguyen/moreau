import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

MOREAU_COLORS = {
    "primary": "#2563EB",       # Blue
    "primary_light": "#60A5FA",
    "primary_dark": "#1D4ED8",
    "accent": "#F97316",        # Orange
    "accent_light": "#FB923C",
    "success": "#10B981",       # Green
    "danger": "#EF4444",        # Red
    "neutral": "#6B7280",       # Gray
    "neutral_light": "#D1D5DB",
    "bg_dark": "#0F172A",       # Slate-900
    "bg_medium": "#1E293B",     # Slate-800
    "text": "#F8FAFC",          # Slate-50
    "text_secondary": "#94A3B8", # Slate-400
}

# Qualitative palette for multi-line plots
MOREAU_PALETTE = [
    "#2563EB", "#F97316", "#10B981", "#EF4444", "#8B5CF6",
    "#EC4899", "#14B8A6", "#F59E0B", "#6366F1", "#84CC16",
]


def set_moreau_style():
    """Apply Moreau's clean, high-DPI matplotlib theme."""
    plt.rcParams.update({
        # Figure
        "figure.facecolor": "white",
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "figure.figsize": (10, 6),

        # Axes
        "axes.facecolor": "white",
        "axes.edgecolor": "#E2E8F0",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelcolor": "#334155",
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.prop_cycle": mpl.cycler(color=MOREAU_PALETTE),

        # Grid
        "grid.color": "#F1F5F9",
        "grid.linewidth": 0.6,

        # Ticks
        "xtick.color": "#64748B",
        "ytick.color": "#64748B",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,

        # Legend
        "legend.frameon": False,
        "legend.fontsize": 10,

        # Font
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 11,

        # Lines
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
    })


def set_moreau_dark_style():
    """Apply dark-background theme for dramatic visualizations."""
    set_moreau_style()
    plt.rcParams.update({
        "figure.facecolor": MOREAU_COLORS["bg_dark"],
        "axes.facecolor": MOREAU_COLORS["bg_dark"],
        "axes.edgecolor": "#334155",
        "axes.labelcolor": MOREAU_COLORS["text"],
        "axes.titlecolor": MOREAU_COLORS["text"],
        "xtick.color": MOREAU_COLORS["text_secondary"],
        "ytick.color": MOREAU_COLORS["text_secondary"],
        "grid.color": "#1E293B",
        "text.color": MOREAU_COLORS["text"],
    })


def color_gradient(n, cmap_name="viridis"):
    """Return n colors sampled from a colormap."""
    cmap = plt.get_cmap(cmap_name)
    return [cmap(i / max(n - 1, 1)) for i in range(n)]


def moreau_gradient(n):
    """Return n colors from Moreau's blue-to-orange gradient."""
    cmap = LinearSegmentedColormap.from_list(
        "moreau", [MOREAU_COLORS["primary"], MOREAU_COLORS["accent"]], N=n
    )
    return [cmap(i / max(n - 1, 1)) for i in range(n)]


def annotate_speedup(ax, baseline, moreau_time, x=0.95, y=0.95):
    """Add a styled speedup annotation to an axes."""
    speedup = baseline / moreau_time
    ax.annotate(
        f"{speedup:.1f}x faster",
        xy=(x, y),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=14,
        fontweight="bold",
        color=MOREAU_COLORS["success"],
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="white",
            edgecolor=MOREAU_COLORS["success"],
            alpha=0.9,
        ),
    )
