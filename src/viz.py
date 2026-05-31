"""
Minimal tactical visualisation.

Dark-theme helpers + a single draw_board() for inspecting any obs (live game
or replay step). Lifted from clean_scripts/orbit_wars_advanced_agent_target_1608_6.py
(apply_dark_theme_fig / apply_dark_theme_ax) plus a stripped-down board plot.

Heavier dashboards (MCTS frontier, opponent-model heatmaps) are NOT included —
add them when we actually have an agent producing those internal states.
"""

from __future__ import annotations

import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from .utils import (
    CENTER_X, CENTER_Y, BOARD_SIZE, SUN_RADIUS, SUN_SAFETY,
    predict_planet_position,
)

BG_DARK    = "#09090E"
BG_PANEL   = "#11111B"
GRID_COL   = "#1E1E2E"
BORDER_COL = "#313244"

# Player colors: -1 = neutral, 0..3 = players
PCOLORS = {-1: "#5A5E6B", 0: "#00F0FF", 1: "#FF007F", 2: "#00FF66", 3: "#FF9900"}


def apply_dark_theme_fig(fig, title: str = ""):
    fig.patch.set_facecolor(BG_DARK)
    if title:
        fig.suptitle(title, color="white", fontsize=16, fontweight="bold", y=0.98)
    return fig


def apply_dark_theme_ax(ax, title: str = "", x_label: str = "", y_label: str = ""):
    ax.set_facecolor(BG_PANEL)
    ax.set_title(title, color="#C9CCDB", fontweight="bold", fontsize=11, pad=8)
    ax.set_xlabel(x_label, color="#A6ADC8", fontsize=9)
    ax.set_ylabel(y_label, color="#A6ADC8", fontsize=9)
    ax.tick_params(colors="#6C7086", labelsize=8)
    ax.grid(color=GRID_COL, linewidth=0.5, alpha=0.8)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER_COL)
        spine.set_linewidth(1.0)
    return ax


def draw_board(obs, ax=None, title: str = "", show_fleets: bool = True):
    """Render planets, fleets, and the sun for a single obs.

    obs is the per-player observation dict from env.steps[t][p].observation.
    Pass ax=None to make a new figure; pass an Axes to draw inside it.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
        apply_dark_theme_fig(fig)
    apply_dark_theme_ax(ax, title=title)
    ax.set_xlim(0, BOARD_SIZE)
    ax.set_ylim(0, BOARD_SIZE)
    ax.set_aspect("equal")

    # Sun + safety buffer
    sun = mpatches.Circle((CENTER_X, CENTER_Y), SUN_RADIUS,
                          color="#FFD27F", alpha=0.9, zorder=2)
    halo = mpatches.Circle((CENTER_X, CENTER_Y), SUN_RADIUS + SUN_SAFETY,
                           edgecolor="#FFD27F", facecolor="none",
                           linestyle="--", alpha=0.4, zorder=2)
    ax.add_patch(sun)
    ax.add_patch(halo)

    planets = obs["planets"] if isinstance(obs, dict) else obs.planets
    for p in planets:
        pid, owner, px, py, pr, ships, prod = p
        color = PCOLORS.get(int(owner), "#FFFFFF")
        ax.add_patch(mpatches.Circle((float(px), float(py)), float(pr),
                                     facecolor=color, edgecolor="white",
                                     linewidth=0.5, alpha=0.85, zorder=3))
        ax.text(float(px), float(py), f"{int(ships)}",
                color="white", ha="center", va="center",
                fontsize=7, zorder=4)

    if show_fleets:
        fleets = obs["fleets"] if isinstance(obs, dict) else obs.fleets
        for f in fleets:
            fid, owner, fx, fy, vx, vy, ships = f
            color = PCOLORS.get(int(owner), "#FFFFFF")
            ax.scatter([float(fx)], [float(fy)],
                       s=20 + int(ships) / 5, c=color,
                       marker="^", alpha=0.9, zorder=5,
                       edgecolors="white", linewidths=0.3)

    return ax
