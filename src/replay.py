"""
Game-trace debugger (T1) — a turn-by-turn replay viewer over a recorded Arena
episode, with fleet death-cause and missed-Shot overlays.

It works from `arena.record_episode(...)`'s `EpisodeTrace` — the episode record,
not a live game — so nothing is re-run. For each turn it draws the board (via
`viz.draw_board`) and overlays every fleet that vanished into the next turn,
labelled with *why* it died (combat / out-of-bounds / sun). A fleet that died to
the sun or off-board never reached a planet, so it's a **missed Shot** (a wasted
launch) and is highlighted differently — the fastest way to confirm by eye that
the verified aim solver actually lands in live games.

The death cause is re-derived from the same physics the interpreter uses
(`utils.fleet_speed` for the step, `utils.segment_intersects_circle` for the
planet sweep, `utils.segment_hits_sun` at the interpreter's exact radius), in the
interpreter's own precedence: planet collision first, then off-board, then sun.
This is a developer tool, validated by eye; only the pure inference is unit-tested.

CLI:  python -m src.replay        # record a short game, write a GIF + summary

Public API
----------
    infer_death_cause(fleet, planets)   -> (cause, x, y)        (pure)
    fleet_deaths(prev_frame, curr_frame) -> list[FleetDeath]    (pure)
    FleetDeath                          (.cause/.x/.y/.missed/...)
    draw_replay_frame(trace, t, ax=None)        -> Axes
    render_replay(trace, out, turns=None)       -> list[Path]
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from .utils import (
    BOARD_SIZE,
    fleet_speed,
    segment_hits_sun,
    segment_intersects_circle,
)

# Death causes.
COMBAT = "combat"
OUT_OF_BOUNDS = "out_of_bounds"
SUN = "sun"


@dataclass(frozen=True)
class FleetDeath:
    """A fleet that was present one turn and gone the next, with the inferred
    cause and an approximate death position."""

    fleet_id: int
    owner: int
    cause: str  # COMBAT | OUT_OF_BOUNDS | SUN
    x: float
    y: float
    ships: int

    @property
    def missed(self) -> bool:
        """True for a wasted launch: it died to the sun or off-board, i.e. never
        reached a planet — a **missed Shot**."""
        return self.cause in (SUN, OUT_OF_BOUNDS)


# ---------------------------------------------------------------------------
# Pure inference (no matplotlib, no env) — unit tested.
# ---------------------------------------------------------------------------


def _step_segment(fleet) -> Tuple[float, float, float, float]:
    """The (x1, y1, x2, y2) a fleet sweeps in one turn — straight line at its
    size-scaled speed, exactly as the interpreter moves it."""
    _, _, x, y, angle, _, ships = fleet
    speed = fleet_speed(int(ships))
    return x, y, x + math.cos(angle) * speed, y + math.sin(angle) * speed


def infer_death_cause(fleet, planets) -> Tuple[str, float, float]:
    """Why a vanished `fleet` died, plus an approximate death position.

    `planets` are the planets present the turn the fleet was last seen (their
    start-of-tick positions). Mirrors the interpreter's precedence: a path that
    grazes a planet is combat even if it would also have left the board or hit
    the sun; then off-board; then sun.
    """
    x1, y1, x2, y2 = _step_segment(fleet)

    # 1. Planet collision -> combat (capture, reinforce, or bounce all land).
    for p in planets:
        _, _, px, py, pr, _, _ = p
        if segment_intersects_circle(x1, y1, x2, y2, px, py, pr):
            return COMBAT, px, py

    # 2. Left the 100x100 board.
    if not (0.0 <= x2 <= BOARD_SIZE and 0.0 <= y2 <= BOARD_SIZE):
        return OUT_OF_BOUNDS, x2, y2

    # 3. Crossed the sun — use the interpreter's exact radius (no safety buffer).
    if segment_hits_sun(x1, y1, x2, y2, safety=0.0):
        return SUN, x2, y2

    # Fallback: swept up by a rotating planet/comet we didn't match above —
    # still an arrival at a planet, so attribute it to combat.
    return COMBAT, x2, y2


def fleet_deaths(prev_frame, curr_frame) -> List[FleetDeath]:
    """Every fleet in `prev_frame` that is gone in `curr_frame`, with its cause.

    Frames are `arena.Frame` (anything with `.fleets` and `.planets`)."""
    survivors = {f[0] for f in curr_frame.fleets}
    deaths: List[FleetDeath] = []
    for f in prev_frame.fleets:
        if f[0] in survivors:
            continue
        cause, dx, dy = infer_death_cause(f, prev_frame.planets)
        deaths.append(
            FleetDeath(
                fleet_id=int(f[0]),
                owner=int(f[1]),
                cause=cause,
                x=dx,
                y=dy,
                ships=int(f[6]),
            )
        )
    return deaths


def death_summary(trace) -> dict:
    """Episode-wide tally of fleet deaths by cause (+ a `missed` total)."""
    tally = {COMBAT: 0, OUT_OF_BOUNDS: 0, SUN: 0}
    for t in range(len(trace.frames) - 1):
        for d in fleet_deaths(trace.frames[t], trace.frames[t + 1]):
            tally[d.cause] += 1
    tally["missed"] = tally[SUN] + tally[OUT_OF_BOUNDS]
    return tally


# ---------------------------------------------------------------------------
# Rendering (matplotlib imported lazily so the pure inference stays light).
# ---------------------------------------------------------------------------


def draw_replay_frame(trace, t: int, ax=None, show_deaths: bool = True):
    """Draw turn `t` of `trace`: the board plus the fleet deaths that happen on
    the way to turn `t+1` (combat = hollow amber ring; missed = red ✕)."""
    import matplotlib.pyplot as plt

    from .viz import apply_dark_theme_fig, draw_board

    frame = trace.frames[t]
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
        apply_dark_theme_fig(fig)

    obs = {"planets": frame.planets, "fleets": frame.fleets}
    draw_board(obs, ax=ax, title=f"turn {frame.step}")

    if show_deaths and t + 1 < len(trace.frames):
        deaths = fleet_deaths(frame, trace.frames[t + 1])
        for d in deaths:
            if d.missed:
                ax.scatter([d.x], [d.y], marker="x", s=90, c="#FF3B3B",
                           linewidths=2.0, zorder=6)
                ax.text(d.x, min(d.y + 2.2, BOARD_SIZE), d.cause,
                        color="#FF3B3B", fontsize=6, ha="center", zorder=6)
            else:
                ax.scatter([d.x], [d.y], marker="o", s=70, facecolors="none",
                           edgecolors="#FFE08A", linewidths=1.4, zorder=6)
        n_missed = sum(1 for d in deaths if d.missed)
        ax.text(1.0, 1.5,
                f"deaths {len(deaths)}  (missed {n_missed})",
                color="#A6ADC8", fontsize=7, ha="left", va="bottom", zorder=6)
    return ax


def render_replay(
    trace, out, turns: Optional[Sequence[int]] = None, dpi: int = 110, fps: int = 6
) -> List[Path]:
    """Render `turns` of `trace` (default: all). If `out` ends in `.gif`, write a
    single animated GIF; otherwise treat `out` as a directory and write one
    `turn_NNNN.png` per turn. Returns the written path(s)."""
    import matplotlib.pyplot as plt

    from .viz import apply_dark_theme_fig

    if turns is None:
        turns = range(len(trace.frames))
    turns = list(turns)
    out = Path(out)

    def _render_one(t):
        fig, ax = plt.subplots(figsize=(7, 7))
        apply_dark_theme_fig(fig)
        draw_replay_frame(trace, t, ax=ax)
        fig.tight_layout()
        return fig

    if out.suffix == ".gif":
        from PIL import Image

        out.parent.mkdir(parents=True, exist_ok=True)
        images = []
        for t in turns:
            fig = _render_one(t)
            fig.canvas.draw()
            images.append(
                Image.frombytes(
                    "RGBA", fig.canvas.get_width_height(),
                    bytes(fig.canvas.buffer_rgba()),
                ).convert("P", palette=Image.ADAPTIVE)
            )
            plt.close(fig)
        images[0].save(
            out, save_all=True, append_images=images[1:],
            duration=int(1000 / fps), loop=0, optimize=True,
        )
        return [out]

    out.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for t in turns:
        fig = _render_one(t)
        path = out / f"turn_{t:04d}.png"
        fig.savefig(path, facecolor=fig.get_facecolor(), dpi=dpi)
        plt.close(fig)
        paths.append(path)
    return paths


# ---------------------------------------------------------------------------
# Demo: record a short game and write a GIF + a death summary.
#   python -m src.replay
# ---------------------------------------------------------------------------


def _demo() -> None:
    from .arena import EpisodeConfig, record_episode

    opp = Path(__file__).parent / "opponents"
    trace = record_episode(
        [str(opp / "weakest_first.py"), str(opp / "production_first.py")],
        EpisodeConfig(num_players=2, seed=2026, episode_steps=80),
    )

    out_dir = Path(__file__).resolve().parents[1] / "analysis" / "replay_demo"
    gif = out_dir / "episode.gif"
    render_replay(trace, gif)

    tally = death_summary(trace)
    print(f"Recorded {len(trace.frames)} turns (seed {trace.seed}).")
    print(
        f"Fleet deaths — combat {tally[COMBAT]}, "
        f"out_of_bounds {tally[OUT_OF_BOUNDS]}, sun {tally[SUN]}  "
        f"=> missed Shots {tally['missed']}"
    )
    print(f"Wrote replay GIF: {gif}")


if __name__ == "__main__":
    _demo()
