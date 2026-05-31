"""Per-game instrumentation for the aim solver and tactical layer.

Lifted from clean_scripts/orbit_wars_physics_helper_module.py:525.
"""

from __future__ import annotations


class PhysicsStats:
    __slots__ = (
        "aims_attempted", "aims_succeeded", "sun_blocked",
        "fleets_sent", "planets_captured",
    )

    def __init__(self):
        self.reset()

    def reset(self):
        self.aims_attempted   = 0
        self.aims_succeeded   = 0
        self.sun_blocked      = 0
        self.fleets_sent      = 0
        self.planets_captured = 0

    def record_aim(self, success: bool, sun_blocked: bool = False):
        self.aims_attempted += 1
        if success:
            self.aims_succeeded += 1
        if sun_blocked:
            self.sun_blocked += 1

    def record_fleet_sent(self, count: int = 1):
        self.fleets_sent += count

    def record_planet_captured(self, count: int = 1):
        self.planets_captured += count

    @property
    def aim_rate(self) -> float:
        return self.aims_succeeded / self.aims_attempted if self.aims_attempted else 0.0

    @property
    def hit_rate(self) -> float:
        return self.planets_captured / self.fleets_sent if self.fleets_sent else 0.0

    def summary(self) -> str:
        return (
            f"physics | aims={self.aims_attempted} ok={self.aims_succeeded} "
            f"({self.aim_rate * 100:.0f}%) sun_blocked={self.sun_blocked} | "
            f"fleets_sent={self.fleets_sent} captures={self.planets_captured} "
            f"hit_rate={self.hit_rate * 100:.0f}%"
        )
