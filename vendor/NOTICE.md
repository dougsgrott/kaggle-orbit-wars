# Vendored third-party code — "The Producer" (slawekbiel)

`vendor/orbit_lite/` and `vendor/producer_runtime.py` are **"The Producer"** by **slawekbiel**, a public
Kaggle *Orbit Wars* submission (notebook `the-producer-agent`), vendored here **UNMODIFIED** as the base for
our agent.

**This is not our work.** All credit to the original author. It is a pure single-turn greedy flow-diff
planner (torch); see [wiki/producer_analysis.md](../wiki/producer_analysis.md) for the breakdown.

The author's notebook asks forkers to *"include a note describing your contributions … a genuine improvement
rather than a small LB tweak."* We honor that: this directory is the unmodified base; **our contributions are
layered on top in separate, clearly-described changes** (tracked in `wiki/measured_log.md` and the issue
docs), never silently entangled with the vendored code.

- Source: `converted_scripts/producer-orbit-wars/` (the corpus copy this was vendored from).
- Requires `torch` (the only non-stdlib dependency); validated on the public LB (sub_07 = 1222).
