# How the energy numbers were measured

Short version of the whitepaper's methodology section, with the operational
details the paper doesn't have room for.

## Rig

- Power: battery fuel gauge, `/sys/class/power_supply/battery/{voltage_now,
  current_now}`, sampled on-device at ~8.5 Hz (`tools/arm_trigger.sh`),
  trapezoid-integrated over the bench window. The gauge reports discharge as
  NEGATIVE current; magnitude is what you want. The gauge itself updates
  slower than either sampling rate, so more Hz buys little.
- **Unplug the charger.** Charging current flows through the same gauge.
- The fan runs off the same battery: its watts are inside every number.
- Frames: GameNative `powercontrol` metrics jsonl (2 Hz), integrating the
  `fps` field over the exact window. `totalFrameCount` is a ~2 s sliding
  window, NOT an odometer: never integrate it. The file grows in place;
  re-pull it after the bench finishes.
- Telemetry also carries `cpuTempC`/`gpuTempC` (during a capped 30 fps bench
  the RP6 sat at ~79-83 C cpu, 76-82 C gpu, fan doing its job). GPU peaked at 83C with the author watching live; no mid-run throttling. Heat-soak past the windows still uncharacterized.

## Carving the window

The capture contains menu, loading and bench. Only bench counts:

- start: where power leaves the loading-screen baseline and ramps;
- end: first sample under 7.5 W after the ramp (or the dead-flat return to a
  locked 30.0 fps when the menu draws the same watts as the bench tail);
- loading screens are excluded on purpose (cap-held; pollute energy/frame
  differently per config).

`tools/carve_energy.py` attempts this automatically; when the capture shape
defeats it (menus draw 8-10 W on this stack, so simple thresholds misfire)
it prints a 10 s power trace: pick `--start/--end`, rerun, and sanity-gate:

1. frame count roughly 2000-2700 frames
   (capped 30 fps arms);
2. mean fps inside the band sibling runs established (±1.6 fps observed
   same-day repeat spread).

**When the power trace and the fps trace disagree about the end, the fps
side wins.**

## Error budget

One honest repeat (native 1080p, same day, twice): power agreed to 0.1%,
mean fps drifted 1.6 fps. So: treat every mJ/frame as ±6%, dominated by the
fps side: a statement about 2 Hz telemetry stability, not the power rig.
Tighter numbers need harder frame counting, not a better gauge.

## Known traps

- The game menu runs the full pipeline at idle: 8-10 W "resting". Measure
  idle with the game closed, not paused.
- Menu idles at fps=0.00 in telemetry and the game-close tail reports 0-14fps: never integrate those spans.
- Heavy scenes are CPU/Box64-bound (~18-21 fps native too); the upscaler
  buys nothing there. Compare like scenes.
- Capped vs uncapped arms used different NPU corners (0x30 vs 0x50) —
  neighbors, not one continuous dataset.
