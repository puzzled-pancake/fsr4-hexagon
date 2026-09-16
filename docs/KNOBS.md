# Daemon knobs (FSR4_* environment variables)

All knobs are environment variables read by the daemon at startup. The
production defaults live in `daemon/launch_production.sh`, and every one of
them can be overridden from the command line without editing anything:

```sh
HTPOWER=0x30 GL_CPU=5 NPU_CPU=1 sh launch_production.sh
# or equivalently:
FSR4_HTPOWER=0x30 FSR4_GL_CPU=5 FSR4_NPU_CPU=1 sh launch_production.sh
```

## NPU power profile (the voltage corner)

`FSR4_HTPOWER`: Hexagon DCVS voltage corner voted through QNN's HTP perf
infrastructure. Lower corner = lower NPU voltage = slower inference.

| Corner | Name | NPU inference | Measured system effect (Lowest preset, uncapped) |
|---|---|---|---|
| `0x30` | SVS2 (floor) | 16.9 ms | 12.2 W, 28.4 fps, **2.33 fr/J**: draws MORE total power than 0x50 |
| `0x50` | SVS | 10.4 ms | 11.5 W, 36.1 fps, **3.14 fr/J**: production pick |
| `0x60` | NOM | 8.6 ms | fastest NPU, highest NPU-side draw | 

Duration beats voltage: the extra 6.5 ms per inference at 0x30 stretches
every frame, and the waiting threads + held-up clocks burn more than the
voltage cut saves. Both 0x30 configs measured (A510 pins and A710 pins)
converged on exactly 2.33 fr/J. Unless you are chasing minimum NPU-side
watts for a capped workload, leave it at 0x50.

## Thread placement

Kalama topology: cpu0-2 = A510 @ 2.0 GHz (little), cpu3-6 = A710 @ 2.8 GHz
(mid), cpu7 = X3 @ 3.2 GHz (prime). GameNative pins the WoW64 game to
cores 4-7, which is why the daemon defaults below avoid that set.

| Knob | Code default | Production | Thread |
|---|---|---|---|
| `FSR4_GL_CPU` | 7 (X3) | **3** (idle A710) | GL: feature + post + rcas shaders |
| `FSR4_NET_CPU` | 5 | **0** (A510) | socket receive + NEON convert |
| `FSR4_SEND_CPU` | 6 | **1** (A510) | socket send |
| `FSR4_NPU_CPU` | 4 | **2** (A510) | graphExecute only |
| `FSR4_AFFINITY` | enabled | 1 | set `0` to disable all pinning |

The X3 story in one line: with the GL thread on cpu7 next to the game's
main thread, GPU fence wait was 21.6 ms; moved to the idle cpu3, 8.0 ms.
Putting everything on the A510s (the all-little-cores config) strangles the pipeline: 26 fps at
2.33 fr/J, dominated on every axis.

## Pipeline structure

- `FSR4_FEAT` / `FSR4_POST`: feature/post compute shader filenames (default `features_real.comp`, `post_real.comp`)
- `FSR4_SPLIT=1`: dedicated NPU thread; postA(N-1) overlaps graphExecute(N). Requires RCAS
- `FSR4_RCAS=0.5`: FSR sharpen amount; `0` disables (and disables SPLIT)
- `FSR4_RCASSRC`: rcas input source override (SSBO10 direct reader is the shipped path)
- `FSR4_FUSE=1`: fused feature/first-conv/quant dispatch
- `FSR4_RAWLR=1`: raw R11G11B10F color path (no shader-side decode)
- `FSR4_MVNORM=1`: normalized motion vectors
- `FSR4_SKIP_SYN=1`: skip synthesis working-set allocation (bank replay only)
- `FSR4_AHBCOPY=1`: zero-copy AHardwareBuffer read bridge for the NPU output
- `FSR4_AHBIN` / `FSR4_DMABUF` / `FSR4_EGLIN`: input-bridge variants (write path: slower, off)
- `FSR4_AHBSYNC`: explicit buffer sync on the bridge (0 = implicit)
- `FSR4_U8OUT`: uint8 output tensor path (default on for the shipped graph)
- `FSR4_HIST` / `FSR4_HIST_TEX` / `FSR4_HIST_CLAMP` / `FSR4_NO_HISTUP`: history handling
- `FSR4_NEONCV=1`: NEON color convert in the net thread
- `FSR4_AUTOEXP`: exposure auto-follow (game runs autoexposure off)

## Temporal quality

- `FSR4_BLENDFLOOR=0.0`: minimum blend weight floor (0.6 was the old tuned value; 0.0 shipped)
- `FSR4_CUTFRAC=0.35`: scene-cut detector threshold (fraction of screen moving)
- `FSR4_CUTMV=0`: motion-vector-magnitude cut trigger (disabled)
- `FSR4_BFTAU`: dist-gated blend floor calibration (scaffolding, default off)
- `FSR4_MVDZ`: MV dead-zone handling

## Diagnostics (leave off in production)

`FSR4_SENTINEL=64` (stale-frame watchdog: 64 texels every 64th frame: keep
this one ON, it has never fired and that is the point), `FSR4_RXDIAG`,
`FSR4_MVLOG`, `FSR4_JCOMP/_JSIGN/_JZERO`, `FSR4_E4B`, `FSR4_SPIN_MS`,
`FSR4_RT`, `FSR4_OUT16_DUMP`, `FSR4_RGB_DUMP_DIR/_FRAMES`, `FSR4_LRA`,
`FSR4_NOPREMAP`. Several of these add per-frame I/O; the logging-cost
lesson is baked into the defaults.
