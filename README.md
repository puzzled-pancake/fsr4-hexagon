# FSR4 on a Phone NPU: The Retroid Pocket 6 DLSS Heist

Rise of the Tomb Raider asks Windows for DLSS upscaling. This project gives it
**AMD FSR4, quantized W8A8, running on the Hexagon NPU of an Android
handheld**: by intercepting the DLSS calls, shipping the frames to a native
ARM64 daemon, and writing the upscaled result back into the game's output
texture. The game never knows. NVIDIA never knew.

The full story, methodology and measured results are in
**[WHITEPAPER.pdf](WHITEPAPER.pdf)** (energy parity with native rendering at
the light end, +22% fps; every config that undervolted the NPU lost).

```
RotTR (Wine/Box64/DXVK)                     Android host
┌─────────────────────────┐    TCP loopback  ┌──────────────────────────────┐
│ game ─> nvngx.dll (us)  │ ───────────────> │ daemon (us)                  │
│  stages color+MVs       │  R11G11B10F raw  │  net ─ GL shaders ─ NPU thr  │
│  d3d12.dll (us) forces  │ <─────────────── │  QNN/HTP: FSR4 W8A8 graph    │
│  DLSS=1 in registry     │  RGBA8 1080p     │  GL: feature/postA/rcas      │
└─────────────────────────┘                  └──────────────────────────────┘
```

## Headline numbers (1080p output, uncapped)

| Preset | Native | FSR4 on NPU | Result |
|---|---|---|---|
| Lowest | 29.5 fps @ 9.4 W (3.14 fr/J) | 36.1 fps @ 11.5 W (3.14 fr/J) | +22% fps, energy/frame parity within noise (±6%) |
| Medium | 18.3 fps @ 8.6 W (2.14 fr/J) | 21.0 fps @ 10.7 W (1.96 fr/J) | +15% fps, 9% premium (near the error band, suggestive) |
| High | 15.5 fps @ 8.6 W (1.80 fr/J) | 17.7 fps @ 10.3 W (1.72 fr/J) | +14% fps, 4% premium (inside the error band) |

![FSR4 vs native: fps against watts](figures/fig1_fps_vs_power.png)

And the counterintuitive one: dropping the NPU to the 0x30 floor voltage
corner made the *whole system* draw more power (2.33 vs 3.14 fr/J).
Duration beats voltage.

## Repo map

| Path | What |
|---|---|
| `WHITEPAPER.pdf` / `.md` + `figures/` | the write-up (charts, in-game screenshots + PDF tooling) |
| `proxy/` | `nvngx.dll` proxy sources + build (1080p and 720p arms) |
| `injector/` | `d3d12.dll` registry injector: the game's DLSS toggle never sticks on a fresh container, so we force-write it |
| `daemon/` | native daemon (live_daemon + qnn_service), GLSL compute shaders, production launch script |
| `model/` | regenerate the NPU context binary from your own FidelityFX SDK (no weights shipped) |
| `tools/` | device deploy script, on-device power sampler, energy carve tool |
| `docs/REPRODUCING.md` | end-to-end build/deploy/run guide |
| `docs/KNOBS.md` | every daemon knob: NPU voltage corner, thread pins, pipeline switches |
| `docs/BENCH_METHOD.md` | how the energy numbers were measured |
| `docs/BUILDS.md` | md5 manifest of the exact measured binaries |

## Prerequisites

Retroid Pocket 6 (or similar Snapdragon 8 Gen 2 device) with the
[npu-unlock](https://github.com/puzzled-pancake/rp6-npu-unlock) applied, the
QAIRT 2.50 SDK, a FidelityFX SDK 2.0.0 checkout, mingw-w64 gcc, Android NDK
r28, your own Steam copy of RotTR, and GameNative v1.2.0. Full list with
versions: `docs/REPRODUCING.md`. Boundaries: `THIRD_PARTY.md`.

## Quickstart

```sh
# 1. rebuild the NPU context binary from your SDK (model/README.md)
# 2. build + deploy the daemon          (daemon/build.sh, tools/android_push.sh)
# 3. build the DLLs                     (proxy/build.sh, injector/build.sh)
# 4. drop nvngx.dll + d3d12.dll + nvapi64.dll in the game dir, dxvk.conf per REPRODUCING
# 5. on device: sh daemon/launch_production.sh   # knobs: docs/KNOBS.md
```


## AI-coded notice

This project was developed with heavy use of AI coding, using the open-source
GLM models 5.3 and 5.3 Flash through the ZCode agent harness, with human
direction and review throughout.

License: MIT for everything original here (see `LICENSE`, `THIRD_PARTY.md`).
Contact: [placelessness@protonmail.com](mailto:placelessness@protonmail.com), [github.com/puzzled-pancake](https://github.com/puzzled-pancake); support via [GitHub Sponsors](https://github.com/sponsors/puzzled-pancake) or [Ko-fi](https://ko-fi.com/ratherpuzzled).
