# Reproducing the whole thing

Hardware/software this was built and measured on:

- Retroid Pocket 6 (Snapdragon 8 Gen 2, HTP v73), Android 13, bootloader
  unlocked, [rp6-npu-unlock](https://github.com/puzzled-pancake/rp6-npu-unlock)
  applied: after boot, `/sys/class/remoteproc/*/state` should read `running`
  ~3.4 s after power-on.
- QAIRT SDK **2.50.0.260828** (Qualcomm, free): headers + aarch64 runtime libs.
- FidelityFX SDK **2.0.0** checkout containing
  `Kits/FidelityFX/upscalers/fsr4/internal/shaders/fsr4_model_v07_i8_quality/`.
- mingw-w64 gcc (proxy + injector), Android NDK **r28.2** (daemon).
- Steam Rise of the Tomb Raider (appid 391220) inside **GameNative 1.2.1, versionCode 23** (paper says 1.2.0; measured device had
  self-updated to 1.2.1): Wine + Box64 + DXVK 2.7.1 + Turnip container.
  Box64/Wine exact build strings: run `box64 --version` / `wine --version`
  inside the container terminal (binaries live in app-private storage and
  aren't readable over plain adb shell).
- The exact binaries measured are pinned by md5 in `docs/BUILDS.md`.

## 1. NPU context binary

Follow `model/README.md` end to end. You end up with `v/bin_cfnhwc.bin`
(clip-free NHWC W8A8 graph, O:3-serialized). Verify md5 both ends after every
`adb push` (adb truncates the last filename character when pushing to a
directory: push explicit full paths; `tools/android_push.sh` does).

## 2. Daemon

```sh
cd daemon && sh build.sh          # NDK r28 + QAIRT headers; see script envs
adb push live_daemon /data/local/tmp/qnn/live_daemon
for s in shaders/*.comp; do adb push $s /data/local/tmp/qnn/$(basename $s); done  # FLAT dir + per-file: the daemon fopen()s bare filenames from its cwd, and glob pushes truncate names
adb push launch_production.sh /data/local/tmp/qnn/launch_production.sh
adb shell chmod +x /data/local/tmp/qnn/live_daemon   # adb push drops the exec bit every time
```

Push the context binary (`adb push v/bin_cfnhwc.bin /data/local/tmp/qnn/v/bin_cfnhwc.bin`) and
generate the gpu_data seeds: `python daemon/gen_seeds.py <fsr4_quality_weights.npz> <tmp>`
then push `pass0_wb.raw hist.raw rec_prev.raw` to `/data/local/tmp/qnn/gpu_data/`
(pass0_wb is md5-checked against the deployed contract; the daemon fatals without all three).

`tools/android_push.sh` stages the QAIRT runtime libs
(`libQnnHtp.so`, `libQnnHtpV73Stub.so`, `libQnnHtpV73Skel.so`, ...) to
`/data/local/tmp/qnn` from your `QNN_SDK_ROOT`. Do NOT bundle
`libcdsprpc.so`: the vendor's own copy in `/vendor` is the one the FastRPC
channel wants.

## 3. Game-side DLLs

```sh
cd proxy && sh build.sh           # nvngx_1080.dll (or nvngx_360.dll for 720p)
cd ../injector && sh build.sh     # d3d12.dll
```

## 4. Exactly how the author's machine was configured

This is the as-measured setup, not a generic suggestion. GameNative
**1.2.1 (versionCode 23)** on the RP6, RotTR Steam appid **391220**; the
per-game Wine home is `Z:\home\xuser-STEAM_391220\` inside the container.

**Files in the GAME directory** (game install dir, next to ROTTR.exe):

| File | What it is |
|---|---|
| `nvngx.dll` | the proxy build renamed (1080p arm; `nvngx_360.dll` for the 720p arm) |
| `d3d12.dll` | the registry injector build (at attach it force-writes nine HKCU Graphics values: DLSS=1, DLSS Previous=1, FXAA=0, plus ScreenEffects/FilmGrain/LensFlares/AmbientOcclusionQuality/AsyncCompute/HighPrecisionRT quality pins; set FSR4_DUMP_REG=1 to dump the pre-patch registry for diffing): the in-menu toggle reverts itself on a fresh registry) |
| `nvapi64.dll` | fakenvapi build (upstream; the 415 KB one) |
| `fsr4cap.ini` | live-mode config as shipped at `proxy/fsr4cap.ini`: `mode=live`, `live_port=48620`, `count=0`, plus `wire=3 mapn2=1 slots3=1 simdmv=1` which select the measured wire class (omit them and the proxy still runs, one class slower). `count=150` is for capture runs |

**DXVK GPU spoof**: file `Z:\home\xuser-STEAM_391220\.config\dxvk.conf`
(GameNative pins `DXVK_CONFIG_FILE`; on the older pre-reinstall container this
lived at `Z:\home\xuser\.config\dxvk.conf`). Exact deployed contents:

```
dxgi.customVendorId = 10de
dxgi.customDeviceId = 7687
dxgi.customDeviceDesc = "NVIDIA GeForce RTX 2080 Ti"
d3d9.customVendorId = 10de
d3d9.customDeviceId = 7687
d3d9.customDeviceDesc = "NVIDIA GeForce RTX 2080 Ti"
dxgi.hideNvidiaGpu = False
```

**Per-game environment** (GameNative's per-game settings, WINEDLLOVERRIDES
mechanism included):

```
PROTON_ENABLE_NVAPI=1
PROTON_FORCE_NVAPI=1
WINEDLLOVERRIDES=nvapi64=n,b;nvngx=n,b;nvofapi64=n,b;d3d12=n,b
```

**In-game settings as benched**: DX12 renderer ON (DLSS does not exist on
the DX11 path), display 1920x1080, DLSS **Quality** (960x540 internal for
the 1080p arm), FXAA off, then the graphics preset per arm (Lowest / Medium /
High). The **30 fps cap for Experiment 1 is GameNative's own per-game
fpsLimiter set to 30** in the app's game settings: not an in-game limiter.

**Order of operations that worked**: start the daemon first, then launch the
game; the injector patches the registry at DLL attach, so a game restart
applies it cleanly. For capture runs the proxy arms on a `START` marker file
(touch `START` next to the daemon, it self-deletes on arm; `rm START` +
re-touch to re-arm: live mode re-reads the ini without a game restart).

## 4a. 720p arm differences

Build the daemon with BOTH macro sets (`-DFSR4D_LW=640 -DFSR4D_LH=360 -DFSR4D_W=1280 -DFSR4D_H=720` for live_daemon.c AND `-DLR_W=640 -DLR_H=360 -DOS_W=1280 -DOS_H=720` for qnn_service.c; miss the second and the two TUs disagree on every dimension) and the proxy with `-DLWV=640u -DLHV=360u`; the context binary is the 640x360 rebuild per `model/README.md` step 6. On device, also copy `shaders/features_fused_360.comp` over `features_fused.comp` (the 360 context runs input scale 0.01512, needing its RCP_S 66.1376; keep 1/RCP_S equal to the daemon init-print scale), copy `shaders/rcas_720.comp` over `rcas.comp`, and set `FSR4_SENTINEL=0`: the sentinel shader's texel indices are 1080p-shaped and would false-fire at 720p.

## 4b. Launch (knobs!)

```sh
adb shell
cd /data/local/tmp/qnn && setsid sh launch_production.sh </dev/null >/dev/null &   # the daemon dies with its adb shell otherwise
```

Success gate: `gl1.log` reads `listening on 127.0.0.1:48620`. Never chain a `pkill -f live_daemon` into the same adb shell as the launch (it self-matches and kills the shell); separate commands.

Defaults are the production config (corner 0x50, GL=cpu3, net=0, send=1,
NPU=2). Override anything without editing: `HTPOWER=0x30 GL_CPU=5 sh
launch_production.sh`. Full reference + the measured corner/pin ladder:
`docs/KNOBS.md`. The daemon listens on **48620** (the bench client's 54345
is a different thing; hours were lost to this once).

## 5. Bench it

`tools/arm_trigger.sh <label> <seconds> <timeout>`: push to device, run
on-device; press **pad A** (BTN_SOUTH) when the bench starts. It samples the
fuel gauge at ~8.5 Hz. The device must be **unplugged** (charging current
flows through the same gauge). Pull `samples_<label>.csv` plus the
GameNative `powercontrol` metrics jsonl at /sdcard/Android/data/app.gamenative/files/powercontrol/tuner-*.jsonl for the same session: the metrics
file grows in place, so re-pull AFTER the bench finishes.

Carve and compute:
`python tools/carve_energy.py samples_<label>.csv metrics_<label>.jsonl`
(auto-cuts attempt the paper's rules; the fallback prints a power trace for
picking `--start/--end`; sanity gates per `docs/BENCH_METHOD.md`).
