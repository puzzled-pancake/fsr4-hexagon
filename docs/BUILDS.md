# Build manifest (md5) of the binaries the whitepaper measured

Scoped to where each ran. Rebuilds from this repo's sources are functionally
equivalent but NOT byte-identical (toolchain drift); pin provenance here.

| Binary | md5 | Role |
|---|---|---|
| `nvngx_v19.dll` | `288f2abe2842149f70468d3712e8bf06` | proxy DLL, capped Experiment 1 FSR4 rows |
| `nvngx_v20_1080.dll` | `940e12d91af97a07db382faeeea59f7c` | proxy DLL as-run for the uncapped 1080p sweep |
| `nvngx_v20_1080c.dll` | `7a85f9f48500fd4cae7bf70268caea04` | clean rebuild twin of the above (same lineage) |
| `nvngx_v20_360b.dll` | `1b7bd7f55487e6d7e8ec5ffc24807c24` | proxy DLL, 720p arm |
| `live_daemon_v33e` | `f49e6ed590caa176fff451749dfc55f0` | daemon, all 1080p arms (capped + uncapped) / production config |
| `live_daemon_v35b_360` | `96a1d1f58e000b002e7127ba1061cab7` | daemon, 720p arm |
| `d3d12.dll` (injector) | `11ea0285972228f2cc601d97e6ed9fc1` | registry injector |
| `arm_trigger.sh` | `f9478145941ab038ae46fc0cfcedfa33` | on-device pad-A power sampler |
| gpu_data seed `pass0_wb.raw` | `0915f25de428aaeee396701e3abfd341` | pass-0 conv weights, 1080p arm (see daemon/gen_seeds.py) |

All proxy rows build from `proxy/full_proxy.c` (default dims = 1080p arm;
`-DLWV=640u -DLHV=360u` = 720p arm). Model context binaries are deliberately
absent: they derive from AMD's FSR4 model; `model/README.md` regenerates them
from your own SDK checkout.
