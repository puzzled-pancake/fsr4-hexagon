# What this repo does NOT contain, and why

Everything needed to build and run is here except other people's property.
You source these yourself:

| Component | Owner | Where you get it | Notes |
|---|---|---|---|
| FSR4 model + weights (`fsr4 v07`) | AMD | FidelityFX SDK 2.0.0 (GPUOpen) | weights/npz/DLC/context binaries never redistributed; `model/` regenerates from your checkout. The SDK's license governs the model |
| NGX ABI knowledge | NVIDIA (interface), OptiScaler (the map) | observed `nvngx` ABI; OptiScaler is GPL-3 | the proxy reproduces the nvngx ABI constants/structs in-source (comment-marked as reproduced from the SDK headers, for interoperability); no NVIDIA SDK files are included, but those definitions are header-derived |
| QAIRT / QNN SDK 2.50 | Qualcomm | qcm.qualcomm.com (free) | daemon build needs its headers; runtime `.so`s deploy to the device; `libcdsprpc.so` must be the vendor's own copy in `/vendor` |
| CDSP firmware | Retroid/Qualcomm | already on the device (modem partition) | the unlock's loader fetches it; never dumped, never redistributed |
| `nvapi64.dll` | fakenvapi project | upstream releases | reference implementation, drop-in for the game dir |
| DXVK 2.7.1 | DXVK project (zlib) | GameNative ships it | only the `dxvk.conf` spoof entry is documented here |
| Rise of the Tomb Raider | Square Enix | your Steam copy (appid 391220) | obviously. The `figures/` screenshots are from the author's own session, included solely to document the pipeline's output; game art remains Square Enix's |
| GameNative v1.2.0 | GameNative | their release | v1.2.x (installed 1.2.0, self-updated 1.2.1); the Wine/Box64/Turnip container this was measured in |

The daemon sources, proxy sources, model-prep scripts, bench tooling and
whitepaper are original work, MIT-licensed (LICENSE). The compute shaders
are ports of AMD FidelityFX kernels where marked in-file.
