#!/bin/sh
# Build the d3d12.dll registry injector (forces DLSS=1 / DLSS Previous=1 / FXAA=0
# into RotTR's registry at DLL ATTACH — the in-menu toggle never sticks on a
# fresh container). Drop next to nvngx.dll in the game dir.
set -e
cd "$(dirname "$0")"
GCC="${GCC:-x86_64-w64-mingw32-gcc}"
$GCC -O2 -shared -o d3d12.dll d3d12_inject.c -lgdi32 -luser32 -lws2_32 -ldxguid
echo "built d3d12.dll"
