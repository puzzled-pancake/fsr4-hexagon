#!/bin/sh
# Build both nvngx.dll variants with mingw-w64 gcc from the ONE parameterized
# source. Dims are compile-time: defaults are 960x540 -> 1920x1080 (1080p arm);
# -DLWV=640u -DLHV=360u gives the 720p arm (640x360 -> 1280x720).
set -e
cd "$(dirname "$0")"
GCC="${GCC:-x86_64-w64-mingw32-gcc}"
$GCC -O2 -shared -o nvngx_1080.dll full_proxy.c live_win.c -lgdi32 -luser32 -lws2_32 -ldxguid
$GCC -O2 -shared -DLWV=640u -DLHV=360u -o nvngx_360.dll full_proxy.c live_win.c -lgdi32 -luser32 -lws2_32 -ldxguid
echo "built nvngx_1080.dll nvngx_360.dll (rename to nvngx.dll in the game dir)"
