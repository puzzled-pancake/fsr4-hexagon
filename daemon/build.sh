#!/bin/sh
# Cross-compile the daemon with NDK r28 clang (author ran 28.2.13676358 + QAIRT
# 2.50.0.260828). Needs the QAIRT SDK headers (free download from Qualcomm).
# 1080p arm: default dims. 720p arm: add -DFSR4D_LW=640 -DFSR4D_LH=360 \
#   -DFSR4D_W=1280 -DFSR4D_H=720 (per model/README.md step 6).
# Linux NDK equivalent: .../toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android28-clang
set -e
cd "$(dirname "$0")"
NDK_CLANG="${NDK_CLANG:?export NDK_CLANG=<your NDK aarch64-linux-android28-clang path>}"
QNN_SDK_ROOT="${QNN_SDK_ROOT:?export QNN_SDK_ROOT=<your QAIRT 2.50 SDK root>}"
# -D_GNU_SOURCE must be on the command line: -include assert.h pulls in
# features.h before the .c files' own #define, so cpu_set_t/sched_setaffinity
# (bionic __USE_GNU) would not be declared without it.
"$NDK_CLANG" -O2 -D_GNU_SOURCE -DFSR4_NO_MAIN -include assert.h \
  -I"$QNN_SDK_ROOT/include/Qnn" -o live_daemon \
  live_daemon.c qnn_service.c -ldl -lm -lEGL -lGLESv3 -landroid
echo "built live_daemon"
