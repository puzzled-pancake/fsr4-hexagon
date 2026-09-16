#!/bin/sh
# Stage the QAIRT 2.50 runtime libs + (optionally) our daemon artifacts to
# /data/local/tmp/qnn on the RP6. Push EXPLICIT full paths (adb TLS transport
# truncates the last char of filenames when pushing to a directory).
# Run as: MSYS2_ARG_CONV_EXCL="/data" sh android_push.sh (Git Bash)
set -e
QNN_SDK_ROOT="${QNN_SDK_ROOT:-C:/Qualcomm/qairt/2.50.0.260828}"
DEV=/data/local/tmp/qnn
adb shell "mkdir -p $DEV/v $DEV/gpu_data"
for f in lib/aarch64-android/libQnnHtp.so lib/aarch64-android/libQnnHtpV73Stub.so \
 lib/aarch64-android/libQnnHtpPrepare.so lib/aarch64-android/libQnnSystem.so \
 lib/aarch64-android/libQnnModelDlc.so lib/aarch64-android/libQnnHtpNetRunExtensions.so \
 lib/hexagon-v73/unsigned/libQnnHtpV73Skel.so; do
 adb push "$QNN_SDK_ROOT/$f" "$DEV/$(basename $f)"
done
# our artifacts (run daemon/build.sh + model/README.md first):
[ -f daemon/live_daemon ] && adb push daemon/live_daemon "$DEV/live_daemon" && adb shell "chmod +x $DEV/live_daemon"
for s in daemon/shaders/*.comp; do adb push "$s" "$DEV/$(basename $s)"; done # FLAT: daemon loads bare filenames
[ -f v/bin_cfnhwc.bin ] && adb push v/bin_cfnhwc.bin "$DEV/v/bin_cfnhwc.bin"
[ -f daemon/launch_production.sh ] && adb push daemon/launch_production.sh "$DEV/launch_production.sh"
echo "staged. generate gpu_data seeds on device per daemon/gen_seeds.py header,"
echo "or push your own to $DEV/gpu_data/ (pass0_wb.raw hist.raw rec_prev.raw)."
