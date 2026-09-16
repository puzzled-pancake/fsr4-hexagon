#!/bin/bash
# energy sampler: reads the battery fuel gauge at ~2Hz with timestamps.
# Usage: bash joules_sample.sh <label> <seconds> (run from repo root)
# Output: samples_<label>.csv with t,voltage_uV,current_uA
# NOTE: device must be UNPLUGGED during measurement (current_now measures
# net battery current — charging masks the SoC draw).
S="${ADB_TARGET:?export ADB_TARGET, e.g. <phone-ip>:5555}"
LABEL=${1:-run}
DUR=${2:-60}
OUT="samples_${LABEL}.csv"
echo "t_sec,voltage_uV,current_uA" > "$OUT"
START=$(date +%s)
echo "sampling '$LABEL' for ${DUR}s -> $OUT (unplug the charger!)"
while [ $(($(date +%s) - START)) -lt "$DUR" ]; do
 T=$(date +%s.%N)
 V=$(adb -s $S shell "cat /sys/class/power_supply/battery/voltage_now" 2>/dev/null | tr -d '\r')
 I=$(adb -s $S shell "cat /sys/class/power_supply/battery/current_now" 2>/dev/null | tr -d '\r')
 [ -n "$V" ] && [ -n "$I" ] && echo "$T,$V,$I" >> "$OUT"
 sleep 0.45
done
echo "done: $(wc -l < "$OUT") samples"
