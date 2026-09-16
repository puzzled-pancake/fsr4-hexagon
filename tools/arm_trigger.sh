#!/bin/bash
# Pad-A benchmark trigger + native 20Hz fuel-gauge sampler.
# Usage (from host): adb shell "sh /data/local/tmp/qnn/arm_trigger.sh <label> [dur=150] [tmo=300]"
# (NO bash on the RP6 — toybox sh only)
# ARM it right before the benchmark's final menu; the next pad-A press
# (BTN_SOUTH 0x130) stamps t0 and starts the sample window at the same
# instant as the bench. getevent only OBSERVES — the game still gets the
# press. One-shot: fires once, then samples for <dur> seconds.
# Output: samples_<label>.csv (t_sec,voltage_uV,current_uA) + t0_<label>.txt
LABEL=${1:-arm}
DUR=${2:-150}
TMO=${3:-300}
PAT=${TEST_PAT:-'0001 0130 00000001'}
DIR=/data/local/tmp/qnn
CSV=$DIR/samples_$LABEL.csv

# every evdev node whose caps carry KEY 0130 (BTN_SOUTH)
NODES=""
for d in /dev/input/event*; do
 getevent -p $d 2>/dev/null | grep -q ' 0130' && NODES="$NODES $d"
done
[ -n "$NODES" ] || { echo "ERROR: no A-capable input node"; exit 1; }
echo "armed '$LABEL': press A (watching:$NODES, ${DUR}s window, timeout ${TMO}s)"

TRIG=$(timeout $TMO getevent $NODES 2>/dev/null | grep -m1 "$PAT")
[ -n "$TRIG" ] || { echo "TIMEOUT: no trigger, disarmed"; exit 2; }
T0=$(date +%s.%N)
echo "$T0" > $DIR/t0_$LABEL.txt
echo "TRIGGERED at $T0 ($TRIG)"

END=$(($(date +%s) + DUR))
echo "t_sec,voltage_uV,current_uA" > $CSV
while [ $(date +%s) -lt $END ]; do
 V=$(cat /sys/class/power_supply/battery/voltage_now)
 I=$(cat /sys/class/power_supply/battery/current_now)
 T=$(date +%s.%N)
 echo "$T,$V,$I" >> $CSV
 usleep 50000 2>/dev/null || sleep 0.05
done
N=$(($(wc -l < $CSV) - 1))
echo "DONE: $N samples -> $CSV (t0=$(cat $DIR/t0_$LABEL.txt))"
