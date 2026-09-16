#!/system/bin/sh
# Production launch (the config measured in the whitepaper).
# Every knob is overridable from the command line without editing this file:
#   HTPOWER=0x30 GL_CPU=5 sh launch_production.sh
# (setting the FSR4_* names directly also works). Full reference: docs/KNOBS.md
cd /data/local/tmp/qnn
export LD_LIBRARY_PATH=/data/local/tmp/qnn
export ADSP_LIBRARY_PATH="/data/local/tmp/qnn;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp"
export FSR4_FEAT="${FSR4_FEAT:-features_real.comp}"
export FSR4_POST="${FSR4_POST:-post_real.comp}"
export FSR4_SPLIT="${FSR4_SPLIT:-1}"
export FSR4_MVNORM="${FSR4_MVNORM:-1}"
export FSR4_SKIP_SYN="${FSR4_SKIP_SYN:-1}"
export FSR4_RCAS="${FSR4_RCAS:-0.5}"
export FSR4_BLENDFLOOR="${FSR4_BLENDFLOOR:-0.0}"
export FSR4_CUTFRAC="${FSR4_CUTFRAC:-0.35}"
export FSR4_CUTMV="${FSR4_CUTMV:-0}"
export FSR4_FUSE="${FSR4_FUSE:-1}"
export FSR4_RAWLR="${FSR4_RAWLR:-1}"
export FSR4_HTPOWER="${FSR4_HTPOWER:-${HTPOWER:-0x50}}"
export FSR4_AHBCOPY="${FSR4_AHBCOPY:-1}"
export FSR4_AHBSYNC="${FSR4_AHBSYNC:-0}"
export FSR4_SENTINEL="${FSR4_SENTINEL:-64}"
export FSR4_GL_CPU="${FSR4_GL_CPU:-${GL_CPU:-3}}"
export FSR4_NET_CPU="${FSR4_NET_CPU:-${NET_CPU:-0}}"
export FSR4_SEND_CPU="${FSR4_SEND_CPU:-${SEND_CPU:-1}}"
export FSR4_NPU_CPU="${FSR4_NPU_CPU:-${NPU_CPU:-2}}"
exec ./live_daemon 48620 /dev/null v/bin_cfnhwc.bin >> gl1.log 2>&1
