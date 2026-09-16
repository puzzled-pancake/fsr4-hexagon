# Rebuilding the NPU model artifacts (no weights in this repo)

Chain: your FidelityFX SDK checkout -> npz -> ONNX -> DLC -> on-device
serialized HTP context (`v/bin_cfnhwc.bin`). Every step is scripted; nothing
here contains AMD's weights.

1. **Extract weights** (needs a FidelityFX SDK 2.0.0 checkout containing
   `Kits/FidelityFX/upscalers/fsr4/internal/shaders/fsr4_model_v07_i8_quality/`;
   edit `ROOT` at the top):
   `python extract/build_weights.py` -> `fsr4_quality_weights.npz` + `graph_spec.json`
   (self-verifying gates against the SDK's own provider cross-check).
2. **Simulator gate** (optional but recommended): `python sim/test_sim.py`.
3. **Build the QDQ ONNX graph** (the shipped graph is clip-free NHWC):
   first `python onnx/build_onnx.py` for the reference QDQ graph, feed it through `python onnx/add_dummy_output.py` to get the `<in_f16dum.onnx>` float graph with the dummy output node, then
   `python onnx/build_qdq_clipfree.py <in_f16dum.onnx> <enc_info.txt> <out.onnx>`
   (its usage line: float graph in, an encoding dump from `qairt-dlc-info` on
   the quantized reference DLC, output). The SDK python tools need the QAIRT
   python env (3.12 exactly + SDK PYTHONPATH).
4. **Convert + quantize** (QAIRT SDK python env; Windows example):
   ```
   qairt-converter -i <in>.onnx --output_path <out>_f16.dlc \
     --source_model_input_layout input_i8 NCHW --desired_input_layout input_i8 NHWC \
     --source_model_output_layout cast_335 NCHW --desired_output_layout cast_335 NHWC
   qairt-quantizer --input_dlc <out>_f16.dlc --output_dlc <out>_a8.dlc \
     --act_bitwidth 8 --weights_bitwidth 8 --bias_bitwidth 32 \
     --act_quantizer_calibration min-max --act_quantizer_schema symmetric \
     --input_list quant_list_qdq.txt
   ```
5. **Serialize the HTP context ON DEVICE** (offline prepare needs a Linux
   host; the aarch64 tool works on the RP6 itself):
   ```
   qairt-dlc-prepare --input_dlc fsr4_a8n.dlc --backend libQairtHtp.so \
     --config_file wrapper.json --binary_file bin_cfnhwc.bin --output_dir v
   ```
   with a wrapper json (pointing at libQairtHtpBackendExtensions.so) whose graphs[0] sets `O: 3, vtcm_mb: 8` and `graph_names` equal to the DLC's INTERNAL graph name, which in this chain FOLLOWS YOUR ONNX/DLC FILENAME. Read it with `qairt-dlc-info` first; a mismatch silently skips O:3 (about 3x slower NPU, zero errors). `--binary_file` resolves relative to `--output_dir`.
6. **720p arm**: same chain with the ONNX reshaped to 640x360 input and
   requantized as its own context (2.8 ms/inference; fits VTCM better than
   linear scaling).
