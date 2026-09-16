"""FSR4 INT8 (fsr4_model_v07_i8_quality) weight/scale extraction with self-verification gates.

Sources (all under FidelityFX-SDK/Kits/FidelityFX/upscalers/fsr4/):
  1. internal/shaders/fsr4_model_v07_i8_quality/initializers.bin  (scale table + 11 tensors)
  2. internal/shaders/fsr4_model_v07_i8_quality/passes_1080.hlsl  (embedded int8 weights + fp16 biases, decls, call sites)
  3. internal/shaders/fsr4_model_v07_i8_quality/post.hlsl         (pass-13 decoder block)
  4. internal/shaders/fsr4_model_v07_i8_quality/pre.hlsl          (pass-0 fp16 weights/bias/scale)
  5. dx12/ffx_provider_fsr4_dx12.cpp                              (weights_quality cross-check)

Output: artifacts/fsr4_quality_weights.npz + artifacts/graph_spec.json
"""
import json, os, re, sys, hashlib
from pathlib import Path
import numpy as np

ROOT = Path(os.environ["FIDELITYFX_SDK_ROOT"]) / "Kits/FidelityFX/upscalers/fsr4"
SH = ROOT / "internal/shaders/fsr4_model_v07_i8_quality"
OUT = Path(__file__).resolve().parents[1] / "artifacts"
OUT.mkdir(exist_ok=True)

failures = []
def gate(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        failures.append(name)

# ---------------------------------------------------------------- array parsing
HEXRE = re.compile(r"0x[0-9a-fA-F]+")
COMMRE = re.compile(r"//\s*([^\n]*)")

def parse_arrays(text):
    """Return {name: (dwords: np.uint32[N], comments: [str])} for `static const uint name_dwords[N] = {...};`"""
    out = {}
    for m in re.finditer(r"static const uint (\w+)_dwords\[(\d+)\] = \{(.*?)\};", text, re.S):
        name, n, body = m.group(1), int(m.group(2)), m.group(3)
        dwords = np.array([int(h, 16) for h in HEXRE.findall(body)], dtype=np.uint32)
        comments = COMMRE.findall(body)
        if len(dwords) != n:
            failures.append(f"{name}: dword count {len(dwords)} != declared {n}")
        out[name] = (dwords, comments)
    return out

def dwords_to_i8(dw):
    return np.frombuffer(dw.astype("<u4").tobytes(), dtype=np.int8).copy()

def dwords_to_f16(dw):
    return np.frombuffer(dw.astype("<u4").tobytes(), dtype="<f2").copy()

def check_comments_i8(name, dw, comments):
    ref = []
    for c in comments:
        vals = [int(v) for v in re.findall(r"-?\d+", c)]
        ref.extend(vals)
    got = [int(v) for v in dwords_to_i8(dw)]
    if len(ref) != len(got):
        return False, f"len {len(ref)} vs {len(got)}"
    bad = sum(1 for a, b in zip(ref, got) if a != b)
    return bad == 0, f"{len(got)} values, {bad} mismatches"

def check_comments_f16(name, dw, comments):
    ref = []
    for c in comments:
        ref.extend(float(v) for v in re.findall(r"-?\d+\.?\d*(?:e-?\d+)?", c))
    got = dwords_to_f16(dw).astype(np.float32)
    if len(ref) != len(got):
        return False, f"len {len(ref)} vs {len(got)}"
    bad = sum(1 for a, b in zip(ref, got) if not (abs(a - b) <= 4e-3 * max(1.0, abs(a))))
    return bad == 0, f"{len(got)} values, {bad} mismatches"

# ---------------------------------------------------------------- tensor decl parsing
DECL_RE = re.compile(
    r"const (QuantizedTensor\d\w+|Tensor\d\w+)<(?:[^<>]|<[^<>]*>)*>\s+(\w+) = \{\s*"
    r"uint([34])\(([^)]*)\), // logicalSize(.*?)\};", re.S)
UINTRE = re.compile(r"uint[34]\(([^)]*)\)")
FLOATRE = re.compile(r"(-?\d+\.\d+(?:e-?\d+)?|-?\d+e-?\d+)")

def parse_decls(text):
    out = {}
    for m in DECL_RE.finditer(text):
        typ, name, nd, logical, body = m.group(1), m.group(2), int(m.group(3)), m.group(4), m.group(5)
        tuples = UINTRE.findall(body)  # sliceStart, sliceSize, storageSize, strides, padB, padE
        floats = FLOATRE.findall(body)
        def ints(s):
            return [int(v) for v in s.split(",")]
        d = dict(type=typ, logical=ints(logical))
        keys = ["sliceStart", "sliceSize", "storageSize", "strides", "padBegin", "padEnd"]
        for k, t in zip(keys, tuples):
            d[k] = ints(t)
        # threadGroupStorageByteOffset is a bare int line, scale a bare float
        moff = re.search(r"(\d+), // threadGroupStorageByteOffset", body)
        d["tgOffset"] = int(moff.group(1)) if moff else 0
        # scratch slices use "threadGroupByteOffsetInTensor + N"
        moff2 = re.search(r"threadGroupByteOffsetInTensor_slice_\d+ \+ (\d+)", body)
        d["scratchOffset"] = int(moff2.group(1)) if moff2 else d["tgOffset"]
        d["scale"] = float(floats[0]) if floats else None
        out[name] = d
    return out

def parse_calls(text):
    out = []
    pat = re.compile(r"(ConvNextBlock|FasterNetBlock<\d+,\s*\d+>|FNB_CT2D_ADD<\d+,\s*\d+>|"
                     r"FusedConv2D_k2s2b_QuantizedOutput|FusedConv2D_DW_Conv2D_PW_Relu_Conv2D_Add_ConvTranspose2D)"
                     r"\((-?\d+\.\d+(?:e-?\d+)?(?:,\s*-?\d+\.\d+(?:e-?\d+)?)*)", re.S)
    for m in pat.finditer(text):
        out.append(dict(op=m.group(1), args=[float(a) for a in m.group(2).split(",")]))
    return out

# ---------------------------------------------------------------- 1. initializers.bin
bin_data = np.frombuffer((SH / "initializers.bin").read_bytes(), dtype=np.uint8)
gate("bin size == 89216", len(bin_data) == 89216, f"{len(bin_data)}")

scale_table = np.frombuffer(bin_data[:9856].tobytes(), dtype="<f4").reshape(77, 32)[:, 0].copy()
gate("scale table entry[1] == 0 (unused)", scale_table[1] == 0.0)

# 11 bin tensors: (short_name, offset, logical(W,H,C,N), strides, layout)
BIN_TENSORS = [
    ("encoder3_downscale_conv_w",     9856, [2,2,32,64],  [32,64,1,128],   "NHWC"),
    ("bottleneck_rb0_spatial_w",     18048, [3,3,16,32],  [16,48,1,144],   "NHWC"),
    ("bottleneck_rb0_pw_expand_w",   22656, [1,1,64,128], [64,64,1,64],    "NHWC"),
    ("bottleneck_rb0_pw_contract_w", 30848, [1,1,128,64], [128,128,1,128], "NHWC"),
    ("bottleneck_rb1_spatial_w",     39040, [3,3,16,32],  [16,48,1,144],   "NHWC"),
    ("bottleneck_rb1_pw_expand_w",   43648, [1,1,64,128], [64,64,1,64],    "NHWC"),
    ("bottleneck_rb1_pw_contract_w", 51840, [1,1,128,64], [128,128,1,128], "NHWC"),
    ("bottleneck_rb2_spatial_w",     60032, [3,3,16,32],  [16,48,1,144],   "NHWC"),
    ("bottleneck_rb2_pw_expand_w",   64640, [1,1,64,128], [64,64,1,64],    "NHWC"),
    ("bottleneck_rb2_pw_contract_w", 72832, [1,1,128,64], [128,128,1,128], "NHWC"),
    ("bottleneck_upscale_convT_w",   81024, [2,2,32,64],  [2048,4096,64,1],"HWCN"),
]
bin_tensors = {}
cover = 9856
ok_contig = True
for name, off, logical, strides, layout in BIN_TENSORS:
    nbytes = int(np.prod(logical))
    if off != cover:
        ok_contig = False
    cover += nbytes
    raw = bin_tensors[name] = bin_data[off:off+nbytes].copy()
    assert len(raw) == nbytes
gate("bin tensors contiguous, cover to EOF", ok_contig and cover == 89216, f"end={cover}")

# scale-table cross-check: bin weight scales must equal the shader literals later
BIN_SCALE_IDX = {"encoder3_downscale_conv_w":28, "bottleneck_rb0_spatial_w":31, "bottleneck_rb0_pw_expand_w":32,
                 "bottleneck_rb0_pw_contract_w":34, "bottleneck_rb1_spatial_w":37, "bottleneck_rb1_pw_expand_w":38,
                 "bottleneck_rb1_pw_contract_w":40, "bottleneck_rb2_spatial_w":43, "bottleneck_rb2_pw_expand_w":44,
                 "bottleneck_rb2_pw_contract_w":46, "bottleneck_upscale_convT_w":48}

# ---------------------------------------------------------------- 2/3. shader-embedded arrays
passes_text = (SH / "passes_1080.hlsl").read_text()
post_text = (SH / "post.hlsl").read_text()
pre_text = (SH / "pre.hlsl").read_text()

arrays = {}
arrays.update(parse_arrays(passes_text))
arrays.update({f"post.{k}": v for k, v in parse_arrays(post_text).items()})
arrays.update({f"pre.{k}": v for k, v in parse_arrays(pre_text).items()})
print(f"parsed {len(arrays)} embedded dword arrays")

# validate every int8 weight array against its comment dump; fp16 for biases & pass0 weights
i8_weights, f16_biases, pass0_w, pass0_b = {}, {}, None, None
n_checked = 0
for name, (dw, comments) in arrays.items():
    if "weight_quant" in name and not name.startswith("pre."):
        ok, det = check_comments_i8(name, dw, comments)
        gate(f"i8 comments match: {name}", ok, det)
        i8_weights[name] = dwords_to_i8(dw)
        n_checked += 1
    elif "_bias" in name:
        ok, det = check_comments_f16(name, dw, comments)
        gate(f"f16 bias comments match: {name}", ok, det)
        f16_biases[name] = dwords_to_f16(dw)
    elif name == "pre.embedded_encoder1_DownscaleStridedConv2x2_downscale_conv_weight":
        ok, det = check_comments_f16(name, dw, comments)
        gate("f16 comments match: pass0 weights", ok, det)
        pass0_w = dwords_to_f16(dw)
gate("parsed >= 15 embedded i8 weight arrays", n_checked >= 15, f"{n_checked}")

# pass0: weight element (kx,ky,c,f) at byte 16*kx+32*ky+2*c+64*f  (fp16 pairs, low half = even c)
w0 = np.zeros((16, 2, 2, 8), dtype="<f2")  # [f, ky, kx, c]
flat = pass0_w  # 2048 fp16 in file order
for f in range(16):
    for ky in range(2):
        for kx in range(2):
            for c in range(8):
                dword_idx = (f*4 + kx + 2*ky)*4 + (c//2)
                lo_hi = c % 2
                w0[f, ky, kx, c] = flat[dword_idx*2 + lo_hi]
gate("pass0 weight ch7 all zero", np.all(w0[..., 7] == 0))

# cross-check pass0 weights against cpp weights_quality
cpp_text = (ROOT / "dx12/ffx_provider_fsr4_dx12.cpp").read_text()
m = re.search(r"weights_quality\[256\] = \{(.*?)\};", cpp_text, re.S)
cpp_dw = np.array([int(h, 16) for h in HEXRE.findall(m.group(1))], dtype=np.uint32)
gate("pass0 weights == cpp weights_quality", np.array_equal(cpp_dw, arrays["pre.embedded_encoder1_DownscaleStridedConv2x2_downscale_conv_weight"][0]))

pass0_scale = float(re.search(r"downscale_quantizationScale = (-?[\d.e-]+);", pre_text).group(1))
pass0_b = f16_biases["pre.embedded_encoder1_DownscaleStridedConv2x2_downscale_conv_bias"]

# ---------------------------------------------------------------- decls & call sites
decls = parse_decls(passes_text)
post_decls = parse_decls(post_text)
decls.update({f"post.{k}": v for k, v in post_decls.items()})
calls = parse_calls(passes_text) + parse_calls(post_text)
gate("operator call sites == 13", len(calls) == 13, f"{len(calls)}")
print(f"parsed {len(decls)} tensor decls, {len(calls)} operator call sites")

# bin scale cross-check vs decl scales (decls carry .quantizationScale)
decl_by_prefix = {}
for dn, d in decls.items():
    decl_by_prefix[dn] = d
ok = True
det = []
for name, idx in BIN_SCALE_IDX.items():
    st = float(scale_table[idx])
    # find matching decl by name fragment
    frag = {"encoder3_downscale_conv_w": "_encoder3_DownscaleStridedConv2x2_downscale_conv_weight",
            "bottleneck_rb0_spatial_w": "bottleneck_ResidualBlock_0_body_spatial_mixing_partial_conv_weight",
            "bottleneck_rb0_pw_expand_w": "bottleneck_ResidualBlock_0_body_pw_expand_weight",
            "bottleneck_rb0_pw_contract_w": "bottleneck_ResidualBlock_0_body_pw_contract_weight",
            "bottleneck_rb1_spatial_w": "bottleneck_ResidualBlock_1_body_spatial_mixing_partial_conv_weight",
            "bottleneck_rb1_pw_expand_w": "bottleneck_ResidualBlock_1_body_pw_expand_weight",
            "bottleneck_rb1_pw_contract_w": "bottleneck_ResidualBlock_1_body_pw_contract_weight",
            "bottleneck_rb2_spatial_w": "bottleneck_ResidualBlock_2_body_spatial_mixing_partial_conv_weight",
            "bottleneck_rb2_pw_expand_w": "bottleneck_ResidualBlock_2_body_pw_expand_weight",
            "bottleneck_rb2_pw_contract_w": "bottleneck_ResidualBlock_2_body_pw_contract_weight",
            "bottleneck_upscale_convT_w": "bottleneck_UpscaleConvTranspose2x2_upscale_conv_weight"}[name]
    matches = [d for dn, d in decls.items() if frag in dn and d["scale"]]
    if not matches or not any(abs(m["scale"] - st) == 0.0 for m in matches):
        ok = False
        det.append(name)
gate("bin scale table == shader decl scales (11 tensors)", ok, ",".join(det) or "all match")

# ---------------------------------------------------------------- reshape helpers
def to_oihw(flat_u8, logical, strides, layout):
    """Reshape raw bytes to [out, in, kH, kW] per WHCN logical + byte strides."""
    flat_i8 = flat_u8.view(np.int8)  # dot4add_i8packed: lanes are SIGNED int8
    W, H, C, N = logical
    if layout == "NHWC":
        # natural NHWC storage: byte = n*sN + ky*sH + kx*sW + c
        assert strides == [C, W*C, 1, H*W*C], f"unexpected NHWC strides {strides}"
        t = flat_i8.reshape(N, H, W, C)
        return t.transpose(0, 3, 1, 2)  # [N(f=out), C(in), ky, kx]
    else:  # HWCN: byte = ky*sH + kx*sW + cout*sC + cin (sN = 1)
        assert strides == [C*N, W*C*N, N, 1], f"unexpected HWCN strides {strides}"
        t = flat_i8.reshape(H, W, C, N)
        return t.transpose(3, 2, 0, 1)  # [N(cin), C(cout), ky, kx]  (ConvTranspose order)

weights = {}
for name, off, logical, strides, layout in BIN_TENSORS:
    weights[name] = to_oihw(bin_tensors[name], logical, strides, layout)

# embedded i8 weights: find each array's decl to get logical size, then reshape NHWC-natural
emb_ok, emb_det = True, []
for name, flat in i8_weights.items():
    cands = [d for dn, d in decls.items() if dn.replace("post.", "") == name.replace("post.", "") and d["type"].startswith("QuantizedTensor4i8")]
    if not cands:
        emb_ok = False; emb_det.append(f"{name}: no decl"); continue
    d = cands[0]
    W, H, C, N = d["logical"]
    if d["strides"] == [C, W*C, 1, H*W*C]:
        weights[name] = flat.reshape(N, H, W, C).transpose(0, 3, 1, 2)
    elif d["strides"][3] == 1 and d["strides"][2] == N and "HWCN" in d["type"]:  # embedded HWCN (CT2D weights)
        weights[name] = flat.reshape(H, W, C, N).transpose(3, 2, 0, 1)
    else:
        emb_ok = False; emb_det.append(f"{name}: strides {d['strides']}")
    # size check
    if len(flat) != W*H*C*N:
        emb_ok = False; emb_det.append(f"{name}: size {len(flat)} != {W*H*C*N}")
gate("embedded i8 weights reshaped to OIHW", emb_ok, "; ".join(emb_det[:4]) or f"{len(i8_weights)} tensors")

_dup_ok = all(np.array_equal(i8_weights[f"post.{k}"], v) for k, v in i8_weights.items() if not k.startswith("post.") and f"post.{k}" in i8_weights)
_dup_ok = _dup_ok and all(np.array_equal(f16_biases[f"post.{k}"], v) for k, v in f16_biases.items() if not k.startswith("post.") and f"post.{k}" in f16_biases)
gate("pass13 weights+biases byte-identical in passes_1080.hlsl and post.hlsl", _dup_ok)

# ---------------------------------------------------------------- assemble npz + spec
artifact = dict(weights)
for name, b in f16_biases.items():
    artifact[f"bias__{name}"] = b.astype(np.float32)
artifact["pass0_weight_fkyxc"] = w0.astype(np.float32)
artifact["pass0_bias"] = pass0_b.astype(np.float32)
artifact["bin_scale_table"] = scale_table
for name, _, _, _, _ in BIN_TENSORS:
    artifact[f"bin_raw__{name}"] = bin_tensors[name]

spec = dict(
    preset="quality", model="fsr4_model_v07_i8",
    pass0=dict(scale=pass0_scale, weight_layout="[f,ky,kx,c(8, ch7=0)]", bias=list(map(float, pass0_b))),
    bin_tensors=[dict(name=n, offset=o, logical=l, strides=s, layout=ly) for n, o, l, s, ly in BIN_TENSORS],
    scale_table=[float(s) for s in scale_table],
    decls={k: {kk: vv for kk, vv in v.items()} for k, v in decls.items()},
    calls=calls,
)
(OUT / "graph_spec.json").write_text(json.dumps(spec, indent=1))
np.savez_compressed(OUT / "fsr4_quality_weights.npz", **artifact)
print(f"\nwrote {OUT/'fsr4_quality_weights.npz'} ({len(artifact)} arrays) and graph_spec.json")
print(f"total int8 weight tensors: {len(weights)}; biases: {len(f16_biases)}")
if failures:
    print(f"\n!!! {len(failures)} GATE FAILURES:")
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("\nALL GATES PASSED")
