"""Build the FSR4 INT8 graph as a static-shape QDQ ONNX model (opset 17, NCHW).

Encodings are the model's exact per-tensor scales (zp=0, symmetric):
  int8 weight -> DequantizeLinear -> Conv(f32, fp16-exact bias) -> QuantizeLinear(next scale)
Grouped two-scale tensors: conv filters split into channel halves, one QuantizeLinear per half.
ReLU-after-quant: Q -> DQ -> Relu on the integer-valued f32 data.
Pass13 output: ConvTranspose -> +bias -> Cast(fp16).
"""
import sys
from pathlib import Path
import numpy as np
import onnx
from onnx import helper, TensorProto as TP

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim"))
import fsr4_sim as S

ART = Path(__file__).resolve().parents[1] / "artifacts"

class G:
    def __init__(self):
        self.nodes, self.inits, self.dbgouts = [], [], []
        self._n = 0
    def nm(self, tag):
        self._n += 1
        return f"{tag}_{self._n}"
    def init(self, arr):
        name = self.nm("init")
        dt = {np.dtype(np.int8): TP.INT8, np.dtype(np.uint8): TP.UINT8, np.dtype(np.int64): TP.INT64, np.dtype(np.float32): TP.FLOAT, np.dtype(np.float16): TP.FLOAT16}[np.dtype(arr.dtype)]
        self.inits.append(helper.make_tensor(name, dt, arr.shape, arr.tobytes(), raw=True))
        return name
    def node(self, op, ins, outs=None, **kw):
        y = self.nm(op.lower())
        self.nodes.append(helper.make_node(op, ins, [y], **kw))
        return y
    W8 = False  # weight-only int8: activations stay float (HTP A16W8), no requant boundaries

    FLOAT = False  # pure float graph: weights pre-dequantized, no quant ops
    D = np.float32  # working float dtype (fp16 mode: --fp16 swaps the interior to fp16)
    DEGCONV = False  # decompose group=2 convs and k=2/s=2 convs into HTP-native ops
    QDQADD = False  # Q before residual Adds so contract/ConvTranspose convs fold natively

    def dq(self, xq, scale, tag):
        if self.FLOAT:
            return xq
        return self.node("DequantizeLinear", [xq, self.init(np.float32(scale)), self.init(np.full((), 128, np.uint8))])
    def spd(self, x, blocksize=2):
        return self.node("SpaceToDepth", [x], blocksize=blocksize)
    def dqw(self, w_i8, scale):
        if self.FLOAT:
            w = np.ascontiguousarray((w_i8.astype(np.float32) * scale).astype(self.D))
            return self.init(w)
        w = np.ascontiguousarray(w_i8.astype(np.int16) + 128, dtype=np.uint8)
        return self.node("DequantizeLinear", [self.init(w), self.init(np.float32(scale)), self.init(np.full((), 128, np.uint8))])
    DBG_TAPS = False
    EXPLICIT_SAT = True  # HTP drops Q-node saturation; make it an explicit Clip op

    def q(self, xf, scale, tag, dbg_shape=None):
        if self.FLOAT:
            lo = self.init(self.D(-128.0 * scale))
            hi = self.init(self.D(127.0 * scale))
            xf = self.node("Clip", [xf, lo, hi])
            if dbg_shape is not None:
                dt = TP.FLOAT16 if self.D is np.float16 else TP.FLOAT
                self.dbgouts.append(helper.make_tensor_value_info(xf, dt, [1] + dbg_shape))
            return xf
        if self.EXPLICIT_SAT:
            lo = self.init(np.float32(-128.0 * scale))
            hi = self.init(np.float32(127.0 * scale))
            xf = self.node("Clip", [xf, lo, hi])
            # fp16 round-trip barrier: stops QAIRT folding Clip into the requant,
            # which silently drops the saturation on HTP
            h = self.node("Cast", [xf], to=TP.FLOAT16)
            xf = self.node("Cast", [h], to=TP.FLOAT)
        y = self.node("QuantizeLinear", [xf, self.init(np.float32(scale)), self.init(np.full((), 128, np.uint8))])
        if dbg_shape is not None:
            self.dbgouts.append(helper.make_tensor_value_info(y, TP.UINT8, [1] + dbg_shape))
        return y
    def conv(self, x, wdq, bias_f32, tag, k=1, s=1, p=0, group=1):
        ins = [x, wdq] + ([self.init(np.asarray(bias_f32, self.D))] if bias_f32 is not None else [])
        attrs = dict(kernel_shape=[k, k], strides=[s, s])
        if p: attrs["pads"] = [p, p, p, p]
        if group != 1: attrs["group"] = group
        return self.node("Conv", ins, **attrs)
    def convT(self, x, wdq, bias=None):
        ins = [x, wdq] + ([self.init(np.asarray(bias, self.D))] if bias is not None else [])
        return self.node("ConvTranspose", ins, kernel_shape=[2, 2], strides=[2, 2])
    def add(self, a, b):
        if isinstance(b, np.ndarray):
            b4 = np.asarray(b, self.D).reshape(1, -1, 1, 1)
            b = self.init(b4)
        return self.node("Add", [a, b])
    def relu(self, x):
        return self.node("Relu", [x])
    def concat(self, a, b):
        return self.node("Concat", [a, b], axis=1)
    def slice_ch(self, x, start, end):
        y = self.nm("slice")
        self.nodes.append(helper.make_node("Slice", [x, self.init(np.array([start], np.int64)),
                                                     self.init(np.array([end], np.int64)),
                                                     self.init(np.array([1], np.int64))], [y]))
        return y

def build(L, H2=540, W2=960, debug=False):
    g = G()
    H, W = H2 * 2, W2 * 2
    sc = lambda k: np.float32(L[k])
    X = "input_i8"
    if g.FLOAT:
        x = X
        if g.D is np.float16:  # f32 APP input, fp16 interior: single boundary Convert on HTP
            x = g.node("Cast", [x], to=TP.FLOAT16)
    elif g.FIO:
        # float graph input: Q/DQ head re-quantizes with the model's exact input scale
        xq_in = g.node("QuantizeLinear", [X, g.init(np.float32(S.S_IN)), g.init(np.full((), 128, np.uint8))], name="headQ")
        x = g.dq(xq_in, S.S_IN, "in")
    else:
        x = g.dq(X, S.S_IN, "in")

    def cnb(xd, prefix, s_mid1, s_mid2, s_out, tag, C, hh, ww, dbg):
        y = g.conv(xd, g.dqw(L[f"{prefix}_conv_dw_weight"], sc(f"scale__{prefix}_conv_dw")),
                   L[f"{prefix}_conv_dw_bias"], tag, k=3, p=1)
        yq = g.q(y, s_mid1, tag, ([C, hh, ww] if dbg and dbg[0] == "h0" else None))
        yd = g.dq(yq, s_mid1, tag)
        y = g.conv(yd, g.dqw(L[f"{prefix}_pw_expand_weight"], sc(f"scale__{prefix}_pw_expand")),
                   L[f"{prefix}_pw_expand_bias"], tag, k=1)
        yq = g.q(y, s_mid2, tag); yd = g.dq(yq, s_mid2, tag); yd = g.relu(yd)
        y = g.conv(yd, g.dqw(L[f"{prefix}_pw_contract_weight"], sc(f"scale__{prefix}_pw_contract")),
                   L[f"{prefix}_pw_contract_bias"], tag, k=1)
        if g.QDQADD:
            # Q before the residual Add: without it the converter float-falls-back the
            # contract conv (Conv->Add->Q is unfoldable), and fallback islands
            # misexecute on HTP. Costs <=0.5 LSB extra rounding to the s_out grid.
            y = g.dq(g.q(y, s_out, tag + "pre"), s_out, tag + "pre")
        y = g.add(y, xd)
        yq = g.q(y, s_out, tag, dbg_shape=[C, hh, ww] if dbg else None)
        return yq, s_out

    def k2s2b(xq, xs, wkey, bkey, skey, q0, q1, tag, hh, ww, dbg):
        """split conv into filter halves -> two int8 tensors at q0/q1"""
        xd = g.dq(xq, xs, tag)
        Wf = L[wkey]; b = L[bkey]; C = Wf.shape[0]
        outs = []
        for i, qf in enumerate([q0, q1]):
            if g.DEGCONV:
                # k2s2 conv == k3 s2 p1 conv: out[y] = sum W3[t]*in[2y+t-1], so the
                # original 2x2 taps live at kernel positions 1..2 (5 border taps zero).
                Wh = Wf[i*C//2:(i+1)*C//2]  # [O, Cin, 2, 2]
                W3 = np.zeros((Wh.shape[0], Wh.shape[1], 3, 3), np.float32)
                W3[:,:, 1:3, 1:3] = Wh
                y = g.conv(xd, g.dqw(W3, sc(skey)), b[i*C//2:(i+1)*C//2], tag+f"h{i}", k=3, s=2, p=1)
            else:
                y = g.conv(xd, g.dqw(Wf[i*C//2:(i+1)*C//2], sc(skey)), b[i*C//2:(i+1)*C//2], tag+f"h{i}", k=2, s=2)
            outs.append(g.q(y, qf, tag + f"h{i}q", dbg_shape=[C//2, hh, ww] if dbg else None))
        return outs  # [int8 half0, int8 half1]

    def fnb(xg, prefix, q0, q1, act, tag, hh, ww, C, variant, o0=None, o1=None, uniform_out=None, dbg=None):
        h0q, h1q = xg
        h0 = g.dq(h0q, q0, tag); h1 = g.dq(h1q, q1, tag)
        grp = 1 if variant == "32" else 2
        Wsp = L[f"{prefix}_spatial_weight"]; bsp = L[f"{prefix}_spatial_bias"]
        ssp = sc(f"scale__{prefix}_spatial")
        if grp == 2 and g.DEGCONV:
            # Grouped conv == group=1 conv with zero-filled cross-group weights:
            # rows [0:Og) only tap inputs [0:Ci), rows [Og:) only tap [Ci:2Ci).
            Ci = Wsp.shape[1]          # input channels per group
            Og = Wsp.shape[0] // 2     # output channels per group
            Wfull = np.zeros((Wsp.shape[0], 2*Ci, 3, 3), np.float32)
            Wfull[:Og,:Ci] = Wsp[:Og]
            Wfull[Og:, Ci:2*Ci] = Wsp[Og:]
            y = g.conv(h0, g.dqw(Wfull, ssp), bsp, tag + "sp", k=3, p=1)
        else:
            y = g.conv(h0, g.dqw(Wsp, ssp), bsp, tag + "sp", k=3, p=1, group=grp)
        yq = g.q(y, q1, tag + "sp"); yd = g.dq(yq, q1, tag + "sp")
        cat = g.concat(yd, h1)
        y = g.conv(cat, g.dqw(L[f"{prefix}_pw_expand_weight"], sc(f"scale__{prefix}_pw_expand")),
                   L[f"{prefix}_pw_expand_bias"], tag + "pe", k=1)
        yq = g.q(y, act, tag + "pe"); yd = g.dq(yq, act, tag + "pe"); yd = g.relu(yd)
        y = g.conv(yd, g.dqw(L[f"{prefix}_pw_contract_weight"], sc(f"scale__{prefix}_pw_contract")),
                   L[f"{prefix}_pw_contract_bias"], tag + "pc", k=1)
        if uniform_out is not None:
            if g.QDQADD:
                y = g.dq(g.q(y, uniform_out, tag + "pre"), uniform_out, tag + "pre")
            y = g.add(y, g.concat(h0, h1))
            return g.q(y, uniform_out, tag + "o", dbg_shape=[C, hh, ww] if dbg else None), uniform_out
        if g.QDQADD:
            # per-half pre-Q at each half's output scale (slice commutes with add)
            ya = g.dq(g.q(g.slice_ch(y, 0, C//2), o0, tag + "pre0"), o0, tag + "pre0")
            yb = g.dq(g.q(g.slice_ch(y, C//2, C), o1, tag + "pre1"), o1, tag + "pre1")
            y = g.concat(ya, yb)
        y = g.add(y, g.concat(h0, h1))
        res = []
        for i, qf in enumerate([o0, o1]):
            ys = g.slice_ch(y, i*C//2, (i+1)*C//2)
            res.append(g.q(ys, qf, tag + f"o{i}", dbg_shape=[C//2, hh, ww] if dbg else None))
        return res  # [int8 half0, int8 half1]

    def fnb_ct2d(xg, prefix, skip_q, skip_s, q0, q1, act, s_fnb, tag, hh, ww, C, variant,
                 wct, bct, sct, o0=None, o1=None, uniform_out=None, dbg=None):
        yq, _ = fnb(xg, prefix, q0, q1, act, tag + "f", hh, ww, C, variant, uniform_out=s_fnb)
        yd = g.dq(yq, s_fnb, tag)
        y = g.convT(yd, g.dqw(L[wct], sc(sct)), L[bct])  # bias folded into the op
        if g.QDQADD:
            if uniform_out is not None:
                y = g.dq(g.q(y, uniform_out, tag + "pre"), uniform_out, tag + "pre")
            else:
                y0 = g.dq(g.q(g.slice_ch(y, 0, C//2), o0, tag + "pre0"), o0, tag + "pre0")
                y1 = g.dq(g.q(g.slice_ch(y, C//2, C), o1, tag + "pre1"), o1, tag + "pre1")
                y = g.concat(y0, y1)
        y = g.add(y, g.dq(skip_q, skip_s, tag))
        if uniform_out is not None:
            return g.q(y, uniform_out, tag + "o", dbg_shape=[C, hh, ww] if dbg else None), uniform_out
        res = []
        for i, qf in enumerate([o0, o1]):
            ys = g.slice_ch(y, i*C//2, (i+1)*C//2)
            res.append(g.q(ys, qf, tag + f"o{i}", dbg_shape=[C//2, hh, ww] if dbg else None))
        return res

    calls = L["_calls"]; c = [calls[i]["args"] for i in range(len(calls))]
    H4, W4 = H2 // 2, W2 // 2
    H8, W8 = H2 // 4, W2 // 4

    yq, ys = cnb(x, "encoder2_RB0", c[0][1], c[0][3], 0.01933070458471775, "p1", 16, H2, W2, debug and "p1")
    yq, ys = cnb(g.dq(yq, ys, "p1d"), "encoder2_RB1", c[1][1], c[1][3], 0.02335178479552269, "p2", 16, H2, W2, debug and "p2")
    skip2 = (yq, ys)
    xg = k2s2b(yq, ys, "enc2_ds_weight", "enc2_ds_bias", "scale__enc2_ds", c[2][0], c[2][1], "p3", H4, W4, debug and "p3")
    xg = fnb(xg, "encoder3_ResidualBlock_0", c[3][0], c[3][1], c[3][2], "p4", H4, W4, 32, "32",
             o0=c[3][3], o1=c[3][4], dbg=debug and "p4")
    yq5, ys5 = fnb(xg, "encoder3_ResidualBlock_1", c[4][0], c[4][1], c[4][2], "p5", H4, W4, 32, "32",
                   uniform_out=0.019347405061125755, dbg=debug and "p5")
    skip3 = (yq5, ys5)
    xg = k2s2b(yq5, ys5, "enc3_ds_weight", "enc3_ds_bias", "scale__enc3_ds", c[5][0], c[5][1], "p6", H8, W8, debug and "p6")
    xg = fnb(xg, "bottleneck_ResidualBlock_0", c[6][0], c[6][1], c[6][2], "p7", H8, W8, 64, "64",
             o0=c[6][3], o1=c[6][4], dbg=debug and "p7")
    xg = fnb(xg, "bottleneck_ResidualBlock_1", c[7][0], c[7][1], c[7][2], "p8", H8, W8, 64, "64",
             o0=c[7][3], o1=c[7][4], dbg=debug and "p8")
    xg = fnb_ct2d(xg, "bottleneck_ResidualBlock_2", skip3[0], skip3[1], c[8][0], c[8][1], c[8][2], c[8][3],
                  "p9", H4, W4, 32, "64", "bottleneck_ct_weight", "bottleneck_ct_bias", "scale__bottleneck_ct",
                  o0=c[8][4], o1=c[8][5], dbg=debug and "p9")
    xg = fnb(xg, "decoder3_ResidualBlock_1", c[9][0], c[9][1], c[9][2], "p10", H4, W4, 32, "32",
             o0=c[9][3], o1=c[9][4], dbg=debug and "p10")
    yq11, ys11 = fnb_ct2d(xg, "decoder3_ResidualBlock_2", skip2[0], skip2[1], c[10][0], c[10][1], c[10][2], c[10][3],
                          "p11", H2, W2, 16, "32", "dec3_ct_weight", "dec3_ct_bias", "scale__dec3_ct",
                          uniform_out=0.027206305414438248, dbg=debug and "p11")
    yq12, ys12 = cnb(g.dq(yq11, ys11, "p11d"), "decoder2_RB1", c[11][1], c[11][3], 0.035173576325178146,
                     "p12", 16, H2, W2, debug and "p12")
    yq13, ys13 = cnb(g.dq(yq12, ys12, "p12d"), "decoder2_ResidualBlock_2", c[12][1], c[12][3],
                     0.06884194910526276, "p13c", 16, H2, W2, False)
    yd = g.dq(yq13, ys13, "p13d")
    y = g.convT(yd, g.dqw(L["dec2_ct_weight"], sc("scale__dec2_ct")), L["dec2_ct_bias"])
    out = g.node("Cast", [y], to=TP.FLOAT16)

    graph = helper.make_graph(
        g.nodes, "fsr4_i8",
        [helper.make_tensor_value_info(X, TP.FLOAT if (g.FIO or g.FLOAT) else TP.INT8, [1, 16, H2, W2])],
        [helper.make_tensor_value_info(out, TP.FLOAT16, [1, 8, H, W])] + g.dbgouts,
        initializer=g.inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model

if __name__ == "__main__":
    import sys
    L = S.load_layers(); L = S.load_with_pass0_scale(L)
    G.FIO = "--fio" in sys.argv
    G.FLOAT = "--float" in sys.argv
    if "--fp16" in sys.argv:  # implies FLOAT with fp16 weights/activations
        G.FLOAT = True
        G.D = np.float16
    if "--noexpsat" in sys.argv:  # QDQ path: let the converter fold Q/DQ (HTP uFxp)
        G.EXPLICIT_SAT = False
    if "--degconv" in sys.argv:  # decompose grouped + k2s2 convs into HTP-native ops
        G.DEGCONV = True
    if "--qdqadd" in sys.argv:  # Q before residual Adds (kills float-fallback islands)
        G.QDQADD = True
    dbg_flag = "--dbg" in sys.argv
    G.DBG_TAPS = "--taps" in sys.argv
    model = build(L, debug=G.DBG_TAPS)
    if "--fp16" in sys.argv and "--noexpsat" in sys.argv:
        out_name = "fsr4_f16_nosat.onnx"
    elif "--fp16" in sys.argv:
        out_name = "fsr4_f16.onnx"
    elif "--qdqadd" in sys.argv:
        out_name = "fsr4_fioq3.onnx"
    elif "--degconv" in sys.argv:
        out_name = "fsr4_fioq2.onnx"
    elif "--noexpsat" in sys.argv:
        out_name = "fsr4_fioq.onnx"
    else:
        out_name = "fsr4_float.onnx" if G.FLOAT else ("fsr4_fio_dbg.onnx" if dbg_flag else ("fsr4_fio.onnx" if G.FIO else "fsr4_i8_1080.onnx"))
    onnx.save(model, ART / out_name)
    print("wrote", out_name)
