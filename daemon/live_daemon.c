// live_daemon.c — in-game bridge.
#define _GNU_SOURCE /* bionic: cpu_set_t / pthread_setaffinity_np */
//
// TCP loopback daemon (127.0.0.1:48620) that serves the service contract
// pipeline to the x86_64 nvngx proxy running in the game (Wine/Box64):
// per frame, the proxy sends:
// color f4 float32 RGBA 960x540 (8,294,400 B) (proxy's fixed-f11 decode)
// mvec f16 u16 RG pairs 960x540 (2,073,600 B) (proxy's dump-path encode)
// the daemon:
// - converts color -> fp16 RGBA (RNE, bank parity: same value chain as
// the reference pipeline's float32-decode -> astype(F16))
// - synthesizes reproj per the reference pipeline math (float32,
// -ffp-contract=off, op order matched to numpy):
// vel_uv = mv / (960, 540); vel = vel_uv at the LR texel covering
// (0.5x, 0.5y); (hx,hy) = (x,y) + vel*(W,H); onscreen mask;
// reproj = bilinear(prev_out, clamp(hx,hy)); 0 outside; A=1
// prev_out = bilinear-2x of the PREVIOUS frame's color (frame 0:
// self-seed) -- the exact semantics the reference implementation was validated against
// - fsr4_execute(lr, mvec, reproj, out) (contract; bootstrap first frame)
// - sends out (fp16 RGBA 1920x1080, 16,588,800 B) back
// Protocol per frame: client -> [8B len-prefixed color][8B len-prefixed mvec];
// daemon -> [8B len-prefixed rgb]. All lengths little-endian u64 (wire-self-
// describing; oversized lengths rejected). One client at a time; a new
// connection replaces a dead one and re-arms the temporal state (invalidate).
//
// Build: aarch64-linux-android28-clang.cmd -O2 -ffp-contract=off
// -Dstatic_assert=_Static_assert -I$QNN_SDK_ROOT/include/QNN
// live_daemon.c qnn_service.c -o live_daemon -lEGL -lGLESv3 -ldl -llog
// (qnn_service.c built with -DFSR4_NO_MAIN)

#include <arpa/inet.h>
#include <errno.h>
#include <math.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <pthread.h>
#include <sched.h>
#include <semaphore.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#include "fsr4_service.h"

#if defined(__ARM_NEON) && defined(__aarch64__)
#include <arm_neon.h>
#endif

/* compile-time overridable dims for the 720p arm. Override via
 * -DFSR4D_LW=640 -DFSR4D_LH=360 -DFSR4D_W=1280 -DFSR4D_H=720 (indirected
 * names: bare H/W -D flags would clobber 'uint32_t H' params in the
 * service translation unit compiled in the same command). */
#ifdef FSR4D_LW
#define LW FSR4D_LW
#define LH FSR4D_LH
#define W FSR4D_W
#define H FSR4D_H
#else
#define LW 960
#define LH 540
#define W 1920
#define H 1080
#endif
#define COLOR_F4_BYTES ((size_t)LW * LH * 4 * 4) /* 8,294,400 */
#define MV_BYTES ((size_t)LW * LH * 2 * 2) /* 2,073,600 */
#define LR_BYTES ((size_t)LH * LW * 4 * 2) /* 4,147,200 */
#define MVF_BYTES ((size_t)LH * LW * 2 * 2) /* 2,073,600 */
#define REP_BYTES ((size_t)H * W * 4 * 2) /* 16,588,800 */
#define OUT_BYTES ((size_t)H * W * 4 * 2) /* 16,588,800 */
#define DEF_PORT 48620

static double now_ms(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
 return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6; }

static int read_full(int fd, void* b, size_t n) {
 unsigned char* p = (unsigned char*)b; size_t got = 0;
 while (got < n) {
 ssize_t r = recv(fd, p + got, n - got, 0);
 if (r == 0) return -1;
 if (r < 0) { if (errno == EINTR) continue; return -1; }
 got += (size_t)r;
 }
 return 0;
}
static int write_full(int fd, const void* b, size_t n) {
 const unsigned char* p = (const unsigned char*)b; size_t put = 0;
 while (put < n) {
 ssize_t r = send(fd, p + put, n - put, MSG_NOSIGNAL);
 if (r < 0) { if (errno == EINTR) continue; return -1; }
 put += (size_t)r;
 }
 return 0;
}
static int recv_frame(int fd, void* b, size_t expect) {
 uint64_t len;
 if (read_full(fd, &len, 8) != 0) return -1;
 if (len != (uint64_t)expect) {
 fprintf(stderr, "live_daemon: bad frame len %llu (expect %zu)\n",
 (unsigned long long)len, expect);
 return -1;
 }
 return read_full(fd, b, expect);
}
static int send_frame(int fd, const void* b, size_t n) {
 uint64_t len = (uint64_t)n;
 if (write_full(fd, &len, 8) != 0) return -1;
 return write_full(fd, b, n);
}

/* ---- wire v2: raw R11G11B10F texels inbound (2,073,600 B) + RGBA8 outbound
 * (8,294,400 B). Auto-detected per frame by the color length prefix:
 * 8,294,400 = v1 decoded-f4 floats, 2,073,600 = v2 raw u32 texels. The old
 * proxy keeps working unchanged (v1); a v2 proxy gets 4x less traffic each
 * way. ---- */

/* fixed-f11 decode of one R11G11B10F texel -> f4 RGBA (alpha 1). Identical
 * exact-dyadic math to the proxy/live_client decode (A/B-validated and
 * GL1 140/140 byte-verified vs the bank chain). */
static float f11_decode(int m, int e, int mbits) {
 if (e == 0) return (float)m / (float)(1u << (14 + mbits));
 if (e >= 31) return 65024.0f;
 if (e >= 15) return (1.0f + (float)m / (float)(1u << mbits)) * (float)(1u << (e - 15));
 return (1.0f + (float)m / (float)(1u << mbits)) / (float)(1u << (15 - e));
}
static void decode_texel(unsigned int u, float* out4) {
 unsigned int rr = u & 0x7FFu, gg = (u >> 11) & 0x7FFu, bb = (u >> 22) & 0x3FFu;
 out4[0] = f11_decode((int)(rr & 0x3Fu), (int)((rr >> 6) & 0x1Fu), 6);
 out4[1] = f11_decode((int)(gg & 0x3Fu), (int)((gg >> 6) & 0x1Fu), 6);
 out4[2] = f11_decode((int)(bb & 0x1Fu), (int)((bb >> 5) & 0x1Fu), 5);
 out4[3] = 1.0f;
}

/* perf: R11G11B10F -> fp16 is a pure bit relocation —
 * f11/f10 and fp16 share exponent bias 15, mantissas shift exactly
 * (f11 m<<4, f10 m<<5), e==0 subnormals land exactly in the fp16 subnormal
 * field, and e==31 saturates to 65024 (0x7BFF) like f11_decode. Produces the
 * SAME bits as decode_texel -> (_Float16) with alpha 0x3C00, ~20x faster.
 * FSR4_NEONCV=0 falls back to the scalar chain for parity A/B. */
#if defined(__ARM_NEON) && defined(__aarch64__)
#include <arm_neon.h>
static inline uint16x4_t f11x4_to_f16(uint32x4_t f) {
 uint32x4_t e = vshrq_n_u32(f, 6);
 uint32x4_t m = vandq_u32(f, vdupq_n_u32(0x3Fu));
 uint32x4_t h = vorrq_u32(vshlq_n_u32(e, 10), vshlq_n_u32(m, 4));
 uint32x4_t s31 = vceqq_u32(e, vdupq_n_u32(31u));
 h = vbslq_u32(s31, vdupq_n_u32(0x7BFFu), h);
 return vmovn_u32(h);
}
static inline uint16x4_t f10x4_to_f16(uint32x4_t f) {
 uint32x4_t e = vshrq_n_u32(f, 5);
 uint32x4_t m = vandq_u32(f, vdupq_n_u32(0x1Fu));
 uint32x4_t h = vorrq_u32(vshlq_n_u32(e, 10), vshlq_n_u32(m, 5));
 uint32x4_t s31 = vceqq_u32(e, vdupq_n_u32(31u));
 h = vbslq_u32(s31, vdupq_n_u32(0x7BFFu), h);
 return vmovn_u32(h);
}
static void rawtex_to_lr16(const unsigned int* raw, unsigned short* dst) {
 const uint32x4_t m11 = vdupq_n_u32(0x7FFu);
 const uint16x4_t a1 = vdup_n_u16(0x3C00u); /* (_Float16)1.0f */
 for (size_t t = 0; t < (size_t)LW * LH; t += 4) {
 uint32x4_t w = vld1q_u32(raw + t);
 uint16x4_t r = f11x4_to_f16(vandq_u32(w, m11));
 uint16x4_t g = f11x4_to_f16(vandq_u32(vshrq_n_u32(w, 11), m11));
 uint16x4_t b = f10x4_to_f16(vshrq_n_u32(w, 22));
 uint16x4x4_t rgba = { { r, g, b, a1 } };
 vst4_u16(dst + t * 4, rgba);
 }
}
/* MV normalize: x/y lane interleaved [x0 y0 x1 y1], vdivq is IEEE
 * correctly-rounded like the scalar '/', vcvt RNE like _Float16. */
static void mvnorm_neon(const unsigned short* src, unsigned short* dst) {
 const float dv_arr[4] = { (float)LW, (float)LH, (float)LW, (float)LH };
 const float32x4_t dv = vld1q_f32(dv_arr);
 for (size_t i = 0; i < (size_t)LW * LH * 2; i += 8) {
 uint16x8_t h = vld1q_u16(src + i);
 float32x4_t lo = vdivq_f32(vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(h))), dv);
 float32x4_t hi = vdivq_f32(vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(h))), dv);
 vst1q_u16(dst + i, vcombine_u16(vreinterpret_u16_f16(vcvt_f16_f32(lo)),
 vreinterpret_u16_f16(vcvt_f16_f32(hi))));
 }
}
#endif
static int neoncv = 1; /* FSR4_NEONCV=0 -> scalar chain (parity A/B) */

/* Name + pin each pipeline thread. 8 Gen 2 layout: cpu7=X3,
 * cpu3-6 = 2xA715+2xA710, cpu0-2 = A510. GL thread gets the prime core
 * (every dispatch/readback segment is driver CPU work + kgsl fence wakeups
 * land on an otherwise-idle core); net and sender get dedicated mids.
 * FSR4_AFFINITY=0 disables; FSR4_{GL,NET,SEND}_CPU override. */
static void pin_thread(const char* name, const char* env, int def_cpu) {
 pthread_setname_np(pthread_self(), name);
 const char* a = getenv("FSR4_AFFINITY");
 if (a && !atoi(a)) return;
 int cpu = def_cpu;
 const char* e = getenv(env);
 if (e) cpu = atoi(e);
 if (cpu < 0 || cpu > 7) return;
 cpu_set_t set;
 CPU_ZERO(&set);
 CPU_SET(cpu, &set);
 if (sched_setaffinity(0, sizeof set, &set) != 0)
 fprintf(stderr, "live_daemon: %s pin cpu%d FAILED errno=%d\n", name, cpu, errno);
 else
 printf("live_daemon: %s pinned to cpu%d\n", name, cpu);
}

/* ---- quality helpers ---- */
static float g_cut_mv = 500.0f; /* FSR4_CUTMV: absolute max |mv| (LR px) cut
 * fallback — fix: 20 fired on
 * 67% of real gameplay frames (bank p50 max
 * 39px, p90 145px during pans); only a true
 * teleport clears 500 */
static float g_cut_frac = 0.05f; /* FSR4_CUTFRAC: reset when >5% of texels move
 * >100px — a CUT moves everything, a pan
 * doesn't (0 disables the fraction test) */
static int g_autoexp = 0; /* FSR4_AUTOEXP=1: SPD-style EV100 exposure
 * from the LR (A/B — default off) */

/* max |mv| and fraction of halves above 100px, one fused NEON scan */
static void mvstats_halves(const unsigned short* mv, size_t n, float* maxv, float* frac100) {
 float m = 0.0f;
 size_t big = 0;
#if defined(__ARM_NEON) && defined(__aarch64__)
 float32x4_t mx = vdupq_n_f32(0.0f);
 uint32x4_t cnt = vdupq_n_u32(0);
 const float32x4_t v100 = vdupq_n_f32(100.0f);
 const uint32x4_t one = vdupq_n_u32(1);
 size_t i = 0;
 for (; i + 8 <= n; i += 8) {
 uint16x8_t h = vld1q_u16(mv + i);
 float32x4_t lo = vabsq_f32(vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(h))));
 float32x4_t hi = vabsq_f32(vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(h))));
 mx = vmaxq_f32(mx, vmaxq_f32(lo, hi));
 /* (third-generation fix): vcgtq returns MASKS,
 * not 1.0 floats — v21b added masks as u32 (=> ~2^32 counts), v21c
 * added them as f32 (NaN => count 0 = fraction rule dead). Mask&1
 * -> u32 accumulate is the correct form; the new
 * -flax-vector-conversions=none build flag makes the whole mistake
 * class a compile error. */
 cnt = vaddq_u32(cnt, vaddq_u32(vandq_u32(vcgtq_f32(lo, v100), one),
 vandq_u32(vcgtq_f32(hi, v100), one)));
 }
 m = vmaxvq_f32(mx);
 big = (size_t)vaddvq_u32(cnt);
#endif
 for (; i < n; i++) {
 _Float16 h; memcpy(&h, &mv[i], 2);
 float v = fabsf((float)h);
 if (v > m) m = v;
 if (v > 100.0f) big++;
 }
 *maxv = m;
 *frac100 = (float)big / (float)n;
}

/* SPD auto-exposure (spd_auto_exposure.hlsl:3-37): Lavg from a strided
 * scan of the fp16 LR, EV100 = log2(Lavg*100/12.5), e = 1/(1.2*2^EV100),
 * clamped [0.25, 4]. Runs on the net thread (~0.5ms). */
static float autoexp_lr16(const unsigned short* lr16) {
 double acc = 0.0;
 size_t n = 0;
 for (size_t t = 0; t < (size_t)LW * LH; t += 4) {
 _Float16 r, g, b;
 memcpy(&r, &lr16[t * 4], 2); memcpy(&g, &lr16[t * 4 + 1], 2); memcpy(&b, &lr16[t * 4 + 2], 2);
 acc += 0.2126 * (double)r + 0.7152 * (double)g + 0.0722 * (double)b;
 n++;
 }
 float lavg = (float)(acc / (double)n);
 if (lavg <= 1e-6f) return 1.0f;
 float ev100 = log2f(lavg * (100.0f / 12.5f));
 float e = 1.0f / (1.2f * exp2f(ev100));
 if (e < 0.25f) e = 0.25f;
 if (e > 4.0f) e = 4.0f;
 return e;
}

#define COLOR_RAW_BYTES ((size_t)LW * LH * 4) /* 2,073,600 */
#define OUT8_BYTES ((size_t)H * W * 4) /* 8,294,400 */

/* RAWLR: per-slot raw texel buffers + mode flag (file scope — the serve
 * thread reads the slot's raw buffer directly when the v2 shaders decode
 * in-GPU; net thread fills slot N+2 only after N was consumed) */
static unsigned int g_rawtex_s[2][LW * LH];
static int g_rawlr = -1;

/* v3: raw f32 MV rows from the proxy (the dll's emulated SSE2
 * convert leaves the game thread entirely). Per-slot like rawtex. */
#define MVF32_BYTES ((size_t)LW * LH * 2 * 4) /* 4,147,200 f32 xy pairs */
#define MVF32_WIRE ((uint64_t)MVF32_BYTES + 24) /* + jx,jy,sx,sy tail */
static float g_mvf32_s[2][LW * LH * 2];

/* f32 rows -> SCALED f16 halves (x*sx, y*sy) into mvraw — the SAME
 * first quantize the dll's mv_f32row_to_f16 did; the existing mvnorm_neon
 * then runs unchanged on mvraw => bit-identical chain to wire v2 by
 * construction. vmul+vcvt only (strict-convert build). */
static void mvf32_to_mvraw_neon(const float* src, unsigned short* dst, float sx, float sy) {
#if defined(__ARM_NEON) && defined(__aarch64__)
 const float scl[4] = { sx, sy, sx, sy };
 const float32x4_t sc = vld1q_f32(scl);
 size_t i = 0;
 for (; i + 4 <= (size_t)LW * LH * 2; i += 4) {
 float32x4_t v = vmulq_f32(vld1q_f32(src + i), sc);
 vst1_u16(dst + i, vreinterpret_u16_f16(vcvt_f16_f32(v)));
 }
 for (; i < (size_t)LW * LH * 2; i++) {
 _Float16 h = (_Float16)(src[i] * ((i & 1) ? sy: sx));
 memcpy(&dst[i], &h, 2);
 }
#else
 for (size_t i = 0; i < (size_t)LW * LH * 2; i++) {
 _Float16 h = (_Float16)(src[i] * ((i & 1) ? sy: sx));
 memcpy(&dst[i], &h, 2);
 }
#endif
}

/* ---- NEON synthesis ----
 * Bit-exactness rules for the vector paths:
 * - vmul/vadd only, NEVER fmla/fmas (the reference build is -ffp-contract=off:
 * every scalar mul+add rounds the product before the add).
 * - same operand order as the scalar reference (a*b + c*d, top before bot,
 * top term before bot term in the final lerp).
 * - fy is always exactly {0, 0.5} and fx {0, 0.5} here (py = yc*0.5), so all
 * weight products are exact powers of two — alpha lane comes out 1.0
 * bit-exactly, matching the scalar o[3] = 1.0f.
 * - reference quirk: bot = c01*(1-fy) + c11*fy (fy in BOTH terms); consecutive
 * output texels xc/xc+1 share x0 (fx = 0 then 0.5) and share bot. ---- */
#if defined(__ARM_NEON) && defined(__aarch64__)
#include <arm_neon.h>

static void upscale2x_f32(const float* lr, float* dst /* [H][W][4] */) {
 for (int yc = 0; yc < H; yc++) {
 float py = (float)yc * 0.5f;
 int y0 = (int)floorf(py);
 float fy = py - (float)y0;
 float gy = 1.0f - fy;
 int y0c = y0 < 0 ? 0: (y0 > LH - 1 ? LH - 1: y0);
 int y1c = y0 + 1 < 0 ? 0: (y0 + 1 > LH - 1 ? LH - 1: y0 + 1);
 const float* r00 = lr + (size_t)y0c * LW * 4;
 const float* r01 = lr + (size_t)y1c * LW * 4;
 float* dstrow = dst + (size_t)yc * W * 4;
 for (int xc = 0; xc + 2 <= W; xc += 2) {
 float px = (float)xc * 0.5f;
 int x0 = (int)px;
 int x0c = x0 < 0 ? 0: (x0 > LW - 1 ? LW - 1: x0);
 int x1c = x0 + 1 < 0 ? 0: (x0 + 1 > LW - 1 ? LW - 1: x0 + 1);
 float32x4_t c00 = vld1q_f32(r00 + (size_t)x0c * 4);
 float32x4_t c10 = vld1q_f32(r00 + (size_t)x1c * 4);
 float32x4_t c01 = vld1q_f32(r01 + (size_t)x0c * 4);
 float32x4_t c11 = vld1q_f32(r01 + (size_t)x1c * 4);
 /* bot shared by both output texels (fy in both terms) */
 float32x4_t bot = vaddq_f32(vmulq_n_f32(c01, gy), vmulq_n_f32(c11, fy));
 float* o = dstrow + (size_t)xc * 4;
 /* even texel: fx = 0 -> top = c00 */
 vst1q_f32(o, vaddq_f32(vmulq_n_f32(c00, gy), vmulq_n_f32(bot, fy)));
 /* odd texel: px = x0 + 0.5 ALWAYS (pair base is even), so the top
 * weight is the fixed 0.5 — the pair's own fx is 0 and must not
 * be reused here */
 float32x4_t top1 = vaddq_f32(vmulq_n_f32(c00, 0.5f), vmulq_n_f32(c10, 0.5f));
 vst1q_f32(o + 4, vaddq_f32(vmulq_n_f32(top1, gy), vmulq_n_f32(bot, fy)));
 }
 if (W & 1) { /* LW*2 == W so this never fires; keep for safety */
 int xc = W - 1;
 float px = (float)xc * 0.5f;
 int x0 = (int)floorf(px);
 float fx = px - (float)x0, gx = 1.0f - fx;
 int x0c = x0 < 0 ? 0: (x0 > LW - 1 ? LW - 1: x0);
 int x1c = x0 + 1 < 0 ? 0: (x0 + 1 > LW - 1 ? LW - 1: x0 + 1);
 const float* c00 = r00 + (size_t)x0c * 4;
 const float* c10 = r00 + (size_t)x1c * 4;
 const float* c01 = r01 + (size_t)x0c * 4;
 const float* c11 = r01 + (size_t)x1c * 4;
 float* o = dstrow + (size_t)xc * 4;
 for (int c = 0; c < 3; c++) {
 float top = c00[c] * gx + c10[c] * fx;
 float bot = c01[c] * gy + c11[c] * fy;
 o[c] = top * gy + bot * fy;
 }
 o[3] = 1.0f;
 }
 }
}

/* fp16 out -> f32 (exact): history feedback buffer */
static void out16_to_f32(const unsigned short* src, float* dst) {
 size_t n = (size_t)H * W * 4, i = 0;
 for (; i + 8 <= n; i += 8) {
 uint16x8_t h = vld1q_u16(src + i);
 vst1q_f32(dst + i, vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(h))));
 vst1q_f32(dst + i + 4, vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(h))));
 }
 for (; i < n; i++) { _Float16 h; memcpy(&h, &src[i], 2); dst[i] = (float)h; }
}

/* fp16 RGBA out -> RGBA8 (proxy encode_px fmt28 math, truncation via FRINTZ —
 * all post-clamp values are positive so floor == truncate) */
static void out16_to_u8(const unsigned short* src, unsigned char* dst) {
 size_t n = (size_t)H * W * 4;
 size_t i = 0;
 float32x4_t v255 = vdupq_n_f32(255.0f), vhalf = vdupq_n_f32(0.5f);
 float32x4_t vzero = vdupq_n_f32(0.0f), vone = vdupq_n_f32(1.0f);
 for (; i + 8 <= n; i += 8) {
 uint16x8_t h = vld1q_u16(src + i);
 float32x4_t lo = vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(h)));
 float32x4_t hi = vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(h)));
 float32x4_t r[2] = { lo, hi };
 uint8x8_t outb;
 uint32x4_t wb[2];
 for (int k = 0; k < 2; k++) {
 /* v<=0 -> 0; v>=1 -> 255; else trunc(v*255+0.5) */
 uint32x4_t is1 = vcgeq_f32(r[k], vone);
 uint32x4_t is0 = vcleq_f32(r[k], vzero);
 float32x4_t scaled = vaddq_f32(vmulq_f32(r[k], v255), vhalf);
 uint32x4_t truncd = vcvtmq_u32_f32(scaled); /* FRINTM: floor == trunc for positives */
 uint32x4_t res = vbslq_u32(is1, vcvtq_u32_f32(v255), truncd);
 res = vbslq_u32(is0, vdupq_n_u32(0), res);
 wb[k] = res;
 }
 outb = vqmovn_u16(vcombine_u16(vqmovn_u32(wb[0]), vqmovn_u32(wb[1])));
 vst1_u8(dst + i, outb);
 }
 for (; i < n; i++) {
 _Float16 h; memcpy(&h, &src[i], 2);
 float v = (float)h;
 dst[i] = (unsigned char)(v <= 0.0f ? 0: v >= 1.0f ? 255: v * 255.0f + 0.5f);
 }
}

#else /* scalar fallback = the reference implementation */

static void upscale2x_f32(const float* lr, float* dst /* [H][W][4] */) {
 for (int yc = 0; yc < H; yc++) {
 float py = (float)yc * 0.5f;
 int y0 = (int)floorf(py);
 float fy = py - (float)y0;
 int y0c = y0 < 0 ? 0: (y0 > LH - 1 ? LH - 1: y0);
 int y1c = y0 + 1 < 0 ? 0: (y0 + 1 > LH - 1 ? LH - 1: y0 + 1);
 for (int xc = 0; xc < W; xc++) {
 float px = (float)xc * 0.5f;
 int x0 = (int)floorf(px);
 float fx = px - (float)x0;
 int x0c = x0 < 0 ? 0: (x0 > LW - 1 ? LW - 1: x0);
 int x1c = x0 + 1 < 0 ? 0: (x0 + 1 > LW - 1 ? LW - 1: x0 + 1);
 const float* c00 = lr + ((size_t)y0c * LW + x0c) * 4;
 const float* c10 = lr + ((size_t)y0c * LW + x1c) * 4;
 const float* c01 = lr + ((size_t)y1c * LW + x0c) * 4;
 const float* c11 = lr + ((size_t)y1c * LW + x1c) * 4;
 float* o = dst + ((size_t)yc * W + xc) * 4;
 for (int c = 0; c < 3; c++) {
 float top = c00[c] * (1.0f - fx) + c10[c] * fx;
 float bot = c01[c] * (1.0f - fy) + c11[c] * fy; /* fy in both */
 o[c] = top * (1.0f - fy) + bot * fy;
 }
 o[3] = 1.0f;
 }
 }
}

static void out16_to_f32(const unsigned short* src, float* dst) {
 size_t n = (size_t)H * W * 4;
 for (size_t i = 0; i < n; i++) { _Float16 h; memcpy(&h, &src[i], 2); dst[i] = (float)h; }
}

static void out16_to_u8(const unsigned short* src, unsigned char* dst) {
 size_t n = (size_t)H * W * 4;
 for (size_t i = 0; i < n; i++) {
 _Float16 h; memcpy(&h, &src[i], 2);
 float v = (float)h;
 dst[i] = (unsigned char)(v <= 0.0f ? 0: v >= 1.0f ? 255: v * 255.0f + 0.5f);
 }
}

#endif

/* ---- reproj synthesis: C port of the reference Python canonicalizer (float32, numpy op
 * order; build with -ffp-contract=off). All coordinates in floats exactly as
 * numpy emits them. ---- */
typedef struct { float x, y; } f2v;

/* bilinear RGBA sample; writes RGB as fp16 halves (alpha lane unused — the
 * caller pre-writes 1.0). Conversion = FCVT RNE, identical to the reference's
 * (_Float16)o[c] in the caller. mn/mx (NEON): optional history-rectification
 * hull — clamp the sample before the fp16 convert. */
static void bilinear_rgba_f32_c(const float* img /* [H][W][4] */, float px, float py,
 unsigned short* out3h,
 const float32x4_t* mn, const float32x4_t* mx) {
 int x0 = (int)floorf(px);
 int y0 = (int)floorf(py);
 float fx = px - (float)x0;
 float fy = py - (float)y0;
 float gfx = 1.0f - fx, gfy = 1.0f - fy;
 int x0c = x0 < 0 ? 0: (x0 > W - 1 ? W - 1: x0);
 int y0c = y0 < 0 ? 0: (y0 > H - 1 ? H - 1: y0);
 int x1c = x0 + 1 < 0 ? 0: (x0 + 1 > W - 1 ? W - 1: x0 + 1);
 int y1c = y0 + 1 < 0 ? 0: (y0 + 1 > H - 1 ? H - 1: y0 + 1);
 const float* c00 = img + ((size_t)y0c * W + x0c) * 4;
 const float* c10 = img + ((size_t)y0c * W + x1c) * 4;
 const float* c01 = img + ((size_t)y1c * W + x0c) * 4;
 const float* c11 = img + ((size_t)y1c * W + x1c) * 4;
#if defined(__ARM_NEON) && defined(__aarch64__)
 /* RGBA-lane vector; same mul/add order as the scalar reference (no fma) */
 float32x4_t v00 = vld1q_f32(c00), v10 = vld1q_f32(c10);
 float32x4_t v01 = vld1q_f32(c01), v11 = vld1q_f32(c11);
 float32x4_t top = vaddq_f32(vmulq_n_f32(v00, gfx), vmulq_n_f32(v10, fx));
 float32x4_t bot = vaddq_f32(vmulq_n_f32(v01, gfy), vmulq_n_f32(v11, fy));
 float32x4_t o = vaddq_f32(vmulq_n_f32(top, gfy), vmulq_n_f32(bot, fy));
 if (mn && mx) o = vminq_f32(vmaxq_f32(o, *mn), *mx);
 uint16_t h4[4];
 vst1_u16(h4, vreinterpret_u16_f16(vcvt_f16_f32(o)));
 memcpy(out3h, h4, 6);
#else
 for (int c = 0; c < 3; c++) { /* RGB only; alpha constant 1.0 */
 float top = c00[c] * (1.0f - fx) + c10[c] * fx;
 float bot = c01[c] * (1.0f - fy) + c11[c] * fy; /* fy in both terms */
 _Float16 h = (_Float16)(top * (1.0f - fy) + bot * fy);
 memcpy(&out3h[c], &h, 2);
 }
#endif
}
/* sample LR at (0.5x, 0.5y) -- reference upscale2x; implementation in the 
 * NEON section below (scalar fallback there is the byte-exact reference). */

/* ---- file-scope server state (v1 serial path + v2 threaded path) ---- */
static void *color_f4, *mv_f16, *lr16, *rep16, *out16, *mvh, *mvnorm;
static float *lr_f32, *prev_out, *cur_out_f;
static int mv_norm, skip_syn, jcomp, jzero, mvlog;
static float jsign, mv_dz;
static int first_frame = 1, have_prev = 0;
static unsigned long frames = 0;
static float pjx = 0.0f, pjy = 0.0f;
static int have_pj = 0;
static uint64_t g_first_clen = 0;
static int g_have_first = 0;

/* ---- depth-2 threaded pipeline (v2 clients) ----
 * The wire protocol is lock-step per request; the depth-2 PROXY (sender/
 * receiver split) puts req N+1 in flight while resp N is still being
 * produced. This side mirrors it: a NET thread reads+converts frame N into
 * slot N&1 while the GL thread runs frame N-1 through the service and sends
 * its response. Period = max(GL ~27ms + send ~3ms, net ~13ms) instead of
 * the sum. Slot safety: the net thread waits cons_sem[p] (posted at the END
 * of execute on that parity) before overwriting slot p two frames later —
 * by then the service consumed its inputs at execute start and the GL
 * thread already sent that parity's response. */
typedef struct {
 unsigned short* lr16;
 unsigned short* mvraw; /* received mv halves (also the !mv_norm path) */
 unsigned short* mvnorm;
 unsigned char* out8;
 float jx, jy;
 int got_jit;
 float expo; /* per-frame exposure (1 = off) */
 int reset; /* cut-reset request */
} PSlot;

typedef struct {
 int cfd;
 PSlot sl[2];
 sem_t in_sem;
 sem_t cons_sem[2];
 sem_t out_sem; /* GL -> sender: response ready */
 sem_t sent_sem[2]; /* sender -> GL: parity response sent */
 volatile int stop;
 volatile int ready_slot; /* parity the last SPLIT execute filled */
 unsigned long fidx, gidx, sidx, tframes;
} Pipe;

static void* net_thread(void* arg) {
 Pipe* P = (Pipe*)arg;
 pin_thread("fsr4-net", "FSR4_NET_CPU", 5);
 int rxd_diag = (getenv("FSR4_RXDIAG") && atoi(getenv("FSR4_RXDIAG")));
 /* FSR4_RAWLR=1 => skip rawtex_to_lr16 and hand the RAW texels to
 * the service (v2 shaders decode in-GPU; pairs with FSR4_FUSE=1).
 * Per-slot buffers at FILE scope: the net thread may fill frame N+2's
 * slot while the GL thread still serves N-1's. */
 if (g_rawlr < 0) {
 const char* e = getenv("FSR4_RAWLR");
 const char* fe = getenv("FSR4_FUSE");
 /* RAWLR without FUSE makes the service
 * upload 4.15MB from a 2.07MB raw slot (slot 1 reads past the array)
 * and the shader consumes raw texels as fp16 = garbage, no error. */
 g_rawlr = (e && atoi(e)) ? ((fe && atoi(fe)) ? 1: 0): 0;
 if (e && atoi(e) && !(fe && atoi(fe)))
 fprintf(stderr, "live_daemon: FSR4_RAWLR=1 needs FSR4_FUSE=1 - rawlr forced OFF (2x overread guard)");
 printf("live_daemon: rawlr=%d (fuse-gated)\n", g_rawlr);
 }
 for (;;) {
 int slot = (int)(P->fidx & 1UL);
 if (P->fidx >= 2UL) sem_wait(&P->cons_sem[slot]);
 uint64_t clen;
 if (g_have_first && P->fidx == 0UL) { clen = g_first_clen; g_have_first = 0; }
 else if (read_full(P->cfd, &clen, 8) != 0) { P->stop = 1; sem_post(&P->in_sem); break; }
 if (clen != (uint64_t)COLOR_RAW_BYTES) {
 fprintf(stderr, "live_daemon: threaded path got color len %llu (v2 only)\n",
 (unsigned long long)clen);
 P->stop = 1; sem_post(&P->in_sem); break;
 }
 if (read_full(P->cfd, g_rawtex_s[slot], COLOR_RAW_BYTES) != 0) { P->stop = 1; sem_post(&P->in_sem); break; }
 unsigned int* rawtex = g_rawtex_s[slot];
 PSlot* s = &P->sl[slot];
 int lr16_done = g_rawlr; /* rawlr skips the decode entirely */
#if defined(__ARM_NEON) && defined(__aarch64__)
 if (neoncv && !g_rawlr) { rawtex_to_lr16(rawtex, s->lr16); lr16_done = 1; } /* also fixed the rawlr double-decode */
#endif
 if (!lr16_done) {
 float* d = (float*)color_f4;
 for (size_t t = 0; t < (size_t)LW * LH; t++) decode_texel(rawtex[t], d + t * 4);
 const float* sp = (const float*)color_f4;
 unsigned short* dp = s->lr16;
 for (size_t i = 0; i < (size_t)LH * LW * 4; i++) {
 _Float16 h = (_Float16)sp[i];
 memcpy(&dp[i], &h, 2);
 }
 for (size_t i = 3; i < (size_t)LH * LW * 4; i += 4) {
 _Float16 h = (_Float16)1.0f;
 memcpy(&dp[i], &h, 2);
 }
 }
 uint64_t mlen;
 s->jx = 0.0f; s->jy = 0.0f; s->got_jit = 0;
 if (read_full(P->cfd, &mlen, 8) != 0) { P->stop = 1; sem_post(&P->in_sem); break; }
 if (mlen == (uint64_t)MV_BYTES) {
 if (read_full(P->cfd, s->mvraw, MV_BYTES) != 0) { P->stop = 1; sem_post(&P->in_sem); break; }
 } else if (mlen == (uint64_t)MV_BYTES + 16) {
 if (read_full(P->cfd, s->mvraw, MV_BYTES) != 0 ||
 read_full(P->cfd, &s->jx, 4) != 0 || read_full(P->cfd, &s->jy, 4) != 0) {
 P->stop = 1; sem_post(&P->in_sem); break;
 }
 s->got_jit = 1;
 } else if (mlen == MVF32_WIRE) {
 /* raw f32 rows + jx,jy,sx,sy tail. Convert into mvraw
 * (the same scaled-f16 form v2 carried), then the existing
 * mvnorm/mvstats pipeline runs unchanged. mlen is disjoint
 * from every v2 value => format discrimination. */
 float v3tail[4]; /* jx, jy, sx, sy — ONE 16B recv (was 4 syscalls) */
 if (read_full(P->cfd, g_mvf32_s[slot], MVF32_BYTES) != 0 ||
 read_full(P->cfd, v3tail, 16) != 0) {
 P->stop = 1; sem_post(&P->in_sem); break;
 }
 s->jx = v3tail[0]; s->jy = v3tail[1];
 float v3sx = v3tail[2], v3sy = v3tail[3];
 s->got_jit = 1;
 /* v3 debug tap: the deployed v22c showed a 549-555 MV cluster +
 * frac100 ~0.5 on fast pans since wire v3 went live (v2-era stats
 * were clean on the same scenes) — blame the wire or the convert.
 * First frames + every 900th: raw f32 lanes in, converted f16
 * lanes out, plus lanes from the middle of a moving region. */
 {
 static long v3dbg = 0;
 int v3do = (v3dbg < 6) || (v3dbg % 900) == 0;
 size_t v3k = (LW * LH < 524288u * 2u) ? (size_t)(LW * LH) / 2: 524288u;
 if (v3do) {
 const float* f = g_mvf32_s[slot];
 printf("live_daemon: v3dbg[%ld] IN f32 @0: %g %g %g %g | @mid: %g %g %g %g | sx=%g sy=%g\n",
 v3dbg, (double)f[0], (double)f[1], (double)f[2], (double)f[3],
 (double)f[v3k], (double)f[v3k + 1], (double)f[v3k + 2], (double)f[v3k + 3],
 (double)v3sx, (double)v3sy);
 }
 mvf32_to_mvraw_neon(g_mvf32_s[slot], s->mvraw, v3sx, v3sy);
 if (v3do) {
 const unsigned short* h = s->mvraw;
 _Float16 q[8];
 for (int k = 0; k < 4; k++) { memcpy(&q[k], &h[k], 2); memcpy(&q[4 + k], &h[v3k + k], 2); }
 printf("live_daemon: v3dbg[%ld] OUT f16 @0: %g %g %g %g | @mid: %g %g %g %g\n",
 v3dbg, (double)q[0], (double)q[1], (double)q[2], (double)q[3],
 (double)q[4], (double)q[5], (double)q[6], (double)q[7]);
 }
 v3dbg++;
 }
 } else {
 fprintf(stderr, "live_daemon: bad mv len %llu\n", (unsigned long long)mlen);
 P->stop = 1; sem_post(&P->in_sem); break;
 }
 if (P->fidx == 0UL)
 printf("live_daemon: mv len %llu (%s)\n", (unsigned long long)mlen,
 (mlen == (uint64_t)MV_BYTES + 16 || mlen == MVF32_WIRE) ? "tail OK": "NO tail");
 if (mv_norm) {
#if defined(__ARM_NEON) && defined(__aarch64__)
 if (!mvlog && mv_dz <= 0.0f) {
 mvnorm_neon(s->mvraw, s->mvnorm);
 } else
#endif
 for (size_t i = 0; i < (size_t)LW * LH; i++) {
 _Float16 h0, h1;
 memcpy(&h0, &s->mvraw[i * 2], 2); memcpy(&h1, &s->mvraw[i * 2 + 1], 2);
 float vx = (float)h0, vy = (float)h1;
 if (mv_dz > 0.0f) {
 if (vx > -mv_dz && vx < mv_dz) vx = 0.0f;
 if (vy > -mv_dz && vy < mv_dz) vy = 0.0f;
 }
 float nx = vx / (float)LW;
 float ny = vy / (float)LH;
 _Float16 a = (_Float16)nx, b = (_Float16)ny;
 memcpy(&s->mvnorm[i * 2], &a, 2); memcpy(&s->mvnorm[i * 2 + 1], &b, 2);
 }
 }
 /* W quality: cut detection — fraction-based (a cut moves
 * >5% of texels >100px; a pan doesn't) with an absolute 500px
 * teleport fallback (the old 20px max fired on 67% of
 * gameplay frames, constantly collapsing history to current) */
 {
 float mvmaxv, frac100;
 mvstats_halves(s->mvraw, (size_t)LW * LH * 2, &mvmaxv, &frac100);
 s->reset = (g_cut_mv > 0.0f && mvmaxv > g_cut_mv) ||
 (g_cut_frac > 0.0f && frac100 > g_cut_frac);
 /* with CORRECT MVs (v13 fix) the thresholds
 * over-fire on gameplay (13+/40 frames RESET at cutfrac=.35
 * in one scene) - they were calibrated on the garbled data.
 * Log the actual distribution so the retune is data-driven. */
 if (s->reset || (P->fidx % 30UL) == 0UL)
 printf("mvstats f%lu max=%.1f frac100=%.3f%s\n",
 P->fidx, (double)mvmaxv, (double)frac100,
 s->reset ? " RESET": "");
 }
 s->expo = 1.0f;
 if (g_autoexp) s->expo = autoexp_lr16(s->lr16);
 if (rxd_diag && (P->fidx % 30) == 0) {
 /* FIONREAD discriminator — bytes ALREADY QUEUED
 * after our reads return late => daemon-side scheduler latency
 * (pinning fixes); bytes arriving late => proxy sender stalling
 * under Box64 load (proxy-side fix) */
 int queued = -1;
 if (ioctl(P->cfd, FIONREAD, &queued) != 0) queued = -1;
 printf("rxdiag f%lu queued=%d\n", P->fidx, queued);
 }
 sem_post(&P->in_sem);
 P->fidx++;
 }
 return NULL;
}

/* sender thread — owns the 8.3MB response send so the GL thread
 * rolls straight into the next frame. sent_sem[p] guards out8[p] reuse:
 * the GL thread waits it before executing frame N+2 on parity p. */
static void* send_thread(void* arg) {
 Pipe* P = (Pipe*)arg;
 pin_thread("fsr4-send", "FSR4_SEND_CPU", 6);
 for (;;) {
 sem_wait(&P->out_sem);
 if (P->stop) break;
 /* in split mode the image parity lags the send count by one
 * (execute k returns frame k-1) — the GL thread records which slot
 * it actually filled */
 int slot = (int)(P->sidx & 1UL);
 (void)slot;
 int s = P->ready_slot;
 if (send_frame(P->cfd, P->sl[s].out8, OUT8_BYTES) != 0) P->stop = 1;
 sem_post(&P->sent_sem[s]);
 if (P->stop) break;
 P->sidx++;
 }
 return NULL;
}

static void serve_threaded(int cfd) {
 Pipe P;
 memset(&P, 0, sizeof P);
 P.cfd = cfd;
 /* the threaded path's out8 slots are OUT8_BYTES — the service
 * memcpy's OS_BYTES when FSR4_U8OUT=0, a silent 8.3MB heap overflow.
 * That A/B knob belongs to the serial path only; refuse here. */
 {
 const char* ue = getenv("FSR4_U8OUT");
 if (ue && !atoi(ue)) {
 fprintf(stderr, "live_daemon: FSR4_U8OUT=0 not supported on the threaded path (8.3MB overflow) - refusing client\n");
 close(cfd);
 return;
 }
 }
 int ok = 1;
 for (int i = 0; i < 2; i++) {
 P.sl[i].lr16 = (unsigned short*)malloc(LR_BYTES);
 P.sl[i].mvraw = (unsigned short*)malloc(MV_BYTES);
 P.sl[i].mvnorm = (unsigned short*)malloc(MV_BYTES);
 P.sl[i].out8 = (unsigned char*)malloc(OUT8_BYTES);
 if (!P.sl[i].lr16 || !P.sl[i].mvraw || !P.sl[i].mvnorm || !P.sl[i].out8) ok = 0;
 }
 if (!ok || sem_init(&P.in_sem, 0, 0) != 0 ||
 sem_init(&P.cons_sem[0], 0, 0) != 0 || sem_init(&P.cons_sem[1], 0, 0) != 0 ||
 sem_init(&P.out_sem, 0, 0) != 0 ||
 sem_init(&P.sent_sem[0], 0, 1) != 0 || sem_init(&P.sent_sem[1], 0, 1) != 0) {
 /* sent_sem starts POSTED — both out slots are trivially
 * free before the first two executes */
 fprintf(stderr, "live_daemon: threaded pipeline alloc failed\n");
 for (int i = 0; i < 2; i++) {
 free(P.sl[i].lr16); free(P.sl[i].mvraw); free(P.sl[i].mvnorm); free(P.sl[i].out8);
 }
 return;
 }
 pthread_t th;
 if (pthread_create(&th, NULL, net_thread, &P) != 0) {
 fprintf(stderr, "live_daemon: net thread create failed\n");
 sem_destroy(&P.in_sem); sem_destroy(&P.cons_sem[0]); sem_destroy(&P.cons_sem[1]);
 sem_destroy(&P.out_sem); sem_destroy(&P.sent_sem[0]); sem_destroy(&P.sent_sem[1]);
 for (int i = 0; i < 2; i++) {
 free(P.sl[i].lr16); free(P.sl[i].mvraw); free(P.sl[i].mvnorm); free(P.sl[i].out8);
 }
 return;
 }
 pthread_t sth;
 if (pthread_create(&sth, NULL, send_thread, &P) != 0) {
 fprintf(stderr, "live_daemon: send thread create failed (falling back inline)\n");
 P.stop = 1;
 sem_post(&P.in_sem);
 pthread_join(th, NULL);
 sem_destroy(&P.in_sem); sem_destroy(&P.cons_sem[0]); sem_destroy(&P.cons_sem[1]);
 sem_destroy(&P.out_sem); sem_destroy(&P.sent_sem[0]); sem_destroy(&P.sent_sem[1]);
 for (int i = 0; i < 2; i++) {
 free(P.sl[i].lr16); free(P.sl[i].mvraw); free(P.sl[i].mvnorm); free(P.sl[i].out8);
 }
 return;
 }
 printf("live_daemon: threaded depth-2 pipeline engaged (+sender)\n");
 pin_thread("fsr4-gl", "FSR4_GL_CPU", 7);
 if (getenv("FSR4_RT") && atoi(getenv("FSR4_RT"))) {
 /* expect EPERM (shell RLIMIT_RTPRIO=0) — log, don't rely */
 struct sched_param sp = { .sched_priority = 1 };
 if (sched_setscheduler(0, SCHED_FIFO, &sp) != 0)
 printf("live_daemon: SCHED_FIFO probe FAILED errno=%d\n", errno);
 else
 printf("live_daemon: SCHED_FIFO granted\n");
 }
 for (;;) {
 sem_wait(&P.in_sem);
 if (P.stop) break;
 int slot = (int)(P.gidx & 1UL);
 PSlot* s = &P.sl[slot];
 if (P.gidx >= 2UL) sem_wait(&P.sent_sem[slot]); /* out8[p] free */
 if (s->got_jit)
 fsr4_set_jitter(jzero ? 0.0f: s->jx * jsign,
 jzero ? 0.0f: s->jy * jsign);
 fsr4_set_exposure(s->expo); /* (1.0 when off) */
 if (s->reset) fsr4_set_reset(1); /* cut handling */
 fsr4_frame_in_t fin;
 fin.lr = g_rawlr ? g_rawtex_s[slot]: s->lr16; fin.mvec = mv_norm ? s->mvnorm: s->mvraw; fin.reproj = rep16;
 fsr4_frame_out_t fout; fout.out = s->out8;
 double t0 = now_ms();
 int er = fsr4_execute(&fin, &fout);
 double t_ex = now_ms();
 sem_post(&P.cons_sem[slot]); /* slot inputs consumed; GL parity done */
 if (er != FSR4_OK) {
 fprintf(stderr, "live_daemon: execute rc=%d\n", er);
 fsr4_invalidate();
 P.stop = 1;
 /* wake the net thread whether parked on cons_sem or blocked in
 * read_full - else join() waits out the client's 15s timeout */
 sem_post(&P.cons_sem[0]); sem_post(&P.cons_sem[1]);
 shutdown(cfd, SHUT_RDWR);
 break;
 }
 /* split mode defers the image one call — the bootstrap
 * execute produces nothing to send */
 if (fsr4_frame_ready()) {
 P.ready_slot = slot;
 sem_post(&P.out_sem); /* sender owns out8 until sent_sem */
 P.tframes++; frames++;
 }
 P.gidx++;
 if ((P.tframes % 30) == 1) {
 fsr4_stats_t st; fsr4_get_stats(&st);
 /* "up" was never uploads — it is the fence1 GPU drain
 * (postA+feat) + copy1. Print the split decomposition. */
 printf("live_daemon[2]: frame %lu svc %.2f | pre %.2f [upld %.2f nwait %.2f c2 %.2f] up %.2f [fence %.2f c1 %.2f] npu %.2f post %.2f tot %.2f | expo %.3f%s\n",
 P.tframes, t_ex - t0,
 st.last_upld_ms + st.last_nwait_ms + st.last_copy2_ms,
 st.last_upld_ms, st.last_nwait_ms, st.last_copy2_ms,
 st.last_upload_ms, st.last_fence_ms, st.last_copy1_ms,
 st.last_npu_ms, st.last_post_ms, st.last_total_ms,
 s->expo, s->reset ? " RESET": "");
 }
 }
 P.stop = 1;
 sem_post(&P.out_sem); /* release the sender */
 pthread_join(sth, NULL);
 pthread_join(th, NULL);
 sem_destroy(&P.in_sem); sem_destroy(&P.cons_sem[0]); sem_destroy(&P.cons_sem[1]);
 sem_destroy(&P.out_sem); sem_destroy(&P.sent_sem[0]); sem_destroy(&P.sent_sem[1]);
 for (int i = 0; i < 2; i++) {
 free(P.sl[i].lr16); free(P.sl[i].mvraw); free(P.sl[i].mvnorm); free(P.sl[i].out8);
 }
}

int main(int argc, char** argv) {
 setvbuf(stdout, NULL, _IONBF, 0);
 int port = DEF_PORT;
 const char* dump_dir = NULL;
 if (argc >= 2) port = atoi(argv[1]);
 if (argc >= 3) dump_dir = argv[2]; // optional: dump synthesized reproj per frame

 signal(SIGPIPE, SIG_IGN);

 fsr4_init_params_t p;
 memset(&p, 0, sizeof(p));
 p.contract_version = FSR4RP6_CONTRACT_VERSION;
 p.backend_so = "libQnnHtp.so";
 p.system_so = "libQnnSystem.so";
 p.model_bin = (argc >= 4) ? argv[3]: "v/bin_cfnhwc.bin";
 p.pre_shader = "pre0_final.comp";
 /* the REAL-semantics shaders are the production path
 * now — default to them (env still overrides for bank-parity A/Bs) */
 p.feat_shader = (getenv("FSR4_FEAT") && *getenv("FSR4_FEAT")) ? getenv("FSR4_FEAT"): "features_real.comp";
 p.post_shader = (getenv("FSR4_POST") && *getenv("FSR4_POST")) ? getenv("FSR4_POST"): "post_real.comp";
 p.weights_seed = "gpu_data/pass0_wb.raw";
 p.hist_seed = "gpu_data/hist.raw";
 p.rec_seed = "gpu_data/rec_prev.raw";
 p.corner.core_corner = 0xA0; p.corner.bus_corner = 0xA0;
 /* (external NPU probe): shared-thermal-budget probe — the
 * HTP pin has been MAX corner (0xA0) all along; FSR4_HTPOWER=<hex> lowers
 * core+bus corners (0x30 SVS2 .. 0xA0 MAX; interesting steps: 0x60 NOM,
 * 0x70 NOM+, 0x80 TURBO) to test whether the package heat drop lets the
 * Adreno governor hold >550MHz warm. NPU has ~6ms slack (7.1ms hidden
 * under 13-20ms of GL work); expect npu ~7.1->8-9ms, still free. */
 {
 const char* hc = getenv("FSR4_HTPOWER");
 if (hc && *hc) {
 unsigned c = (unsigned)strtoul(hc, NULL, 0);
 if (c >= 0x30 && c <= 0xA0) {
 p.corner.core_corner = c; p.corner.bus_corner = c;
 printf("live_daemon: FSR4_HTPOWER corner 0x%x\n", c);
 }
 }
 }
 p.slot_ms = 16.667f;
 p.warmup = 3;
 /* optional service-level dumps (rgb_f<n>/in_f<n>/npuout_f<n>) for the frame
 * numbers in FSR4_RGB_DUMP_FRAMES — used to capture what live actually
 * returns in-game (the argv[2] path only dumps the daemon's inputs) */
 {
 const char* dd = getenv("FSR4_RGB_DUMP_DIR");
 const char* df = getenv("FSR4_RGB_DUMP_FRAMES");
 if (dd && *dd) p.debug_dump_dir = dd;
 if (df && *df) p.debug_dump_frames = df;
 }
 int rc = fsr4_init(&p);
 if (rc != FSR4_OK) { fprintf(stderr, "live_daemon: init rc=%d\n", rc); return 1; }
 printf("live_daemon: service initialized (port %d, bin %s)\n", port, p.model_bin);

 // per-frame buffers (file-scope statics — shared with the threaded path)
 color_f4 = malloc(COLOR_F4_BYTES); // received (proxy decode)
 mv_f16 = malloc(MV_BYTES); // received
 lr16 = malloc(LR_BYTES); // contract lr
 rep16 = malloc(REP_BYTES); // contract reproj
 out16 = malloc(OUT_BYTES); // contract out (v1 / u8out=0)
 mvh = (unsigned short*)malloc(MV_BYTES); // mv viewed as halves
 if (!color_f4 || !mv_f16 || !lr16 || !rep16 || !out16 || !mvh) {
 fprintf(stderr, "live_daemon: OOM\n"); return 1;
 }
 /* F10: the synthesis working set (~100MB) is only consumed by the
 * dead serial-path reference reproj synthesis — skip when syn skipped */
 if (!skip_syn) {
 lr_f32 = (float*)malloc((size_t)LH * LW * 4 * 4);
 prev_out = (float*)malloc((size_t)H * W * 4 * 4);
 cur_out_f = (float*)malloc((size_t)H * W * 4 * 4);
 if (!lr_f32 || !prev_out || !cur_out_f) { fprintf(stderr, "live_daemon: OOM\n"); return 1; }
 }

 first_frame = 1;
 have_prev = 0;
 frames = 0;

 /* the feature shader consumes NORMALIZED MVs
 * (vel * displaySize); the game/proxy sends LR-pixels-per-frame. Without
 * this, any real motion throws every history warp offscreen (live-only
 * degeneracy; the bank's MVs were pre-normalized). FSR4_MVNORM=0 for
 * bank-replay parity. */
 int mv_norm_l = 1;
 {
 const char* e = getenv("FSR4_MVNORM");
 if (e) mv_norm_l = atoi(e);
 }
 mv_norm = mv_norm_l;
 /* post_real blends with the pre-pass rectified history (SSBO 9);
 * the daemon's SSBO 6 reproj synthesis is dead weight there — skip it. */
 int skip_syn_l = 1;
 {
 const char* e = getenv("FSR4_SKIP_SYN");
 if (e) skip_syn_l = atoi(e);
 }
 skip_syn = skip_syn_l;
 unsigned short* mvnorm_l = (unsigned short*)malloc(MV_BYTES);
 if (!mvnorm_l) { fprintf(stderr, "live_daemon: OOM\n"); return 1; }
 mvnorm = mvnorm_l;
 if (skip_syn) memset(rep16, 0, REP_BYTES); // SSBO 6 unused by post_real

 /* jitter compensation: the proxy appends a 16-byte tail (2 floats:
 * the game's current jitter, LR-px units) to the mv frame (mv length
 * becomes MV_BYTES+16 — length-tagged, v1 clients unaffected). The
 * history warp gains the jitter delta term and the model's sampling
 * kernel gets the true jitter via fsr4_set_jitter. */
 jcomp = 0; /* 0 = off (AMD reference default); 1 = +2*(j_prev - j_cur); 2 = opposite */
 jsign = 1.0f; /* game Jitter.Offset sign convention A/B */
 {
 const char* e = getenv("FSR4_JSIGN");
 if (e) jsign = (float)atof(e);
 }
 pjx = 0.0f; pjy = 0.0f;
 have_pj = 0;
 {
 const char* e = getenv("FSR4_JCOMP");
 if (e) jcomp = atoi(e);
 }

 {
 const char* e = getenv("FSR4_NEONCV"); if (e) neoncv = atoi(e);
 printf("live_daemon: neoncv=%d\n", neoncv);
 e = getenv("FSR4_CUTMV"); if (e) g_cut_mv = (float)atof(e);
 e = getenv("FSR4_CUTFRAC"); if (e) g_cut_frac = (float)atof(e);
 e = getenv("FSR4_AUTOEXP"); if (e) g_autoexp = atoi(e);
 printf("live_daemon: cutmv=%.0f cutfrac=%.3f autoexp=%d\n", g_cut_mv, g_cut_frac, g_autoexp);
 }

 /* shimmer hunt. External video characterization of the live
 * capture: ~20 Hz buzz/crawl on STATIC high-frequency detail (carpet,
 * laptop keys, pinboard) — the game's per-frame sub-pixel jitter phases
 * don't integrate across our async 20 fps pipeline. Three levers:
 * FSR4_JZERO=1 pin the shader jitter to (0,0): phase-locks the
 * feature/post sampling taps so consecutive NPU outputs
 * stop re-tiling with the game's jitter phase; the baked
 * render jitter then acts as temporal SSAA the history
 * path integrates.
 * FSR4_MVDZ=px deadzone: zero accumulated MVs below |v| LR px. A
 * static camera with noisy MVs warps history randomly
 * each NPU frame => exactly "carpet buzz".
 * FSR4_MVLOG=1 per-frame log: game jitter (phase sequence) + max/mean
 * |mv| in LR px (post-accumulation, pre-normalization). */
 jzero = 0; mvlog = 0;
 mv_dz = 0.0f;
 {
 const char* e = getenv("FSR4_JZERO"); if (e) jzero = atoi(e);
 e = getenv("FSR4_MVDZ"); if (e) mv_dz = (float)atof(e);
 e = getenv("FSR4_MVLOG"); if (e) mvlog = atoi(e);
 }
 printf("live_daemon: jsign=%+.3f jzero=%d mvdz=%.3f mvlog=%d\n",
 jsign, jzero, mv_dz, mvlog);

 /* history policy: 1 (default) = feedback of the ACTUAL previous
 * NPU output, motion-warped + clamped to the local LR 2x2 hull — real
 * temporal detail accumulation (the reference bilinear-upscale stand-in caps
 * the image at 540p-upscale quality forever). 0 = the reference stand-in, kept for
 * the byte-parity gate vs the d1bank reference. */
 int hist_mode = 1;
 int hist_clamp = 1;
 {
 const char* e = getenv("FSR4_HIST");
 if (e) hist_mode = atoi(e);
 e = getenv("FSR4_HIST_CLAMP");
 if (e) hist_clamp = atoi(e); else hist_clamp = 2; /* default: 3x3 rectification */
 printf("live_daemon: hist_mode=%d clamp=%d\n", hist_mode, hist_clamp);
 }

 int lsock = socket(AF_INET, SOCK_STREAM, 0);
 if (lsock < 0) { perror("socket"); return 1; }
 int one = 1;
 setsockopt(lsock, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
 struct sockaddr_in sa; memset(&sa, 0, sizeof sa);
 sa.sin_family = AF_INET; sa.sin_addr.s_addr = htonl(INADDR_LOOPBACK); sa.sin_port = htons(port);
 if (bind(lsock, (struct sockaddr*)&sa, sizeof sa) != 0) { perror("bind"); return 1; }
 if (listen(lsock, 1) != 0) { perror("listen"); return 1; }
 printf("live_daemon: listening on 127.0.0.1:%d\n", port);

 for (;;) {
 int cfd = accept(lsock, NULL, NULL);
 if (cfd < 0) { if (errno == EINTR) continue; perror("accept"); return 1; }
 int nb = 16 * 1024 * 1024; /* >= the 8.3MB response frame
 * (4MB window stalled the send path) */
 setsockopt(cfd, SOL_SOCKET, SO_RCVBUF, &nb, sizeof nb);
 setsockopt(cfd, SOL_SOCKET, SO_SNDBUF, &nb, sizeof nb);
 int nd = 1;
 setsockopt(cfd, IPPROTO_TCP, TCP_NODELAY, &nd, sizeof nd);
 printf("live_daemon: client connected\n");
 if (!first_frame) { // fresh client = fresh temporal state
 if (fsr4_invalidate() != FSR4_OK) { /* a failed reseed would serve stale history */
 fprintf(stderr, "live_daemon: invalidate FAILED - refusing client\n");
 close(cfd);
 continue;
 }
 }
 first_frame = 0;
 have_prev = 0;
 have_pj = 0;

 /* read the FIRST length prefix here to route the connection:
 * v2 (raw texels) -> threaded depth-2 pipeline; v1 (f4 floats, the
 * bank-replay harness) -> the original serial loop */
 if (read_full(cfd, &g_first_clen, 8) != 0) { close(cfd); continue; }
 g_have_first = 1;
 if (g_first_clen == (uint64_t)COLOR_RAW_BYTES) {
 serve_threaded(cfd);
 close(cfd);
 printf("live_daemon: client disconnected (threaded, frames this connection: %lu)\n", frames);
 continue;
 }

 for (;;) {
 double t0 = now_ms();
 /* color: length prefix selects v1 (decoded f4, 8,294,400) or
 * v2 (raw R11G11B10F texels, 2,073,600); response format follows */
 uint64_t clen;
 int v2 = 0;
 if (g_have_first) { clen = g_first_clen; g_have_first = 0; }
 else if (read_full(cfd, &clen, 8) != 0) break;
 if (clen == (uint64_t)COLOR_F4_BYTES) {
 if (read_full(cfd, color_f4, COLOR_F4_BYTES) != 0) break;
 } else if (clen == (uint64_t)COLOR_RAW_BYTES) {
 static unsigned int rawtex[LW * LH];
 if (read_full(cfd, rawtex, COLOR_RAW_BYTES) != 0) break;
 v2 = 1;
 /* NEON path fills lr16 DIRECTLY (bit relocation,
 * identical to decode_texel -> (_Float16)); the f32 staging
 * color_f4 is skipped entirely on v2+neoncv */
#if defined(__ARM_NEON) && defined(__aarch64__)
 if (neoncv) rawtex_to_lr16(rawtex, lr16);
 else
#endif
 {
 float* d = (float*)color_f4;
 for (size_t t = 0; t < (size_t)LW * LH; t++) decode_texel(rawtex[t], d + t * 4);
 }
 } else {
 fprintf(stderr, "live_daemon: bad color len %llu\n", (unsigned long long)clen);
 break;
 }
 /* mv: length prefix selects no-jitter (MV_BYTES) or jitter tail
 * (MV_BYTES+16: 2 floats jx, jy appended — */
 uint64_t mlen;
 float jx = 0.0f, jy = 0.0f;
 int got_jit = 0;
 if (read_full(cfd, &mlen, 8) != 0) break;
 if (mlen == (uint64_t)MV_BYTES) {
 if (read_full(cfd, mv_f16, MV_BYTES) != 0) break;
 } else if (mlen == (uint64_t)MV_BYTES + 16) {
 if (read_full(cfd, mv_f16, MV_BYTES) != 0) break;
 if (read_full(cfd, &jx, 4) != 0 || read_full(cfd, &jy, 4) != 0) break;
 got_jit = 1;
 } else {
 fprintf(stderr, "live_daemon: bad mv len %llu\n", (unsigned long long)mlen);
 break;
 }
 if (frames == 0)
 printf("live_daemon: mv len %llu (%s)\n", (unsigned long long)mlen,
 (mlen == (uint64_t)MV_BYTES + 16 || mlen == MVF32_WIRE) ? "tail OK": "NO tail");
 double t_rx = now_ms();

 // color f32 RGBA -> fp16 RGBA (bank parity: same value chain as
 // reference float32-decode -> astype(F16); RNE on aarch64 FCVT).
 // skipped on v2+neoncv (rawtex_to_lr16 already wrote fp16).
 int lr16_done =
#if defined(__ARM_NEON) && defined(__aarch64__)
 (v2 && neoncv);
#else
 0;
#endif
 if (!lr16_done) {
 const float* s = (const float*)color_f4;
 unsigned short* d = (unsigned short*)lr16;
 for (size_t i = 0; i < (size_t)LH * LW * 4; i++) {
 _Float16 h = (_Float16)s[i];
 memcpy(&d[i], &h, 2);
 }
 // alpha channel: proxy f4 alpha is already 1.0; force it anyway
 for (size_t i = 3; i < (size_t)LH * LW * 4; i += 4) {
 _Float16 h = (_Float16)1.0f;
 memcpy(&d[i], &h, 2);
 }
 }
 memcpy(mvh, mv_f16, MV_BYTES); // mv halves (RG pairs per texel)
 float mv_max2 = 0.0f, mv_sum = 0.0f;
 if (mv_norm) {
#if defined(__ARM_NEON) && defined(__aarch64__)
 /* fast path when no per-texel extras are needed */
 if (!mvlog && mv_dz <= 0.0f) {
 mvnorm_neon(mvh, mvnorm);
 } else
#endif
 for (size_t i = 0; i < (size_t)LW * LH; i++) {
 _Float16 h0, h1;
 memcpy(&h0, &mvh[i * 2], 2); memcpy(&h1, &mvh[i * 2 + 1], 2);
 float vx = (float)h0, vy = (float)h1;
 if (mv_dz > 0.0f) {
 if (vx > -mv_dz && vx < mv_dz) vx = 0.0f;
 if (vy > -mv_dz && vy < mv_dz) vy = 0.0f;
 }
 if (mvlog) {
 float m2 = vx * vx + vy * vy;
 if (m2 > mv_max2) mv_max2 = m2;
 mv_sum += fabsf(vx) + fabsf(vy);
 }
 float nx = vx / (float)LW;
 float ny = vy / (float)LH;
 _Float16 a = (_Float16)nx, b = (_Float16)ny;
 memcpy(&mvnorm[i * 2], &a, 2); memcpy(&mvnorm[i * 2 + 1], &b, 2);
 }
 }
 if (mvlog)
 printf("mv f%lu j=%+.4f,%+.4f max=%.3f mean=%.5f lrpx\n",
 frames, jx, jy, sqrtf(mv_max2),
 mv_sum / (float)(LW * LH));
 double t_cv = now_ms();

 // lr f32 working copy for the synthesis — ONLY the (dead in live)
 // reference synthesis consumes it; skip entirely when syn skipped
 // (bank parity note preserved: the reference pipeline upscaled the PRE-fp16 float32)
 if (!skip_syn) {
 const float* s = (const float*)color_f4;
 for (size_t i = 0; i < (size_t)LH * LW * 4; i++) lr_f32[i] = s[i];
 for (size_t i = 3; i < (size_t)LH * LW * 4; i += 4) lr_f32[i] = 1.0f;
 }

 // reproj synthesis (reference math) — only consumed by the OLD post (SSBO 6)
 if (!skip_syn) {
 float* src = have_prev ? prev_out: NULL;
 static float self[(size_t)H * W * 4]; // 8 MB; heap-equivalent (single-thread)
 if (!src) { upscale2x_f32(lr_f32, self); src = self; } // frame 0: self-seed
 unsigned short* rp = (unsigned short*)rep16;
 /* alpha halves are constant 1.0 (0x3C00): write once */
 for (size_t k = 0; k < (size_t)H * W; k++) memcpy(&rp[k * 4 + 3], &(const _Float16){1.0f}, 2);
 size_t i = 0;
 for (int yc = 0; yc < H; yc++) {
 int row = (int)((float)yc * 0.5f);
 if (row < 0) row = 0; if (row > LH - 1) row = LH - 1;
 const unsigned short* mvrow = mvh + ((size_t)row * LW) * 2;
 const float* lrrow = lr_f32 + (size_t)(yc >> 1) * LW * 4;
 for (int xc = 0; xc < W; xc++, i += 4) {
 int col = (int)((float)xc * 0.5f);
 if (col < 0) col = 0; if (col > LW - 1) col = LW - 1;
 const unsigned short* mvh2 = mvrow + (size_t)col * 2;
 _Float16 h0; memcpy(&h0, &mvh2[0], 2);
 _Float16 h1; memcpy(&h1, &mvh2[1], 2);
 float velx = ((float)h0) / (float)LW;
 float vely = ((float)h1) / (float)LH;
 float hx = (float)xc + velx * (float)W;
 float hy = (float)yc + vely * (float)H;
 if (jcomp && have_pj) {
 /* the game jittered frame N by (jx,jy) and
 * frame N-1 by (pjx,pjy) — the history must be
 * pulled by the jitter delta on top of mv */
 float djx = (jcomp == 2) ? (jx - pjx): (pjx - jx);
 float djy = (jcomp == 2) ? (jy - pjy): (pjy - jy);
 hx += djx * 2.0f; /* LR px -> output px */
 hy += djy * 2.0f;
 }
 unsigned short o3[4];
 if (hx >= 0.0f && hx <= (float)(W - 1) && hy >= 0.0f && hy <= (float)(H - 1)) {
 float cx = hx < 0.0f ? 0.0f: (hx > (float)(W - 1) ? (float)(W - 1): hx);
 float cy = hy < 0.0f ? 0.0f: (hy > (float)(H - 1) ? (float)(H - 1): hy);
#if defined(__ARM_NEON) && defined(__aarch64__)
 float32x4_t mn4, mx4;
 float32x4_t* pmn = NULL;
 if (hist_mode && hist_clamp) {
 /* history rectification:
 * clamp=1: 2x2 LR hull (bilinear envelope — too tight,
 * kills accumulation; kept for reference)
 * clamp=2: 3x3 LR hull — spans +-3 output px, lets
 * accumulated subpixel detail survive */
 int cx2 = (int)(xc >> 1); if (cx2 > (int)LW - 2) cx2 = (int)LW - 2;
 int cy2 = (int)(yc >> 1); if (cy2 > (int)LH - 2) cy2 = (int)LH - 2;
 int rad = (hist_clamp >= 2) ? 1: 0;
 int x0c3 = cx2 - rad < 0 ? 0: cx2 - rad;
 int y0c3 = cy2 - rad < 0 ? 0: cy2 - rad;
 int x1c3 = cx2 + 1 + rad > (int)LW - 1 ? (int)LW - 1: cx2 + 1 + rad;
 int y1c3 = cy2 + 1 + rad > (int)LH - 1 ? (int)LH - 1: cy2 + 1 + rad;
 const float* qa = lr_f32 + ((size_t)y0c3 * LW + x0c3) * 4;
 const float* qb = lr_f32 + ((size_t)y0c3 * LW + x1c3) * 4;
 const float* qc = lr_f32 + ((size_t)y1c3 * LW + x0c3) * 4;
 const float* qd = lr_f32 + ((size_t)y1c3 * LW + x1c3) * 4;
 float32x4_t q0 = vld1q_f32(qa), q1 = vld1q_f32(qb);
 float32x4_t q2 = vld1q_f32(qc), q3 = vld1q_f32(qd);
 mn4 = vminq_f32(vminq_f32(q0, q1), vminq_f32(q2, q3));
 mx4 = vmaxq_f32(vmaxq_f32(q0, q1), vmaxq_f32(q2, q3));
 if (rad) {
 /* extend to the 3x3 ring corners (rad=1) */
 int xm = cx2 < 1 ? 0: cx2 - 1;
 int xp = cx2 + 2 > (int)LW - 1 ? (int)LW - 1: cx2 + 2;
 int ym = cy2 < 1 ? 0: cy2 - 1;
 int yp = cy2 + 2 > (int)LH - 1 ? (int)LH - 1: cy2 + 2;
 const float* r0 = lr_f32 + ((size_t)ym * LW + xm) * 4;
 const float* r1 = lr_f32 + ((size_t)ym * LW + xp) * 4;
 const float* r2 = lr_f32 + ((size_t)cy2 * LW + xm) * 4;
 const float* r3 = lr_f32 + ((size_t)cy2 * LW + xp) * 4;
 const float* r4 = lr_f32 + ((size_t)yp * LW + xm) * 4;
 const float* r5 = lr_f32 + ((size_t)yp * LW + xp) * 4;
 float32x4_t e0 = vld1q_f32(r0), e1 = vld1q_f32(r1);
 float32x4_t e2 = vld1q_f32(r2), e3 = vld1q_f32(r3);
 float32x4_t e4 = vld1q_f32(r4), e5 = vld1q_f32(r5);
 mn4 = vminq_f32(mn4, vminq_f32(vminq_f32(e0, e1), vminq_f32(vminq_f32(e2, e3), vminq_f32(e4, e5))));
 mx4 = vmaxq_f32(mx4, vmaxq_f32(vmaxq_f32(e0, e1), vmaxq_f32(vmaxq_f32(e2, e3), vmaxq_f32(e4, e5))));
 }
 pmn = &mn4;
 (void)lrrow;
 }
 bilinear_rgba_f32_c(src, cx, cy, o3, pmn, pmn ? &mx4: NULL);
#else
 bilinear_rgba_f32_c(src, cx, cy, o3, NULL, NULL);
#endif
 } else {
 o3[0] = o3[1] = o3[2] = 0; /* (+0.0 halves) */
 }
 rp[i] = o3[0]; rp[i + 1] = o3[1]; rp[i + 2] = o3[2];
 }
 }
 // history for the NEXT frame
 if (hist_mode)
 out16_to_f32(out16, prev_out); // the ACTUAL output (detail accumulates)
 else
 upscale2x_f32(lr_f32, prev_out); // reference stand-in (byte-gate parity)
 have_prev = 1;
 pjx = jx; pjy = jy; have_pj = 1;
 }
 double t_syn = now_ms();

 fsr4_frame_in_t fin; fin.lr = lr16; fin.mvec = mv_norm ? mvnorm: mv_f16; fin.reproj = rep16;
 /* with FSR4_U8OUT (default) the service reads back the
 * post shader's RGBA8 mirror directly — the daemon sends it
 * as-is (out16_to_u8 leaves the hot path). v1 (f4) clients need
 * the fp16 frame and are rejected in this mode. */
 static unsigned char out8[OUT8_BYTES];
 int u8out = (getenv("FSR4_U8OUT") && !atoi(getenv("FSR4_U8OUT"))) ? 0: 1;
 if (!v2 && u8out) {
 fprintf(stderr, "live_daemon: v1 (f4) client needs FSR4_U8OUT=0\n");
 break;
 }
 fsr4_frame_out_t fout; fout.out = u8out ? out8: out16;
 if (got_jit) fsr4_set_jitter(jzero ? 0.0f: jx * jsign,
 jzero ? 0.0f: jy * jsign);
 int er = fsr4_execute(&fin, &fout);
 if (er != FSR4_OK) {
 fprintf(stderr, "live_daemon: execute rc=%d at frame %lu\n", er, frames);
 fsr4_invalidate();
 break;
 }
 double t_ex = now_ms();

 if (dump_dir) {
 char pp[600];
 snprintf(pp, sizeof pp, "%s/reproj_f%lu.raw", dump_dir, frames);
 FILE* f = fopen(pp, "wb"); if (f) { fwrite(rep16, 1, REP_BYTES, f); fclose(f); }
 snprintf(pp, sizeof pp, "%s/lr_f%lu.raw", dump_dir, frames);
 f = fopen(pp, "wb"); if (f) { fwrite(lr16, 1, LR_BYTES, f); fclose(f); }
 const char* od = getenv("FSR4_OUT16_DUMP");
 if (od && *od && !u8out) { // fp16 frame only exists when U8OUT=0
 snprintf(pp, sizeof pp, "%s/out16_f%lu.raw", dump_dir, frames);
 FILE* g = fopen(pp, "wb"); if (g) { fwrite(out16, 1, OUT_BYTES, g); fclose(g); }
 }
 }

 if (fsr4_frame_ready()) { /* split bootstrap emits nothing */
 if (v2) {
 if (u8out) {
 if (send_frame(cfd, out8, OUT8_BYTES) != 0) break;
 } else {
 static unsigned char out8c[OUT8_BYTES];
 out16_to_u8(out16, out8c);
 if (send_frame(cfd, out8c, OUT8_BYTES) != 0) break;
 }
 } else {
 if (send_frame(cfd, out16, OUT_BYTES) != 0) break;
 }
 }
 frames++;
 if ((frames % 30) == 1) {
 fsr4_stats_t st; fsr4_get_stats(&st);
 printf("live_daemon: frame %lu rx %.2f cv %.2f syn %.2f exe %.2f ms "
 "| svc up %.2f gl %.2f npu %.2f post %.2f rb %.2f tot %.2f\n",
 frames, t_rx - t0, t_cv - t_rx, t_syn - t_cv, t_ex - t_syn,
 st.last_upload_ms, st.last_gl_ms, st.last_npu_ms,
 st.last_post_ms, st.last_readback_ms, st.last_total_ms);
 }
 }
 close(cfd);
 printf("live_daemon: client disconnected (frames this connection ran: %lu)\n", frames);
 }
}
