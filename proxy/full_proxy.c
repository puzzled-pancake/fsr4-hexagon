/* fsr4_ngx_proxy.c — nvngx.dll seat proxy for the RotTR capture.
 *
 * Takes the NGX application-API seat (OptiScaler pattern): implements enough of the
 * NVIDIA NGX ABI that the game's DLSS integration runs, logs every call, and on
 * EvaluateFeature dumps the per-frame inputs (color -> RGBA16F 960x540,
 * motion vectors -> RG16F 960x540, meta) to shared storage while producing a naive
 * bilinear upscale so the game keeps rendering.
 *
 * ABI ground truth: NVIDIA nvsdk_ngx.h / nvsdk_ngx_params.h / nvsdk_ngx_defs.h
 * (SDK 0x15 headers mirrored in OptiScaler external/nvngx_dlss_sdk). The Parameter
 * object vtable (17 slots) and all key strings are reproduced EXACTLY from those
 * headers — no invented ABI.
 *
 * Capture control (polled per eval, all under the capture outdir):
 * START file present -> arm capture for the next `count` evals
 * STOP file present -> disarm
 * fsr4cap.ini -> [cap] count=N outdir=... (read once, lazy)
 * CAPTURE_DONE -> written by the proxy when the armed run completes
 *
 * Build (host, MinGW x86_64):
 * x86_64-w64-mingw32-gcc -O2 -shared -mdll -o nvngx.dll fsr4_ngx_proxy.c nvngx.def \
 * -ld3d11 -ldxgi -ldxguid -luuid -static-libgcc
 */

#include <windows.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <d3d11.h>
#include <dxgi.h>

/* ------------------------------------------------------------------ */
/* NGX ABI (verbatim values from the NVIDIA SDK headers) */
/* ------------------------------------------------------------------ */

#define NVSDK_NGX_VERSION_API_MACRO 0x0000015

typedef enum NVSDK_NGX_Result {
 NVSDK_NGX_Result_Success = 0x1,
 NVSDK_NGX_Result_Fail = 0xBAD00000,
 NVSDK_NGX_Result_FAIL_FeatureNotSupported = NVSDK_NGX_Result_Fail | 1,
 NVSDK_NGX_Result_FAIL_PlatformError = NVSDK_NGX_Result_Fail | 2,
 NVSDK_NGX_Result_FAIL_FeatureAlreadyExists = NVSDK_NGX_Result_Fail | 3,
 NVSDK_NGX_Result_FAIL_FeatureNotFound = NVSDK_NGX_Result_Fail | 4,
 NVSDK_NGX_Result_FAIL_InvalidParameter = NVSDK_NGX_Result_Fail | 5,
 NVSDK_NGX_Result_FAIL_ScratchBufferTooSmall = NVSDK_NGX_Result_Fail | 6,
 NVSDK_NGX_Result_FAIL_NotInitialized = NVSDK_NGX_Result_Fail | 7,
 NVSDK_NGX_Result_FAIL_UnsupportedInputFormat= NVSDK_NGX_Result_Fail | 8,
 NVSDK_NGX_Result_FAIL_RWFlagMissing = NVSDK_NGX_Result_Fail | 9,
 NVSDK_NGX_Result_FAIL_MissingInput = NVSDK_NGX_Result_Fail | 10,
 NVSDK_NGX_Result_FAIL_UnableToInitializeFeature = NVSDK_NGX_Result_Fail | 11,
 NVSDK_NGX_Result_FAIL_OutOfDate = NVSDK_NGX_Result_Fail | 12,
 NVSDK_NGX_Result_FAIL_OutOfGPUMemory = NVSDK_NGX_Result_Fail | 13,
 NVSDK_NGX_Result_FAIL_UnsupportedFormat = NVSDK_NGX_Result_Fail | 14,
 NVSDK_NGX_Result_FAIL_UnableToWriteToAppDataPath = NVSDK_NGX_Result_Fail | 15,
 NVSDK_NGX_Result_FAIL_UnsupportedParameter = NVSDK_NGX_Result_Fail | 16,
 NVSDK_NGX_Result_FAIL_Denied = NVSDK_NGX_Result_Fail | 17,
 NVSDK_NGX_Result_FAIL_NotImplemented = NVSDK_NGX_Result_Fail | 18,
} NVSDK_NGX_Result;

typedef enum NVSDK_NGX_Feature {
 NVSDK_NGX_Feature_Reserved0 = 0,
 NVSDK_NGX_Feature_SuperSampling = 1,
 NVSDK_NGX_Feature_ImageSuperResolution = 3,
 NVSDK_NGX_Feature_ImageSignalProcessing = 9,
} NVSDK_NGX_Feature;

typedef enum NVSDK_NGX_PerfQuality_Value {
 NVSDK_NGX_PerfQuality_Value_MaxPerf = 0,
 NVSDK_NGX_PerfQuality_Value_Balanced = 1,
 NVSDK_NGX_PerfQuality_Value_MaxQuality = 2,
 NVSDK_NGX_PerfQuality_Value_UltraPerformance = 3,
 NVSDK_NGX_PerfQuality_Value_UltraQuality = 4,
 NVSDK_NGX_PerfQuality_Value_DLAA = 5,
} NVSDK_NGX_PerfQuality_Value;

typedef struct NVSDK_NGX_Handle { unsigned int Id; } NVSDK_NGX_Handle;

typedef struct NVSDK_NGX_PathListInfo { wchar_t const* const* Path; unsigned int Length; } NVSDK_NGX_PathListInfo;
typedef struct NVSDK_NGX_FeatureCommonInfo_Internal NVSDK_NGX_FeatureCommonInfo_Internal;
typedef void (*NVSDK_NGX_AppLogCallback)(const char* message, int loggingLevel, int sourceComponent);
typedef struct NVSDK_NGX_LoggingInfo {
 NVSDK_NGX_AppLogCallback LoggingCallback; int MinimumLoggingLevel; int DisableOtherLoggingSinks;
} NVSDK_NGX_LoggingInfo;
typedef struct NVSDK_NGX_FeatureCommonInfo {
 NVSDK_NGX_PathListInfo PathListInfo;
 NVSDK_NGX_FeatureCommonInfo_Internal* InternalData;
 NVSDK_NGX_LoggingInfo LoggingInfo;
} NVSDK_NGX_FeatureCommonInfo;

typedef struct NVSDK_NGX_FeatureRequirement {
 unsigned int FeatureSupported; /* NVSDK_NGX_Feature_Support_Result bitfield, 0 = supported */
 unsigned int MinHWArchitecture; /* NV_GPU_ARCHITECTURE_ID class */
 char MinOSVersion[255];
} NVSDK_NGX_FeatureRequirement;

/* NVSDK_NGX_Parameter — the C++ interface from nvsdk_ngx_params.h. The game was
 * compiled against this exact declaration; the vtable below must keep the same
 * 17-slot order (simple single-inheritance abstract class: identical layout in
 * MSVC and GCC on x64). */
typedef struct NVSDK_NGX_Parameter NVSDK_NGX_Parameter;

/* keys (nvsdk_ngx_defs.h) */
#define K_SSR_Available "SuperSampling.Available"
#define K_SSR_NeedsUpdatedDrv "SuperSampling.NeedsUpdatedDriver"
#define K_SSR_MinDrvMajor "SuperSampling.MinDriverVersionMajor"
#define K_SSR_MinDrvMinor "SuperSampling.MinDriverVersionMinor"
#define K_SSR_FeatInitResult "SuperSampling.FeatureInitResult"
#define K_DLSS_OptSettingsCb "DLSSOptimalSettingsCallback"
#define K_DLSS_GetStatsCb "DLSSGetStatsCallback"
#define K_Width "Width"
#define K_Height "Height"
#define K_OutWidth "OutWidth"
#define K_OutHeight "OutHeight"
#define K_Sharpness "Sharpness"
#define K_Color "Color"
#define K_MV "MotionVectors"
#define K_Output "Output"
#define K_Depth "Depth"
#define K_JitterX "Jitter.Offset.X"
#define K_JitterY "Jitter.Offset.Y"
#define K_MVScaleX "MV.Scale.X"
#define K_MVScaleY "MV.Scale.Y"
#define K_MVOffX "MV.Offset.X"
#define K_MVOffY "MV.Offset.Y"
#define K_PerfQ "PerfQualityValue"
#define K_RTValue "RTXValue"
#define K_DLSSMode "DLSSMode"
#define K_CreateFlags "DLSS.Feature.Create.Flags"
#define K_PreExposure "DLSS.Pre.Exposure"
#define K_Exposure "ExposureTexture"
#define K_Scratch "Scratch"
#define K_ScratchBytes "Scratch.SizeInBytes"
#define K_CreationNodeMask "CreationNodeMask"
#define K_VisibilityNodeMask "VisibilityNodeMask"
#define K_Format "Format"
#define K_Model "Model"
#define K_DynMaxW "DLSS.Get.Dynamic.Max.Render.Width"
#define K_DynMaxH "DLSS.Get.Dynamic.Max.Render.Height"
#define K_DynMinW "DLSS.Get.Dynamic.Min.Render.Width"
#define K_DynMinH "DLSS.Get.Dynamic.Min.Render.Height"
/* legacy (SDK 1.x-era) key family — probed as fallback, every Get is logged */
#define K_ColorL "__ColorInput"
#define K_MVL "__MVecInput"
#define K_OutputL "__Output"

/* ------------------------------------------------------------------ */
/* logging */
/* ------------------------------------------------------------------ */

static FILE* g_log = NULL;
static CRITICAL_SECTION g_logcs;
static LONG g_init_done = 0;
static LONG g_logcs_init = 0;
static char g_dll_dir[MAX_PATH] = {0};
static char g_outdir[MAX_PATH] = {0}; /* chosen capture/log dir */
static char g_dll_path[MAX_PATH] = {0};

static long long now_epoch_ms(void) {
 FILETIME ft; GetSystemTimeAsFileTime(&ft);
 long long t = ((long long)ft.dwHighDateTime << 32) | ft.dwLowDateTime;
 return t / 10000LL - 11644473600000LL;
}

static void log_open(void) {
 if (g_log) return;
 if (!g_logcs_init) { InitializeCriticalSection(&g_logcs); g_logcs_init = 1; }
 /* candidate capture dirs: D: = public Download (adb-visible), E: = app storage, then fallbacks */
 static const char* cands[] = { "D:\\fsr4cap", "E:\\fsr4cap", "Z:\\fsr4cap", "C:\\fsr4cap" };
 int i;
 for (i = 0; i < 4; i++) {
 CreateDirectoryA(cands[i], NULL);
 char p[MAX_PATH];
 snprintf(p, sizeof(p), "%s\\fsr4_ngx.log", cands[i]);
 FILE* f = fopen(p, "a");
 if (f) {
 strcpy(g_outdir, cands[i]);
 g_log = f;
 fprintf(g_log, "[%lld] log opened, outdir=%s, dll=%s\n", now_epoch_ms(), g_outdir, g_dll_path);
 fflush(g_log);
 return;
 }
 }
}

static void LOG(const char* fmt, ...) {
 if (!g_init_done) return;
 if (!g_log) return;
 va_list ap; va_start(ap, fmt);
 EnterCriticalSection(&g_logcs);
 fprintf(g_log, "[%lld] ", now_epoch_ms());
 vfprintf(g_log, fmt, ap);
 fprintf(g_log, "\n");
 fflush(g_log);
 LeaveCriticalSection(&g_logcs);
 va_end(ap);
}

static void ensure_init(void) {
 if (InterlockedCompareExchange(&g_init_done, 1, 0) == 0) {
 /* first export call: one-time setup (outside DllMain on purpose) */
 if (!g_logcs_init) { InitializeCriticalSection(&g_logcs); g_logcs_init = 1; }
 log_open();
 LOG("proxy init (NGX seat, sdk-version target 0x%x)", NVSDK_NGX_VERSION_API_MACRO);
 }
}

/* ------------------------------------------------------------------ */
/* fp16 / R11G11B10F conversion */
/* ------------------------------------------------------------------ */

static float h2f(unsigned short h) {
 unsigned int sign = (h & 0x8000u) << 16;
 unsigned int exp = (h & 0x7C00u) >> 10;
 unsigned int man = (h & 0x03FFu);
 unsigned int bits;
 if (exp == 0) {
 if (man == 0) { bits = sign; }
 else {
 int e = -1; unsigned int m = man;
 do { e++; m <<= 1; } while ((m & 0x0400u) == 0);
 bits = sign | ((unsigned int)(127 - 15 - e) << 23) | ((m & 0x03FFu) << 13);
 }
 } else if (exp == 31) {
 bits = sign | 0x7F800000u | (man << 13);
 } else {
 bits = sign | ((exp - 15 + 127) << 23) | (man << 13);
 }
 float f; memcpy(&f, &bits, 4); return f;
}

static unsigned short f2h(float f) {
 unsigned int bits; memcpy(&bits, &f, 4);
 unsigned int sign = (bits >> 16) & 0x8000u;
 int exp = (int)((bits >> 23) & 0xFFu) - 127 + 15;
 unsigned int man = bits & 0x007FFFFFu;
 if (((bits >> 23) & 0xFFu) == 0xFF) /* inf/nan */
 return (unsigned short)(sign | 0x7C00u | (man ? 0x0200u: 0u));
 if (exp >= 31) return (unsigned short)(sign | 0x7C00u); /* overflow -> inf */
 if (exp <= 0) { /* subnormal / zero */
 if (exp < -10) return (unsigned short)sign;
 man |= 0x00800000u;
 unsigned int shift = (unsigned int)(1 - exp);
 /* Dormant since: the old form shifted `man >> shift`
 * (13 bits too few) and tied at 1<<(shift-1) — every f32 in
 * [2^-25, 2^-14) came out ~8192x too large before the u16 truncation
 * (f2h(2^-23) = 0x4000 = 2.0 instead of 0x0002). Every host harness
 * carried the correct form; production never got it. */
 unsigned int half = man >> (shift + 13);
 unsigned int rem = (man >> shift) & 0x1FFFu;
 if (rem > 0x1000u || (rem == 0x1000u && (half & 1u))) half++;
 return (unsigned short)(sign | half);
 }
 unsigned int half = (unsigned int)(exp << 10) | (man >> 13);
 unsigned int rem = man & 0x1FFFu;
 if (rem > 0x1000u || (rem == 0x1000u && (half & 1u))) half++;
 return (unsigned short)(sign | half);
}

typedef struct { float r, g, b, a; } f4;

/* decode a 5-exp/m-mantissa float (bias 15) as used by R11G11B10F channels */
static float f11(int m, int e, int mbits) {
 if (e == 0) return (float)m / (float)(1u << (14 + mbits));
 if (e >= 31) return 65024.0f;
 /* e < 15 needs a NEGATIVE shift — never write 1u << (e-15): x86 masks the
 * count mod 32, silently multiplying sub-1.0 values by 2^32 (this exact bug
 * faked the "Inf flood" across every earlier capture) */
 if (e >= 15) return (1.0f + (float)m / (float)(1u << mbits)) * (float)(1u << (e - 15));
 return (1.0f + (float)m / (float)(1u << mbits)) / (float)(1u << (15 - e));
}

/* pack to a 5-exp float channel (naive display path only) */
static unsigned int pack11(float x, int mbits) {
 if (!(x > 0.0f)) return 0;
 int ee;
 float mx = frexpf(x, &ee);
 mx *= 2.0f; ee -= 1;
 int e = ee + 15;
 if (e < 0) e = 0;
 if (e == 0) {
 /* everything below 2^-14 (true subnormals + the clamped
 * range) used to fall through to the max code (inf-class). Encode the
 * subnormal value instead: mx is the fraction in [0.5, 1) scaled to
 * [1, 2) above, so value = mx * 2^-15 and the subnormal code is
 * round(mx * 2^(mbits+1)) for 6-bit, rounded into 5-bit for B. */
 float sub = mx * (float)(1u << (mbits + 1));
 unsigned int code = (unsigned int)(sub + 0.5f);
 unsigned int maxc = (1u << mbits) - 1u;
 if (code > maxc) code = maxc; /* clamped-from-below lands at max subnormal, never inf */
 return code;
 }
 if (e >= 1 && e < 31) {
 int m = (int)((mx - 1.0f) * (float)(1u << mbits) + 0.5f);
 if (m >= (1 << mbits)) { m = 0; e++; }
 if (e < 31) return ((unsigned)e << mbits) | (unsigned)m;
 }
 return (mbits == 6) ? 0x7C0u: 0x3E0u;
}

/* decode one pixel of fmt into linear-ish float4 (as stored; no colorspace math) */
static int decode_px(unsigned int fmt, const unsigned char* src, f4* out) {
 switch (fmt) {
 case 10: { /* R16G16B16A16_FLOAT */
 const unsigned short* h = (const unsigned short*)src;
 out->r = h2f(h[0]); out->g = h2f(h[1]); out->b = h2f(h[2]); out->a = h2f(h[3]);
 return 8; }
 case 2: { /* R32G32B32A32_FLOAT */
 const float* f = (const float*)src;
 out->r = f[0]; out->g = f[1]; out->b = f[2]; out->a = f[3];
 return 16; }
 case 26: { /* R11G11B10_FLOAT: bits[10:0]=R(5e/6m) [21:11]=G(5e/6m) [31:22]=B(5e/5m), bias 15 */
 unsigned int v; memcpy(&v, src, 4);
 unsigned int rr = v & 0x7FFu, gg = (v >> 11) & 0x7FFu, bb = (v >> 22) & 0x3FFu;
 out->r = f11((int)(rr & 0x3Fu), (int)((rr >> 6) & 0x1Fu), 6);
 out->g = f11((int)(gg & 0x3Fu), (int)((gg >> 6) & 0x1Fu), 6);
 out->b = f11((int)(bb & 0x1Fu), (int)((bb >> 5) & 0x1Fu), 5);
 out->a = 1.0f;
 return 4; }
 case 34: { /* R16G16_FLOAT */
 const unsigned short* h = (const unsigned short*)src;
 out->r = h2f(h[0]); out->g = h2f(h[1]); out->b = 0.0f; out->a = 1.0f;
 return 4; }
 case 41: { /* R32_FLOAT was stageable but undecodable (silent zero-fill) */
 float fv; memcpy(&fv, src, 4); out->r = fv; out->g = 0; out->b = 0; out->a = 1; return 4;
 }
 case 54: { /* R16_FLOAT */
 const unsigned short* h = (const unsigned short*)src;
 out->r = h2f(h[0]); out->g = 0.0f; out->b = 0.0f; out->a = 1.0f;
 return 2; }
 case 16: { /* R32G32_FLOAT */
 const float* f = (const float*)src;
 out->r = f[0]; out->g = f[1]; out->b = 0.0f; out->a = 1.0f;
 return 8; }
 case 28: case 29: { /* RGBA8_UNORM / _SRGB */
 float x = src[0] / 255.0f, y = src[1] / 255.0f, z = src[2] / 255.0f;
 if (fmt == 29) { /* approx srgb->linear */
 x = (x <= 0.04045f) ? x / 12.92f: (float)pow((double)x, 2.4) * 1.055 - 0.055;
 y = (y <= 0.04045f) ? y / 12.92f: (float)pow((double)y, 2.4) * 1.055 - 0.055;
 z = (z <= 0.04045f) ? z / 12.92f: (float)pow((double)z, 2.4) * 1.055 - 0.055;
 }
 out->r = x; out->g = y; out->b = z; out->a = src[3] / 255.0f;
 return 4; }
 case 87: case 91: { /* BGRA8_UNORM / _SRGB */
 float x = src[2] / 255.0f, y = src[1] / 255.0f, z = src[0] / 255.0f;
 if (fmt == 91) {
 x = (x <= 0.04045f) ? x / 12.92f: (float)pow((double)x, 2.4) * 1.055 - 0.055;
 y = (y <= 0.04045f) ? y / 12.92f: (float)pow((double)y, 2.4) * 1.055 - 0.055;
 z = (z <= 0.04045f) ? z / 12.92f: (float)pow((double)z, 2.4) * 1.055 - 0.055;
 }
 out->r = x; out->g = y; out->b = z; out->a = src[3] / 255.0f;
 return 4; }
 default:
 return 0;
 }
}

static int encode_px(unsigned int fmt, const f4* in, unsigned char* dst) {
 switch (fmt) {
 case 10: { /* RGBA16F */
 unsigned short* h = (unsigned short*)dst;
 h[0] = f2h(in->r); h[1] = f2h(in->g); h[2] = f2h(in->b); h[3] = f2h(in->a);
 return 8; }
 case 2: { /* RGBA32F */
 float* f = (float*)dst;
 f[0] = in->r; f[1] = in->g; f[2] = in->b; f[3] = in->a;
 return 16; }
 case 26: { /* R11G11B10_FLOAT (naive display path only) */
 unsigned int v = pack11(in->r, 6) | (pack11(in->g, 6) << 11) | (pack11(in->b, 5) << 22);
 memcpy(dst, &v, 4);
 return 4; }
 case 28: case 29: {
 dst[0] = (unsigned char)(in->r <= 0 ? 0: in->r >= 1 ? 255: in->r * 255.0f + 0.5f);
 dst[1] = (unsigned char)(in->g <= 0 ? 0: in->g >= 1 ? 255: in->g * 255.0f + 0.5f);
 dst[2] = (unsigned char)(in->b <= 0 ? 0: in->b >= 1 ? 255: in->b * 255.0f + 0.5f);
 dst[3] = (unsigned char)(in->a <= 0 ? 0: in->a >= 1 ? 255: in->a * 255.0f + 0.5f);
 return 4; }
 case 87: case 91: {
 dst[0] = (unsigned char)(in->b <= 0 ? 0: in->b >= 1 ? 255: in->b * 255.0f + 0.5f);
 dst[1] = (unsigned char)(in->g <= 0 ? 0: in->g >= 1 ? 255: in->g * 255.0f + 0.5f);
 dst[2] = (unsigned char)(in->r <= 0 ? 0: in->r >= 1 ? 255: in->r * 255.0f + 0.5f);
 dst[3] = (unsigned char)(in->a <= 0 ? 0: in->a >= 1 ? 255: in->a * 255.0f + 0.5f);
 return 4; }
 default:
 return 0;
 }
}

static int fmt_bytes(unsigned int fmt) {
 switch (fmt) {
 case 10: return 8; /* R16G16B16A16_FLOAT */
 case 2: return 16; /* R32G32B32A32_FLOAT */
 case 26: return 4; /* R11G11B10_FLOAT */
 case 16: return 8; /* R32G32_FLOAT */
 case 34: return 4; /* R16G16_FLOAT */
 case 41: return 4; /* R32_FLOAT (the 1x1 exposure texture) */
 case 54: return 2; /* R16_FLOAT */
 case 28: case 29: case 87: case 91: return 4; /* RGBA8/BGRA8 (+sRGB) */
 default: return 0; /* incl. 35 R16G16_UNORM / 37 R16G16_SNORM / 50 R8G8_UINT:
 byte-size known but no decoder -> refuse staging rather
 than silently zero-fill (correction) */
 }
}

/* ------------------------------------------------------------------ */
/* Parameter object — exact 17-slot vtable order from nvsdk_ngx_params.h */
/* ------------------------------------------------------------------ */

#define PMAX 96
typedef struct {
 char key[64];
 int type; /* 0=ull 1=f 2=d 3=ui 4=i 5=pointer */
 unsigned long long ull;
 float f;
 double d;
 void* p;
} PEntry;

typedef struct {
 void* vtbl;
 PEntry e[PMAX];
 int n;
 int owned; /* 1 = free with DestroyParameters, 0 = SDK-persistent */
} FsParam;

static void fsp_set(FsParam* s, const char* name, int t, unsigned long long ull, float f, double d, void* p) {
 int i;
 for (i = 0; i < s->n; i++)
 if (strcmp(s->e[i].key, name) == 0) break;
 if (i == s->n) {
 if (s->n >= PMAX) { LOG("PARAM SET overflow, key=%s", name); return; }
 s->n++;
 snprintf(s->e[i].key, sizeof(s->e[i].key), "%s", name);
 }
 s->e[i].type = t; s->e[i].ull = ull; s->e[i].f = f; s->e[i].d = d; s->e[i].p = p;
}

static int fsp_get(FsParam* s, const char* name, int t, unsigned long long* ull, float* f, double* d, void** p) {
 int i;
 for (i = 0; i < s->n; i++)
 if (strcmp(s->e[i].key, name) == 0) {
 /* NGX tolerant typing: serve the stored variant for any numeric/pointer ask */
 switch (s->e[i].type) {
 case 0: if (ull) *ull = s->e[i].ull; if (f) *f = (float)(long long)s->e[i].ull; if (d) *d = (double)(long long)s->e[i].ull;
 if (p) *p = (void*)(uintptr_t)s->e[i].ull; break;
 case 1: { /* float->int of NaN/out-of-range is UB — sanitize */
 double sv = (s->e[i].f != s->e[i].f) ? 0.0: (double)s->e[i].f;
 if (sv > 9.2e18) sv = 9.2e18; if (sv < -9.2e18) sv = -9.2e18;
 if (ull) *ull = (unsigned long long)(long long)sv; if (f) *f = s->e[i].f; if (d) *d = (double)s->e[i].f;
 if (p) *p = (void*)(uintptr_t)(long long)sv; break; }
 case 2: { double sv = (s->e[i].d != s->e[i].d) ? 0.0: s->e[i].d;
 if (sv > 9.2e18) sv = 9.2e18; if (sv < -9.2e18) sv = -9.2e18;
 if (ull) *ull = (unsigned long long)(long long)sv; if (f) *f = (float)s->e[i].d; if (d) *d = s->e[i].d;
 if (p) *p = (void*)(uintptr_t)(long long)sv; break; }
 case 3: case 4: if (ull) *ull = (unsigned long long)(long long)s->e[i].ull; if (f) *f = (float)(int)s->e[i].ull;
 if (d) *d = (double)(int)s->e[i].ull; if (p) *p = (void*)(uintptr_t)s->e[i].ull; break;
 default: if (ull) *ull = (unsigned long long)(uintptr_t)s->e[i].p;
 if (f) *f = (float)(intptr_t)s->e[i].p; if (d) *d = (double)(intptr_t)s->e[i].p;
 if (p) *p = s->e[i].p; break;
 }
 return (int)NVSDK_NGX_Result_Success;
 }
 return (int)NVSDK_NGX_Result_FAIL_InvalidParameter;
}

/* vtable slots — declaration order from nvsdk_ngx_params.h */
/* per-parameter logging gated (10-30 fprintf+fflush per eval under
 * Box64 = several ms/frame); re-enable with ini key "plog=1" */
int g_plog = 0; /* ini key plog=1 enables; default OFF (10-30
 * fprintf/eval under Box64 = several ms/frame) */
static void FSP_SetULL (NVSDK_NGX_Parameter* t, const char* n, unsigned long long v) {
 FsParam* s = (FsParam*)t; fsp_set(s, n, 0, v, 0, 0, NULL); if (g_plog) LOG("P.SetULL \"%s\" = %llu", n, v);
}
static void FSP_SetF (NVSDK_NGX_Parameter* t, const char* n, float v) {
 FsParam* s = (FsParam*)t; fsp_set(s, n, 1, 0, v, 0, NULL); if (g_plog) LOG("P.SetF \"%s\" = %.6g", n, v);
}
static void FSP_SetD (NVSDK_NGX_Parameter* t, const char* n, double v) {
 FsParam* s = (FsParam*)t; fsp_set(s, n, 2, 0, 0, v, NULL); if (g_plog) LOG("P.SetD \"%s\" = %.6g", n, v);
}
static void FSP_SetUI (NVSDK_NGX_Parameter* t, const char* n, unsigned int v) {
 FsParam* s = (FsParam*)t; fsp_set(s, n, 3, (unsigned long long)v, 0, 0, NULL); if (g_plog) LOG("P.SetUI \"%s\" = %u", n, v);
}
static void FSP_SetI (NVSDK_NGX_Parameter* t, const char* n, int v) {
 FsParam* s = (FsParam*)t; fsp_set(s, n, 4, (unsigned long long)(long long)v, 0, 0, NULL); if (g_plog) LOG("P.SetI \"%s\" = %d", n, v);
}
static void FSP_SetD11 (NVSDK_NGX_Parameter* t, const char* n, ID3D11Resource* v) {
 FsParam* s = (FsParam*)t; fsp_set(s, n, 5, 0, 0, 0, (void*)v); if (g_plog) LOG("P.SetD3D11 \"%s\" = %p", n, (void*)v);
}
static void FSP_SetD12 (NVSDK_NGX_Parameter* t, const char* n, void* v) {
 FsParam* s = (FsParam*)t; fsp_set(s, n, 5, 0, 0, 0, v); if (g_plog) LOG("P.SetD3D12 \"%s\" = %p", n, v);
}
static void FSP_SetVP (NVSDK_NGX_Parameter* t, const char* n, void* v) {
 FsParam* s = (FsParam*)t; fsp_set(s, n, 5, 0, 0, 0, v); if (g_plog) LOG("P.SetPtr \"%s\" = %p", n, v);
}
static NVSDK_NGX_Result FSP_GetULL(NVSDK_NGX_Parameter* t, const char* n, unsigned long long* v) {
 FsParam* s = (FsParam*)t; int r = fsp_get(s, n, 0, v, NULL, NULL, NULL); if (g_plog) LOG("P.GetULL \"%s\" -> 0x%x", n, r); return r;
}
static NVSDK_NGX_Result FSP_GetF (NVSDK_NGX_Parameter* t, const char* n, float* v) {
 FsParam* s = (FsParam*)t; int r = fsp_get(s, n, 1, NULL, v, NULL, NULL); if (g_plog) LOG("P.GetF \"%s\" -> 0x%x", n, r); return r;
}
static NVSDK_NGX_Result FSP_GetD (NVSDK_NGX_Parameter* t, const char* n, double* v) {
 FsParam* s = (FsParam*)t; int r = fsp_get(s, n, 2, NULL, NULL, v, NULL); if (g_plog) LOG("P.GetD \"%s\" -> 0x%x", n, r); return r;
}
static NVSDK_NGX_Result FSP_GetUI(NVSDK_NGX_Parameter* t, const char* n, unsigned int* v) {
 FsParam* s = (FsParam*)t; unsigned long long tmp = 0; int r = fsp_get(s, n, 3, &tmp, NULL, NULL, NULL);
 if (r == (int)NVSDK_NGX_Result_Success && v) *v = (unsigned int)tmp; if (g_plog) LOG("P.GetUI \"%s\" -> 0x%x", n, r); return r;
}
static NVSDK_NGX_Result FSP_GetI (NVSDK_NGX_Parameter* t, const char* n, int* v) {
 FsParam* s = (FsParam*)t; unsigned long long tmp = 0; int r = fsp_get(s, n, 4, &tmp, NULL, NULL, NULL);
 if (r == (int)NVSDK_NGX_Result_Success && v) *v = (int)(long long)tmp; if (g_plog) LOG("P.GetI \"%s\" -> 0x%x", n, r); return r;
}
static NVSDK_NGX_Result FSP_GetD11(NVSDK_NGX_Parameter* t, const char* n, ID3D11Resource** v) {
 FsParam* s = (FsParam*)t; void* p = NULL; int r = fsp_get(s, n, 5, NULL, NULL, NULL, &p);
 if (r == (int)NVSDK_NGX_Result_Success && v) *v = (ID3D11Resource*)p;
 if (g_plog) LOG("P.GetD3D11 \"%s\" -> 0x%x (%p)", n, r, p); return r;
}
static NVSDK_NGX_Result FSP_GetD12(NVSDK_NGX_Parameter* t, const char* n, void** v) {
 FsParam* s = (FsParam*)t; int r = fsp_get(s, n, 5, NULL, NULL, NULL, v); if (g_plog) LOG("P.GetD3D12 \"%s\" -> 0x%x", n, r); return r;
}
static NVSDK_NGX_Result FSP_GetVP (NVSDK_NGX_Parameter* t, const char* n, void** v) {
 FsParam* s = (FsParam*)t; int r = fsp_get(s, n, 5, NULL, NULL, NULL, v); if (g_plog) LOG("P.GetPtr \"%s\" -> 0x%x", n, r); return r;
}
static void FSP_Reset(NVSDK_NGX_Parameter* t) {
 FsParam* s = (FsParam*)t; s->n = 0; if (g_plog) LOG("P.Reset");
}

/* vtable order matches the SDK generation RotTR was built against (empirically
 * verified from parameter-call logs: float-set reaches slot 6, pointer-set
 * reaches slot 2, uint/int at 3/4). GP-ABI slots (int/uint/ptr) are placed at
 * the slots the game uses for pointers; XMM slots (float/double) only where the
 * game sends floating-point values. */
static void* g_param_vtbl[17] = {
 (void*)FSP_SetULL, (void*)FSP_SetULL, (void*)FSP_SetVP, (void*)FSP_SetUI, (void*)FSP_SetI,
 (void*)FSP_SetD11, (void*)FSP_SetF, (void*)FSP_SetD,
 (void*)FSP_GetULL, (void*)FSP_GetULL, (void*)FSP_GetVP, (void*)FSP_GetUI, (void*)FSP_GetI,
 (void*)FSP_GetD11, (void*)FSP_GetF, (void*)FSP_GetD,
 (void*)FSP_Reset
};

static FsParam* param_new(int owned) {
 FsParam* s = (FsParam*)HeapAlloc(GetProcessHeap(), HEAP_ZERO_MEMORY, sizeof(FsParam));
 if (!s) return NULL;
 s->vtbl = (void*)g_param_vtbl;
 s->owned = owned;
 return s;
}

static NVSDK_NGX_Result __cdecl FsOptimalSettingsCb(NVSDK_NGX_Parameter* p);
static NVSDK_NGX_Result __cdecl FsGetStatsCb(NVSDK_NGX_Parameter* p);

static void param_populate_capabilities(FsParam* s) {
 fsp_set(s, K_SSR_Available, 4, 1, 0, 0, NULL);
 fsp_set(s, K_SSR_NeedsUpdatedDrv, 4, 0, 0, 0, NULL);
 fsp_set(s, K_SSR_MinDrvMajor, 4, 511, 0, 0, NULL);
 fsp_set(s, K_SSR_MinDrvMinor, 4, 23, 0, 0, NULL);
 fsp_set(s, K_SSR_FeatInitResult, 4, (unsigned long long)NVSDK_NGX_Result_Success, 0, 0, NULL);
 fsp_set(s, K_DLSS_OptSettingsCb, 5, 0, 0, 0, (void*)&FsOptimalSettingsCb);
 fsp_set(s, K_DLSS_GetStatsCb, 5, 0, 0, 0, (void*)&FsGetStatsCb);
 fsp_set(s, K_DynMinW, 4, 640, 0, 0, NULL);
 fsp_set(s, K_DynMinH, 4, 360, 0, 0, NULL);
}

#include <stdint.h>
/* DLSS optimal-settings callback: game supplies Width/Height (display) + PerfQualityValue;
 * we return the recommended render resolution. Performance preset = 0.5 scale so the
 * pinned capture shapes come out exactly 960x540 @ 1080p. */
static NVSDK_NGX_Result __cdecl FsOptimalSettingsCb(NVSDK_NGX_Parameter* p) {
 unsigned int w = 0, h = 0; int pq = 0;
 FsParam* s = (FsParam*)p;
 unsigned long long tmp = 0;
 if (fsp_get(s, K_Width, 4, &tmp, NULL, NULL, NULL) == (int)NVSDK_NGX_Result_Success) w = (unsigned int)tmp;
 if (fsp_get(s, K_Height, 4, &tmp, NULL, NULL, NULL) == (int)NVSDK_NGX_Result_Success) h = (unsigned int)tmp;
 if (fsp_get(s, K_PerfQ, 4, &tmp, NULL, NULL, NULL) == (int)NVSDK_NGX_Result_Success) pq = (int)(long long)tmp;
 float scale = 0.5f;
 switch (pq) {
 case NVSDK_NGX_PerfQuality_Value_UltraPerformance: scale = 1.0f / 3.0f; break;
 case NVSDK_NGX_PerfQuality_Value_MaxPerf: scale = 0.5f; break;
 case NVSDK_NGX_PerfQuality_Value_Balanced: scale = 0.58f; break;
 case NVSDK_NGX_PerfQuality_Value_MaxQuality: scale = 2.0f / 3.0f; break;
 case NVSDK_NGX_PerfQuality_Value_UltraQuality: scale = 0.77f; break;
 case NVSDK_NGX_PerfQuality_Value_DLAA: scale = 1.0f; break;
 default: scale = 0.5f; break;
 }
 unsigned int ow = (unsigned int)(w * scale / 2.0f + 0.5f) * 2u;
 unsigned int oh = (unsigned int)(h * scale / 2.0f + 0.5f) * 2u;
 if (ow < 64) ow = 64;
 if (oh < 64) oh = 64;
 fsp_set(s, K_OutWidth, 4, ow, 0, 0, NULL);
 fsp_set(s, K_OutHeight, 4, oh, 0, 0, NULL);
 fsp_set(s, K_Sharpness, 1, 0, 0.0f, 0, NULL);
 fsp_set(s, K_DynMaxW, 4, w, 0, 0, NULL);
 fsp_set(s, K_DynMaxH, 4, h, 0, 0, NULL);
 fsp_set(s, K_DynMinW, 4, ow, 0, 0, NULL);
 fsp_set(s, K_DynMinH, 4, oh, 0, 0, NULL);
 LOG("OptimalSettingsCb w=%u h=%u pq=%d -> out %ux%u", w, h, pq, ow, oh);
 return NVSDK_NGX_Result_Success;
}

static NVSDK_NGX_Result __cdecl FsGetStatsCb(NVSDK_NGX_Parameter* p) {
 (void)p; LOG("GetStatsCb"); return NVSDK_NGX_Result_Success;
}
/* ------------------------------------------------------------------ */
/* capture + feature state */
/* ------------------------------------------------------------------ */

static ID3D11Device* g_dev = NULL;
static NVSDK_NGX_Handle g_handle = { 1 };
static unsigned int g_feat_w = 0, g_feat_h = 0, g_feat_ow = 0, g_feat_oh = 0;
static int g_feat_flags = 0, g_feat_pq = 0;
static int g_created = 0;

/* ini-gated features (all default OFF = exact v12c behavior).
 * Re-read at every arm => each feature flips with an ini edit + START
 * touch, NO dll swap. slots3 is STICKY (locks at wire-ensure: semaphore
 * max counts and slot allocation must not change mid-session). */
static int g_opt_simdmv = 0, g_opt_slots3 = 0, g_opt_zerowait = 0, g_opt_qicache = 0;
static int g_opt_mapn2 = 0; /* mapn2=1 pops the deferred-staging
 * ring at depth 2 (map N-2's copy) — the
 * lag-1 Map still drained the GPU queue
 * at saturation. NOT latched: ini flip +
 * START re-arm toggles it live */
static int g_opt_fpscap = 0; /* fpscap=N (0=off) — eval-end QPC
 * metronome. With mapn2 the game thread
 * sprints and the freed CPU contends the
 * daemon's socket read + floods the GPU
 * queue; a paced eval floor fixes both.
 * NOT latched: ini flip + START re-arm */
static int g_opt_wire = 2; /* wire=3 ships raw f32 MV rows
 * (daemon NEON converts) — latch at ensure */
static int g_slots_locked = 0;
static int g_nslots = 2; /* fixed at live_wire_ensure */
static volatile int g_live_fails; /* fwd decl: read_ini resets it */
static int g_ini_done = 0;
static int g_cap_target = 150;
static int g_armed = 0;
static int g_cap_n = 0;
static long long g_eval_n = 0;

/* mode globals (declared before read_ini; see live_output below) */
int g_live = 0; /* ini mode=live */
static int g_live_port = 48620;
static int g_live_sock = -1; /* connected socket, -1 = down */
/* fd numbers recycle under Wine (lowest-free), so
 * comparing fd VALUES cannot prove identity. Every socket swap goes through
 * g_live_cs and bumps g_live_gen; readers cache (fd, gen) and act only when
 * the generation still matches — the receiver can never abort the sender's
 * fresh connection or a game-owned recycled fd. */
static volatile long g_live_gen = 0;
static ULONGLONG g_conn_retry_until = 0; /* one 10s heal window per OUTAGE; re-armed on ini re-arm */
static int g_drop_run = 0; /* consecutive-drop run (accumulator fold cap) */
static int g_raw_slot_filled = 0; /* raw refreshed this eval (fmt-gate drop guard) */
static CRITICAL_SECTION g_live_cs; /* fwd tentative decl — ensure_sock/strike precede the worker block */
int live_net_connect(const char* host, int port, int timeout_ms);
int live_net_send_all(int s, const char* b, size_t n);
int live_net_recv_all(int s, char* b, size_t n);
void live_net_close(int s);
void live_net_abort(int s);

static void read_ini(void) {
 /* re-read at every arm so ini edits between captures take effect
 * without a game restart (single-process sessions are the norm here) */
 char p[MAX_PATH];
 snprintf(p, sizeof(p), "%s\\fsr4cap.ini", g_outdir);
 FILE* f = fopen(p, "r");
 if (!f && g_dll_dir[0]) { snprintf(p, sizeof(p), "%s\\fsr4cap.ini", g_dll_dir); f = fopen(p, "r"); }
 if (!f) { LOG("ini: none (count stays %d)", g_cap_target); return; }
 char line[256];
 while (fgets(line, sizeof(line), f)) {
 int v;
 char mstr[32];
 if (sscanf(line, " count = %d", &v) == 1 || sscanf(line, "count=%d", &v) == 1)
 g_cap_target = v;
 if (sscanf(line, " mode = %31s", mstr) == 1 || sscanf(line, "mode=%31s", mstr) == 1)
 g_live = (_stricmp(mstr, "live") == 0);
 if (sscanf(line, " live_port = %d", &v) == 1 || sscanf(line, "live_port=%d", &v) == 1)
 g_live_port = v;
 if (sscanf(line, " simdmv = %d", &v) == 1 || sscanf(line, "simdmv=%d", &v) == 1)
 g_opt_simdmv = v ? 1: 0;
 if (sscanf(line, " slots3 = %d", &v) == 1 || sscanf(line, "slots3=%d", &v) == 1)
 g_opt_slots3 = v ? 1: 0;
 if (sscanf(line, " zerowait = %d", &v) == 1 || sscanf(line, "zerowait=%d", &v) == 1)
 g_opt_zerowait = v ? 1: 0;
 if (sscanf(line, " qicache = %d", &v) == 1 || sscanf(line, "qicache=%d", &v) == 1)
 g_opt_qicache = v ? 1: 0;
 if (sscanf(line, " wire = %d", &v) == 1 || sscanf(line, "wire=%d", &v) == 1)
 g_opt_wire = (v == 3) ? 3: 2;
 if (sscanf(line, " mapn2 = %d", &v) == 1 || sscanf(line, "mapn2=%d", &v) == 1)
 g_opt_mapn2 = v ? 1: 0;
 if (sscanf(line, " fpscap = %d", &v) == 1 || sscanf(line, "fpscap=%d", &v) == 1)
 g_opt_fpscap = (v >= 0 && v <= 240) ? v: 0;
 }
 fclose(f);
 g_live_fails = 0; /* a re-arm clears stale strikes */
 g_conn_retry_until = 0; /* ...and re-arms the connect-heal window */
 LOG("ini: count=%d live=%d port=%d | v13 opts simdmv=%d slots3=%d zerowait=%d qicache=%d mapn2=%d wire=%d fpscap=%d%s",
 g_cap_target, g_live, g_live_port,
 g_opt_simdmv, g_opt_slots3, g_opt_zerowait, g_opt_qicache, g_opt_mapn2, g_opt_wire, g_opt_fpscap,
 g_slots_locked ? " (slots3 LOCKED from first wire init)": "");
}

static int file_exists(const char* p) {
 DWORD a = GetFileAttributesA(p);
 return a != INVALID_FILE_ATTRIBUTES;
}

static void capture_poll(void) {
 char p[MAX_PATH];
 /* first eval: read ini even without an arm, so mode=live engages from
 * frame 1 (the START-only read left live off until a trigger file existed) */
 if (!g_ini_done) { read_ini(); g_ini_done = 1; }
 snprintf(p, sizeof(p), "%s\\STOP", g_outdir);
 if (g_armed && file_exists(p)) {
 g_armed = 0;
 DeleteFileA(p); /* consume STOP like START — a stale one aborted every future arm */
 snprintf(p, sizeof(p), "%s\\CAPTURE_ABORTED", g_outdir);
 FILE* f = fopen(p, "w"); if (f) { fprintf(f, "captured=%d\n", g_cap_n); fclose(f); }
 LOG("CAPTURE ABORT by STOP at %d", g_cap_n);
 return;
 }
 if (!g_armed) {
 snprintf(p, sizeof(p), "%s\\START", g_outdir);
 if (file_exists(p)) {
 read_ini();
 g_armed = 1; g_cap_n = 0;
 /* consume the trigger: without this the next poll re-arms 1 frame
 * after CAPTURE_DONE (correction — observed 33ms re-arm) */
 if (DeleteFileA(p)) LOG("START consumed (deleted)");
 else LOG("START delete FAILED err=%lu (next poll will re-arm!)", GetLastError());
 LOG("CAPTURE ARM (target=%d)", g_cap_target);
 }
 }
}

static void capture_maybe_done(void) {
 char p[MAX_PATH];
 if (g_armed && g_cap_n >= g_cap_target) {
 g_armed = 0;
 snprintf(p, sizeof(p), "%s\\CAPTURE_DONE", g_outdir);
 FILE* f = fopen(p, "w"); if (f) { fprintf(f, "captured=%d target=%d\n", g_cap_n, g_cap_target); fclose(f); }
 LOG("CAPTURE DONE n=%d", g_cap_n);
 }
}

/* ------------------------------------------------------------------ */
/* resource plumbing */
/* ------------------------------------------------------------------ */

static ID3D11Texture2D* g_pending_stage = NULL;
static ID3D11Texture2D* g_pending_tex = NULL;
static int g_pending_stage_cached = 0; /* staging cache slot — Unmap only */

#include <xmmintrin.h> /* MXCSR pin in sp_eval_begin precedes the emmintrin block */
static unsigned int g_eval_csr = 0x1F80; /* foreign MXCSR saved per eval, restored at part3's exits */
/* ---------------- QPC span instrumentation ----------------
 * Coarser 16-span form: every blocking/copy phase of the
 * eval gets a microsecond span; a 32-entry ring collects them; every 30
 * evals we flush p50/p90 + flags + up to 3 SPIKE tuples. Replaces the
 * stale-prone g_t_wait/g_t_write logging (miss-frames never stamped them;
 * the redisplay write was never timed; "wait" swallowed the cache memcpy).
 * Game thread is the ONLY writer — no locks. Falls back to the legacy
 * 1/30 line if QPC is unusable. */
typedef enum {
 SP_TOT = 0, SP_PRE, SP_MAPC, SP_MAPM, SP_RSV, SP_DEC, SP_CMP, SP_MVC,
 SP_SUB, SP_REL, SP_WAIT, SP_CPY, SP_WRT, SP_NAI, SP_UNM, SP_DSUB,
 SP_COUNT
} SpanId;
static const char* g_sp_name[SP_COUNT] = {
 "tot","pre","mapC","mapM","rsv","dec","cmp","mvc","sub","rel","wait","cpy","wrt","nai","unm","dsub"
};
static LARGE_INTEGER g_qpc_freq = { 0 };
static int g_qpc_ok = 0;
static LARGE_INTEGER g_sp_t[SP_COUNT];
static unsigned short g_sp_us[32][SP_COUNT];
static unsigned char g_sp_fl[32];
static unsigned g_sp_n = 0, g_sp_i = 0, g_sp_drops = 0, g_sp_fresh = 0, g_sp_cache = 0;
#define SPFL_DROP 0x01
#define SPFL_FRESH 0x02
#define SPFL_CACHE 0x04
#define SPFL_MVPRE 0x08
#define SPFL_SKIPC 0x10
#define SPFL_NAIVE 0x20
#define SP_BEGIN(id) do { if (g_qpc_ok) QueryPerformanceCounter(&g_sp_t[id]); } while (0)
#define SP_END(id) do { if (g_qpc_ok) { LARGE_INTEGER _n; QueryPerformanceCounter(&_n); unsigned long long _us = (unsigned long long)((_n.QuadPart - g_sp_t[id].QuadPart) * 1000000ULL / g_qpc_freq.QuadPart); g_sp_us[g_sp_i][id] = (unsigned short)(_us > 65535 ? 65535: _us); } } while (0)
#define SP_FLAG(b) do { g_sp_fl[g_sp_i] |= (b); } while (0)
static void sp_eval_begin(void) {
 static int qpc_checked = 0; /* lazy init: part1's ensure_init is frozen */
 if (!qpc_checked) {
 qpc_checked = 1;
 LARGE_INTEGER f;
 if (QueryPerformanceFrequency(&f) && f.QuadPart > 0) {
 g_qpc_freq = f;
 g_qpc_ok = 1;
 LOG("v13: QPC instrumentation on (freq=%I64d Hz)", f.QuadPart);
 } else {
 LOG("v13: QPC unusable — legacy 1/30 timing only");
 }
 }
 /* cvtps_epi32 honors MXCSR.RC — pin RC to nearest for the
 * vector subnormal path. SAVE the foreign mode first; part3 restores
 * it at the eval exits (never silently mutate the game's FP env).
 * DAZ/FTZ are provably irrelevant here: subnormal-path inputs are f32
 * NORMALS; true f32-denormals take the designed zero path either way. */
 g_eval_csr = _mm_getcsr();
 _mm_setcsr((g_eval_csr & ~0x6000u) | 0x0000u);
 g_sp_i = g_sp_n & 31u;
 g_sp_fl[g_sp_i] = 0;
 memset(g_sp_us[g_sp_i], 0, sizeof(g_sp_us[g_sp_i]));
 SP_BEGIN(SP_TOT);
}
static unsigned sp_pctl(int id, double pct) {
 unsigned short v[32]; unsigned n = g_sp_n > 32u ? 32u: g_sp_n;
 if (!n) return 0;
 for (unsigned i = 0; i < n; i++) v[i] = g_sp_us[i][id];
 for (unsigned i = 1; i < n; i++) { unsigned short k = v[i]; unsigned j = i;
 while (j && v[j-1] > k) { v[j] = v[j-1]; j--; } v[j] = k; }
 unsigned idx = (unsigned)(pct * (n - 1) + 0.5);
 return v[idx];
}
static void sp_flush(long long eval_n) {
 unsigned n = g_sp_n > 32u ? 32u: g_sp_n;
 if (!n) { g_sp_n = 0; return; }
 LOG("S30 eval=%lld n=%u drops=%u fresh=%u cached=%u | p50us tot=%u pre=%u mapC=%u mapM=%u rsv=%u dec=%u cmp=%u mvc=%u sub=%u rel=%u wait=%u cpy=%u wrt=%u nai=%u unm=%u dsub=%u",
 eval_n, n, g_sp_drops, g_sp_fresh, g_sp_cache,
 sp_pctl(SP_TOT,.5), sp_pctl(SP_PRE,.5), sp_pctl(SP_MAPC,.5), sp_pctl(SP_MAPM,.5),
 sp_pctl(SP_RSV,.5), sp_pctl(SP_DEC,.5), sp_pctl(SP_CMP,.5), sp_pctl(SP_MVC,.5),
 sp_pctl(SP_SUB,.5), sp_pctl(SP_REL,.5), sp_pctl(SP_WAIT,.5), sp_pctl(SP_CPY,.5),
 sp_pctl(SP_WRT,.5), sp_pctl(SP_NAI,.5), sp_pctl(SP_UNM,.5), sp_pctl(SP_DSUB,.5));
 LOG("S90 eval=%lld p90us tot=%u mapC=%u mapM=%u mvc=%u sub=%u wait=%u wrt=%u dsub=%u",
 eval_n, sp_pctl(SP_TOT,.9), sp_pctl(SP_MAPC,.9), sp_pctl(SP_MAPM,.9),
 sp_pctl(SP_MVC,.9), sp_pctl(SP_SUB,.9), sp_pctl(SP_WAIT,.9), sp_pctl(SP_WRT,.9),
 sp_pctl(SP_DSUB,.9));
 /* SPIKE: raw tuples of the 3 worst frames (tot tail anatomy) */
 for (int s = 0; s < 3; s++) {
 int wi = -1; unsigned wv = 0;
 for (unsigned i = 0; i < n; i++)
 if (g_sp_us[i][SP_TOT] > wv) { wv = g_sp_us[i][SP_TOT]; wi = (int)i; }
 if (wi < 0 || wv < 20000) break; /* only report >=20ms frames */
 LOG("SPIKE tot=%u mapC=%u mapM=%u rsv=%u dec=%u cmp=%u mvc=%u sub=%u rel=%u wait=%u cpy=%u wrt=%u nai=%u unm=%u dsub=%u fl=0x%x",
 g_sp_us[wi][SP_TOT], g_sp_us[wi][SP_MAPC], g_sp_us[wi][SP_MAPM],
 g_sp_us[wi][SP_RSV], g_sp_us[wi][SP_DEC], g_sp_us[wi][SP_CMP],
 g_sp_us[wi][SP_MVC], g_sp_us[wi][SP_SUB], g_sp_us[wi][SP_REL],
 g_sp_us[wi][SP_WAIT], g_sp_us[wi][SP_CPY], g_sp_us[wi][SP_WRT],
 g_sp_us[wi][SP_NAI], g_sp_us[wi][SP_UNM], g_sp_us[wi][SP_DSUB],
 (unsigned)g_sp_fl[wi]);
 g_sp_us[wi][SP_TOT] = 0; /* marked reported */
 }
 g_sp_n = 0; g_sp_drops = 0; g_sp_fresh = 0; g_sp_cache = 0;
}

/* (ini qicache=1): AddRef-held resource->(tex,desc) cache — kills the
 * per-eval QueryInterface+GetDesc pairs in live_write_out8/dfr_push_one.
 * The cache's own AddRef keeps the address from being recycled; the caller
 * still releases ITS ref, so refcounts stay neutral. Game-thread-only. */
typedef struct { ID3D11Resource* res; ID3D11Texture2D* tex; D3D11_TEXTURE2D_DESC desc; } qcent;
static qcent g_qcache[3] = { 0 };
static int qcache_lookup(ID3D11Resource* res, ID3D11Texture2D** tex, D3D11_TEXTURE2D_DESC* od) {
 if (res)
 for (int k = 0; k < 3; k++)
 if (g_qcache[k].res == res && g_qcache[k].tex) { *tex = g_qcache[k].tex; *od = g_qcache[k].desc; return 1; }
 return 0;
}
static void qcache_store(ID3D11Resource* res, ID3D11Texture2D* tex, const D3D11_TEXTURE2D_DESC* od) {
 if (!res || !tex) return;
 for (int k = 0; k < 3; k++)
 if (g_qcache[k].res == res) {
 if (g_qcache[k].tex) g_qcache[k].tex->lpVtbl->Release(g_qcache[k].tex);
 g_qcache[k].tex = tex; g_qcache[k].desc = *od;
 tex->lpVtbl->AddRef(tex);
 return;
 }
 int e = 0;
 for (int k = 0; k < 3; k++)
 if (!g_qcache[k].res) { e = k; goto have; }
 if (g_qcache[e].tex) g_qcache[e].tex->lpVtbl->Release(g_qcache[e].tex);
have:
 g_qcache[e].res = res; g_qcache[e].tex = tex; g_qcache[e].desc = *od;
 tex->lpVtbl->AddRef(tex);
}

/* staging-texture cache shared by the sync and deferred paths */
static ID3D11Texture2D* sc_tex[2] = { NULL, NULL };
static unsigned int sc_fmt[2] = { 0, 0 }, sc_w[2] = { 0, 0 }, sc_h[2] = { 0, 0 };
static ID3D11Texture2D* staging_for_desc(const D3D11_TEXTURE2D_DESC* csd) {
 int slot = -1;
 for (int k = 0; k < 2; k++)
 if (sc_tex[k] && sc_fmt[k] == csd->Format && sc_w[k] == csd->Width && sc_h[k] == csd->Height)
 return sc_tex[k];
 slot = (sc_tex[0] == NULL) ? 0: ((sc_tex[1] == NULL) ? 1: 0);
 if (sc_tex[slot]) sc_tex[slot]->lpVtbl->Release(sc_tex[slot]);
 sc_tex[slot] = NULL;
 D3D11_TEXTURE2D_DESC sd = *csd; /* COM method takes non-const */
 if (FAILED(g_dev->lpVtbl->CreateTexture2D(g_dev, &sd, NULL, &sc_tex[slot])) || !sc_tex[slot]) {
 sc_tex[slot] = NULL;
 return NULL;
 }
 sc_fmt[slot] = csd->Format; sc_w[slot] = csd->Width; sc_h[slot] = csd->Height;
 return sc_tex[slot];
}

static float* g_cbuf = NULL; /* color decoded as f4 (4 floats/px) */
static float* g_mbuf = NULL; /* mv decoded as f4 (r,g used) */
static long long g_mbuf_eval = -1; /* eval that filled it (stale-MV guard) */
static size_t g_cbuf_n = 0, g_mbuf_n = 0;
static unsigned int g_cw = 0, g_ch = 0, g_mw = 0, g_mh = 0;

static int fbuf_get(float** buf, size_t* cap, size_t need_px) {
 if (*buf == NULL || *cap < need_px) {
 if (*buf) HeapFree(GetProcessHeap(), 0, *buf);
 *buf = (float*)HeapAlloc(GetProcessHeap(), HEAP_ZERO_MEMORY, need_px * 4 * sizeof(float));
 *cap = need_px;
 if (!*buf) return 0;
 }
 return 1;
}

static int stage_to_cpu(ID3D11DeviceContext* ctx, ID3D11Resource* res, const char* what,
 D3D11_TEXTURE2D_DESC* desc, D3D11_MAPPED_SUBRESOURCE* map) {
 ID3D11Texture2D* tex = NULL;
 memset(desc, 0, sizeof(*desc)); memset(map, 0, sizeof(*map));
 g_pending_stage = NULL; g_pending_tex = NULL;
 if (FAILED(((IUnknown*)res)->lpVtbl->QueryInterface((IUnknown*)res, &IID_ID3D11Texture2D, (void**)&tex)) || !tex) {
 LOG("eval %s: not a Texture2D", what);
 return 0;
 }
 tex->lpVtbl->GetDesc(tex, desc);
 if (g_plog) LOG("eval %s: %ux%u fmt=%u sample=%u mips=%u array=%u bind=%u misc=%u", what,
 desc->Width, desc->Height, desc->Format, desc->SampleDesc.Count,
 desc->MipLevels, desc->ArraySize, desc->BindFlags, desc->MiscFlags);
 if (desc->SampleDesc.Count != 1) { LOG("eval %s: MSAA unsupported", what); tex->lpVtbl->Release(tex); return 0; }
 if (fmt_bytes(desc->Format) == 0) { LOG("eval %s: format %u unsupported", what, desc->Format); tex->lpVtbl->Release(tex); return 0; }

 D3D11_TEXTURE2D_DESC sd = *desc;
 sd.MipLevels = 1; sd.ArraySize = 1;
 sd.Usage = D3D11_USAGE_STAGING;
 sd.BindFlags = 0;
 sd.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
 sd.MiscFlags = 0;
 /* staging-texture cache — CreateTexture2D+Release twice per frame
 * (color+mv) is pure churn under Wine/D3D11; 2 slots keyed by
 * (fmt,w,h), Unmap-only release. lookup extracted so the
 * deferred path can share it. */
 ID3D11Texture2D* stage = staging_for_desc(&sd);
 if (!stage) {
 LOG("eval %s: staging create FAILED", what); tex->lpVtbl->Release(tex); return 0;
 }
 ctx->lpVtbl->CopySubresourceRegion(ctx, (ID3D11Resource*)stage, 0, 0, 0, 0, (ID3D11Resource*)tex, 0, NULL);
 HRESULT hr = ctx->lpVtbl->Map(ctx, (ID3D11Resource*)stage, 0, D3D11_MAP_READ, 0, map);
 if (FAILED(hr)) {
 LOG("eval %s: map FAILED hr=0x%lx", what, (long)hr);
 tex->lpVtbl->Release(tex); return 0;
 }
 g_pending_stage = stage;
 g_pending_stage_cached = 1;
 g_pending_tex = tex;
 return 1;
}

static void stage_release(ID3D11DeviceContext* ctx) {
 if (g_pending_stage) {
 ctx->lpVtbl->Unmap(ctx, (ID3D11Resource*)g_pending_stage, 0);
 if (!g_pending_stage_cached) g_pending_stage->lpVtbl->Release(g_pending_stage);
 g_pending_stage = NULL;
 g_pending_stage_cached = 0;
 }
 if (g_pending_tex) { g_pending_tex->lpVtbl->Release(g_pending_tex); g_pending_tex = NULL; }
}

/* ---- 3-entry deferred-staging ring (ini mapn2) ----
 * The old deferred staging mapped the copy queued one eval ago (lag 1).
 * At GPU saturation the game's own render is ~1 frame deep in the queue,
 * so that Map still drained it — the 17-28ms mapC spikes behind the last
 * of the rubber-banding. mapn2=1 pops the oldest entry only when TWO are
 * queued (map N-2's copy): two frames of GPU grace, blocking ~never fires.
 * Cost: +1 frame display latency; color, MV and their jitter/scale params
 * all ride the same entry, so the shipped eval state stays a consistent
 * pair. mapn2=0 = the v14 lag-1 behavior (same code path, pop-at-1).
 * Each entry owns its staging textures (created on first touch, recreated
 * on desc change) — unlike v14's desc-keyed cache there is no eviction
 * between the copy and the map: the mapped texture is always the one that
 * was copied into. Push/pop counters are monotonic mod 3, so a mid-session
 * mapn2 flip can only add one sync-path bootstrap eval, never alias an
 * in-flight copy with the entry being mapped (queue depth <= 2 < 3). */
#define DFR_N 3
typedef struct {
 ID3D11Texture2D* tc; unsigned ck[3]; /* color staging tex + fmt/w/h key */
 ID3D11Texture2D* tm; unsigned mk[3]; /* mv staging tex + fmt/w/h key */
 D3D11_TEXTURE2D_DESC dc, dm; /* ORIGINAL descs of the copied frame */
 int okc, okm;
 float jx, jy, sx, sy; /* params of the frame riding this entry */
} DFREntry;
static DFREntry g_dfr[DFR_N];
static unsigned g_dfr_u = 0, g_dfr_p = 0; /* pushes/pops (monotonic; idx = ctr % DFR_N) */
static int dfr_depth(void) { return (int)(g_dfr_u - g_dfr_p); }
static int dfr_need(void) { return g_opt_mapn2 ? 2: 1; }

static ID3D11Texture2D* dfr_stage_for(ID3D11Texture2D** slot, unsigned* key,
 const D3D11_TEXTURE2D_DESC* od) {
 if (*slot && key[0] == od->Format && key[1] == od->Width && key[2] == od->Height)
 return *slot;
 if (*slot) { (*slot)->lpVtbl->Release(*slot); *slot = NULL; }
 D3D11_TEXTURE2D_DESC sd = *od; /* COM method takes non-const */
 sd.MipLevels = 1; sd.ArraySize = 1;
 sd.Usage = D3D11_USAGE_STAGING;
 sd.BindFlags = 0; sd.CPUAccessFlags = D3D11_CPU_ACCESS_READ; sd.MiscFlags = 0;
 if (FAILED(g_dev->lpVtbl->CreateTexture2D(g_dev, &sd, NULL, slot)) || !*slot) {
 *slot = NULL;
 return NULL;
 }
 key[0] = od->Format; key[1] = od->Width; key[2] = od->Height;
 return *slot;
}

static int dfr_push_one(ID3D11DeviceContext* ctx, ID3D11Resource* res,
 ID3D11Texture2D** tex_slot, unsigned* key,
 D3D11_TEXTURE2D_DESC* dout) {
 ID3D11Texture2D* tex = NULL;
 memset(dout, 0, sizeof(*dout));
 int dcached = 0;
 if (g_opt_qicache && qcache_lookup(res, &tex, dout)) {
 dcached = 1;
 } else {
 if (FAILED(((IUnknown*)res)->lpVtbl->QueryInterface((IUnknown*)res, &IID_ID3D11Texture2D, (void**)&tex)) || !tex) {
 LOG("dfr: not a Texture2D");
 return 0;
 }
 tex->lpVtbl->GetDesc(tex, dout);
 if (g_opt_qicache) qcache_store(res, tex, dout);
 }
 if (dout->SampleDesc.Count != 1 || fmt_bytes(dout->Format) == 0) {
 LOG("dfr: fmt %u unsupported", dout->Format);
 if (!dcached) tex->lpVtbl->Release(tex);
 return 0;
 }
 ID3D11Texture2D* st = dfr_stage_for(tex_slot, key, dout);
 if (!st) { if (!dcached) tex->lpVtbl->Release(tex); return 0; }
 ctx->lpVtbl->CopySubresourceRegion(ctx, (ID3D11Resource*)st, 0, 0, 0, 0, (ID3D11Resource*)tex, 0, NULL);
 if (!dcached) tex->lpVtbl->Release(tex);
 return 1;
}

/* decode a whole mapped texture into an f4 buffer (returns 1 on success) */
static int decode_all(D3D11_TEXTURE2D_DESC* desc, D3D11_MAPPED_SUBRESOURCE* map,
 float** buf, size_t* cap, unsigned int* ow, unsigned int* oh) {
 unsigned int w = desc->Width, h = desc->Height;
 size_t n = (size_t)w * h;
 int bb = fmt_bytes(desc->Format);
 if (bb == 0) return 0;
 if (!fbuf_get(buf, cap, n)) return 0;
 unsigned char* base = (unsigned char*)map->pData;
 for (unsigned int y = 0; y < h; y++) {
 unsigned char* row = base + (size_t)y * map->RowPitch;
 f4* dst = (f4*)(*buf + (size_t)y * w * 4);
 for (unsigned int x = 0; x < w; x++) {
 f4 v; int used = decode_px(desc->Format, row + (size_t)x * (unsigned)bb, &v);
 if (!used) { v.r = v.g = v.b = 0; v.a = 1; }
 dst[x] = v;
 }
 }
 *ow = w; *oh = h;
 return 1;
}

static void dump_file(const char* suffix, const void* data, size_t bytes, unsigned int idx) {
 char p[MAX_PATH];
 snprintf(p, sizeof(p), "%s\\f%08u_%s", g_outdir, idx, suffix);
 FILE* f = fopen(p, "wb");
 if (!f) { LOG("dump %s OPEN FAILED", suffix); return; }
 size_t wr = fwrite(data, 1, bytes, f);
 fclose(f);
 if (wr != bytes) LOG("dump %s SHORT WRITE %zu/%zu", suffix, wr, bytes);
}

/* stage + dump a small auxiliary resource (ExposureTexture): raw bytes of the
 * top-left <=64x64 region plus a word-level log line for instant diagnosis */
static void dump_small_resource(ID3D11DeviceContext* ctx, ID3D11Resource* res,
 const char* what, unsigned int idx) {
 D3D11_TEXTURE2D_DESC d; D3D11_MAPPED_SUBRESOURCE m;
 if (!stage_to_cpu(ctx, res, what, &d, &m)) return;
 unsigned int w = d.Width > 64 ? 64: d.Width;
 unsigned int h = d.Height > 64 ? 64: d.Height;
 int bb = fmt_bytes(d.Format);
 if (bb > 0 && w && h) {
 size_t rb = (size_t)w * (size_t)bb;
 unsigned char* raw = (unsigned char*)HeapAlloc(GetProcessHeap(), 0, rb * h);
 if (raw) {
 const unsigned char* base = (const unsigned char*)m.pData;
 for (unsigned int y = 0; y < h; y++)
 memcpy(raw + (size_t)y * rb, base + (size_t)y * m.RowPitch, rb);
 char sfx[48];
 snprintf(sfx, sizeof(sfx), "%s.bin", what);
 dump_file(sfx, raw, rb * h, idx);
 const unsigned int* wd = (const unsigned int*)raw;
 if (rb >= 32)
 LOG("%s[%u] fmt=%u words: %08x %08x %08x %08x %08x %08x %08x %08x",
 what, idx, d.Format, wd[0], wd[1], wd[2], wd[3], wd[4], wd[5], wd[6], wd[7]);
 else if (rb >= 4)
 LOG("%s[%u] w=%u h=%u fmt=%u bytes=%zu first=%08x", what, idx,
 w, h, d.Format, rb * h, wd[0]);
 HeapFree(GetProcessHeap(), 0, raw);
 }
 }
 stage_release(ctx);
}

static void write_meta(unsigned int idx, D3D11_TEXTURE2D_DESC* cd, D3D11_TEXTURE2D_DESC* md,
 float jx, float jy, float sx, float sy, float preexp) {
 char p[MAX_PATH];
 snprintf(p, sizeof(p), "%s\\f%08u_meta.txt", g_outdir, idx);
 FILE* f = fopen(p, "w");
 if (!f) return;
 fprintf(f, "eval=%lld cap=%u\n", g_eval_n, idx);
 fprintf(f, "color_fmt=%u color_w=%u color_h=%u\n", cd->Format, cd->Width, cd->Height);
 fprintf(f, "mv_fmt=%u mv_w=%u mv_h=%u\n", md->Format, md->Width, md->Height);
 fprintf(f, "jitter_x=%.9g jitter_y=%.9g\n", (double)jx, (double)jy);
 fprintf(f, "mv_scale_x=%.9g mv_scale_y=%.9g\n", (double)sx, (double)sy);
 fprintf(f, "pre_exposure=%.9g\n", (double)preexp);
 fprintf(f, "create_flags=%d pq=%d\n", g_feat_flags, g_feat_pq);
 fprintf(f, "feat_create=%ux%u feat_out=%ux%u\n", g_feat_w, g_feat_h, g_feat_ow, g_feat_oh);
 fclose(f);
}

/* bilinear upsample color buffer into the output resource (naive display path) */
static int naive_output(ID3D11DeviceContext* ctx, ID3D11Resource* out_res) {
 if (!g_cbuf || g_cw == 0 || g_ch == 0) return 0;
 ID3D11Texture2D* otex = NULL;
 if (FAILED(((IUnknown*)out_res)->lpVtbl->QueryInterface((IUnknown*)out_res, &IID_ID3D11Texture2D, (void**)&otex)) || !otex) return 0;
 D3D11_TEXTURE2D_DESC od; otex->lpVtbl->GetDesc(otex, &od);
 int bb = fmt_bytes(od.Format);
 if (bb == 0 || od.SampleDesc.Count != 1) { otex->lpVtbl->Release(otex); return 0; }
 size_t n = (size_t)od.Width * od.Height;
 unsigned char* obuf = (unsigned char*)HeapAlloc(GetProcessHeap(), 0, n * (size_t)bb);
 if (!obuf) { otex->lpVtbl->Release(otex); return 0; }
 for (unsigned int y = 0; y < od.Height; y++) {
 float fy = ((float)y + 0.5f) * (float)g_ch / (float)od.Height - 0.5f;
 if (fy < 0) fy = 0;
 if (fy > (float)(g_ch - 1)) fy = (float)(g_ch - 1);
 unsigned int y0 = (unsigned int)fy;
 unsigned int y1 = (y0 + 1 < g_ch) ? y0 + 1: g_ch - 1;
 float ty = fy - (float)y0;
 unsigned char* orow = obuf + (size_t)y * od.Width * (unsigned)bb;
 for (unsigned int x = 0; x < od.Width; x++) {
 float fx = ((float)x + 0.5f) * (float)g_cw / (float)od.Width - 0.5f;
 if (fx < 0) fx = 0;
 if (fx > (float)(g_cw - 1)) fx = (float)(g_cw - 1);
 unsigned int x0 = (unsigned int)fx;
 unsigned int x1 = (x0 + 1 < g_cw) ? x0 + 1: g_cw - 1;
 float tx = fx - (float)x0;
 f4* c00 = (f4*)(g_cbuf + ((size_t)y0 * g_cw + x0) * 4);
 f4* c01 = (f4*)(g_cbuf + ((size_t)y0 * g_cw + x1) * 4);
 f4* c10 = (f4*)(g_cbuf + ((size_t)y1 * g_cw + x0) * 4);
 f4* c11 = (f4*)(g_cbuf + ((size_t)y1 * g_cw + x1) * 4);
 f4 v;
 v.r = c00->r * (1 - tx) * (1 - ty) + c01->r * tx * (1 - ty) + c10->r * (1 - tx) * ty + c11->r * tx * ty;
 v.g = c00->g * (1 - tx) * (1 - ty) + c01->g * tx * (1 - ty) + c10->g * (1 - tx) * ty + c11->g * tx * ty;
 v.b = c00->b * (1 - tx) * (1 - ty) + c01->b * tx * (1 - ty) + c10->b * (1 - tx) * ty + c11->b * tx * ty;
 v.a = 1.0f;
 encode_px(od.Format, &v, orow + (size_t)x * (unsigned)bb);
 }
 }
 D3D11_TEXTURE2D_DESC td = od;
 td.Usage = D3D11_USAGE_DEFAULT;
 td.CPUAccessFlags = 0;
 ID3D11Texture2D* tmp = NULL;
 if (FAILED(g_dev->lpVtbl->CreateTexture2D(g_dev, &td, NULL, &tmp)) || !tmp) {
 HeapFree(GetProcessHeap(), 0, obuf); otex->lpVtbl->Release(otex); return 0;
 }
 ctx->lpVtbl->UpdateSubresource(ctx, (ID3D11Resource*)tmp, 0, NULL, obuf, od.Width * (unsigned)bb, 0);
 ctx->lpVtbl->CopyResource(ctx, out_res, (ID3D11Resource*)tmp);
 tmp->lpVtbl->Release(tmp);
 HeapFree(GetProcessHeap(), 0, obuf);
 otex->lpVtbl->Release(otex);
 return 1;
}

/* ------------------------------------------------------------------ */
/* live mode — serve DLSS execute from the ARM64 NPU daemon over */
/* TCP loopback. Fallback = naive_output. */
/* pipelined transport: a worker thread owns the socket and the */
/* round-trip; the game's render thread drops the raw texels into a */
/* slot and picks up the most recent COMPLETED frame (bounded wait). */
/* Wire v2: send [len8 rawtex][rawtex][len8 mv][mv], */
/* recv [len8 rgba8][rgba8]. The daemon auto-detects v2 by */
/* the color length prefix (2,073,600 vs the old 8,294,400 f4 bytes). */
/* ------------------------------------------------------------------ */
/* dims compile-time overridable for the 720p arm (-DLWV=640u -DLHV=360u);
 * defaults = production 540p->1080p */
#ifndef LWV
#define LWV 960u
#define LHV 540u
#endif
#define RAWTEX_BYTES ((size_t)LWV * LHV * 4u) /* 2,073,600 */
#define MVH_BYTES ((size_t)LWV * LHV * 4u) /* 2,073,600 */
#define OUT8_BYTES ((size_t)(LWV * 2u) * (LHV * 2u) * 4u)/* OS = 2x LR */
#define MVF32_BYTES ((size_t)LWV * LHV * 2 * 4)/* 4,147,200 f32 xy pairs (wire v3) */
static float* g_mvraw_f32 = NULL; /* v3: staged f32 rows */
static int g_wire_v3 = 0; /* latched at ensure */
static float* g_mvf32_acc = NULL; /* drop accumulator (f32 domain) */
static int g_mvf32_acc_valid = 0;

static int live_ensure_sock(void) {
 if (g_live_sock >= 0) return g_live_sock;
 /* a single refused connect used to zero g_live INSTANTLY —
 * the "10fps naive" trap on every daemon restart (the 2-strike cushion
 * only covered mid-stream failures). Now refusals retry inside a 10s
 * window without touching g_live (heals any restart); after the window
 * the normal strike path owns the disable. */
 ULONGLONG now = GetTickCount64(); /* g_conn_retry_until is file-scope (read_ini re-arms it) */
 if (g_conn_retry_until == 0) g_conn_retry_until = now + 10000;
 for (;;) {
 int ns = live_net_connect("127.0.0.1", g_live_port, 500);
 if (ns >= 0) {
 EnterCriticalSection(&g_live_cs);
 g_live_sock = ns;
 g_live_gen++;
 LeaveCriticalSection(&g_live_cs);
 g_conn_retry_until = 0;
 LOG("live: connected to daemon (gen %ld)", g_live_gen);
 return ns;
 }
 if (GetTickCount64() >= g_conn_retry_until) break;
 Sleep(250);
 }
 LOG("live: daemon unreachable on 127.0.0.1:%d (retry window elapsed; "
 "strikes decide)", g_live_port);
 return -1;
}


/* write a completed RGBA8 frame into the output resource. RotTR's out is
 * fmt 28 (RGBA8) 1920x1080 -> direct byte path (out-desc ground truth from
 * the session log); any other format falls through encode_px (u8/255). */
static int live_write_out8_core(ID3D11DeviceContext* ctx, ID3D11Resource* out_res,
 const unsigned char* rgb8) {
 ID3D11Texture2D* otex = NULL;
 D3D11_TEXTURE2D_DESC od;
 int cached = 0;
 if (g_opt_qicache && qcache_lookup(out_res, &otex, &od)) {
 cached = 1; /* borrowed from the cache — do NOT Release */
 } else {
 if (FAILED(((IUnknown*)out_res)->lpVtbl->QueryInterface((IUnknown*)out_res,
 &IID_ID3D11Texture2D, (void**)&otex)) || !otex) return 0;
 otex->lpVtbl->GetDesc(otex, &od);
 if (g_opt_qicache) qcache_store(out_res, otex, &od);
 }
 {
 static int od_logged = 0;
 if (!od_logged) {
 LOG("live: out desc fmt=%u %ux%u sample=%u bind=%u misc=%u usage=%u "
 "mips=%u array=%u (direct path: fmt 28/29 @1920x1080, mips=1 array=1)",
 od.Format, od.Width, od.Height, od.SampleDesc.Count,
 od.BindFlags, od.MiscFlags, od.Usage, od.MipLevels, od.ArraySize);
 od_logged = 1;
 }
 }
 if (od.SampleDesc.Count != 1) { if (!cached) otex->lpVtbl->Release(otex); return 0; }
 /* the generic path indexed the fixed 1080p frame by the OUTPUT
 * size — a larger output overread the 8.3MB heap buffer by megabytes. */
 if (od.Width > (unsigned)(LWV * 2u) || od.Height > (unsigned)(LHV * 2u)) { if (!cached) otex->lpVtbl->Release(otex); return 0; } /* dims-derived */
 size_t n = (size_t)od.Width * od.Height;
 int bb = fmt_bytes(od.Format);
 if (bb == 0) { if (!cached) otex->lpVtbl->Release(otex); return 0; }

 const unsigned char* src = rgb8;
 unsigned char* obuf = NULL;
 if (!((od.Format == 28 || od.Format == 29) && od.Width == (unsigned)(LWV * 2u) && od.Height == (unsigned)(LHV * 2u))) {
 /* generic: RGBA8 -> f4 -> encode_px (naive-encoder semantics) */
 obuf = (unsigned char*)HeapAlloc(GetProcessHeap(), 0, n * (size_t)bb);
 if (!obuf) { if (!cached) otex->lpVtbl->Release(otex); return 0; }
 for (size_t i = 0; i < n; i++) {
 f4 v;
 v.r = rgb8[i * 4 + 0] / 255.0f;
 v.g = rgb8[i * 4 + 1] / 255.0f;
 v.b = rgb8[i * 4 + 2] / 255.0f;
 v.a = rgb8[i * 4 + 3] / 255.0f;
 encode_px(od.Format, &v, obuf + i * (size_t)bb);
 }
 src = obuf;
 }
 D3D11_TEXTURE2D_DESC td = od;
 td.Usage = D3D11_USAGE_DEFAULT;
 td.CPUAccessFlags = 0;
 /* A plain DEFAULT-usage texture takes
 * UpdateSubresource DIRECTLY — the cached-tmp + CopyResource detour put
 * an extra full-frame GPU copy inside the GAME's render stream every
 * frame. Exotic descs keep the tmp path. */
 if (od.Usage == D3D11_USAGE_DEFAULT && od.CPUAccessFlags == 0
 && od.MipLevels == 1 && od.ArraySize == 1) {
 ctx->lpVtbl->UpdateSubresource(ctx, out_res, 0, NULL, src,
 od.Width * (unsigned)bb, 0);
 HeapFree(GetProcessHeap(), 0, obuf);
 if (!cached) otex->lpVtbl->Release(otex);
 return 1;
 }
 /* cache the upload texture — CreateTexture2D (8.3MB alloc in the
 * driver) + Release PER EVAL was pure churn under Wine; keyed by
 * format+size (always fmt 28 @1920x1080 for RotTR) */
 static ID3D11Texture2D* tmp = NULL;
 static DXGI_FORMAT tkf = (DXGI_FORMAT)0;
 static unsigned tkw = 0, tkh = 0;
 if (tmp && (tkf != od.Format || tkw != od.Width || tkh != od.Height)) {
 tmp->lpVtbl->Release(tmp); tmp = NULL;
 }
 if (!tmp) {
 if (FAILED(g_dev->lpVtbl->CreateTexture2D(g_dev, &td, NULL, &tmp)) || !tmp) {
 HeapFree(GetProcessHeap(), 0, obuf); if (!cached) otex->lpVtbl->Release(otex); return 0;
 }
 tkf = od.Format; tkw = od.Width; tkh = od.Height;
 }
 ctx->lpVtbl->UpdateSubresource(ctx, (ID3D11Resource*)tmp, 0, NULL, src,
 od.Width * (unsigned)bb, 0);
 ctx->lpVtbl->CopyResource(ctx, out_res, (ID3D11Resource*)tmp);
 HeapFree(GetProcessHeap(), 0, obuf);
 if (!cached) otex->lpVtbl->Release(otex);
 return 1;
}

/* wrapper: EVERY output write (fresh AND cached-redisplay) lands in the
 * SP_WRT span — the old g_t_write never timed the redisplay path. */
static int live_write_out8(ID3D11DeviceContext* ctx, ID3D11Resource* out_res,
 const unsigned char* rgb8) {
 SP_BEGIN(SP_WRT);
 int ok = live_write_out8_core(ctx, out_res, rgb8);
 SP_END(SP_WRT);
 return ok;
}

/* ---- worker: sender + receiver threads (depth-2 wire) ----
 * The old single worker did a lock-step round trip per slot: req N+1 never
 * entered the wire until resp N was consumed, so the daemon could never
 * overlap transport with compute. Split: the SENDER ships any FULL slot in
 * submission order; the RECEIVER blocks on the oldest SENT slot's response.
 * Up to 2 requests in flight; the daemon's threaded pipeline reads N+1
 * while it executes N. TCP preserves order both ways; responses arrive in
 * request order. */
typedef enum { LS_EMPTY = 0, LS_FULL = 1, LS_SENT = 2, LS_PROC = 3, LS_DONE = 4, LS_FILL = 5 } LState;
typedef struct {
 volatile LState state;
 unsigned seq; /* submission order */
 unsigned int raw[RAWTEX_BYTES / 4];
 unsigned short mvh[MVH_BYTES / 2];
 unsigned char* out8; /* heap buffer (3-slot pointer-swap cache) */
 float* mvf32; /* v3: raw f32 MV rows (heap) */
 float sx, sy; /* MV scales ride the wire tail */
 float j[2]; /* game jitter this frame */
} LSlot;
static LSlot g_slots[3]; /* slots3 indexes [2] */
static HANDLE g_work_sem = NULL, g_done_sem = NULL;
static HANDLE g_sender = NULL, g_receiver = NULL;
static CRITICAL_SECTION g_live_cs;
static volatile int g_live_fails = 0; /* consecutive transfer failures */
/* compacted RAW colortexels of the staged color (filled at staging, part3) */
static unsigned int* g_rawtex = NULL;
int g_fill_slot = -1; /* Slot reserved for THIS eval's
 * fills (game thread only); -1 = drop path */
int g_mv_slot_filled = 0; /* part3's fast path wrote slot->mvh directly */
void live_fill_rawtex(const unsigned char* base, unsigned rowpitch,
 unsigned w, unsigned h) {
 if (w != LWV || h != LHV) return;
 /* write straight into the reserved slot's wire buffer when one
 * is held — the g_rawtex -> slot memcpy (2.07MB/eval) disappears */
 unsigned int* dst = (g_fill_slot >= 0) ? g_slots[g_fill_slot].raw: g_rawtex;
 if (g_fill_slot >= 0) g_raw_slot_filled = 1; /* raw refreshed this eval */
 if (!dst) {
 /* postmortem: g_rawtex is STILL the drop-path buffer AND the
 * live_output entry guard — never remove its allocation */
 if (!g_rawtex) g_rawtex = (unsigned int*)HeapAlloc(GetProcessHeap(), 0, RAWTEX_BYTES);
 if (!g_rawtex) return;
 dst = g_rawtex;
 }
 for (unsigned int y = 0; y < LHV; y++)
 memcpy((unsigned char*)dst + (size_t)y * (RAWTEX_BYTES / LHV),
 base + (size_t)y * rowpitch, RAWTEX_BYTES / LHV);
}

/* reserve an EMPTY slot for direct filling (short CS; state -> LS_FILL) */
int live_try_reserve(void) {
 int idx = -1;
 EnterCriticalSection(&g_live_cs);
 for (int k = 0; k < g_nslots; k++)
 if (g_slots[k].state == LS_EMPTY) { g_slots[k].state = LS_FILL; idx = k; break; }
 LeaveCriticalSection(&g_live_cs);
 return idx;
}

/* (v12c): the wire CS, semaphores and the g_rawtex guard buffer
 * must exist BEFORE the first reserve. v12b called live_try_reserve from
 * staging while InitializeCriticalSection still sat inside live_output:
 * (1) EnterCriticalSection on a zeroed CS reads as already-held under
 * Wine (legacy LockCount scheme: 0 = locked) -> render thread wedged
 * with 100% CPU = BOTH v12 black screens, no daemon connection, no
 * worker-start log; and
 * (2) with a slot reserved, live_fill_rawtex wrote the slot's inline
 * buffer so g_rawtex was never allocated and live_output's entry
 * guard `if (!g_rawtex) return 0` fired forever - live silently
 * never engaged.
 * Idempotent; part3 gates every reserve on it. */
static int g_wire_cs_init = 0, g_wire_init = 0;
int live_wire_ensure(void) {
 if (g_wire_init) return 1;
 /* slots3 locks HERE (first ensure — after read_ini at eval 1):
 * semaphore max counts and slot-buffer allocation must never change
 * mid-session, so later ini edits to slots3 are ignored (logged) */
 if (!g_slots_locked) {
 g_slots_locked = 1;
 g_nslots = g_opt_slots3 ? 3: 2;
 g_wire_v3 = (g_opt_wire == 3);
 LOG("wire: %d slots (slots3=%d locked), wire=%d", g_nslots, g_opt_slots3, g_wire_v3 ? 3: 2);
 }
 if (g_wire_v3) {
 if (!g_mvraw_f32) g_mvraw_f32 = (float*)HeapAlloc(GetProcessHeap(), 0, MVF32_BYTES);
 if (!g_mvf32_acc) g_mvf32_acc = (float*)HeapAlloc(GetProcessHeap(), HEAP_ZERO_MEMORY, MVF32_BYTES);
 for (int k = 0; k < g_nslots; k++)
 if (!g_slots[k].mvf32)
 g_slots[k].mvf32 = (float*)HeapAlloc(GetProcessHeap(), 0, MVF32_BYTES);
 if (!g_mvraw_f32 || !g_mvf32_acc) return 0;
 for (int k = 0; k < g_nslots; k++)
 if (!g_slots[k].mvf32) return 0;
 }
 /* partial-failure retries must not re-init the CS or
 * orphan live semaphores — init each piece exactly once, retry only
 * the pieces that are still missing */
 if (!g_wire_cs_init) { InitializeCriticalSection(&g_live_cs); g_wire_cs_init = 1; }
 if (!g_work_sem) g_work_sem = CreateSemaphoreW(NULL, 0, g_nslots, NULL);
 if (!g_done_sem) g_done_sem = CreateSemaphoreW(NULL, 0, g_nslots, NULL);
 if (!g_rawtex) g_rawtex = (unsigned int*)HeapAlloc(GetProcessHeap(), 0, RAWTEX_BYTES);
 for (int k = 0; k < g_nslots; k++)
 if (!g_slots[k].out8)
 g_slots[k].out8 = (unsigned char*)HeapAlloc(GetProcessHeap(), 0, OUT8_BYTES);
 if (!g_work_sem || !g_done_sem || !g_rawtex) return 0;
 for (int k = 0; k < g_nslots; k++)
 if (!g_slots[k].out8) return 0;
 g_wire_init = 1;
 return 1;
}

static volatile unsigned g_live_seq = 0;

static void live_fail_strike(const char* who) {
 /* plain closesocket() does not abort the receiver's blocked
 * recv under Wine — shutdown first, or the receiver never wakes and
 * done_sem starves while the sender happily serves the new connection.
 * swap under the CS with a generation bump (fd numbers recycle). */
 EnterCriticalSection(&g_live_cs);
 if (g_live_sock >= 0) { live_net_abort(g_live_sock); g_live_sock = -1; }
 g_live_gen++;
 LeaveCriticalSection(&g_live_cs);
 g_live_fails++;
 if (g_live_fails >= 2) {
 LOG("live: %s failed %dx — live disabled until next ini arm", who, g_live_fails);
 g_live = 0;
 } else if (g_live) {
 LOG("live: %s failed once — reconnecting", who);
 }
}

static int live_send_slot(int s, LSlot* sl) {
 uint64_t L;
 L = RAWTEX_BYTES;
 if (live_net_send_all(s, (char*)&L, 8) != 0) return -1;
 if (live_net_send_all(s, (const char*)sl->raw, RAWTEX_BYTES) != 0) return -1;
 if (g_wire_v3) {
 /* raw f32 rows + jx,jy,sx,sy tail — the daemon NEON-converts
 * (same first quantize our SSE2 path did => bit-identical chain).
 * the 16B tail goes as ONE send (was 4 tiny NODELAY segments). */
 struct { float jx, jy, sx, sy; } tail = { sl->j[0], sl->j[1], sl->sx, sl->sy };
 L = MVF32_BYTES + 24;
 if (live_net_send_all(s, (char*)&L, 8) != 0) return -1;
 if (live_net_send_all(s, (const char*)sl->mvf32, MVF32_BYTES) != 0) return -1;
 if (live_net_send_all(s, (const char*)&tail, 16) != 0) return -1;
 return 0;
 }
 L = MVH_BYTES + 16; /* jitter tail */
 if (live_net_send_all(s, (char*)&L, 8) != 0) return -1;
 if (live_net_send_all(s, (const char*)sl->mvh, MVH_BYTES) != 0) return -1;
 if (live_net_send_all(s, (const char*)sl->j, 8) != 0) return -1;
 return 0;
}

static DWORD WINAPI live_sender_proc(LPVOID arg) {
 (void)arg;
 for (;;) {
 if (WaitForSingleObject(g_work_sem, INFINITE) != WAIT_OBJECT_0) return 0;
 int idx = -1; unsigned best = 0xFFFFFFFFu;
 EnterCriticalSection(&g_live_cs);
 for (int k = 0; k < g_nslots; k++)
 if (g_slots[k].state == LS_FULL && g_slots[k].seq < best) { best = g_slots[k].seq; idx = k; }
 if (idx >= 0) g_slots[idx].state = LS_SENT;
 LeaveCriticalSection(&g_live_cs);
 if (idx < 0) continue; /* spurious wake (slot already collected) */
 int s = live_ensure_sock();
 if (s < 0 || live_send_slot(s, &g_slots[idx]) != 0) {
 EnterCriticalSection(&g_live_cs);
 g_slots[idx].state = LS_EMPTY;
 LeaveCriticalSection(&g_live_cs);
 live_fail_strike("send");
 }
 }
}

static DWORD WINAPI live_recv_proc(LPVOID arg) {
 (void)arg;
 for (;;) {
 int idx = -1; unsigned best = 0xFFFFFFFFu; int s = -1; long sgen = -1;
 for (;;) {
 EnterCriticalSection(&g_live_cs);
 for (int k = 0; k < g_nslots; k++)
 if (g_slots[k].state == LS_SENT && g_slots[k].seq < best) { best = g_slots[k].seq; idx = k; }
 s = g_live_sock;
 sgen = g_live_gen;
 LeaveCriticalSection(&g_live_cs);
 if (idx >= 0 && s >= 0) break;
 /* idle backoff when live is down (was a ~500Hz poll forever) */
 if (!g_live) Sleep(50); else Sleep(2);
 }
 unsigned bseq = best;
 uint64_t L = 0;
 int ok = live_net_recv_all(s, (char*)&L, 8) == 0 && L == OUT8_BYTES
 && live_net_recv_all(s, (char*)g_slots[idx].out8, OUT8_BYTES) == 0;
 if (!ok) {
 /* a failed/partial/short response misaligns the stream
 * forever on this socket; a fresh connection is the only resync
 * boundary. abort ONLY if this is still the same connection
 * GENERATION — the fd number may have been recycled into the
 * sender's fresh connection (or a game socket). */
 EnterCriticalSection(&g_live_cs);
 if (g_live_sock == s && g_live_gen == sgen) {
 live_net_abort(s);
 g_live_sock = -1;
 g_live_gen++;
 }
 LeaveCriticalSection(&g_live_cs);
 }
 EnterCriticalSection(&g_live_cs);
 if (g_slots[idx].state == LS_SENT && g_slots[idx].seq == bseq) {
 /* slot unchanged during the (blocking) recv — safe to act on it.
 * If a strike recycled it mid-recv, leave the state alone and
 * DISCARD this response (writing it would clobber the recycled
 * slot or tear the displayed cache buffer). */
 if (ok) {
 g_slots[idx].state = LS_DONE;
 g_live_fails = 0;
 } else {
 g_slots[idx].state = LS_EMPTY;
 }
 } else if (ok) {
 LOG("live: recv raced a recycle (seq %u) — response discarded", bseq);
 }
 LeaveCriticalSection(&g_live_cs);
 /* ONLY the sender owns strikes — a dead socket shows up as
 * one redisplay frame, then the sender's next send fails and does
 * the close+strike. The old double-strike disabled live for the
 * whole session on a single daemon blip. */
 if (!ok) LOG("live: recv failed (strike deferred to sender)");
 else ReleaseSemaphore(g_done_sem, 1, NULL);
 }
}
/* anti-flicker: hold the LAST completed NPU frame when the worker is
 * behind (alternating naive-bilinear/NPU frames reads as flicker); and
 * accumulate motion vectors across DROPPED frames so the daemon's history
 * warp covers the real elapsed motion, not one frame of it. */
static unsigned char* g_live_cache = NULL;
int g_live_cache_valid = 0; /* non-static — part3 gates the color
 * decode skip on having a frame to display */
static unsigned short* g_mvraw = NULL; /* staged halves */
static int g_mvraw_valid = 0;
unsigned long g_t_wait = 0, g_t_write = 0; /* stage timers (ms) */

/* SSE2 4-wide f32->f16 (RNE) — same semantics as the scalar f2h()
 * for normal lanes; blocks containing zero/subnormal/inf/overflow lanes
 * fall back to scalar (rare for motion vectors). 518k texels x2 channels
 * went through ~20-instruction soft-float converts per eval under Box64. */
#include <emmintrin.h>
static __m128i f32x4_to_f16x4_sse2(__m128 v, __m128i* slow) {
 __m128i f = _mm_castps_si128(v);
 __m128i e8 = _mm_srli_epi32(_mm_and_si128(f, _mm_set1_epi32(0x7F800000)), 23);
 *slow = _mm_or_si128(_mm_cmpeq_epi32(e8, _mm_setzero_si128()),
 _mm_or_si128(_mm_cmpeq_epi32(e8, _mm_set1_epi32(255)),
 _mm_or_si128(_mm_cmplt_epi32(e8, _mm_set1_epi32(113)),
 _mm_cmpgt_epi32(e8, _mm_set1_epi32(142)))));
 __m128i man = _mm_and_si128(f, _mm_set1_epi32(0x007FFFFF));
 __m128i e15 = _mm_sub_epi32(e8, _mm_set1_epi32(112));
 __m128i half = _mm_or_si128(_mm_slli_epi32(e15, 10), _mm_srli_epi32(man, 13));
 __m128i rem = _mm_and_si128(man, _mm_set1_epi32(0x1FFF));
 __m128i gt = _mm_cmpgt_epi32(rem, _mm_set1_epi32(0x1000));
 __m128i eq = _mm_cmpeq_epi32(rem, _mm_set1_epi32(0x1000));
 __m128i odd = _mm_and_si128(half, _mm_set1_epi32(1));
 __m128i inc = _mm_and_si128(_mm_or_si128(gt, _mm_and_si128(eq, odd)), _mm_set1_epi32(1));
 half = _mm_add_epi32(half, inc);
 __m128i sign = _mm_and_si128(_mm_srli_epi32(f, 16), _mm_set1_epi32(0x8000));
 return _mm_or_si128(half, sign);
}

/* zero-fast variant — MV data is zero-heavy (static pixels) and the
 * base helper's slow mask (e8==0) degenerated whole blocks to scalar redo,
 * making the "fast" path the exception. Here +-0 converts IN the vector
 * path (0*scale = +-0 exactly -> half +-0). inf/nan AND f16-subnormals
 * ALSO convert in the vector path now:
 * - inf/nan -> canonical 0x7C00/0x7E00|sign = exactly f2h's special form
 * (cut/loading frames are inf-heavy; the scalar fallback made mvc
 * explode to 48-65ms at every scene transition).
 * - subnormals (0<e8<113): TRUE-RNE via cvtps(v * 2^24) — the menu's
 * slow-drift MVs sit just under 2^-14 and the old scalar redo made the
 * MENU run at 7fps. Divergence from legacy f2h: +1 subnormal ulp on
 * ~0.006% of lanes (f2h truncates tie detection to 13 bits and rounds
 * near-ties down; vector is the correct RNE) — |delta| <= 2^-24.
 * Host-proven over ~44M lanes incl. inf-heavy and subnormal sweeps.
 * Only f16-overflow (e8>142, |v| > 65520) stays slow — garbage beyond the
 * f16 range, vanishingly rare. */
static __m128i f32x4_to_f16x4_z(__m128 v, __m128i* slow) {
 __m128i f = _mm_castps_si128(v);
 __m128i e8 = _mm_srli_epi32(_mm_and_si128(f, _mm_set1_epi32(0x7F800000)), 23);
 __m128i man = _mm_and_si128(f, _mm_set1_epi32(0x007FFFFF));
 __m128i isspecial = _mm_cmpeq_epi32(e8, _mm_set1_epi32(255));
 /* e8==255 satisfies e8>142, so the plain compare kept inf/nan
 * lanes in the scalar redo — the inf/nan vector path was dead code
 * and cut frames still took the 48-65ms stalls. Mask specials out. */
 *slow = _mm_andnot_si128(isspecial, _mm_cmpgt_epi32(e8, _mm_set1_epi32(142)));
 __m128i e15 = _mm_sub_epi32(e8, _mm_set1_epi32(112));
 __m128i half = _mm_or_si128(_mm_slli_epi32(e15, 10), _mm_srli_epi32(man, 13));
 __m128i rem = _mm_and_si128(man, _mm_set1_epi32(0x1FFF));
 __m128i gt = _mm_cmpgt_epi32(rem, _mm_set1_epi32(0x1000));
 __m128i eq = _mm_cmpeq_epi32(rem, _mm_set1_epi32(0x1000));
 __m128i odd = _mm_and_si128(half, _mm_set1_epi32(1));
 __m128i inc = _mm_and_si128(_mm_or_si128(gt, _mm_and_si128(eq, odd)), _mm_set1_epi32(1));
 half = _mm_add_epi32(half, inc);
 __m128i isz = _mm_cmpeq_epi32(e8, _mm_setzero_si128());
 half = _mm_or_si128(_mm_and_si128(isz, _mm_setzero_si128()), _mm_andnot_si128(isz, half));
 /* subnormal lanes: k = RNE(v * 2^24) — the exact f16-subnormal integer
 * (power-of-2 scale is exact; cvtps is nearest-even; the 0x400 boundary
 * rounds up into min-normal naturally, matching f2h's half++ path) */
 const __m128 two24 = _mm_set1_ps(16777216.0f);
 __m128i issub = _mm_and_si128(_mm_cmpgt_epi32(e8, _mm_setzero_si128()),
 _mm_cmplt_epi32(e8, _mm_set1_epi32(113)));
 __m128i k = _mm_cvtps_epi32(_mm_mul_ps(v, two24));
 __m128i km = _mm_srai_epi32(k, 31); /* all-ones if negative */
 k = _mm_sub_epi32(_mm_xor_si128(k, km), km); /* |k| */
 half = _mm_or_si128(_mm_and_si128(issub, k), _mm_andnot_si128(issub, half));
 /* inf (man==0) -> 0x7C00, nan (man!=0) -> 0x7E00, both |sign */
 __m128i spec = _mm_or_si128(
 _mm_and_si128(_mm_cmpeq_epi32(man, _mm_setzero_si128()), _mm_set1_epi32(0x7C00)),
 _mm_and_si128(_mm_cmpgt_epi32(man, _mm_setzero_si128()), _mm_set1_epi32(0x7E00)));
 half = _mm_or_si128(_mm_and_si128(isspecial, spec), _mm_andnot_si128(isspecial, half));
 __m128i sign = _mm_and_si128(_mm_srli_epi32(f, 16), _mm_set1_epi32(0x8000));
 return _mm_or_si128(half, sign);
}

/* one staged R32G32_FLOAT row -> fp16 halves (x*sx, y*sy), w texels */
static void mv_f32row_to_f16(const float* src, unsigned short* dst, unsigned w, float sx, float sy) {
 __m128 mul = _mm_setr_ps(sx, sy, sx, sy);
 unsigned i = 0;
 for (; i + 4 <= w; i += 4) {
 __m128 a = _mm_mul_ps(_mm_loadu_ps(src + i * 2), mul);
 __m128 b = _mm_mul_ps(_mm_loadu_ps(src + i * 2 + 4), mul);
 __m128i s0, s1;
 __m128i h0 = f32x4_to_f16x4_z(a, &s0);
 __m128i h1 = f32x4_to_f16x4_z(b, &s1);
 if (_mm_movemask_epi8(_mm_or_si128(s0, s1))) {
 for (unsigned k = 0; k < 8; k++)
 dst[i * 2 + k] = f2h(src[i * 2 + k] * ((k & 1) ? sy: sx));
 } else {
 /* CRITICAL FIX: the helpers return
 * UNPACKED lanes — the old unpacklo_epi64 store wrote
 * [a0,0,a1,0,b0,0,b1,0]: every other half a structural zero and
 * half the values misplaced. Fast-path (all-normal) blocks =
 * MOVING regions got garbled MVs while zero-heavy
 * static blocks took the correct scalar redo — matching the
 * chronic motion-shimmer signature. Correct unsigned pack via
 * the 0x8000 bias trick (packs_epi32 saturates signed). */
 const __m128i b32 = _mm_set1_epi32(0x8000);
 __m128i p = _mm_packs_epi32(_mm_sub_epi32(h0, b32), _mm_sub_epi32(h1, b32));
 p = _mm_xor_si128(p, _mm_set1_epi16((short)0x8000));
 _mm_storeu_si128((__m128i*)(dst + i * 2), p);
 }
 }
 for (; i < w; i++) {
 dst[i * 2] = f2h(src[i * 2] * sx);
 dst[i * 2 + 1] = f2h(src[i * 2 + 1] * sy);
 }
}
static unsigned short* g_mv_acc = NULL;
static int g_mv_acc_valid = 0;

/* (ini simdmv=1): vectorized dst[i] = f2h(h2f(a[i]) + h2f(b[i])) over
 * 8-half blocks. Widen f16->f32 via bit math (normal lanes exact; e5==0
 * with mant!=0 (subnormal) or e5==31 (inf/nan) -> scalar block), add in
 * f32 (same rounding as the scalar path), convert back with the existing
 * RNE SSE2 helper (its slow lanes -> scalar). Per-lane bit-identical to
 * the scalar loops it replaces; kills 8-20ms of Box64 soft-float on every
 * drop/merge eval. */
static void mvh_merge_sse2(unsigned short* dst, const unsigned short* a, const unsigned short* b, size_t n) {
 size_t i = 0;
 for (; i + 8 <= n; i += 8) {
 __m128i ha = _mm_loadu_si128((const __m128i*)(a + i));
 __m128i hb = _mm_loadu_si128((const __m128i*)(b + i));
 __m128i wa = _mm_unpacklo_epi16(ha, _mm_setzero_si128());
 __m128i xa = _mm_unpackhi_epi16(ha, _mm_setzero_si128());
 __m128i wb = _mm_unpacklo_epi16(hb, _mm_setzero_si128());
 __m128i xb = _mm_unpackhi_epi16(hb, _mm_setzero_si128());
 int slow = 0;
 __m128 fa, fb, fc, fd;
 {
 const __m128i e5m = _mm_set1_epi32(0x7C00), m5m = _mm_set1_epi32(0x03FF);
 __m128i pairs[4] = { wa, xa, wb, xb };
 __m128* outs[4] = { &fa, &fb, &fc, &fd };
 for (int q = 0; q < 4; q++) {
 __m128i x = pairs[q];
 __m128i e5 = _mm_srli_epi32(_mm_and_si128(x, e5m), 10);
 __m128i man = _mm_and_si128(x, m5m);
 /* subnormal (e5==0 & man!=0) or inf/nan (e5==31) -> slow */
 __m128i bad = _mm_or_si128(
 _mm_and_si128(_mm_cmpeq_epi32(e5, _mm_setzero_si128()),
 _mm_cmpgt_epi32(man, _mm_setzero_si128())),
 _mm_cmpeq_epi32(e5, _mm_set1_epi32(31)));
 if (_mm_movemask_epi8(bad)) slow = 1;
 /* sign must land at f32 bit 31 (<<16, was
 * OR'd into the mantissa), and e5==0&man==0 (true zero)
 * must widen to +-0 (the (0+112)<<23 form made 2^-15) */
 __m128i sign = _mm_slli_epi32(_mm_and_si128(x, _mm_set1_epi32(0x8000)), 16);
 __m128i norm = _mm_or_si128(_mm_slli_epi32(man, 13),
 _mm_slli_epi32(_mm_add_epi32(e5, _mm_set1_epi32(112)), 23));
 __m128i isz = _mm_cmpeq_epi32(e5, _mm_setzero_si128());
 __m128i bits = _mm_or_si128(sign,
 _mm_or_si128(_mm_and_si128(isz, _mm_setzero_si128()),
 _mm_andnot_si128(isz, norm)));
 *outs[q] = _mm_castsi128_ps(bits);
 }
 }
 if (slow) {
 for (unsigned k = 0; k < 8; k++)
 dst[i + k] = f2h(h2f(a[i + k]) + h2f(b[i + k]));
 } else {
 __m128 s0 = _mm_add_ps(fa, fc); /* lanes 0-3: a + b */
 __m128 s1 = _mm_add_ps(fb, fd);
 __m128i sl0, sl1;
 __m128i h0 = f32x4_to_f16x4_z(s0, &sl0);
 __m128i h1 = f32x4_to_f16x4_z(s1, &sl1);
 if (_mm_movemask_epi8(_mm_or_si128(sl0, sl1))) {
 for (unsigned k = 0; k < 8; k++)
 dst[i + k] = f2h(h2f(a[i + k]) + h2f(b[i + k]));
 } else {
 /* same packing bug as the row converter —
 * pack the UNPACKED helper lanes via the bias trick */
 const __m128i b32 = _mm_set1_epi32(0x8000);
 __m128i p = _mm_packs_epi32(_mm_sub_epi32(h0, b32), _mm_sub_epi32(h1, b32));
 p = _mm_xor_si128(p, _mm_set1_epi16((short)0x8000));
 _mm_storeu_si128((__m128i*)(dst + i), p);
 }
 }
 }
 for (; i < n; i++)
 dst[i] = f2h(h2f(a[i]) + h2f(b[i]));
}
#define MVH_MERGE(dst, a, b, n2) do { if (g_opt_simdmv) mvh_merge_sse2((dst), (a), (b), (n2)); else for (size_t _i = 0; _i < (size_t)(n2); _i++) (dst)[_i] = f2h(h2f((a)[_i]) + h2f((b)[_i])); } while (0)

static int g_v13_last_ok = 0; /* live_output's return via the helpers */
static int v13_submit(float jx, float jy, float sx, float sy, const unsigned short* mv_src);
static int v13_collect(ID3D11DeviceContext* ctx, ID3D11Resource* out_res);
static int live_output(ID3D11DeviceContext* ctx, ID3D11Resource* out_res,
 float sx, float sy, float jx, float jy) {
 if (!g_rawtex) return 0;
 /* g_cbuf may be NULL in live mode (decode skipped once the cache
 * exists) — dims + rawtex are what the wire needs */
 /* dims gates derive from LWV/LHV (v20_360 missed these two —
 * hardcoded 960/540 refused 640x360 input before the wire engaged) */
 if (g_cw != LWV || g_ch != LHV) return 0;
 /* v12c: g_mbuf is only allocated by the SLOW MV decode; in pure live
 * mode the fast path fills halves straight into mvraw/slot and
 * decode never runs — requiring g_mbuf here kept live from EVER
 * engaging (silent naive fallback; the old START-touch capture arm was
 * what bootstrapped g_mbuf in deployments) */
 if (g_mw != LWV || g_mh != LHV || (!g_mvraw_valid && !g_mbuf)) return 0;
 /* a decode-filled g_mbuf is only fresh for its own eval —
 * shipping current color with minutes-old MVs is the ghosting class */
 if (!g_mvraw_valid && g_mbuf && g_eval_n - g_mbuf_eval > 1) {
 LOG("live: stale MV decode (eval %lld data at %lld) — naive this eval", g_mbuf_eval, g_eval_n);
 return 0;
 }
 if (!live_wire_ensure()) return 0;
 /* a one-sided CreateThread failure used to give up
 * forever (silent naive) — create only the missing thread so the next
 * eval retries; log exactly once when both are up */
 if (!g_sender || !g_receiver) {
 if (!g_sender) g_sender = CreateThread(NULL, 0, live_sender_proc, NULL, 0, NULL);
 if (!g_receiver) g_receiver = CreateThread(NULL, 0, live_recv_proc, NULL, 0, NULL);
 if (!g_sender || !g_receiver) return 0;
 LOG("live: sender+receiver threads started (depth-2 wire, v2 raw in, RGBA8 out)");
 }

 /* submit and collect are split so zerowait (ini zerowait=1,
 * requires slots3) can COLLECT FIRST, then submit — the game thread
 * never sleeps on the daemon; a held display frame can't starve the
 * wire because the 3rd slot absorbs it. Default order and semantics
 * (2ms wait, memcpy cache) are exactly v12c. */
 static unsigned short* cur_mv = NULL;
 if (!cur_mv) cur_mv = (unsigned short*)HeapAlloc(GetProcessHeap(), 0, MVH_BYTES);
 if (!g_mv_acc) g_mv_acc = (unsigned short*)HeapAlloc(GetProcessHeap(), HEAP_ZERO_MEMORY, MVH_BYTES);
 if (!cur_mv || !g_mv_acc) return 0;
 /* raw staged halves when the fast path hit — skip the f4 decode +
 * re-encode loops entirely (the accumulation merge still does the math) */
 const unsigned short* mv_src;
 if (g_wire_v3) {
 /* the f32 source is g_mvraw_f32 (part3's fast path staged the
 * rows directly); the decode fallback materializes it from g_mbuf.
 * Drop-merges run in the f32 domain (plain adds — H12:
 * slightly better than v2's fp16-domain acc). */
 /* was a ONE-SHOT static that staged PRE-SCALED rows — the
 * daemon multiplies by sx/sy again (x*sx^2) and every later fallback
 * re-shipped frame 1's data verbatim. Now: RAW values, refreshed per
 * fallback eval (the daemon's multiply is the only one). */
 if (!g_mvraw_valid && g_mbuf) {
 f4* m = (f4*)g_mbuf;
 for (size_t i = 0; i < (size_t)LWV * LHV; i++) {
 g_mvraw_f32[i * 2] = m[i].r;
 g_mvraw_f32[i * 2 + 1] = m[i].g;
 }
 }
 mv_src = NULL; /* v3 path below uses g_mvraw_f32 */
 } else if (g_mvraw_valid) {
 mv_src = g_mvraw;
 } else {
 f4* m = (f4*)g_mbuf;
 for (size_t i = 0; i < (size_t)LWV * LHV; i++) {
 cur_mv[i * 2] = f2h(m[i].r * sx);
 cur_mv[i * 2 + 1] = f2h(m[i].g * sy);
 }
 mv_src = cur_mv;
 }
 if (g_opt_zerowait && g_nslots >= 3) {
 v13_collect(ctx, out_res);
 v13_submit(jx, jy, sx, sy, mv_src);
 } else {
 v13_submit(jx, jy, sx, sy, mv_src);
 v13_collect(ctx, out_res);
 }
 return g_v13_last_ok;
}

/* helpers (file-scope, game-thread-only). Order-sensitive state
 * (g_fill_slot) matches the v12c flow. */

static int v13_submit(float jx, float jy, float sx, float sy, const unsigned short* mv_src) {
 int submitted = 0;
 int rk = g_fill_slot; /* pre-reserved, raw already inside */
 SP_BEGIN(SP_SUB);
 /* v12c: single balanced CS scope over BOTH submit paths */
 EnterCriticalSection(&g_live_cs);
 if (g_wire_v3) {
 /* v3 submit: f32 rows in, scales ride the slot */
 if (rk >= 0 && !g_raw_slot_filled) rk = -2; /* stale raw = drop, not submit */
 if (rk >= 0) {
 LSlot* rsl = &g_slots[rk];
 if (!g_mv_slot_filled)
 memcpy(rsl->mvf32, g_mvraw_f32, MVF32_BYTES);
 if (g_mvf32_acc_valid) {
 for (size_t i = 0; i < MVF32_BYTES / 4; i++)
 rsl->mvf32[i] += g_mvf32_acc[i];
 g_mvf32_acc_valid = 0;
 }
 rsl->j[0] = jx; rsl->j[1] = jy;
 rsl->sx = sx; rsl->sy = sy;
 rsl->seq = g_live_seq++;
 rsl->state = LS_FULL;
 submitted = 1;
 } else {
 for (int k = 0; k < g_nslots && !submitted; k++) {
 if (g_slots[k].state == LS_EMPTY) {
 memcpy(g_slots[k].raw, g_rawtex, RAWTEX_BYTES);
 if (g_mvf32_acc_valid) {
 for (size_t i = 0; i < MVF32_BYTES / 4; i++)
 g_slots[k].mvf32[i] = g_mvraw_f32[i] + g_mvf32_acc[i];
 g_mvf32_acc_valid = 0;
 } else {
 memcpy(g_slots[k].mvf32, g_mvraw_f32, MVF32_BYTES);
 }
 g_slots[k].j[0] = jx; g_slots[k].j[1] = jy;
 g_slots[k].sx = sx; g_slots[k].sy = sy;
 g_slots[k].seq = g_live_seq++;
 g_slots[k].state = LS_FULL;
 submitted = 1;
 }
 }
 }
 if (!submitted) {
 if (rk == -2) {
 /* release the stale-raw reservation back to EMPTY */
 if (g_fill_slot >= 0) { g_slots[g_fill_slot].state = LS_EMPTY; g_fill_slot = -1; }
 }
 /* cap the fold — a 10s outage summed ~400 frames of
 * motion into one field (garbage warp). Beyond 8 consecutive
 * drops, discard the accumulator: the next submit ships fresh. */
 g_drop_run++;
 if (g_drop_run > 8) g_mvf32_acc_valid = 0;
 if (g_mvf32_acc_valid) {
 for (size_t i = 0; i < MVF32_BYTES / 4; i++)
 g_mvf32_acc[i] += g_mvraw_f32[i];
 } else {
 for (size_t i = 0; i < MVF32_BYTES / 4; i++)
 g_mvf32_acc[i] = g_mvraw_f32[i];
 g_mvf32_acc_valid = 1;
 }
 SP_FLAG(SPFL_DROP);
 g_sp_drops++;
 } else {
 g_drop_run = 0;
 if (g_mv_slot_filled) SP_FLAG(SPFL_MVPRE);
 }
 LeaveCriticalSection(&g_live_cs);
 g_fill_slot = -1;
 SP_END(SP_SUB);
 if (submitted) {
 SP_BEGIN(SP_REL);
 ReleaseSemaphore(g_work_sem, 1, NULL);
 SP_END(SP_REL);
 }
 return submitted;
 }
 if (rk >= 0 && !g_raw_slot_filled) rk = -2; /* stale raw = drop, not submit */
 if (rk >= 0) {
 LSlot* rsl = &g_slots[rk];
 /* v12c: never accumulate onto stale slot MVs — fresh first, THEN
 * fold motion accumulated across dropped frames */
 if (!g_mv_slot_filled)
 memcpy(rsl->mvh, mv_src, MVH_BYTES);
 if (g_mv_acc_valid) {
 MVH_MERGE(rsl->mvh, rsl->mvh, g_mv_acc, (size_t)LWV * LHV * 2);
 g_mv_acc_valid = 0;
 }
 rsl->j[0] = jx; rsl->j[1] = jy;
 rsl->seq = g_live_seq++;
 rsl->state = LS_FULL;
 submitted = 1;
 } else {
 for (int k = 0; k < g_nslots && !submitted; k++) {
 if (g_slots[k].state == LS_EMPTY) {
 memcpy(g_slots[k].raw, g_rawtex, RAWTEX_BYTES);
 if (g_mv_acc_valid) {
 MVH_MERGE(g_slots[k].mvh, mv_src, g_mv_acc, (size_t)LWV * LHV * 2);
 g_mv_acc_valid = 0;
 } else {
 memcpy(g_slots[k].mvh, mv_src, MVH_BYTES);
 }
 g_slots[k].j[0] = jx; g_slots[k].j[1] = jy;
 g_slots[k].seq = g_live_seq++;
 g_slots[k].state = LS_FULL;
 submitted = 1;
 }
 }
 }
 if (!submitted) {
 if (rk == -2) {
 /* release the stale-raw reservation back to EMPTY */
 if (g_fill_slot >= 0) { g_slots[g_fill_slot].state = LS_EMPTY; g_fill_slot = -1; }
 }
 /* fold cap — see the v3 branch comment */
 g_drop_run++;
 if (g_drop_run > 8) g_mv_acc_valid = 0;
 if (g_mv_acc_valid) {
 MVH_MERGE(g_mv_acc, g_mv_acc, mv_src, (size_t)LWV * LHV * 2);
 } else {
 memcpy(g_mv_acc, mv_src, MVH_BYTES);
 g_mv_acc_valid = 1;
 }
 SP_FLAG(SPFL_DROP);
 g_sp_drops++;
 } else {
 g_drop_run = 0;
 if (g_mv_slot_filled) SP_FLAG(SPFL_MVPRE);
 }
 LeaveCriticalSection(&g_live_cs);
 g_fill_slot = -1;
 SP_END(SP_SUB);
 if (submitted) {
 SP_BEGIN(SP_REL);
 ReleaseSemaphore(g_work_sem, 1, NULL);
 SP_END(SP_REL);
 }
 return submitted;
}

static int v13_collect(ID3D11DeviceContext* ctx, ID3D11Resource* out_res) {
 int got = -1; unsigned gotseq = 0xFFFFFFFFu;
 SP_BEGIN(SP_WAIT);
 if (WaitForSingleObject(g_done_sem, (g_opt_zerowait && g_nslots >= 3) ? 0: 2) == WAIT_OBJECT_0) {
 EnterCriticalSection(&g_live_cs);
 for (int k = 0; k < g_nslots; k++)
 /* pick the DONE slot by lowest seq (was array index) —
 * with 2+ responses queued the index order can display the newer
 * frame first = one frame of time travel after every redisplay */
 if (g_slots[k].state == LS_DONE && g_slots[k].seq < gotseq) { gotseq = g_slots[k].seq; got = k; }
 LeaveCriticalSection(&g_live_cs);
 }
 SP_END(SP_WAIT);
 if (got < 0) {
 /* no fresh result this eval: REDISPLAY the last NPU frame instead of
 * falling back to naive bilinear (soft/sharp alternation = flicker) */
 if (g_live_cache_valid) {
 g_sp_cache++;
 SP_FLAG(SPFL_CACHE);
 g_v13_last_ok = live_write_out8(ctx, out_res, g_live_cache);
 return -1;
 }
 g_v13_last_ok = 0;
 return -1;
 }
 if (!g_live_cache) g_live_cache = (unsigned char*)HeapAlloc(GetProcessHeap(), 0, OUT8_BYTES);
 SP_BEGIN(SP_CPY);
 if (!g_live_cache) {
 /* OOM on the FIRST cache alloc — write straight from
 * the slot buffer (v12c behavior) instead of passing NULL */
 SP_END(SP_CPY);
 g_sp_fresh++;
 SP_FLAG(SPFL_FRESH);
 int ok0 = live_write_out8(ctx, out_res, g_slots[got].out8);
 EnterCriticalSection(&g_live_cs);
 g_slots[got].state = LS_EMPTY;
 LeaveCriticalSection(&g_live_cs);
 g_v13_last_ok = ok0;
 return got;
 }
 if (g_live_cache) {
 if (g_nslots >= 3) {
 /* pointer swap: the slot's out8 buffer BECOMES the cache;
 * the stale cache buffer returns to the slot for the receiver to
 * refill — the 8.3MB per-fresh-frame memcpy disappears. The
 * write below then reads the (new) cache pointer, which stays
 * game-thread-owned until the next swap. */
 unsigned char* t = g_live_cache;
 g_live_cache = g_slots[got].out8;
 g_slots[got].out8 = t;
 } else {
 memcpy(g_live_cache, g_slots[got].out8, OUT8_BYTES);
 }
 g_live_cache_valid = 1;
 }
 SP_END(SP_CPY);
 g_sp_fresh++;
 SP_FLAG(SPFL_FRESH);
 int ok = live_write_out8(ctx, out_res, g_live_cache);
 EnterCriticalSection(&g_live_cs);
 g_slots[got].state = LS_EMPTY;
 LeaveCriticalSection(&g_live_cs);
 g_v13_last_ok = ok;
 return got;
}

/* ------------------------------------------------------------------ */
/* D3D11 implementation (the API RotTR uses) */
/* ------------------------------------------------------------------ */

static FsParam* g_params_persistent = NULL; /* GetParameters (SDK-owned) */
static FsParam* g_caps_persistent = NULL; /* GetCapabilityParameters (SDK-owned) */

/* deferred staging is the 3-entry g_dfr ring in part2 — each entry
 * owns its staging textures plus the copied frame's descs and params. */

static void read_dims(FsParam* p, const char* k, unsigned int* dst) {
 unsigned long long tmp = 0;
 if (fsp_get(p, k, 4, &tmp, NULL, NULL, NULL) == (int)NVSDK_NGX_Result_Success) *dst = (unsigned int)tmp;
}

static void maybe_remember_device(ID3D11DeviceContext* ctx) {
 if (g_dev || !ctx) return;
 ctx->lpVtbl->GetDevice(ctx, &g_dev);
 LOG("device recovered from ctx: %p", (void*)g_dev);
}

static NVSDK_NGX_Result eval_d3d11(ID3D11DeviceContext* ctx, const NVSDK_NGX_Handle* h, NVSDK_NGX_Parameter* pIn) {
 FsParam* p = (FsParam*)pIn;
 g_eval_n++;
 /* FUSE GetFileAttributes x2 per eval is real ms under Wine —
 * poll every 8th (adds ~0.4s latency to START/STOP detection) */
 sp_eval_begin(); /* TOT covers capture_poll too */
 if ((g_eval_n & 7) == 0 || g_armed) capture_poll();
 ULONGLONG eval_t0 = GetTickCount64(); /* legacy fallback timer */
 (void)h;

 SP_BEGIN(SP_PRE);
 float jx = 0.0f, jy = 0.0f, sx = 1.0f, sy = 1.0f, preexp = -1.0f;
 unsigned long long tmp = 0;
 fsp_get(p, K_JitterX, 1, NULL, &jx, NULL, NULL);
 fsp_get(p, K_JitterY, 1, NULL, &jy, NULL, NULL);
 fsp_get(p, K_MVScaleX, 1, NULL, &sx, NULL, NULL);
 fsp_get(p, K_MVScaleY, 1, NULL, &sy, NULL, NULL);
 fsp_get(p, K_PreExposure, 1, NULL, &preexp, NULL, NULL);

 ID3D11Resource* col = NULL, * mv = NULL, * out = NULL;
 if (fsp_get(p, K_Color, 5, NULL, NULL, NULL, (void**)&col) != (int)NVSDK_NGX_Result_Success)
 fsp_get(p, K_ColorL, 5, NULL, NULL, NULL, (void**)&col);
 if (fsp_get(p, K_MV, 5, NULL, NULL, NULL, (void**)&mv) != (int)NVSDK_NGX_Result_Success)
 LOG("eval %lld: no MotionVectors param", g_eval_n);
 if (fsp_get(p, K_Output, 5, NULL, NULL, NULL, (void**)&out) != (int)NVSDK_NGX_Result_Success)
 fsp_get(p, K_OutputL, 5, NULL, NULL, NULL, (void**)&out);

 if (!col || !out) {
 LOG("eval %lld: missing inputs col=%p mv=%p out=%p", g_eval_n, (void*)col, (void*)mv, (void*)out);
 _mm_setcsr(g_eval_csr); /* restore (sp_eval_begin already pinned) */
 return NVSDK_NGX_Result_Success;
 }
 if (g_eval_n <= 3 || (g_eval_n % 120) == 0)
 LOG("eval %lld jit=(%.5f,%.5f) mvs=(%.3f,%.3f) preexp=%.6g armed=%d cap=%d", g_eval_n,
 (double)jx, (double)jy, (double)sx, (double)sy, (double)preexp, g_armed, g_cap_n);

 /* exposure-texture probe: research A/B discriminator (exposure path vs GameWorks path).
 * No AddRef/Release — we borrow the game's pointer exactly like col/mv/out. */
 {
 ID3D11Resource* ex = NULL;
 if (fsp_get(p, K_Exposure, 5, NULL, NULL, NULL, (void**)&ex) != (int)NVSDK_NGX_Result_Success || !ex)
 fsp_get(p, "Exposure", 5, NULL, NULL, NULL, (void**)&ex);
 if (ex) {
 if (g_armed)
 dump_small_resource(ctx, ex, "exposure", (unsigned)g_cap_n);
 else if (g_eval_n <= 3)
 dump_small_resource(ctx, ex, "exposure_pre", 0xFFFF0000u + (unsigned)g_eval_n);
 } else if (g_eval_n <= 3) {
 LOG("eval %lld: game did NOT set ExposureTexture param", g_eval_n);
 }
 }
 SP_END(SP_PRE);

 D3D11_TEXTURE2D_DESC cd, md;
 memset(&cd, 0, sizeof(cd)); memset(&md, 0, sizeof(md));
 D3D11_MAPPED_SUBRESOURCE cm, mm;
 int have_c = 0, have_m = 0;
 g_mvraw_valid = 0; /* recomputed per eval */
 /* stash THIS eval's params before deferred fill clobbers them */
 float cjx = jx, cjy = jy, csx = sx, csy = sy;

 /* pop the OLDEST queued ring entry and map its copies. mapn2=1
 * requires TWO entries queued (maps N-2's copy — two frames of GPU
 * grace, the Map doesn't drain the queue); mapn2=0 keeps the v14 lag-1
 * behavior. A shallow queue (session bootstrap, or right after a
 * mid-session mapn2 flip) falls through to the sync path — the first
 * live eval needs the decode for the naive fallback anyway. */
 /* a mapn2 1->0 ini flip never drained the ring (depth is a
 * threshold, not a drain) — lag-2 frames kept shipping under a lag-1
 * config. Clamp once per eval; dropping the extra entry is one skipped
 map, benign. */
 if (dfr_depth() > dfr_need()) g_dfr_p = g_dfr_u - dfr_need();
 int df_mode = (g_live && dfr_depth() >= dfr_need());
 DFREntry* de = NULL;
 ID3D11Texture2D *st_c = NULL, *st_m = NULL;
 if (df_mode) {
 de = &g_dfr[g_dfr_p % DFR_N];
 g_dfr_p++;
 SP_BEGIN(SP_MAPC);
 if (de->okc && de->tc) {
 if (SUCCEEDED(ctx->lpVtbl->Map(ctx, (ID3D11Resource*)de->tc, 0, D3D11_MAP_READ, 0, &cm))) {
 cd = de->dc; have_c = 1; st_c = de->tc;
 }
 }
 SP_END(SP_MAPC);
 SP_BEGIN(SP_MAPM);
 if (de->okm && de->tm) {
 if (SUCCEEDED(ctx->lpVtbl->Map(ctx, (ID3D11Resource*)de->tm, 0, D3D11_MAP_READ, 0, &mm))) {
 md = de->dm; have_m = 1; st_m = de->tm;
 }
 }
 SP_END(SP_MAPM);
 jx = de->jx; jy = de->jy; sx = de->sx; sy = de->sy;
 }

 int c_staged = df_mode ? have_c: stage_to_cpu(ctx, col, "color", &cd, &cm);
 if (c_staged) {
 /* Reserve a wire slot BEFORE filling — the
 * color compaction and MV convert then write straight into it and
 * the 4MB of intermediate copies (g_rawtex/g_mvraw -> slot) die */
 SP_BEGIN(SP_RSV);
 g_fill_slot = (g_live && !g_armed && live_wire_ensure()) ? live_try_reserve(): -1;
 SP_END(SP_RSV);
 g_mv_slot_filled = 0;
 /* GROUND TRUTH: raw texel bytes, decoded on host. Dumped for every
 * armed frame (2 MB each) + dual-interpretation sample of the first
 * words every 300 evals (f11 layout vs raw-float read). */
 if (g_armed && cd.Width && cd.Height && cm.pData) {
 unsigned int rw = cd.Width * (unsigned)fmt_bytes(cd.Format); /* was hardcoded 4B/px */
 unsigned char* raw = (unsigned char*)HeapAlloc(GetProcessHeap(), 0, (size_t)rw * cd.Height);
 if (raw) {
 for (unsigned int y = 0; y < cd.Height; y++)
 memcpy(raw + (size_t)y * rw, (unsigned char*)cm.pData + (size_t)y * cm.RowPitch, rw);
 dump_file("colortexel.raw", raw, (size_t)rw * cd.Height, (unsigned)g_cap_n);
 HeapFree(GetProcessHeap(), 0, raw);
 }
 }
 if ((g_eval_n % 300) == 1 && cm.pData) {
 const unsigned int* wd = (const unsigned int*)cm.pData;
 for (int k = 0; k < 4; k++) {
 unsigned int v = wd[k];
 float rf; memcpy(&rf, &v, 4);
 /* f11 interpretation of the same word */
 unsigned int rr = v & 0x7FFu, gg = (v >> 11) & 0x7FFu, bb = (v >> 22) & 0x3FFu;
 LOG("texel[%d] eval=%lld fmt=%u raw=%08x floatread=%.6g f11=[%.6g,%.6g,%.6g]",
 k, g_eval_n, cd.Format, v, (double)rf,
 (double)f11((int)(rr & 0x3Fu), (int)((rr >> 6) & 0x1Fu), 6),
 (double)f11((int)(gg & 0x3Fu), (int)((gg >> 6) & 0x1Fu), 6),
 (double)f11((int)(bb & 0x1Fu), (int)((bb >> 5) & 0x1Fu), 5));
 }
 }
 /* perf: in live mode with a cached NPU frame to display, skip
 * the f11 decode entirely (10+ ms/frame of Box64-emulated CPU) —
 * live_output only needs the dims + the raw texels staged above.
 * Decode resumes when capture arms or before the first result. */
 /* v12c: skip ONLY when dims match the decoded buffer — overwriting
 * g_cw/g_ch with larger staged dims left naive_output reading past
 * g_cbuf's allocation (heap OOB on any mid-session res change) */
 SP_BEGIN(SP_DEC);
 if (g_live && !g_armed && g_live_cache_valid
 && cd.Width == g_cw && cd.Height == g_ch) {
 g_cbuf_n = 0;
 have_c = 1;
 SP_FLAG(SPFL_SKIPC);
 } else {
 have_c = decode_all(&cd, &cm, &g_cbuf, &g_cbuf_n, &g_cw, &g_ch);
 }
 SP_END(SP_DEC);
 /* wire v2: compact the RAW R11G11B10F texels (row pitch!) for
 * the live worker — the daemon decodes, saving 4x upload + host f11 */
 SP_BEGIN(SP_CMP);
 /* the raw ship assumes 4B/px R11G11B10F texels — gate on
 * the format too (dims alone let an RGBA16F switch ship garbage). */
 if (have_c && g_live && cd.Format == 26) /* 26 = R11G11B10_FLOAT per fmt_bytes table (29 is RGBA8-SRGB!) */
 live_fill_rawtex((const unsigned char*)cm.pData, cm.RowPitch,
 cd.Width, cd.Height);
 else if (have_c && g_live) {
 static int fmt_warned = 0;
 if (!fmt_warned) { LOG("live: color fmt %u != R11G11B10F — raw ship off this eval", cd.Format); fmt_warned = 1; }
 }
 SP_END(SP_CMP);
 if (!df_mode) stage_release(ctx);
 /* Inf-flood instrumentation: % non-finite + max finite, every 30 evals.
 * Drift curve discriminates accumulation-divergence (finite values grow)
 * from spreading-overwrite (finite count shrinks at constant max). */
 if (have_c && g_cbuf && g_cbuf_n != 0 && ((g_eval_n % 30) == 1)) {
 size_t n = (size_t)g_cw * g_ch;
 const f4* c = (const f4*)g_cbuf;
 unsigned long long inf = 0, fin = 0;
 double mx = 0.0;
 for (size_t i = 0; i < n; i++) {
 float r = c[i].r, g = c[i].g, b = c[i].b;
 int bad = isnan(r) || isinf(r) || isnan(g) || isinf(g) || isnan(b) || isinf(b);
 if (bad) inf++;
 else {
 fin++;
 double m = (double)(r > g ? (r > b ? r: b): (g > b ? g: b));
 if (m > mx) mx = m;
 }
 }
 LOG("stat eval=%lld fmt=%u inf=%llu (%.4f%%) finite=%llu maxfin=%.6g",
 g_eval_n, cd.Format, inf, 100.0 * (double)inf / (double)n, fin, mx);
 }
 }
 int m_staged = df_mode ? have_m: (mv && stage_to_cpu(ctx, mv, "mv", &md, &mm));
 if (m_staged) {
 /* The wire format for MVs IS fp16 halves — copy/convert the
 * staged rows directly instead of f32-decoding 518k texels into f4
 * and re-encoding (two Box64-emulated scalar loops per eval).
 * fmt 34 = R16G16_FLOAT (pure memcpy); fmt 16 = R32G32_FLOAT (SSE2
 * 4-wide convert — RotTR's actual format). */
 SP_BEGIN(SP_MVC);
 if (g_live && !g_armed && md.Width == (unsigned)LWV && md.Height == (unsigned)LHV) {
 /* with a slot reserved and nothing accumulated, write
 * the halves straight into the slot's wire buffer */
 int mv_wrote = 0; /* flag on ACTUAL write, not
 * pointer identity (unknown MV format at
 * 960x540 must not publish stale slot MVs) */
 if (g_wire_v3) {
 /* v3: ship the raw f32 rows — the emulated SSE2
 * convert LEAVES the game thread (daemon NEON does the same
 * first quantize). fmt16 = plain row memcpy; fmt34 = widen. */
 float* fdst = (g_fill_slot >= 0 && !g_mvf32_acc_valid)
 ? g_slots[g_fill_slot].mvf32: g_mvraw_f32;
 if (fdst) {
 if (md.Format == 16) {
 for (unsigned int y = 0; y < (unsigned)LHV; y++)
 memcpy(fdst + (size_t)y * LWV * 2,
 (unsigned char*)mm.pData + (size_t)y * mm.RowPitch, LWV * 8u);
 mv_wrote = 1;
 } else if (md.Format == 34 && g_wire_v3) { /* v2's tail carries no sx/sy — fmt34 shipped unscaled */
 for (unsigned int y = 0; y < (unsigned)LHV; y++) {
 const unsigned short* row = (const unsigned short*)
 ((unsigned char*)mm.pData + (size_t)y * mm.RowPitch);
 for (unsigned int x = 0; x < LWV * 2u; x++)
 fdst[(size_t)y * LWV * 2 + x] = h2f(row[x]);
 }
 mv_wrote = 1;
 }
 if (mv_wrote) {
 g_mvraw_valid = 1; /* means "f32 rows staged" in v3 mode */
 have_m = 1; g_mw = (int)LWV; g_mh = (int)LHV; g_mbuf_n = 0;
 if (g_fill_slot >= 0 && fdst == g_slots[g_fill_slot].mvf32)
 g_mv_slot_filled = 1;
 }
 }
 } else {
 unsigned short* mvdst = (g_fill_slot >= 0 && !g_mv_acc_valid)
 ? g_slots[g_fill_slot].mvh: NULL;
 if (md.Format == 34) {
 if (!mvdst && !g_mvraw) g_mvraw = (unsigned short*)HeapAlloc(GetProcessHeap(), 0, (size_t)LWV * LHV * 4u);
 if (!mvdst) mvdst = g_mvraw;
 if (mvdst) {
 for (unsigned int y = 0; y < (unsigned)LHV; y++)
 memcpy(mvdst + (size_t)y * LWV * 2,
 (unsigned char*)mm.pData + (size_t)y * mm.RowPitch, LWV * 4u);
 g_mvraw_valid = 1;
 have_m = 1; g_mw = (int)LWV; g_mh = (int)LHV; g_mbuf_n = 0;
 mv_wrote = 1;
 }
 } else if (md.Format == 16) {
 if (!mvdst && !g_mvraw) g_mvraw = (unsigned short*)HeapAlloc(GetProcessHeap(), 0, (size_t)LWV * LHV * 4u);
 if (!mvdst) mvdst = g_mvraw;
 if (mvdst) {
 for (unsigned int y = 0; y < (unsigned)LHV; y++) {
 const float* row = (const float*)((unsigned char*)mm.pData + (size_t)y * mm.RowPitch);
 mv_f32row_to_f16(row, mvdst + (size_t)y * LWV * 2, (int)LWV, sx, sy);
 }
 g_mvraw_valid = 1;
 have_m = 1; g_mw = (int)LWV; g_mh = (int)LHV; g_mbuf_n = 0;
 mv_wrote = 1;
 }
 }
 if (mv_wrote && g_fill_slot >= 0 && mvdst == g_slots[g_fill_slot].mvh) g_mv_slot_filled = 1;
 }
 }
 SP_END(SP_MVC);
 if (!g_mvraw_valid) {
 have_m = decode_all(&md, &mm, &g_mbuf, &g_mbuf_n, &g_mw, &g_mh);
 if (have_m) g_mbuf_eval = g_eval_n; /* stale-MV guard stamp */
 }
 if (!df_mode) stage_release(ctx);
 }

 /* live mode: NPU daemon output (falls back to naive on any failure) */
 int live_done = 0;
 ULONGLONG tlc0 = GetTickCount64(), tmc0 = tlc0; /* stage timers */
 (void)tlc0; (void)tmc0;
 if (g_live && have_c && have_m)
 live_done = live_output(ctx, out, sx, sy, jx, jy);
 unsigned long t_live = (unsigned long)(GetTickCount64() - tlc0);
 SP_BEGIN(SP_NAI);
 if (!live_done && have_c) { naive_output(ctx, out); SP_FLAG(SPFL_NAIVE); }
 SP_END(SP_NAI);

 if (g_armed && have_c) {
 size_t n = (size_t)g_cw * g_ch;
 unsigned short* d = (unsigned short*)HeapAlloc(GetProcessHeap(), 0, n * 8);
 if (d) {
 f4* c = (f4*)g_cbuf;
 for (size_t i = 0; i < n; i++) {
 unsigned short* px = d + i * 4;
 px[0] = f2h(c[i].r); px[1] = f2h(c[i].g); px[2] = f2h(c[i].b); px[3] = f2h(c[i].a);
 }
 dump_file("color.raw", d, n * 8, (unsigned)g_cap_n);
 HeapFree(GetProcessHeap(), 0, d);
 }
 if (have_m) {
 size_t n2 = (size_t)g_mw * g_mh;
 unsigned short* d2 = (unsigned short*)HeapAlloc(GetProcessHeap(), 0, n2 * 4);
 if (d2) {
 f4* m = (f4*)g_mbuf;
 for (size_t i = 0; i < n2; i++) {
 d2[i * 2] = f2h(m[i].r * sx);
 d2[i * 2 + 1] = f2h(m[i].g * sy);
 }
 dump_file("mv.raw", d2, n2 * 4, (unsigned)g_cap_n);
 HeapFree(GetProcessHeap(), 0, d2);
 }
 } else {
 LOG("cap %d: NO MV dumped", g_cap_n);
 }
 write_meta((unsigned)g_cap_n, &cd, &md, jx, jy, sx, sy, preexp);
 g_cap_n++;
 capture_maybe_done();
 }
 /* release the popped entry's Maps (st_c/st_m ARE de->tc/de->tm —
 * the ring slot is reused by a push 3 evals from now, never in flight) */
 if (df_mode) {
 SP_BEGIN(SP_UNM);
 if (st_c) ctx->lpVtbl->Unmap(ctx, (ID3D11Resource*)st_c, 0);
 if (st_m) ctx->lpVtbl->Unmap(ctx, (ID3D11Resource*)st_m, 0);
 SP_END(SP_UNM);
 }
 /* defensive reservation release (live_output clears its own) */
 if (g_fill_slot >= 0) {
 EnterCriticalSection(&g_live_cs);
 if (g_slots[g_fill_slot].state == LS_FILL) g_slots[g_fill_slot].state = LS_EMPTY;
 LeaveCriticalSection(&g_live_cs);
 g_fill_slot = -1;
 }
 if (g_live) {
 /* queue THIS frame's copies into the next ring slot
 * (non-blocking). The push only counts when the color copy landed —
 * no color = no consistent pair to ship two evals from now. */
 SP_BEGIN(SP_DSUB);
 DFREntry* pe = &g_dfr[g_dfr_u % DFR_N];
 pe->okc = col ? dfr_push_one(ctx, col, &pe->tc, pe->ck, &pe->dc): 0;
 pe->okm = mv ? dfr_push_one(ctx, mv, &pe->tm, pe->mk, &pe->dm): 0;
 pe->jx = cjx; pe->jy = cjy; pe->sx = csx; pe->sy = csy;
 if (pe->okc) g_dfr_u++;
 SP_END(SP_DSUB);
 } else {
 /* live disabled (strike/fail): drop any queued entries so a later
 * re-arm bootstraps from a FRESH sync-path frame instead of
 * shipping pre-fail copies (v14 redisplayed one stale frame here) */
 g_dfr_p = g_dfr_u;
 /* don't fold pre-outage motion into the first post-rearm frame */
 g_mv_acc_valid = 0;
 g_mvf32_acc_valid = 0;
 }
 _mm_setcsr(g_eval_csr); /* restore the game's FP mode (pin was in sp_eval_begin) */
 /* Legacy line (QPC-fallback only); real budget = S30/S90 */
 SP_END(SP_TOT);
 /* (ini fpscap=N): eval-end metronome. After mapn2 removed the mapC
 * governor the game thread sprints (~2.5ms evals): the freed CPU
 * contends the daemon's 6.2MB/frame socket read (up 18-22ms => svc
 * 32-36ms > frame budget => redisplay judder + flash) and runs the GPU
 * queue 2-3 deep (wrt staging-recycle spikes). A QPC-paced FLOOR on the
 * eval interval paces the whole frame; the Sleep yields the core to the
 * daemon. Pure floor — a game slower than the cap never waits here; a
 * >1-period stall resyncs instead of bursting. Skipped when armed so
 * capture timing stays true. */
 if (g_live && !g_armed && g_opt_fpscap > 0 && g_qpc_ok) {
 static LARGE_INTEGER fcap_next;
 static int fcap_have = 0;
 const long long fper = g_qpc_freq.QuadPart / (long long)g_opt_fpscap;
 LARGE_INTEGER fnow;
 QueryPerformanceCounter(&fnow);
 if (!fcap_have || fcap_next.QuadPart - fnow.QuadPart < -fper) {
 /* anchoring to fnow made the resync frame wait a FULL
 * period — a grid, not a floor (a 40fps game under a 60 cap ran
 * at 24). Anchor one period BACK: this frame releases now and
 * pacing starts with the next interval. */
 fcap_next.QuadPart = fnow.QuadPart - fper;
 fcap_have = 1;
 }
 fcap_next.QuadPart += fper;
 for (;;) {
 long long left = fcap_next.QuadPart - fnow.QuadPart;
 if (left <= 0) break;
 if (left > g_qpc_freq.QuadPart / 500) { /* >2ms: sleep, yield the core */
 Sleep(1);
 QueryPerformanceCounter(&fnow);
 } else { /* final <=2ms: spin */
 do { QueryPerformanceCounter(&fnow); } while (fnow.QuadPart < fcap_next.QuadPart);
 break;
 }
 }
 }
 if ((g_eval_n % 30) == 0) {
 if (g_qpc_ok) sp_flush(g_eval_n);
 else
 LOG("eval %lld total %lums | live %lu (wait %lu write %lu) df=%d",
 g_eval_n, (unsigned long)(GetTickCount64() - eval_t0), t_live, g_t_wait, g_t_write,
 df_mode);
 }
 g_sp_n++;
 return NVSDK_NGX_Result_Success;
}

/* ---- exported D3D11 ---- */

NVSDK_NGX_Result __declspec(dllexport) NVSDK_NGX_D3D11_Init(
 unsigned long long InApplicationId, const wchar_t* InApplicationDataPath,
 ID3D11Device* InDevice, const NVSDK_NGX_FeatureCommonInfo* InFeatureInfo,
 unsigned long long InSDKVersion) {
 ensure_init();
 if (InDevice) { g_dev = InDevice; }
 LOG("D3D11_Init appid=%llu sdkver=0x%llx dev=%p path=%ls", InApplicationId, InSDKVersion,
 (void*)InDevice, InApplicationDataPath ? InApplicationDataPath: L"(null)");
 (void)InFeatureInfo;
 return NVSDK_NGX_Result_Success;
}

NVSDK_NGX_Result __declspec(dllexport) NVSDK_NGX_D3D11_Init_with_ProjectID(
 const char* InProjectId, int InEngineType, const char* InEngineVersion,
 const wchar_t* InApplicationDataPath, ID3D11Device* InDevice,
 const NVSDK_NGX_FeatureCommonInfo* InFeatureInfo, unsigned long long InSDKVersion) {
 ensure_init();
 if (InDevice) { g_dev = InDevice; }
 LOG("D3D11_Init_ProjectID project=%s engine=%d ver=%s sdkver=0x%llx dev=%p",
 InProjectId ? InProjectId: "(null)", InEngineType, InEngineVersion ? InEngineVersion: "(null)",
 InSDKVersion, (void*)InDevice);
 (void)InApplicationDataPath; (void)InFeatureInfo;
 return NVSDK_NGX_Result_Success;
}

NVSDK_NGX_Result __declspec(dllexport) NVSDK_NGX_D3D11_Init_Ext(
 unsigned long long InApplicationId, const wchar_t* InApplicationDataPath,
 ID3D11Device* InDevice, unsigned long long InSDKVersion,
 const NVSDK_NGX_FeatureCommonInfo* InFeatureInfo) {
 ensure_init();
 if (InDevice) { g_dev = InDevice; }
 LOG("D3D11_Init_Ext appid=%llu sdkver=0x%llx dev=%p", InApplicationId, InSDKVersion, (void*)InDevice);
 (void)InApplicationDataPath; (void)InFeatureInfo;
 return NVSDK_NGX_Result_Success;
}

NVSDK_NGX_Result __declspec(dllexport) NVSDK_NGX_D3D11_Shutdown1(ID3D11Device* InDevice) {
 ensure_init();
 LOG("D3D11_Shutdown1 dev=%p", (void*)InDevice);
 (void)InDevice;
 return NVSDK_NGX_Result_Success;
}

NVSDK_NGX_Result __declspec(dllexport) NVSDK_NGX_D3D11_Shutdown(void) {
 ensure_init();
 LOG("D3D11_Shutdown");
 return NVSDK_NGX_Result_Success;
}

NVSDK_NGX_Result __declspec(dllexport) NVSDK_NGX_D3D11_GetParameters(NVSDK_NGX_Parameter** OutParameters) {
 ensure_init();
 if (!g_params_persistent) { g_params_persistent = param_new(0); param_populate_capabilities(g_params_persistent); }
 if (!g_params_persistent || !OutParameters) return NVSDK_NGX_Result_FAIL_PlatformError;
 *OutParameters = (NVSDK_NGX_Parameter*)g_params_persistent;
 LOG("D3D11_GetParameters -> %p", (void*)g_params_persistent);
 return NVSDK_NGX_Result_Success;
}

NVSDK_NGX_Result __declspec(dllexport) NVSDK_NGX_D3D11_AllocateParameters(NVSDK_NGX_Parameter** OutParameters) {
 ensure_init();
 FsParam* s = param_new(1);
 if (!s || !OutParameters) return NVSDK_NGX_Result_FAIL_PlatformError;
 *OutParameters = (NVSDK_NGX_Parameter*)s;
 LOG("D3D11_AllocateParameters -> %p", (void*)s);
 return NVSDK_NGX_Result_Success;
}

NVSDK_NGX_Result __declspec(dllexport) NVSDK_NGX_D3D11_GetCapabilityParameters(NVSDK_NGX_Parameter** OutParameters) {
 ensure_init();
 if (!g_caps_persistent) { g_caps_persistent = param_new(0); param_populate_capabilities(g_caps_persistent); }
 if (!g_caps_persistent || !OutParameters) return NVSDK_NGX_Result_FAIL_PlatformError;
 *OutParameters = (NVSDK_NGX_Parameter*)g_caps_persistent;
 LOG("D3D11_GetCapabilityParameters -> %p", (void*)g_caps_persistent);
 return NVSDK_NGX_Result_Success;
}

NVSDK_NGX_Result __declspec(dllexport) NVSDK_NGX_D3D11_DestroyParameters(NVSDK_NGX_Parameter* InParameters) {
 ensure_init();
 FsParam* s = (FsParam*)InParameters;
 LOG("D3D11_DestroyParameters %p owned=%d", (void*)s, s ? s->owned: -1);
 if (s && s->owned) HeapFree(GetProcessHeap(), 0, s);
 return NVSDK_NGX_Result_Success;
}

NVSDK_NGX_Result __declspec(dllexport) NVSDK_NGX_D3D11_GetScratchBufferSize(
 int InFeatureId, const NVSDK_NGX_Parameter* InParameters, size_t* OutSizeInBytes) {
 ensure_init();
 LOG("D3D11_GetScratchBufferSize feature=%d -> 0", InFeatureId);
 (void)InParameters;
 if (OutSizeInBytes) *OutSizeInBytes = 0;
 return NVSDK_NGX_Result_Success;
}

NVSDK_NGX_Result __declspec(dllexport) NVSDK_NGX_D3D11_CreateFeature(
 ID3D11DeviceContext* InDevCtx, int InFeatureID, NVSDK_NGX_Parameter* InParameters,
 NVSDK_NGX_Handle** OutHandle) {
 ensure_init();
 LOG("D3D11_CreateFeature feature=%d", InFeatureID);
 if (InFeatureID != NVSDK_NGX_Feature_SuperSampling && InFeatureID != NVSDK_NGX_Feature_ImageSuperResolution) {
 LOG("CreateFeature: unsupported feature %d", InFeatureID);
 return NVSDK_NGX_Result_FAIL_FeatureNotSupported;
 }
 FsParam* p = (FsParam*)InParameters;
 read_dims(p, K_Width, &g_feat_w);
 read_dims(p, K_Height, &g_feat_h);
 read_dims(p, K_OutWidth, &g_feat_ow);
 read_dims(p, K_OutHeight, &g_feat_oh);
 {
 unsigned long long tmp = 0;
 if (fsp_get(p, K_CreateFlags, 4, &tmp, NULL, NULL, NULL) == (int)NVSDK_NGX_Result_Success) g_feat_flags = (int)(long long)tmp;
 if (fsp_get(p, K_PerfQ, 4, &tmp, NULL, NULL, NULL) == (int)NVSDK_NGX_Result_Success) g_feat_pq = (int)(long long)tmp;
 }
 if (!g_dev) maybe_remember_device(InDevCtx);
 if (!OutHandle) return NVSDK_NGX_Result_FAIL_InvalidParameter;
 g_handle.Id = 1;
 *OutHandle = &g_handle;
 g_created = 1;
 LOG("CreateFeature OK: %ux%u -> %ux%u flags=0x%x pq=%d%s",
 g_feat_w, g_feat_h, g_feat_ow, g_feat_oh, g_feat_flags, g_feat_pq,
 (g_feat_w == (int)LWV && g_feat_h == (int)LHV && g_feat_ow == (int)(LWV * 2u) && g_feat_oh == (int)(LHV * 2u))
 ? " [PINNED SHAPES MATCH]": " [SHAPE MISMATCH — capture will be resampled or re-configured]");
 return NVSDK_NGX_Result_Success;
}

NVSDK_NGX_Result __declspec(dllexport) NVSDK_NGX_D3D11_ReleaseFeature(NVSDK_NGX_Handle* InHandle) {
 ensure_init();
 LOG("D3D11_ReleaseFeature id=%u", InHandle ? InHandle->Id: 0);
 return NVSDK_NGX_Result_Success;
}

NVSDK_NGX_Result __declspec(dllexport) NVSDK_NGX_D3D11_EvaluateFeature(
 ID3D11DeviceContext* InDevCtx, const NVSDK_NGX_Handle* InFeatureHandle,
 NVSDK_NGX_Parameter* InParameters, void* InCallback) {
 ensure_init();
 (void)InCallback;
 if (!g_dev) maybe_remember_device(InDevCtx);
 return eval_d3d11(InDevCtx, InFeatureHandle, InParameters);
}

NVSDK_NGX_Result __declspec(dllexport) NVSDK_NGX_D3D11_EvaluateFeature_C(
 ID3D11DeviceContext* InDevCtx, const NVSDK_NGX_Handle* InFeatureHandle,
 const NVSDK_NGX_Parameter* InParameters, void* InCallback) {
 ensure_init();
 (void)InCallback;
 if (!g_dev) maybe_remember_device(InDevCtx);
 return eval_d3d11(InDevCtx, InFeatureHandle, (NVSDK_NGX_Parameter*)InParameters);
}

NVSDK_NGX_Result __declspec(dllexport) NVSDK_NGX_D3D11_GetFeatureRequirements(
 void* Adapter, const void* FeatureDiscoveryInfo, NVSDK_NGX_FeatureRequirement* OutSupported) {
 ensure_init();
 LOG("D3D11_GetFeatureRequirements adapter=%p", Adapter);
 (void)Adapter; (void)FeatureDiscoveryInfo;
 if (!OutSupported) return NVSDK_NGX_Result_FAIL_InvalidParameter;
 OutSupported->FeatureSupported = 0; /* 0 = supported */
 OutSupported->MinHWArchitecture = 0x190; /* Ada-class: >= Turing requirement satisfied */
 strcpy(OutSupported->MinOSVersion, "10.0.19041");
 return NVSDK_NGX_Result_Success;
}

/* ---- parameter C-helper exports (dispatch through the vtable) ---- */

NVSDK_NGX_Result __declspec(dllexport) NVSDK_NGX_Parameter_SetULL(NVSDK_NGX_Parameter* p, const char* n, unsigned long long v) { FSP_SetULL(p, n, v); return NVSDK_NGX_Result_Success; }
NVSDK_NGX_Result __declspec(dllexport) NVSDK_NGX_Parameter_SetF(NVSDK_NGX_Parameter* p, const char* n, float v) { FSP_SetF(p, n, v); return NVSDK_NGX_Result_Success; }
NVSDK_NGX_Result __declspec(dllexport) NVSDK_NGX_Parameter_SetD(NVSDK_NGX_Parameter* p, const char* n, double v) { FSP_SetD(p, n, v); return NVSDK_NGX_Result_Success; }
NVSDK_NGX_Result __declspec(dllexport) NVSDK_NGX_Parameter_SetUI(NVSDK_NGX_Parameter* p, const char* n, unsigned int v) { FSP_SetUI(p, n, v); return NVSDK_NGX_Result_Success; }
NVSDK_NGX_Result __declspec(dllexport) NVSDK_NGX_Parameter_SetI(NVSDK_NGX_Parameter* p, const char* n, int v) { FSP_SetI(p, n, v); return NVSDK_NGX_Result_Success; }
NVSDK_NGX_Result __declspec(dllexport) NVSDK_NGX_Parameter_SetD3d11Resource(NVSDK_NGX_Parameter* p, const char* n, ID3D11Resource* v) { FSP_SetD11(p, n, v); return NVSDK_NGX_Result_Success; }
NVSDK_NGX_Result __declspec(dllexport) NVSDK_NGX_Parameter_SetD3d12Resource(NVSDK_NGX_Parameter* p, const char* n, void* v) { FSP_SetD12(p, n, v); return NVSDK_NGX_Result_Success; }
NVSDK_NGX_Result __declspec(dllexport) NVSDK_NGX_Parameter_SetVoidPointer(NVSDK_NGX_Parameter* p, const char* n, void* v) { FSP_SetVP(p, n, v); return NVSDK_NGX_Result_Success; }
NVSDK_NGX_Result __declspec(dllexport) NVSDK_NGX_Parameter_GetULL(NVSDK_NGX_Parameter* p, const char* n, unsigned long long* v) { return FSP_GetULL(p, n, v); }
NVSDK_NGX_Result __declspec(dllexport) NVSDK_NGX_Parameter_GetF(NVSDK_NGX_Parameter* p, const char* n, float* v) { return FSP_GetF(p, n, v); }
NVSDK_NGX_Result __declspec(dllexport) NVSDK_NGX_Parameter_GetD(NVSDK_NGX_Parameter* p, const char* n, double* v) { return FSP_GetD(p, n, v); }
NVSDK_NGX_Result __declspec(dllexport) NVSDK_NGX_Parameter_GetUI(NVSDK_NGX_Parameter* p, const char* n, unsigned int* v) { return FSP_GetUI(p, n, v); }
NVSDK_NGX_Result __declspec(dllexport) NVSDK_NGX_Parameter_GetI(NVSDK_NGX_Parameter* p, const char* n, int* v) { return FSP_GetI(p, n, v); }
NVSDK_NGX_Result __declspec(dllexport) NVSDK_NGX_Parameter_GetD3d11Resource(NVSDK_NGX_Parameter* p, const char* n, ID3D11Resource** v) { return FSP_GetD11(p, n, v); }
NVSDK_NGX_Result __declspec(dllexport) NVSDK_NGX_Parameter_GetD3d12Resource(NVSDK_NGX_Parameter* p, const char* n, void** v) { return FSP_GetD12(p, n, v); }
NVSDK_NGX_Result __declspec(dllexport) NVSDK_NGX_Parameter_GetVoidPointer(NVSDK_NGX_Parameter* p, const char* n, void** v) { return FSP_GetVP(p, n, v); }

/* ---- misc common exports ---- */

NVSDK_NGX_Result __declspec(dllexport) NVSDK_NGX_UpdateFeature(const void* ApplicationId, int FeatureID) {
 ensure_init();
 LOG("UpdateFeature feature=%d", FeatureID);
 (void)ApplicationId;
 return NVSDK_NGX_Result_Success;
}

const wchar_t* __declspec(dllexport) GetNGXResultAsString(NVSDK_NGX_Result r) {
 (void)r;
 return L"FSR4PROXY";
}

unsigned int __declspec(dllexport) NVSDK_NGX_GetSnippetVersion(void) {
 return NVSDK_NGX_VERSION_API_MACRO;
}

/* ---- D3D12 / CUDA / VULKAN stubs (RotTR is DX11; unknown signatures are
 * declared parameterless — x64 callers may pass extra ignored args) ---- */

NVSDK_NGX_Result __declspec(dllexport) NVSDK_NGX_D3D12_Init(void) { ensure_init(); LOG("D3D12_Init (stub)"); return NVSDK_NGX_Result_Success; }
NVSDK_NGX_Result __declspec(dllexport) NVSDK_NGX_D3D12_Init_with_ProjectID(void) { ensure_init(); LOG("D3D12_Init_ProjectID (stub)"); return NVSDK_NGX_Result_Success; }
NVSDK_NGX_Result __declspec(dllexport) NVSDK_NGX_D3D12_Init_Ext(void) { ensure_init(); LOG("D3D12_Init_Ext (stub)"); return NVSDK_NGX_Result_Success; }
NVSDK_NGX_Result __declspec(dllexport) NVSDK_NGX_D3D12_Shutdown(void) { ensure_init(); LOG("D3D12_Shutdown (stub)"); return NVSDK_NGX_Result_Success; }
NVSDK_NGX_Result __declspec(dllexport) NVSDK_NGX_D3D12_Shutdown1(void) { return NVSDK_NGX_Result_Success; }
NVSDK_NGX_Result __declspec(dllexport) NVSDK_NGX_D3D12_GetParameters(NVSDK_NGX_Parameter** o) { return NVSDK_NGX_D3D11_GetParameters(o); }
NVSDK_NGX_Result __declspec(dllexport) NVSDK_NGX_D3D12_AllocateParameters(NVSDK_NGX_Parameter** o) { return NVSDK_NGX_D3D11_AllocateParameters(o); }
NVSDK_NGX_Result __declspec(dllexport) NVSDK_NGX_D3D12_GetCapabilityParameters(NVSDK_NGX_Parameter** o) { return NVSDK_NGX_D3D11_GetCapabilityParameters(o); }
NVSDK_NGX_Result __declspec(dllexport) NVSDK_NGX_D3D12_DestroyParameters(NVSDK_NGX_Parameter* p) { return NVSDK_NGX_D3D11_DestroyParameters(p); }
NVSDK_NGX_Result __declspec(dllexport) NVSDK_NGX_D3D12_GetScratchBufferSize(void) { return NVSDK_NGX_Result_Success; }
NVSDK_NGX_Result __declspec(dllexport) NVSDK_NGX_D3D12_CreateFeature(void) { ensure_init(); LOG("D3D12_CreateFeature (stub) -> not supported"); return NVSDK_NGX_Result_FAIL_FeatureNotSupported; }
NVSDK_NGX_Result __declspec(dllexport) NVSDK_NGX_D3D12_ReleaseFeature(void) { return NVSDK_NGX_Result_Success; }
NVSDK_NGX_Result __declspec(dllexport) NVSDK_NGX_D3D12_EvaluateFeature(void) { ensure_init(); LOG("D3D12_EvaluateFeature (stub) -> not supported"); return NVSDK_NGX_Result_FAIL_FeatureNotSupported; }
NVSDK_NGX_Result __declspec(dllexport) NVSDK_NGX_D3D12_GetFeatureRequirements(void* Adapter, const void* Fdi, NVSDK_NGX_FeatureRequirement* Out) { return NVSDK_NGX_D3D11_GetFeatureRequirements(Adapter, Fdi, Out); }

NVSDK_NGX_Result __declspec(dllexport) NVSDK_NGX_CUDA_Init(void) { ensure_init(); LOG("CUDA_Init (stub)"); return NVSDK_NGX_Result_Success; }
NVSDK_NGX_Result __declspec(dllexport) NVSDK_NGX_CUDA_Init_with_ProjectID(void) { return NVSDK_NGX_Result_Success; }
NVSDK_NGX_Result __declspec(dllexport) NVSDK_NGX_CUDA_Shutdown(void) { return NVSDK_NGX_Result_Success; }
NVSDK_NGX_Result __declspec(dllexport) NVSDK_NGX_CUDA_GetParameters(NVSDK_NGX_Parameter** o) { return NVSDK_NGX_D3D11_GetParameters(o); }
NVSDK_NGX_Result __declspec(dllexport) NVSDK_NGX_CUDA_AllocateParameters(NVSDK_NGX_Parameter** o) { return NVSDK_NGX_D3D11_AllocateParameters(o); }
NVSDK_NGX_Result __declspec(dllexport) NVSDK_NGX_CUDA_GetCapabilityParameters(NVSDK_NGX_Parameter** o) { return NVSDK_NGX_D3D11_GetCapabilityParameters(o); }
NVSDK_NGX_Result __declspec(dllexport) NVSDK_NGX_CUDA_DestroyParameters(NVSDK_NGX_Parameter* p) { return NVSDK_NGX_D3D11_DestroyParameters(p); }
NVSDK_NGX_Result __declspec(dllexport) NVSDK_NGX_CUDA_GetScratchBufferSize(void) { return NVSDK_NGX_Result_Success; }
NVSDK_NGX_Result __declspec(dllexport) NVSDK_NGX_CUDA_CreateFeature(void) { return NVSDK_NGX_Result_FAIL_FeatureNotSupported; }
NVSDK_NGX_Result __declspec(dllexport) NVSDK_NGX_CUDA_ReleaseFeature(void) { return NVSDK_NGX_Result_Success; }
NVSDK_NGX_Result __declspec(dllexport) NVSDK_NGX_CUDA_EvaluateFeature(void) { return NVSDK_NGX_Result_FAIL_FeatureNotSupported; }

NVSDK_NGX_Result __declspec(dllexport) NVSDK_NGX_VULKAN_RequiredExtensionCount(void) { return NVSDK_NGX_Result_Success; }
NVSDK_NGX_Result __declspec(dllexport) NVSDK_NGX_VULKAN_RequiredExtensions(void) { return NVSDK_NGX_Result_Success; }
NVSDK_NGX_Result __declspec(dllexport) NVSDK_NGX_VULKAN_Init(void) { ensure_init(); LOG("VULKAN_Init (stub)"); return NVSDK_NGX_Result_Success; }
NVSDK_NGX_Result __declspec(dllexport) NVSDK_NGX_VULKAN_Init_with_ProjectID(void) { return NVSDK_NGX_Result_Success; }
NVSDK_NGX_Result __declspec(dllexport) NVSDK_NGX_VULKAN_Init_Ext(void) { return NVSDK_NGX_Result_Success; }
NVSDK_NGX_Result __declspec(dllexport) NVSDK_NGX_VULKAN_Init_Ext2(void) { return NVSDK_NGX_Result_Success; }
NVSDK_NGX_Result __declspec(dllexport) NVSDK_NGX_VULKAN_Shutdown(void) { return NVSDK_NGX_Result_Success; }
NVSDK_NGX_Result __declspec(dllexport) NVSDK_NGX_VULKAN_GetParameters(NVSDK_NGX_Parameter** o) { return NVSDK_NGX_D3D11_GetParameters(o); }
NVSDK_NGX_Result __declspec(dllexport) NVSDK_NGX_VULKAN_AllocateParameters(NVSDK_NGX_Parameter** o) { return NVSDK_NGX_D3D11_AllocateParameters(o); }
NVSDK_NGX_Result __declspec(dllexport) NVSDK_NGX_VULKAN_GetCapabilityParameters(NVSDK_NGX_Parameter** o) { return NVSDK_NGX_D3D11_GetCapabilityParameters(o); }
NVSDK_NGX_Result __declspec(dllexport) NVSDK_NGX_VULKAN_DestroyParameters(NVSDK_NGX_Parameter* p) { return NVSDK_NGX_D3D11_DestroyParameters(p); }
NVSDK_NGX_Result __declspec(dllexport) NVSDK_NGX_VULKAN_GetScratchBufferSize(void) { return NVSDK_NGX_Result_Success; }
NVSDK_NGX_Result __declspec(dllexport) NVSDK_NGX_VULKAN_CreateFeature(void) { return NVSDK_NGX_Result_FAIL_FeatureNotSupported; }
NVSDK_NGX_Result __declspec(dllexport) NVSDK_NGX_VULKAN_ReleaseFeature(void) { return NVSDK_NGX_Result_Success; }
NVSDK_NGX_Result __declspec(dllexport) NVSDK_NGX_VULKAN_EvaluateFeature(void) { return NVSDK_NGX_Result_FAIL_FeatureNotSupported; }

/* ---- DLL bootstrap ---- */

BOOL WINAPI DllMain(HINSTANCE hinst, DWORD reason, LPVOID reserved) {
 (void)reserved;
 if (reason == DLL_PROCESS_ATTACH) {
 DisableThreadLibraryCalls(hinst);
 GetModuleFileNameA(hinst, g_dll_path, MAX_PATH);
 char* s = strrchr(g_dll_path, '\\');
 if (s) { *s = 0; strcpy(g_dll_dir, g_dll_path); *s = '\\'; }
 }
 return TRUE;
}

/* additions to match the proven nvngx.dll seat surface (OptiScaler export set) */
NVSDK_NGX_Result __declspec(dllexport) NVSDK_NGX_D3D11_Init_ProjectID(
 const char* InProjectId, int InEngineType, const char* InEngineVersion,
 const wchar_t* InApplicationDataPath, ID3D11Device* InDevice,
 const NVSDK_NGX_FeatureCommonInfo* InFeatureInfo, unsigned long long InSDKVersion) {
 return NVSDK_NGX_D3D11_Init_with_ProjectID(InProjectId, InEngineType, InEngineVersion,
 InApplicationDataPath, InDevice, InFeatureInfo, InSDKVersion);
}

NVSDK_NGX_Result __declspec(dllexport) NVSDK_NGX_D3D11_PopulateParameters_Impl(NVSDK_NGX_Parameter* InParameters) {
 ensure_init();
 LOG("D3D11_PopulateParameters_Impl %p", (void*)InParameters);
 if (InParameters) param_populate_capabilities((FsParam*)InParameters);
 return NVSDK_NGX_Result_Success;
}

NVSDK_NGX_Result __declspec(dllexport) NVSDK_NGX_D3D12_Init_ProjectID(void) { return NVSDK_NGX_Result_Success; }
NVSDK_NGX_Result __declspec(dllexport) NVSDK_NGX_D3D12_PopulateParameters_Impl(NVSDK_NGX_Parameter* p) { ensure_init(); if (p) param_populate_capabilities((FsParam*)p); return NVSDK_NGX_Result_Success; }
NVSDK_NGX_Result __declspec(dllexport) NVSDK_NGX_VULKAN_CreateFeature1(void) { return NVSDK_NGX_Result_FAIL_FeatureNotSupported; }
NVSDK_NGX_Result __declspec(dllexport) NVSDK_NGX_VULKAN_GetFeatureInstanceExtensionRequirements(void) { return NVSDK_NGX_Result_Success; }
NVSDK_NGX_Result __declspec(dllexport) NVSDK_NGX_VULKAN_GetFeatureDeviceExtensionRequirements(void) { return NVSDK_NGX_Result_Success; }
NVSDK_NGX_Result __declspec(dllexport) NVSDK_NGX_VULKAN_GetFeatureRequirements(void) { return NVSDK_NGX_Result_Success; }
NVSDK_NGX_Result __declspec(dllexport) NVSDK_NGX_VULKAN_Init_ProjectID(void) { return NVSDK_NGX_Result_Success; }
NVSDK_NGX_Result __declspec(dllexport) NVSDK_NGX_VULKAN_Init_ProjectID_Ext(void) { return NVSDK_NGX_Result_Success; }
NVSDK_NGX_Result __declspec(dllexport) NVSDK_NGX_VULKAN_Shutdown1(void) { return NVSDK_NGX_Result_Success; }
NVSDK_NGX_Result __declspec(dllexport) NVSDK_NGX_VULKAN_PopulateParameters_Impl(NVSDK_NGX_Parameter* p) { ensure_init(); if (p) param_populate_capabilities((FsParam*)p); return NVSDK_NGX_Result_Success; }
