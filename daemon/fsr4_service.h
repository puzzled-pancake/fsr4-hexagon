// fsr4_service.h — FSR4RP6 integration contract, version 1.
//
// In-process library. Single process;
// QNN executes on the CALLING thread (contract (e)); GL runs on the same
// thread (a single-stream loop was adopted — no GL worker thread was
// adopted, so no GL-worker clause applies;
// the harness's sem/pthread skeleton is inert dead code).
// Single executing client v1: init/execute/invalidate/shutdown MUST be called
// from the same thread (the service thread). NOTE: init creates the service's
// OWN private EGL context on that thread and makes it current — any
// pre-existing current context on the thread is NOT saved/restored, and
// shutdown terminates the default EGL display. Use a dedicated thread if the
// host owns EGL state.
//
// Shapes are pinned (contract (a)): lr 960x540 (WxH) RGBA16F, mvec 960x540
// RG16F, reproj 1920x1080 (WxH) RGBA16F, out 1920x1080 (WxH) RGBA16F. Others
// are rejected with FSR4_ERR_STATE (201; SHAPE class — resolution profiles
// are not in v1); letterboxing = caller.
//
// One LR color buffer feeds BOTH the post stage's lrc and the feat stage's
// u_lr — the capture bank double-writes identical bytes to lrcolor/lrA.
// Note:
// the live harness's one-frame lrA/mv skew (its feat(fr) consumes frame fr-1's
// uploads) is a software-pipelining artifact of the BENCHMARK HARNESS, not of
// this contract; the service is skew-free by design (frame n's own inputs).
//
// Service-owned temporal state (contract (d)): the rec texture (evolved by the
// post stage every frame) and the hist texture (STATIC seed — a recorded
// fidelity deviation; it never evolves in v1). invalidate() re-seeds both
// AND re-arms the bootstrap state, so the frame after invalidate is
// byte-identical to the frame after init for identical caller inputs. A client
// disconnect/reconnect maps to invalidate(); a disconnect NEVER triggers
// shutdown or state loss on its own — only explicit invalidate()/shutdown()
// change service state. (With a single blocking execute() on the calling
// thread, a mid-execute disconnect is structurally unrepresentable in v1 —
// no mid-execute disconnect handling is required in v1.)
//
// NPU-facing formats (internal, contract (c)): int8 NHWC [1,540,960,16]
// symmetric scale 0.015001929365 in / [1,1080,1920,8] scale 0.303211569786
// out (pinned transitively by model_bin = the ship context bin).
//
// Buffer ownership (contract (b)): the CALLER owns all four frame buffers;
// the service copies in and copies out (persistent rpcmem lives inside the
// service, allocated once at init; per-call zero-copy is a measured extension,
// not v1). Steady-state execute() is log-silent (per-frame logging would pollute
// 36k-frame endurance record); errors are logged once per state change.
//
// Pacing (contract (h)): execute() BLOCKS for the full frame (uploads + GL +
// NPU + readback); v1 adds NO internal sleep — slot pacing is the caller's
// duty (FFX_JITTER_MS-class variance is caller-side in v1).
//
// Version check (contract (i)): fsr4_init fails with FSR4_ERR_VERSION unless
// the caller's FSR4RP6_CONTRACT_VERSION matches; the check is performed FIRST,
// before any library load or resource allocation, so a version mismatch
// requires no cleanup. v1 makes NO backwards-compat promise.
//
// Init preconditions: ADSP_LIBRARY_PATH must be set (in-process
// setenv: "<nativeLibDir>;/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp") BEFORE the
// first QNN library load, and libQnnHtpV73Stub/Skel.so bundled in the
// process's native lib dir; NEVER bundle libcdsprpc.so (bind vendor-public).
// Violations surface as INIT-class failures (14001-class loader errors); the
// full recipe lives in docs/REPRODUCING.md.
#ifndef FSR4_SERVICE_H
#define FSR4_SERVICE_H

#ifdef __cplusplus
extern "C" {
#endif

#define FSR4RP6_CONTRACT_VERSION 1u

/* error taxonomy (contract (f)) — class == code/100 (class 0 = OK);
 * retryability per class:
 * INIT (1xx): service not usable; init has fully released its partial
 * state; shutdown is safe afterwards. NOT retryable.
 * SHAPE (2xx): caller-side contract violation (shapes, state); fix
 * the call. NOT retryable as-is.
 * BUFFER (3xx): null caller buffer. NOT retryable as-is.
 * QNN (4xx): NPU execute failure (transient HTP errors possible):
 * ONE immediate retry of the same execute is allowed and
 * is EXEMPT from the invalidate-after-error rule; any
 * other error class, or a second consecutive QNN failure,
 * requires invalidate() (or shutdown + re-init) first —
 * the service does not auto-recover.
 * GL (5xx): GL/EGL failure (context loss possible). NOT retryable; re-init.
 * PACING (6xx): reserved — never raised by the service in v1 (pacing
 * is caller-side). FSR4_ERR_PACING = 600 defined for
 * literal completeness of "each class has a defined code".
 * TEARDOWN (7xx): shutdown-time cleanup failure (recorded, never fatal
 * to the caller; process exit follows). n/a
 */
enum {
 FSR4_OK = 0,
 FSR4_ERR_VERSION = 100, /* INIT: contract version mismatch */
 FSR4_ERR_INIT_QNN = 101, /* INIT: backend/system/context/graph failure */
 FSR4_ERR_INIT_GL = 102, /* INIT: EGL context / buffer creation failure */
 FSR4_ERR_INIT_MEM = 103, /* INIT: rpcmem alloc / QnnMem_register failure */
 FSR4_ERR_INIT_SHADER = 104, /* INIT: shader/seed file missing or compile failure */
 FSR4_ERR_INIT_MODEL = 105, /* INIT: context bin missing/unreadable */
 FSR4_ERR_STATE = 201, /* SHAPE: call outside the lifecycle state machine
 * (execute/invalidate before init or after
 * shutdown; init while already initialized) */
 FSR4_ERR_BUFFER = 300, /* BUFFER: null frame buffer or null in/out struct */
 FSR4_ERR_QNN_EXECUTE = 400, /* QNN: graphExecute failure */
 FSR4_ERR_GL_STAGE = 500, /* GL: upload/dispatch/barrier failure */
 FSR4_ERR_READBACK = 501, /* GL: final-buffer map/readback failure */
 FSR4_ERR_PACING = 600, /* PACING: reserved; never raised in v1 */
 FSR4_ERR_TEARDOWN = 700, /* TEARDOWN: non-clean teardown step (recorded) */
};

/* corner preset (contract: C2 corner policy; 0xA0 = the project's MAX control
 * corner per C2/C7; other corners via the C2 sweep mapping: 0x60/0x70/0x80).
 * {0,0} is NOT a valid corner (it is passed straight to the DCVS config);
 * always set both fields explicitly. */
typedef struct {
 unsigned core_corner; /* HTP core voltage corner, e.g. 0xA0 */
 unsigned bus_corner; /* HTP bus voltage corner, e.g. 0xA0 */
} fsr4_corner_t;

typedef struct {
 unsigned contract_version; /* MUST be FSR4RP6_CONTRACT_VERSION */
 const char* backend_so; /* libQnnHtp.so path */
 const char* system_so; /* libQnnSystem.so path */
 const char* model_bin; /* context bin (ship: bin_cfnhwc.bin) */
 const char* pre_shader; /* pre0_final.comp */
 const char* feat_shader; /* features_real.comp */
 const char* post_shader; /* post_real.comp */
 const char* weights_seed; /* raw uint32 blob, pre-stage SSBO binding 2
 (harness argv[7] weights.raw/pass0_wb.raw).
 REQUIRED; unreadable => FSR4_ERR_INIT_SHADER. */
 const char* hist_seed; /* raw RGBA16F 1920x1080 (gpu_data/hist.raw) */
 const char* rec_seed; /* raw RGBA16F 1920x1080 (gpu_data/rec_prev.raw) */
 fsr4_corner_t corner;
 float slot_ms; /* init bookkeeping only; v1 does not sleep and
 stats do not carry it */
 unsigned warmup; /* discarded executes at init (pinned count;
 0 allowed) */
 /* v1 diagnostics (both NULL = silent, production default):
 * when debug_dump_dir is set, the service writes in_f<n>/npuout_f<n>/rgb_f<n>
 * dumps (A3-style) for every frame in debug_dump_frames ("50,51,100,101"
 * or "a-b" syntax; frame 0 = bootstrap excluded). Never set by production
 * callers. */
 const char* debug_dump_dir;
 const char* debug_dump_frames;
 /* All const char* fields are BORROWED: read only during fsr4_init (libs
 * dlopen'd, shaders compiled, seed files copied into service-owned GL
 * buffers/textures). The caller may free or reuse them once fsr4_init
 * returns. init(NULL) returns FSR4_ERR_STATE. */
} fsr4_init_params_t;

/* buffer requirements (lr, mvec, reproj, out — all four):
 * - capacity: at least the pinned byte size above (lr 4,147,200 B;
 * mvec 2,073,600 B; reproj/out 16,588,800 B). The service reads/writes
 * exactly the pinned size and CANNOT verify capacity (no size field in
 * v1): smaller buffers are caller-side undefined behavior.
 * - alignment: none required in v1 — the service moves bytes with memcpy.
 * - lifetime: read (in) / written (out) only during the execute() call;
 * the service retains no pointer to them after execute() returns. */
typedef struct {
 const void* lr; /* RGBA16F 960x540 (4,147,200 B) — feeds post lrc + feat u_lr */
 const void* mvec; /* RG16F 960x540 (2,073,600 B) — feat motion vectors */
 const void* reproj; /* RGBA16F 1920x1080 (16,588,800 B) — post reprojection */
} fsr4_frame_in_t;

typedef struct {
 void* out; /* RGBA16F 1920x1080 (16,588,800 B) — final RGB, alpha=1.0 */
} fsr4_frame_out_t;

/* init: version-checked FIRST (see above). Loads QNN + context bin, creates
 * the EGL pbuffer context + the three programs + the persistent rpcmem/SSBO
 * set, applies the power vote (always-on), runs `warmup` discarded
 * executes, seeds hist/rec. On ANY init failure the service fully releases its
 * partially built state (QNN, GL, rpcmem) before returning; the service is
 * then in the shutdown state — fsr4_shutdown() afterwards is SAFE (and
 * returns FSR4_OK as a no-op). NULL/missing required init strings return
 * FSR4_ERR_BUFFER; an unreadable seed file returns FSR4_ERR_INIT_SHADER.
 * Strings/paths are read during init only. Lifecycle owner: the HOST APP
 * (in-process library; no daemon in v1).
 * Thread: caller becomes the service thread. Returns FSR4_OK or an INIT-class
 * error.
 *
 * PINNED PER-FRAME ORDER (single stream, one GL context, calling thread;
 * bootstrap frame = steps 2-4 skipped, step 1 still performed in full;
 * ADSP_LIBRARY_PATH is the HOST'S/environment's duty — fsr4_init does not
 * modify the process environment):
 * 1. upload: lr -> post lrc SSBO (binding 5) AND feat u_lr texture (unit 2);
 * mvec -> feat SSBO (binding 12); reproj -> post SSBO (binding 6)
 * 2. feat dispatch 240x135 (samples u_hist=0/u_rec=1/u_lr=2 + SSBO 12,
 * rewrites the feats SSBO 0 fully) + glMemoryBarrier(SHADER_STORAGE_BIT)
 * 3. pre dispatch 120x68 (feats SSBO 0 + weights SSBO 2 -> SSBO 1)
 * + glMemoryBarrier(SHADER_STORAGE_BIT) + glFinish
 * 4. copy1: SSBO 1 -> NPU input rpcmem (dmabuf RW|START before memcpy,
 * WRITE|END after)
 * 5. graphExecute — synchronous on the calling thread
 * 6. copy2: NPU output rpcmem -> post SSBO 4 (dmabuf RW|START before memcpy,
 * READ|END after)
 * 7. post dispatch 240x135 (SSBO 4/5/6 -> RGB SSBO 7; imageStore rec, unit 1)
 * + glMemoryBarrier(GL_ALL_BARRIER_BITS) (MANDATORY: imageStore->fetch)
 * 8. glFinish; copy SSBO 7 -> caller out (alpha = 1.0)
 * FIXED uniforms (never caller-visible; the validated values):
 * pre IN_W=1920 OUT_W=960 OUT_H=540; feat IN_W=1920 IN_H=1080 LR_W=960
 * LR_H=540; post u_params=(1.0, 0.25, -0.125, 0.0),
 * u_dims=(1920.0, 1080.0, 960.0, 540.0).
 * Only graph output[0] (16,588,800 B cast_335) is consumed — the ship bin's
 * second (dummy) output is ignored, matching the adopted harness.
 * The feats SSBO 0 is created zero-filled (its seed content in the harness is
 * dead data — feat fully rewrites it before any read). */
int fsr4_init(const fsr4_init_params_t* params);

/* execute: one full frame per the PINNED PER-FRAME ORDER above.
 *
 * BOOTSTRAP FRAME (pinned; mirrors the adopted harness's frame 0, ffxpipe2.c):
 * the FIRST execute after fsr4_init() — and again after every
 * fsr4_invalidate() — is the BOOTSTRAP frame. Steps 2-4 are SKIPPED on it
 * (no feat/pre dispatch precedes the harness loop, and the service performs
 * no copy1 either): the NPU input rpcmem is ZERO — zero-filled at
 * registration and re-zeroed by init/invalidate (the harness reaches the
 * same bytes by copying its zero-filled pre-output buffer with its frame-0
 * copy1; the service keeps the NPU input zero-filled and skips the copy).
 * The uploads (step 1) still occur and feed post (post consumes lrc/reproj
 * on every frame, including the bootstrap); the caller's lr/mvec are first
 * CONSUMED by frame 1's feat. post(bootstrap) writes out (alpha=1.0) and
 * imageStores the rec texture — rec after the bootstrap frame = post's
 * recurrent output for that frame. Every subsequent execute runs the full
 * recurrence: feat(n) samples hist(seed) + rec(n-1); NO cross-frame software
 * pipelining (blocking contract (h)).
 *
 * After any execute error the temporal state is UNDEFINED and the caller MUST
 * invalidate() before the next execute. Returns FSR4_OK or a
 * SHAPE/BUFFER/QNN/GL-class error. */
int fsr4_execute(const fsr4_frame_in_t* in, fsr4_frame_out_t* out);

/* invalidate: re-seeds rec + hist from the init seeds AND re-arms the
 * bootstrap state (zeroed pre-output/NPU-input buffers), so the observable
 * behavior of frame-after-invalidate equals frame-after-init for identical
 * caller inputs. Required after client reconnect; safe any time after init.
 * Returns FSR4_OK or an INIT/GL-class error. */
int fsr4_invalidate(void);

/* shutdown: full teardown (GL release first, then deregister -> rpcmem_free ->
 * contextFree -> deviceFree -> backendFree -> logFree; the GL/QNN release
 * order is not load-bearing — disjoint subsystems). After a
 * SUCCESSFUL init this returns FSR4_OK or TEARDOWN-class (recorded; process
 * may exit afterwards). Before init, after a failed init, or after a previous
 * shutdown it is a no-op returning FSR4_OK.
 * DIAGNOSTIC KNOB (never set by production callers): env
 * FSR4_E4B switches shutdown to the partial-teardown exit — deregister +
 * rpcmem_free, SKIP the QNN frees, fflush + fsync, _exit(0) — which does NOT
 * return. All contract semantics above apply to FSR4_E4B unset. */
int fsr4_shutdown(void);

/* stats (diagnostics; v1 informational only). Stage boundaries (pinned):
 * NOTE these are service-stage buckets, not 1:1 with the harness ftlog
 * columns (the ftlog's "post" contains feat+pre and its "copy2" is separate;
 * compare totals, or map via the pinned step numbers below).
 * Call fsr4_get_stats from the service thread (single-thread v1 makes this
 * trivially safe). Buckets and frames update ONLY on successful executes.
 * When debug dumps are enabled they add their file I/O to
 * the bucket they fall in (in-dump -> npu bucket, npuout-dump -> post,
 * rgb-dump -> readback); production (dumps NULL) is unaffected. slot_ms is
 * init-parameter bookkeeping only — stats do not carry it.
 * buckets:
 * last_upload_ms = step 1
 * last_gl_ms = steps 2-4 (feat + pre + glFinish + copy1)
 * last_npu_ms = step 5 wall
 * last_post_ms = steps 6-7 (copy2 + post dispatch + ALL_BARRIER + glFinish)
 * last_readback_ms = step 8
 * last_total_ms = whole execute() call
 * frames_over_100ms counts executes whose last_total_ms exceeded 100 (the
 * stall-gate quantity, also derivable caller-side). frames = executes since
 * init (not reset by invalidate). */
typedef struct {
 double last_upload_ms, last_gl_ms, last_npu_ms, last_post_ms, last_readback_ms;
 double last_total_ms;
 /* split-path sub-buckets: the classic five MISLABEL execute_split —
 * "up" is really the fence1 GPU drain (postA(N-1)+feat(N)) + copy1, and
 * the uploads/NPU-wait/copy2 sit in an unbucketed t0->t_npu span. These
 * decompose the split call so the next rounds are data-driven:
 * last_upld_ms = t0 -> mv+lrc SubData done
 * last_nwait_ms = residual sem_wait(npu_done) for frame N-1
 * last_copy2_ms = copy2 memcpy + postA submit (-> t_npu)
 * last_fence_ms = glClientWaitSync fs0 drain of postA+feat (the GPU wall)
 * last_copy1_ms = copy1 memcpy pre_map -> inB (-> t_gl) */
 double last_upld_ms, last_nwait_ms, last_copy2_ms, last_fence_ms, last_copy1_ms;
 double execute_total_ms; /* cumulative execute() wall time since init */
 double execute_max_ms; /* worst single execute() wall time since init */
 unsigned long frames;
 unsigned long frames_over_100ms;
} fsr4_stats_t;
/* jitter passthrough (contract v1 addendum): sets the game's CURRENT
 * frame jitter (LR-pixel units, e.g. [-0.5, 0.5]). When called, execute()
 * feeds these to the post pass u_params.yz (sampling kernel centering)
 * instead of the pinned (0.25, -0.125). Call before fsr4_execute. */
void fsr4_set_jitter(float jx, float jy);

/* quality additions. Exposure: multiplies LR BEFORE mu-law and is
 * divided out at the end (1.0 = off). Reset: one-shot cut handling — the
 * next feat pass seeds history with the current color and zeroes the
 * recurrent (SDK pre_common reset semantics). Call before fsr4_execute. */
void fsr4_set_exposure(float e);
void fsr4_set_reset(int r);

/* split mode: execute() output is deferred one call — the bootstrap
 * frame produces no image. Returns 0 for that call (skip sending it), 1
 * when out carries a real frame. */
int fsr4_frame_ready(void);

void fsr4_get_stats(fsr4_stats_t* out);

#ifdef __cplusplus
}
#endif
#endif /* FSR4_SERVICE_H */
