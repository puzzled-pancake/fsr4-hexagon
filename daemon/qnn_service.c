// qnn_service.c — FSR4RP6 service v3: the full
// adopted pipeline (GL feat/pre/post_real + QNN HTP) folded behind the service
// contract (fsr4_service.h). Supersedes the v2 TCP daemon (git history keeps
// it); the Wine-side-GL-pass remark is superseded.
//
// Process model: in-process library. Single process; QNN executes
// synchronously on the CALLING thread (contract (e) — mirrors the adopted
// loop's synchronous graphExecute); GL runs on the same thread (single-stream
// loop — no GL worker thread).
//
// Per-frame order (skew-free; feat(n) samples rec written by post(n-1)):
// uploads (SSBO5 lr, SSBO6 reproj, SSBO12 mvec, unit-2 lr tex)
// -> feat dispatch + SSBO barrier -> pre dispatch (pre_out_pp[slot])
// -> glFinish -> copy1 (pre_out_pp -> inB, dmabuf sync)
// -> graphExecute (inB -> outB) -> copy2 (outB -> post_in SSBO4)
// -> post_real dispatch + ALL_BARRIER (writes final RGB SSBO7 + rec texture)
// -> glFinish -> readback (post_f -> caller out).
// Bootstrap frame (first after init/invalidate): NPU consumes the zero-filled
// internal input (A2 convention), feat/pre skipped — matches the adopted
// loop's frame 0 byte-for-byte (pre_out_pp created zero-filled, no feat/pre
// dispatch before the loop's first copy1).
//
// main() (this translation unit also builds as an executable) is the GE2
// driver: static seeds identical to ffxpipe2_instr's argv, N frames, A3-style
// in/npuout/rgb dumps at the pinned frames through the SERVICE internals, and
// the lifecycle checks (invalidate mid-stream via argv; restart = the gate's
// fresh-process leg).

#define _GNU_SOURCE /* bionic: cpu_set_t for the per-thread pin */
#include <android/hardware_buffer.h>
#include <sys/mman.h>
#include <dirent.h>
#include <dlfcn.h>
#include <errno.h>
#include <pthread.h>
#include <sched.h>
#include <semaphore.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <EGL/egl.h>
#include <GLES3/gl32.h>

#include "QnnInterface.h"
#include "System/QnnSystemInterface.h"
#include "QnnMem.h"
#include "HTP/QnnHtpDevice.h"
#include "HTP/QnnHtpPerfInfrastructure.h"

#include "fsr4_service.h"

typedef QNN_INTERFACE_VER_TYPE iface_t;
typedef QNN_SYSTEM_INTERFACE_VER_TYPE sysiface_t;
typedef Qnn_ErrorHandle_t (*BackendGetProvidersFn_t)(const QnnInterface_t***, uint32_t*);
typedef Qnn_ErrorHandle_t (*SystemGetProvidersFn_t)(const QnnSystemInterface_t***, uint32_t*);

typedef void* (*rpcmem_alloc_fn_t)(int heapid, uint32_t flags, int size);
typedef void (*rpcmem_free_fn_t)(void* po);
typedef int (*rpcmem_to_fd_fn_t)(void* po);
#define RPCMEM_HEAP_ID_SYSTEM 25
#define RPCMEM_DEFAULT_FLAGS 1

struct dmabuf_sync_s { uint64_t flags; };
#define DMABUF_SYNC_READ (1 << 0)
#define DMABUF_SYNC_WRITE (2 << 0)
#define DMABUF_SYNC_RW (DMABUF_SYNC_READ | DMABUF_SYNC_WRITE)
#define DMABUF_SYNC_START (0 << 2)
#define DMABUF_SYNC_END (1 << 2)
#define DMABUF_BASE 'b'
#define DMABUF_IOCTL_SYNC _IOW(DMABUF_BASE, 0, struct dmabuf_sync_s)

// ---- tiny utils (ports of the adopted harness helpers) ----
static double now_ms(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
 return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6; }

static char* load_text(const char* path) {
 FILE* f = fopen(path, "rb"); if (!f) return NULL;
 fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
 char* b = (char*)malloc((size_t)n + 1);
 if (!b || fread(b, 1, (size_t)n, f) != (size_t)n) { fclose(f); free(b); return NULL; }
 b[n] = 0; fclose(f); return b;
}
static int load_file(const char* path, void** out, size_t* n) {
 FILE* f = fopen(path, "rb"); if (!f) return -1;
 fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
 void* b = malloc((size_t)sz);
 if (!b || fread(b, 1, (size_t)sz, f) != (size_t)sz) { fclose(f); free(b); return -1; }
 fclose(f); *out = b; *n = (size_t)sz; return 0;
}

// ---- service state (single executing client v1: one static instance) ----
typedef struct { void* ptr; int fd; Qnn_MemHandle_t handle; size_t nbytes; } MemBuf;

static struct {
 unsigned inited;
 // QNN
 void* rpc_lib; void* be_lib; void* sys_lib;
 rpcmem_alloc_fn_t p_alloc; rpcmem_free_fn_t p_free; rpcmem_to_fd_fn_t p_tofd;
 iface_t iface;
 Qnn_LogHandle_t logh; Qnn_BackendHandle_t backend; Qnn_DeviceHandle_t device;
 Qnn_ContextHandle_t context; Qnn_GraphHandle_t graph;
 Qnn_Tensor_t* in_info; Qnn_Tensor_t* out_info;
 uint32_t nIn, nOut;
 MemBuf inB[2][2], outB[2][2];
 Qnn_Tensor_t inT[2][2], outT[2][2];
 unsigned have_vote;
 QnnHtpDevice_Infrastructure_t* di; // held for vote teardown bookkeeping
 // GL
 EGLDisplay dpy; EGLSurface surf; EGLContext ctx;
 GLuint pre_prog, feat_prog, post_prog;
 GLuint bf; // SSBO 0: feat output (feats)
 GLuint pre_out_pp[2]; // pre outputs, ping-pong (NOT bound at creation: A1)
 GLuint wbuf_ssbo; // SSBO 2: pre weights
 GLuint post_in; // SSBO 4
 GLuint lrc_ssbo, rep_ssbo, out_ssbo; // SSBO 5/6/7 (lrc_ssbo = lrc_pp[0] alias)
 GLuint lrc_pp[2]; // SSBO5 ping-pong — upload lrc(N) into the
 // partner while postA(N-1) still reads this one;
 // lets the upload hide inside the NPU wait
 GLuint rpj_ssbo, hst_ssbo; // SSBO 9/10 (real-semantics)
 GLuint out8_ssbo; // SSBO 11 (RGBA8 readback)
 void* pre_map[2]; // persistent READ maps (GL_EXT_buffer_storage)
 void* post_in_map; // persistent WRITE map for copy2 (WC memcpy)
 void* out8_map; // transient map/unmap READs run uncached on
 int have_bstorage; // Adreno (~2.4GB/s); persistent+coherent maps
 // read at RAM speed. Fallback = old path.
 GLuint mulr_ssbo; // SSBO 13 (mu-law'd LR)
 GLuint outc2_ssbo; // SSBO 14 (rcas fp16 out)
 GLuint dist_ssbo; // SSBO 15 (compact chroma-dist, 1 word/px)
 /* dma-buf import (GL_EXT_memory_object_fd) — postA reads the NPU
 * outB tensor directly as SSBO4; copy2's 16.6MB memcpy + ioctls die.
 * FSR4_DMABUF=1 enables (copy2 side only); FSR4_NOPREMAP=1 records the
 * persistent-map write-path delta for the record. */
 int have_memfd, dmabuf_in, dmabuf_pending, dmabuf_last_fd, npremap;
 GLuint post_in_dm[2], post_in_mem[2];
 /* EGL dma-buf import (FSR4_EGLIN bit0) — postA reads outB via an
 * imported dma-buf TEXTURE (posta_egl.comp); copy2 dies. The buffer
 * path is driver-blocked (but the texture path may not be. */
 int egl_in;
 GLuint npuimg_tex[2];
 /* FSR4_AHBCOPY — outB[.][0] backed by gralloc AHB; GL reads it
 * via EGLImage (posta_egl.comp imageLoads), QNN via DMA_BUF
 * registration of the fd-scanned backing. copy2 dies. */
 int ahb_in;
 int ahb_sync; /* FSR4_AHBSYNC (default 1) — 0 skips the AHB cpu-access brackets */
 /* FSR4_AHBIN — the copy1 bridge: inB[.][0] backed by gralloc,
 * feat imageStores it directly (features_fused_egl.comp); the 8.29MB
 * c1 memcpy dies. */
 int ahb_in2;
 AHardwareBuffer* ahb_inkeep[2];
 GLuint in_img_tex[2];
 double spin_ms; /* FSR4_SPIN_MS — pure CPU wait in the c2 slot */
 /* sentinel: every Nth frame, GPU-read bytes of the
 * AHB are compared against the mmap'd dma-buf view — catches a COHERENT
 * STALE FRAME, which md5 parity cannot. FSR4_SENTINEL=N (0 = off). */
 int sentinel_n;
 GLuint sent_prog, sent_ssbo;
 unsigned long sent_checked, sent_mismatch;
 void* dmabuf_last_ptr;
 AHardwareBuffer* ahb_out[2];
 GLuint mulr_prog;
 GLuint rcas_prog;
 float rcas_sharp; // 0 = off; game-style [0,1]
 int raw_lr; // FSR4_RAWLR=1 — raw upload, v2 shaders decode (mulr deleted)
 GLint ul_feat_expo, ul_feat_bftau, ul_post_expo; // uniform caches
 int fuse; // FSR4_FUSE=1 — fused feat+pre0 (features_fused.comp @ LR dispatch)
 float expo; // frame exposure (1 = off)
 int reset; // one-shot cut-reset flag
 GLint ul_mulr_expo, ul_feat_jit, ul_feat_reset, ul_post_params,
 ul_post_rcas, ul_post_bftau, ul_rcas_expo, ul_rcas_sharp, ul_rcas_out16;
 /* NPU-thread split — postA(N-1)+rcas(N-1)+
 * readback(N-1) run on the GL thread while thread B executes the NPU
 * on frame N. graphExecute is a CPU-side FastRPC call (no GL) so a
 * dedicated thread IS the async mechanism; depth 1, strict ordering. */
 int split_mode;
 pthread_t npu_thr;
 sem_t npu_submit_sem, npu_done_sem; /* strict depth-1 alternation */
 volatile unsigned npu_cur_slot; /* written pre-post, read post-wait */
 volatile int npu_err[2];
 volatile int npu_stop;
 double npu_wall[2];
 struct { unsigned valid; unsigned slot; } tail;
 volatile int npu_inflight; /* submits minus waits (GL thread only) */
 int emitted; /* last execute produced a readback frame */
 float jit_prev[2]; /* the split tail's frame-N-1 jitter */
 GLuint mv_ssbo; // SSBO 12
 GLuint hist_tex; // unit 0 (static seed)
 GLuint rec_tex; // unit 1 (immutable, imageStore'd by post_real)
 GLuint lrA_tex; // unit 2
 void* hist_seed_data; void* rec_seed_data; // kept for invalidate()
 // frame bookkeeping
 unsigned first_frame; unsigned long frames_done;
 float jit[2]; int jit_set; // jitter passthrough (default pinned)
 float blend_floor; // history-weight floor -> u_params.w
 float bf_tau; // chroma-dist floor-decay scale (0=off)
 int lra_upload; // legacy u_lr texture upload (off for SSBO13 feats)
 // diagnostics
 char dump_dir[512]; int dump_frames[128]; int n_dump;
 fsr4_stats_t stats;
} S;

/* forward decls (defined before fsr4_execute) */
static void* npu_thread(void* arg);
static void npu_submit(unsigned slot);
static int fsr4_execute_split(const fsr4_frame_in_t* in, fsr4_frame_out_t* out);

static int dmabuf_sync(int fd, uint64_t flags) {
 struct dmabuf_sync_s s; s.flags = flags;
 if (ioctl(fd, DMABUF_IOCTL_SYNC, &s) != 0) {
 fprintf(stderr, "warn: fsr4_service dmabuf sync failed errno=%d\n", errno);
 return -1;
 }
 return 0;
}

static size_t dtype_elem_size(Qnn_DataType_t dt) {
 switch (dt) {
 case QNN_DATATYPE_INT_8: case QNN_DATATYPE_UINT_8:
 case QNN_DATATYPE_SFIXED_POINT_8: case QNN_DATATYPE_UFIXED_POINT_8: return 1;
 case QNN_DATATYPE_INT_16: case QNN_DATATYPE_UINT_16:
 case QNN_DATATYPE_SFIXED_POINT_16: case QNN_DATATYPE_UFIXED_POINT_16:
 case QNN_DATATYPE_FLOAT_16: case QNN_DATATYPE_BFLOAT_16: return 2;
 default: return 4;
 }
}
static const char* t_name(const Qnn_Tensor_t* t) { return t->version == QNN_TENSOR_VERSION_1 ? t->v1.name: t->v2.name; }
static uint32_t t_rank(const Qnn_Tensor_t* t) { return t->version == QNN_TENSOR_VERSION_1 ? t->v1.rank: t->v2.rank; }
static const uint32_t* t_dims(const Qnn_Tensor_t* t) { return t->version == QNN_TENSOR_VERSION_1 ? t->v1.dimensions: t->v2.dimensions; }
static Qnn_DataType_t t_dtype(const Qnn_Tensor_t* t) { return t->version == QNN_TENSOR_VERSION_1 ? t->v1.dataType: t->v2.dataType; }
static size_t tensor_nbytes(const Qnn_Tensor_t* t) {
 size_t n = dtype_elem_size(t_dtype(t));
 for (uint32_t i = 0; i < t_rank(t); i++) n *= t_dims(t)[i];
 return n;
}
static Qnn_Tensor_t* copy_tensors(const Qnn_Tensor_t* src, uint32_t n) {
 Qnn_Tensor_t* dst = (Qnn_Tensor_t*)calloc(n, sizeof(Qnn_Tensor_t));
 if (!dst) return NULL;
 for (uint32_t i = 0; i < n; i++) {
 dst[i] = src[i];
 const char* name = t_name(&src[i]);
 char* nc = (char*)malloc(strlen(name) + 1); if (!nc) { free(dst); return NULL; }
 strcpy(nc, name);
 uint32_t rank = t_rank(&src[i]);
 uint32_t* dims = (uint32_t*)malloc(rank * sizeof(uint32_t));
 if (!dims) { free(nc); free(dst); return NULL; }
 memcpy(dims, t_dims(&src[i]), rank * sizeof(uint32_t));
 if (dst[i].version == QNN_TENSOR_VERSION_1) { dst[i].v1.name = nc; dst[i].v1.dimensions = dims; }
 else { dst[i].v2.name = nc; dst[i].v2.dimensions = dims; }
 }
 return dst;
}
static int register_mem_buf(Qnn_ContextHandle_t ctxh, Qnn_Tensor_t* t, MemBuf* out) {
 size_t nbytes = tensor_nbytes(t);
 out->ptr = S.p_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, (int)nbytes);
 if (!out->ptr) return -1;
 memset(out->ptr, 0, nbytes);
 out->fd = S.p_tofd(out->ptr);
 if (out->fd < 0) { S.p_free(out->ptr); out->ptr = NULL; return -1; }
 out->nbytes = nbytes;
 Qnn_MemDescriptor_t desc = QNN_MEM_DESCRIPTOR_INIT;
 desc.memShape.numDim = t_rank(t);
 desc.memShape.dimSize = (uint32_t*)t_dims(t);
 desc.memShape.shapeConfig = NULL;
 desc.dataType = t_dtype(t);
 desc.memType = QNN_MEM_TYPE_ION;
 desc.ionInfo.fd = out->fd;
 if (S.iface.memRegister(ctxh, &desc, 1, &out->handle) != 0 || out->handle == NULL) {
 S.p_free(out->ptr); out->ptr = NULL; return -1;
 }
 return 0;
}

/* AHB-backed MemBuf for out tensor 0 — gralloc allocates (linear,
 * uncompressed via CPU_READ_OFTEN usage), the backing dma-buf is found by
 * a /proc/self/fd size-exact scan, QNN registers it as DMA_BUF, and the GL
 * attaches it as an EGLImage after context creation. Stride must equal
 * width or the tensor's HWC rows would not line up with texel rows. */
static int register_mem_buf_ahb(Qnn_ContextHandle_t ctxh, Qnn_Tensor_t* t, MemBuf* out, AHardwareBuffer** keep, uint32_t H) {
 size_t nbytes = tensor_nbytes(t);
 if (nbytes % ((size_t)H * 4)) return -1;
 uint32_t W = (uint32_t)(nbytes / ((size_t)H * 4));
 AHardwareBuffer_Desc abd = {0};
 abd.width = W; abd.height = H; abd.layers = 1;
 abd.format = AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM;
 abd.usage = AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE | AHARDWAREBUFFER_USAGE_GPU_COLOR_OUTPUT | AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN;
 int fds_before[4096]; int nb = 0;
 DIR* dp = opendir("/proc/self/fd");
 if (dp) { struct dirent* de; while ((de = readdir(dp)) && nb < 4096) { int f = atoi(de->d_name); if (f > 2) fds_before[nb++] = f; } closedir(dp); }
 AHardwareBuffer* ahb = NULL;
 if (AHardwareBuffer_allocate(&abd, &ahb) != 0 || !ahb) return -1;
 AHardwareBuffer_Desc gb; AHardwareBuffer_describe(ahb, &gb);
 if (gb.stride != W) {
 fprintf(stderr, "fsr4_service: AHB stride %u != width %u (padding) - refusing\n", gb.stride, W);
 AHardwareBuffer_release(ahb); return -1;
 }
 int found = -1; off_t want = (off_t)gb.stride * (off_t)H * 4;
 dp = opendir("/proc/self/fd");
 if (dp) {
 struct dirent* de;
 while ((de = readdir(dp))) {
 int f = atoi(de->d_name); if (f <= 2) continue;
 int known = 0;
 for (int i = 0; i < nb; i++) if (fds_before[i] == f) { known = 1; break; }
 if (known) continue;
 if (lseek(f, 0, SEEK_END) == want) { found = f; break; }
 }
 closedir(dp);
 }
 if (found < 0) { fprintf(stderr, "fsr4_service: AHB fd scan missed the %lld B dma-buf\n", (long long)want); AHardwareBuffer_release(ahb); return -1; }
 out->fd = dup(found);
 out->nbytes = nbytes;
 out->ptr = mmap(NULL, nbytes, PROT_READ | PROT_WRITE, MAP_SHARED, out->fd, 0);
 if (out->ptr == MAP_FAILED) { close(out->fd); AHardwareBuffer_release(ahb); out->ptr = NULL; return -1; }
 memset(out->ptr, 0, nbytes);
 Qnn_MemDescriptor_t desc = QNN_MEM_DESCRIPTOR_INIT;
 desc.memShape.numDim = t_rank(t);
 desc.memShape.dimSize = (uint32_t*)t_dims(t);
 desc.memShape.shapeConfig = NULL;
 desc.dataType = t_dtype(t);
 /* QNN_MEM_TYPE_DMA_BUF segfaults inside libQnnHtp (first
 * attempt, tombstone: memRegister -> near-NULL deref). The ION import
 * path (FastRPC ion import = dma-buf attach on modern kernels) is what
 * rpcmem uses — register the gralloc fd through it instead. */
 desc.memType = QNN_MEM_TYPE_ION;
 desc.ionInfo.fd = out->fd;
 if (S.iface.memRegister(ctxh, &desc, 1, &out->handle) != 0 || out->handle == NULL) {
 fprintf(stderr, "fsr4_service: AHB QNN DMA_BUF register FAILED\n");
 munmap(out->ptr, nbytes); close(out->fd); AHardwareBuffer_release(ahb); out->ptr = NULL; return -1;
 }
 *keep = ahb;
 printf("fsr4_service: AHB out %ux%u stride %u fd %d (%lld B) QNN DMA_BUF OK\n",
 W, H, gb.stride, out->fd, (long long)want);
 return 0;
}

static int parse_dump_frames(const char* spec, int* list, int cap) {
 // "50,51,100,101" syntax; frame 0 skipped (bootstrap); range "a-b" allowed
 int n = 0; const char* p = spec;
 while (*p && n < cap) {
 char* end = NULL;
 long v = strtol(p, &end, 10);
 if (end == p) return -1;
 if (v < 0) return -1;
 if (*end == '-') {
 char* end2 = NULL;
 long v2 = strtol(end + 1, &end2, 10);
 if (end2 == end + 1 || v2 < v) return -1;
 end = end2;
 for (long k = v; k <= v2 && n < cap; k++)
 if (k > 0) list[n++] = (int)k;
 else fprintf(stderr, "fsr4_service: dump frame 0 (bootstrap) excluded\n");
 } else {
 if (v > 0) list[n++] = (int)v;
 else fprintf(stderr, "fsr4_service: dump frame 0 (bootstrap) excluded\n");
 }
 p = (*end == ',') ? end + 1: end;
 }
 return n;
}

// ---- GL helpers (ports; error-returning variants) ----
static GLuint svc_make_prog(const char* path, int* err) {
 char* src = load_text(path);
 if (!src) { *err = FSR4_ERR_INIT_SHADER; return 0; }
 GLuint cs = glCreateShader(GL_COMPUTE_SHADER);
 glShaderSource(cs, 1, (const GLchar* const*)&src, NULL);
 glCompileShader(cs);
 GLint ok = 0; glGetShaderiv(cs, GL_COMPILE_STATUS, &ok);
 if (!ok) { char lg[2048]; glGetShaderInfoLog(cs, 2048, NULL, lg);
 fprintf(stderr, "fsr4_service: compile %s:\n%s\n", path, lg);
 *err = FSR4_ERR_INIT_SHADER; free(src); return 0; }
 GLuint p = glCreateProgram(); glAttachShader(p, cs); glLinkProgram(p);
 glGetProgramiv(p, GL_LINK_STATUS, &ok);
 if (!ok) { char lg[2048]; glGetProgramInfoLog(p, 2048, NULL, lg);
 fprintf(stderr, "fsr4_service: link %s:\n%s\n", path, lg);
 *err = FSR4_ERR_INIT_SHADER; free(src); return 0; }
 free(src); *err = FSR4_OK; return p;
}
/* GL_EXT_buffer_storage constants (GLES3 headers don't ship them; spec values) */
#define GL_MAP_PERSISTENT_BIT_EXT_ 0x0040
#define GL_DYNAMIC_STORAGE_BIT_EXT_ 0x0100 /* required for SubData on immutable storage */
#define GL_MAP_COHERENT_BIT_EXT_ 0x0080
#define GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT_EXT_ 0x00004000

/* persistent READ-mapped SSBO (GL_EXT_buffer_storage). Adreno's transient
 * glMapBufferRange(READ) mappings come back uncached (~2.4GB/s — the pre_out
 * 8.29MB and out8 4.15MB reads were ~5ms/frame combined); persistent+coherent
 * maps are cacheable. Reads need GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT ordering
 * against the dispatches that wrote the buffer. */
typedef void (*glBufferStorageEXTPROC_t)(GLenum, GLsizeiptr, const void*, GLbitfield);
static glBufferStorageEXTPROC_t pglBufferStorageEXT;

/* GL_EXT_memory_object_fd (spec values) — dma-buf import for the NPU
 * bridges; typed pointers resolved like pglBufferStorageEXT above. */
#define GL_HANDLE_TYPE_OPAQUE_FD_EXT_ 0x8DA0
typedef void (*glCreateMemoryObjectsEXT_t)(GLsizei, GLuint*);
typedef void (*glImportMemoryFdEXT_t)(GLuint, int64_t, GLenum, GLint);
typedef void (*glBufferStorageMemEXT_t)(GLenum, int64_t, GLuint, GLuint64);
typedef void (*glMemoryObjectParameterivEXT_t)(GLuint, GLenum, const GLint*);
/* EGL_EXT_image_dma_buf_import plumbing (spec values) */
#define EGL_LINUX_DMA_BUF_EXT_ 0x3270
#define EGL_LINUX_DRM_FOURCC_EXT_ 0x3271
#define EGL_DMA_BUF_PLANE0_FD_EXT_ 0x3272
#define EGL_DMA_BUF_PLANE0_OFFSET_EXT_ 0x3273
#define EGL_DMA_BUF_PLANE0_PITCH_EXT_ 0x3274
#define DRM_FORMAT_ABGR8888_ 0x34324241 /* fourcc 'AB24' — byte0 -> .x */
typedef void* EGLImageKHR_;
typedef EGLImageKHR_ (*eglCreateImageKHR_t)(EGLDisplay, EGLContext, GLenum, EGLClientBuffer, const EGLint*);
typedef void (*glEGLImageTargetTexture2DOES_t)(GLenum, void*);
static eglCreateImageKHR_t pglCreateImageKHR_;
static glEGLImageTargetTexture2DOES_t pglEGLImageTargetTexture2DOES;
static glCreateMemoryObjectsEXT_t pglCreateMemoryObjectsEXT;
static glImportMemoryFdEXT_t pglImportMemoryFdEXT;
static glBufferStorageMemEXT_t pglBufferStorageMemEXT;
static glMemoryObjectParameterivEXT_t pglMemoryObjectParameterivEXT;
#define GL_DEDICATED_MEMORY_OBJECT_EXT_ 0x9581
static int svc_has_ext(const char* want) {
 GLint n = 0; glGetIntegerv(GL_NUM_EXTENSIONS, &n);
 for (GLint i = 0; i < n; i++) {
 const char* x = (const char*)glGetStringi(GL_EXTENSIONS, (GLuint)i);
 if (x && strcmp(x, want) == 0) return 1;
 }
 return 0;
}
static GLuint ssbo_create_persistent_write(size_t n, void** mapout) {
 GLuint buf; glGenBuffers(1, &buf);
 glBindBuffer(GL_SHADER_STORAGE_BUFFER, buf);
 pglBufferStorageEXT(GL_SHADER_STORAGE_BUFFER, (GLsizeiptr)n, NULL,
 GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT_EXT_ | GL_MAP_COHERENT_BIT_EXT_
 | GL_DYNAMIC_STORAGE_BIT_EXT_);
 GLenum ge = glGetError();
 void* m = (ge == GL_NO_ERROR)
 ? glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, (GLsizeiptr)n,
 GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT_EXT_ | GL_MAP_COHERENT_BIT_EXT_)
: NULL;
 if (!m) {
 GLenum e2 = glGetError();
 fprintf(stderr, "fsr4_service: persistent write map FAILED (storage 0x%x map 0x%x)\n", ge, e2);
 glDeleteBuffers(1, &buf);
 return 0;
 }
 glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
 *mapout = m;
 return buf;
}

static GLuint ssbo_create_persistent(size_t n, void** mapout) {
 GLuint buf; glGenBuffers(1, &buf);
 glBindBuffer(GL_SHADER_STORAGE_BUFFER, buf);
 pglBufferStorageEXT(GL_SHADER_STORAGE_BUFFER, (GLsizeiptr)n, NULL,
 GL_MAP_READ_BIT | GL_MAP_PERSISTENT_BIT_EXT_ | GL_MAP_COHERENT_BIT_EXT_
 | GL_DYNAMIC_STORAGE_BIT_EXT_);
 GLenum ge = glGetError();
 void* m = (ge == GL_NO_ERROR)
 ? glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, (GLsizeiptr)n,
 GL_MAP_READ_BIT | GL_MAP_PERSISTENT_BIT_EXT_ | GL_MAP_COHERENT_BIT_EXT_)
: NULL;
 if (!m) {
 GLenum e2 = glGetError(); /* drain the map's own sticky error or the fallback poisons init */
 fprintf(stderr, "fsr4_service: persistent map FAILED (storage 0x%x map 0x%x)\n", ge, e2);
 glDeleteBuffers(1, &buf);
 return 0;
 }
 glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
 *mapout = m;
 return buf;
}

static GLuint ssbo_create(const void* data, size_t n, bool dynamic) { GLuint buf; glGenBuffers(1, &buf);
 glBindBuffer(GL_SHADER_STORAGE_BUFFER, buf);
 if (data) {
 glBufferData(GL_SHADER_STORAGE_BUFFER, (GLsizeiptr)n, data, dynamic ? GL_DYNAMIC_COPY: GL_STATIC_DRAW);
 } else {
 void* z = calloc(1, n); // explicit zero-fill (A2): never trust NULL zeros
 if (z) { glBufferData(GL_SHADER_STORAGE_BUFFER, (GLsizeiptr)n, z, dynamic ? GL_DYNAMIC_COPY: GL_STATIC_DRAW); free(z); }
 }
 glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
 return buf;
}
static GLuint make_ssbo(uint32_t binding, const void* data, size_t n, bool dynamic) {
 GLuint buf = ssbo_create(data, n, dynamic);
 glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding, buf);
 return buf;
}
static GLuint make_tex_rgba16f(uint32_t unit, uint32_t w, uint32_t h, const void* data) {
 GLuint t; glGenTextures(1, &t);
 glActiveTexture(GL_TEXTURE0 + unit);
 glBindTexture(GL_TEXTURE_2D, t);
 glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, w, h, 0, GL_RGBA, GL_HALF_FLOAT, data);
 glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
 glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
 glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
 glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
 return t;
}

/* dimensions compile-time overridable (-DLR_W=640 -DLR_H=360 for the
 * 720p arm; OS = 2x LR by design). Defaults = production 540p->1080p. */
#ifndef LR_W
#define LR_W 960
#define LR_H 540
#define OS_W 1920
#define OS_H 1080
#endif
#define LR_BYTES ((size_t)LR_H * LR_W * 4 * 2) /* 4,147,200 RGBA16F */
#define MV_BYTES ((size_t)LR_H * LR_W * 2 * 2) /* 2,073,600 RG16F */
#define OS_BYTES ((size_t)OS_H * OS_W * 4 * 2) /* 16,588,800 RGBA16F */
#define COLOR_RAW_BYTES ((size_t)LR_W * LR_H * 4) /* 2,073,600 R11G11B10F */

static void release_gl(void);
static void release_qnn(void);

// seed hist + rec textures from the retained seed blobs (init + invalidate)
// Texture-unit invariant (harness parity: unit0=hist, unit1=rec,
// unit2=lrA; active unit restored to TEXTURE0. Binding rec while unit 0 is
// active silently retargets feat's u_hist sampler — the bug this once was.
static int seed_temporal(void) {
 if (!S.hist_seed_data || !S.rec_seed_data) return FSR4_ERR_INIT_SHADER;
 glActiveTexture(GL_TEXTURE0);
 glBindTexture(GL_TEXTURE_2D, S.hist_tex);
 glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, OS_W, OS_H, GL_RGBA, GL_HALF_FLOAT, S.hist_seed_data);
 glActiveTexture(GL_TEXTURE1);
 glBindTexture(GL_TEXTURE_2D, S.rec_tex);
 glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, OS_W, OS_H, GL_RGBA, GL_UNSIGNED_BYTE, S.rec_seed_data); /* rec is RGBA8 now (seed converted at load) */
 glActiveTexture(GL_TEXTURE0);
 /* the real feat shader reads the history from SSBO 10 — seed it
 * there too (the hist_tex upload above only serves the legacy shaders) */
 if (S.hst_ssbo) {
 glBindBuffer(GL_SHADER_STORAGE_BUFFER, S.hst_ssbo);
 glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, (GLsizeiptr)OS_BYTES, S.hist_seed_data);
 }
 { GLenum e0 = glGetError(); if (e0 != GL_NO_ERROR) { fprintf(stderr, "fsr4_service: seed_temporal early GL 0x%x\\n", e0); return FSR4_ERR_INIT_GL; } }
 // zero the ping-pong pre outputs + NPU input so the post-invalidate
 // bootstrap matches the fresh-init bootstrap (A2 zero-fill state)
 /* SSBO9 (rectified history) was the ONE
 * temporal buffer the reseed never touched — the first frame after every
 * reconnect blended a ghost of the pre-disconnect scene. */
 if (S.rpj_ssbo) {
 glBindBuffer(GL_SHADER_STORAGE_BUFFER, S.rpj_ssbo);
 void* zr = calloc(1, (size_t)OS_BYTES);
 if (zr) { glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, (GLsizeiptr)OS_BYTES, zr); free(zr); }
 }
 for (int i = 0; i < 2; i++) {
 glBindBuffer(GL_SHADER_STORAGE_BUFFER, S.pre_out_pp[i]);
 if (S.pre_map[i]) {
 /* immutable glBufferStorage storage — INVALIDATE maps are
 * illegal (GL_INVALID_OPERATION, broke init w/ rc=102); the A2
 * deterministic zero-fill goes through SubData instead */
 void* z = calloc(1, (size_t)LR_BYTES * 2);
 if (!z) return FSR4_ERR_INIT_GL;
 glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, (GLsizeiptr)(LR_BYTES * 2), z);
 free(z);
 { GLenum ez = glGetError(); if (ez != GL_NO_ERROR) { fprintf(stderr, "fsr4_service: prezero SubData GL 0x%x\\n", ez); return FSR4_ERR_INIT_GL; } }
 } else {
 void* m = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, (GLsizeiptr)LR_BYTES * 2, GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT);
 if (!m) return FSR4_ERR_INIT_GL;
 memset(m, 0, LR_BYTES * 2);
 glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
 }
 }
 dmabuf_sync(S.inB[0][0].fd, DMABUF_SYNC_RW | DMABUF_SYNC_START);
 memset(S.inB[0][0].ptr, 0, S.inB[0][0].nbytes);
 dmabuf_sync(S.inB[0][0].fd, DMABUF_SYNC_WRITE | DMABUF_SYNC_END);
 dmabuf_sync(S.inB[1][0].fd, DMABUF_SYNC_RW | DMABUF_SYNC_START);
 memset(S.inB[1][0].ptr, 0, S.inB[1][0].nbytes);
 dmabuf_sync(S.inB[1][0].fd, DMABUF_SYNC_WRITE | DMABUF_SYNC_END);
 S.first_frame = 1;
 return FSR4_OK;
}

int fsr4_init(const fsr4_init_params_t* p) {
 if (!p) return FSR4_ERR_STATE;
 if (p->contract_version != FSR4RP6_CONTRACT_VERSION) return FSR4_ERR_VERSION;
 if (S.inited) return FSR4_ERR_STATE;
 memset(&S, 0, sizeof(S)); // clean slate; preserves nothing across failed inits
 if (!p->backend_so || !p->system_so || !p->model_bin || !p->pre_shader ||
 !p->feat_shader || !p->post_shader || !p->weights_seed ||
 !p->hist_seed || !p->rec_seed)
 return FSR4_ERR_BUFFER;

 // ---- QNN ----
 S.rpc_lib = dlopen("libcdsprpc.so", RTLD_NOW | RTLD_LOCAL);
 if (!S.rpc_lib) return FSR4_ERR_INIT_MEM;
 S.p_alloc = (rpcmem_alloc_fn_t)dlsym(S.rpc_lib, "rpcmem_alloc");
 S.p_free = (rpcmem_free_fn_t)dlsym(S.rpc_lib, "rpcmem_free");
 S.p_tofd = (rpcmem_to_fd_fn_t)dlsym(S.rpc_lib, "rpcmem_to_fd");
 if (!S.p_alloc || !S.p_free || !S.p_tofd) { release_qnn(); return FSR4_ERR_INIT_MEM; }
 S.be_lib = dlopen(p->backend_so, RTLD_NOW | RTLD_LOCAL);
 S.sys_lib = dlopen(p->system_so, RTLD_NOW | RTLD_LOCAL);
 if (!S.be_lib || !S.sys_lib) { release_qnn(); return FSR4_ERR_INIT_QNN; }
 BackendGetProvidersFn_t bgp = (BackendGetProvidersFn_t)dlsym(S.be_lib, "QnnInterface_getProviders");
 SystemGetProvidersFn_t sgp = (SystemGetProvidersFn_t)dlsym(S.sys_lib, "QnnSystemInterface_getProviders");
 if (!bgp || !sgp) { release_qnn(); return FSR4_ERR_INIT_QNN; }
 QnnInterface_t** prov = NULL; uint32_t npv = 0;
 if (bgp((const QnnInterface_t***)&prov, &npv) != 0 || !prov || npv == 0) { release_qnn(); return FSR4_ERR_INIT_QNN; }
 S.iface = prov[0]->QNN_INTERFACE_VER_NAME;
 for (uint32_t i = 0; i < npv; i++)
 if (prov[i]->apiVersion.coreApiVersion.major == QNN_API_VERSION_MAJOR &&
 prov[i]->apiVersion.coreApiVersion.minor >= QNN_API_VERSION_MINOR)
 S.iface = prov[i]->QNN_INTERFACE_VER_NAME;
 if (S.iface.logCreate(NULL, QNN_LOG_LEVEL_ERROR, &S.logh) != 0) { release_qnn(); return FSR4_ERR_INIT_QNN; }
 if (S.iface.backendCreate(S.logh, NULL, &S.backend) != 0) { release_qnn(); return FSR4_ERR_INIT_QNN; }
 if (S.iface.deviceCreate(S.logh, NULL, &S.device) != 0) { release_qnn(); return FSR4_ERR_INIT_QNN; }

 // always-on power vote (corner from params)
 {
 QnnDevice_Infrastructure_t di = NULL;
 if (S.iface.deviceGetInfrastructure(&di) != 0 || di == NULL) { release_qnn(); return FSR4_ERR_INIT_QNN; }
 QnnHtpDevice_PerfInfrastructure_t* perf = &((QnnHtpDevice_Infrastructure_t*)di)->perfInfra;
 uint32_t pid = 0;
 if (perf->createPowerConfigId(0, 0, &pid) != 0) { release_qnn(); return FSR4_ERR_INIT_QNN; }
 QnnHtpPerfInfrastructure_PowerConfig_t dcvs, rl, rp;
 memset(&dcvs, 0, sizeof(dcvs)); memset(&rl, 0, sizeof(rl)); memset(&rp, 0, sizeof(rp));
 dcvs.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
 dcvs.dcvsV3Config.contextId = pid;
 dcvs.dcvsV3Config.setDcvsEnable = 1;
 dcvs.dcvsV3Config.dcvsEnable = 0;
 dcvs.dcvsV3Config.setSleepDisable = 1;
 dcvs.dcvsV3Config.sleepDisable = 1;
 dcvs.dcvsV3Config.setSleepLatency = 1;
 dcvs.dcvsV3Config.sleepLatency = 40;
 dcvs.dcvsV3Config.setBusParams = 1;
 dcvs.dcvsV3Config.busVoltageCornerMin = p->corner.bus_corner;
 dcvs.dcvsV3Config.busVoltageCornerTarget = p->corner.bus_corner;
 dcvs.dcvsV3Config.busVoltageCornerMax = p->corner.bus_corner;
 dcvs.dcvsV3Config.setCoreParams = 1;
 dcvs.dcvsV3Config.coreVoltageCornerMin = p->corner.core_corner;
 dcvs.dcvsV3Config.coreVoltageCornerTarget = p->corner.core_corner;
 dcvs.dcvsV3Config.coreVoltageCornerMax = p->corner.core_corner;
 rl.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_CONTROL_LATENCY;
 rl.rpcControlLatencyConfig = 100;
 rp.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_POLLING_TIME;
 rp.rpcPollingTimeConfig = 9999;
 const QnnHtpPerfInfrastructure_PowerConfig_t* cfgs[4] = {&dcvs, &rl, &rp, NULL};
 if (perf->setPowerConfig(pid, cfgs) != 0) { release_qnn(); return FSR4_ERR_INIT_QNN; }
 S.have_vote = 1;
 printf("fsr4_service: HTP vote core=0x%x bus=0x%x (always-on)\n",
 p->corner.core_corner, p->corner.bus_corner);
 }

 {
 QnnSystemInterface_t** sprov = NULL; uint32_t nsp = 0;
 if (sgp((const QnnSystemInterface_t***)&sprov, &nsp) != 0 || !sprov || nsp == 0) { release_qnn(); return FSR4_ERR_INIT_QNN; }
 sysiface_t sysIface = sprov[0]->QNN_SYSTEM_INTERFACE_VER_NAME;
 QnnSystemContext_Handle_t sysCtx = NULL;
 uint32_t numGraphs = 0; const QnnSystemContext_GraphInfo_t* graphs = NULL;
 void* ctx_buf = NULL; size_t ctx_size = 0;
 if (sysIface.systemContextCreate(&sysCtx) != 0) { release_qnn(); return FSR4_ERR_INIT_QNN; }
 if (load_file(p->model_bin, &ctx_buf, &ctx_size) != 0) { release_qnn(); return FSR4_ERR_INIT_MODEL; }
 const QnnSystemContext_BinaryInfo_t* bi = NULL; Qnn_ContextBinarySize_t bis = 0;
 Qnn_ErrorHandle_t ge = sysIface.systemContextGetBinaryInfo(sysCtx, ctx_buf, ctx_size, &bi, &bis);
 if (ge != 0 || bi == NULL) { free(ctx_buf); release_qnn(); return FSR4_ERR_INIT_MODEL; }
 switch (bi->version) {
 case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1: graphs = bi->contextBinaryInfoV1.graphs; numGraphs = bi->contextBinaryInfoV1.numGraphs; break;
 case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2: graphs = bi->contextBinaryInfoV2.graphs; numGraphs = bi->contextBinaryInfoV2.numGraphs; break;
 case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3: graphs = bi->contextBinaryInfoV3.graphs; numGraphs = bi->contextBinaryInfoV3.numGraphs; break;
 default: graphs = NULL;
 }
 if (!graphs || numGraphs < 1) { free(ctx_buf); release_qnn(); return FSR4_ERR_INIT_MODEL; }
 if (S.iface.contextCreateFromBinary(S.backend, S.device, NULL, ctx_buf, ctx_size, &S.context, NULL) != 0) {
 free(ctx_buf); release_qnn(); return FSR4_ERR_INIT_QNN;
 }
 char gname_buf[256];
 snprintf(gname_buf, sizeof(gname_buf), "%s", graphs[0].graphInfoV1.graphName);
 free(ctx_buf); // QNN owns BinaryInfo independently (16/16 GE2 empirically); copy defensively
 const char* gname = gname_buf;
 S.nIn = graphs[0].graphInfoV1.numGraphInputs;
 S.nOut = graphs[0].graphInfoV1.numGraphOutputs;
 if (S.nIn > 2 || S.nOut > 2) { /* inB/outB/inT are [2][2] */
 fprintf(stderr, "fsr4_service: graph has %d in / %d out (max 2/2)\n", (int)S.nIn, (int)S.nOut);
 release_gl(); release_qnn(); return FSR4_ERR_INIT_QNN;
 }
 S.in_info = copy_tensors(graphs[0].graphInfoV1.graphInputs, S.nIn);
 S.out_info = copy_tensors(graphs[0].graphInfoV1.graphOutputs, S.nOut);
 sysIface.systemContextFree(sysCtx);
 if (!S.in_info || !S.out_info) { release_qnn(); return FSR4_ERR_INIT_QNN; }
 printf("fsr4_service: graph '%s': %u in, %u out\n", gname, S.nIn, S.nOut);
 /* diagnosis: dump the graph's REAL dtypes + quantization encodings
 * (our shader dequant is signed*scale with NO zero-point — verify against
 * what the graph actually declares) */
 for (uint32_t i = 0; i < S.nIn; i++) {
 const Qnn_Tensor_t* t = &S.in_info[i];
 if (t->version == QNN_TENSOR_VERSION_1)
 printf("fsr4_service: IN[%u] dtype=0x%x enc=%d scale=%g offset=%d\n", i,
 (unsigned)t->v1.dataType, (int)t->v1.quantizeParams.quantizationEncoding,
 (t->v1.quantizeParams.quantizationEncoding == QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) ? (double)t->v1.quantizeParams.scaleOffsetEncoding.scale: -1.0,
 (t->v1.quantizeParams.quantizationEncoding == QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) ? (int)t->v1.quantizeParams.scaleOffsetEncoding.offset: -1);
 else if (t->v2.quantizeParams.quantizationEncoding == QNN_QUANTIZATION_ENCODING_SCALE_OFFSET)
 printf("fsr4_service: IN[%u] v2 dtype=0x%x SCALE_OFFSET scale=%g offset=%d\n", i,
 (unsigned)t->v2.dataType, (double)t->v2.quantizeParams.scaleOffsetEncoding.scale,
 (int)t->v2.quantizeParams.scaleOffsetEncoding.offset);
 else
 printf("fsr4_service: IN[%u] v2 dtype=0x%x enc=%d\n", i,
 (unsigned)t->v2.dataType, (int)t->v2.quantizeParams.quantizationEncoding);
 }
 for (uint32_t i = 0; i < S.nOut; i++) {
 const Qnn_Tensor_t* t = &S.out_info[i];
 if (t->version == QNN_TENSOR_VERSION_1)
 printf("fsr4_service: OUT[%u] dtype=0x%x enc=%d scale=%g offset=%d\n", i,
 (unsigned)t->v1.dataType, (int)t->v1.quantizeParams.quantizationEncoding,
 (t->v1.quantizeParams.quantizationEncoding == QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) ? (double)t->v1.quantizeParams.scaleOffsetEncoding.scale: -1.0,
 (t->v1.quantizeParams.quantizationEncoding == QNN_QUANTIZATION_ENCODING_SCALE_OFFSET) ? (int)t->v1.quantizeParams.scaleOffsetEncoding.offset: -1);
 else if (t->v2.quantizeParams.quantizationEncoding == QNN_QUANTIZATION_ENCODING_SCALE_OFFSET)
 printf("fsr4_service: OUT[%u] v2 dtype=0x%x SCALE_OFFSET scale=%g offset=%d\n", i,
 (unsigned)t->v2.dataType, (double)t->v2.quantizeParams.scaleOffsetEncoding.scale,
 (int)t->v2.quantizeParams.scaleOffsetEncoding.offset);
 else
 printf("fsr4_service: OUT[%u] v2 dtype=0x%x enc=%d\n", i,
 (unsigned)t->v2.dataType, (int)t->v2.quantizeParams.quantizationEncoding);
 }
 if (S.iface.graphRetrieve(S.context, gname, &S.graph) != 0) { release_qnn(); return FSR4_ERR_INIT_QNN; }
 }

 // two registered ping-pong slots
 S.ahb_in = (getenv("FSR4_AHBCOPY") && atoi(getenv("FSR4_AHBCOPY"))) ? 1: 0;
 S.ahb_in2 = (getenv("FSR4_AHBIN") && atoi(getenv("FSR4_AHBIN"))) ? 1: 0;
 S.ahb_sync = getenv("FSR4_AHBSYNC") ? atoi(getenv("FSR4_AHBSYNC")): 1;
 S.spin_ms = getenv("FSR4_SPIN_MS") ? atof(getenv("FSR4_SPIN_MS")): 0.0;
 S.sentinel_n = getenv("FSR4_SENTINEL") ? atoi(getenv("FSR4_SENTINEL")): 0;
 for (uint32_t i = 0; i < S.nIn; i++) S.inT[0][i] = S.in_info[i];
 for (uint32_t i = 0; i < S.nOut; i++) S.outT[0][i] = S.out_info[i];
 for (uint32_t i = 0; i < S.nIn; i++) S.inT[1][i] = S.in_info[i];
 for (uint32_t i = 0; i < S.nOut; i++) S.outT[1][i] = S.out_info[i];
 for (uint32_t s = 0; s < 2; s++) {
 for (uint32_t i = 0; i < S.nIn; i++) {
 /* FSR4_AHBIN — in tensor 0 from gralloc (H=540: 8,294,400 B
 * = 3840x540x4); fallback to rpcmem on failure. */
 if (S.ahb_in2 && i == 0) {
 if (register_mem_buf_ahb(S.context, &S.inT[s][i], &S.inB[s][i], &S.ahb_inkeep[s], LR_H) != 0) {
 fprintf(stderr, "fsr4_service: AHB in slot %u FAILED - rpcmem fallback\n", s);
 S.ahb_in2 = 0;
 }
 }
 if (!(S.ahb_in2 && i == 0 && S.ahb_inkeep[s]))
 if (register_mem_buf(S.context, &S.inT[s][i], &S.inB[s][i]) != 0) { release_qnn(); return FSR4_ERR_INIT_MEM; }
 }
 for (uint32_t i = 0; i < S.nOut; i++) {
 /* FSR4_AHBCOPY — out tensor 0 from gralloc (GL reads it
 * via EGLImage; QNN via DMA_BUF). Any failure falls back to
 * rpcmem for that slot (and disables the GL read path). */
 if (S.ahb_in && i == 0) {
 if (register_mem_buf_ahb(S.context, &S.outT[s][i], &S.outB[s][i], &S.ahb_out[s], OS_H) != 0) {
 fprintf(stderr, "fsr4_service: AHB out slot %u FAILED - rpcmem fallback\n", s);
 S.ahb_in = 0;
 }
 }
 if (!(S.ahb_in && i == 0 && S.ahb_out[s]))
 if (register_mem_buf(S.context, &S.outT[s][i], &S.outB[s][i]) != 0) { release_qnn(); return FSR4_ERR_INIT_MEM; }
 }
 }
 for (uint32_t s = 0; s < 2; s++)
 for (uint32_t i = 0; i < S.nIn; i++) {
 if (S.inT[s][i].version == QNN_TENSOR_VERSION_1) { S.inT[s][i].v1.memType = QNN_TENSORMEMTYPE_MEMHANDLE; S.inT[s][i].v1.memHandle = S.inB[s][i].handle; }
 else { S.inT[s][i].v2.memType = QNN_TENSORMEMTYPE_MEMHANDLE; S.inT[s][i].v2.memHandle = S.inB[s][i].handle; }
 }
 for (uint32_t s = 0; s < 2; s++)
 for (uint32_t i = 0; i < S.nOut; i++) {
 if (S.outT[s][i].version == QNN_TENSOR_VERSION_1) { S.outT[s][i].v1.memType = QNN_TENSORMEMTYPE_MEMHANDLE; S.outT[s][i].v1.memHandle = S.outB[s][i].handle; }
 else { S.outT[s][i].v2.memType = QNN_TENSORMEMTYPE_MEMHANDLE; S.outT[s][i].v2.memHandle = S.outB[s][i].handle; }
 }
 printf("fsr4_service: rpcmem slots registered (%zu B in, %zu B out)\n", S.inB[0][0].nbytes, S.outB[0][0].nbytes);

 // ---- GL (EGL pbuffer, same thread) ----
 if (eglBindAPI(EGL_OPENGL_ES_API) != EGL_TRUE) { release_gl(); release_qnn(); return FSR4_ERR_INIT_GL; }
 S.dpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);
 if (S.dpy == EGL_NO_DISPLAY || eglInitialize(S.dpy, NULL, NULL) != EGL_TRUE) { release_gl(); release_qnn(); return FSR4_ERR_INIT_GL; }
 const EGLint cfga[] = { EGL_SURFACE_TYPE, EGL_PBUFFER_BIT, EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT, EGL_NONE };
 EGLConfig cfg; EGLint ncfg = 0;
 if (eglChooseConfig(S.dpy, cfga, &cfg, 1, &ncfg) != EGL_TRUE || ncfg < 1) { release_gl(); release_qnn(); return FSR4_ERR_INIT_GL; }
 const EGLint pb[] = { EGL_WIDTH, 1, EGL_HEIGHT, 1, EGL_NONE };
 S.surf = eglCreatePbufferSurface(S.dpy, cfg, pb);
 if (S.surf == EGL_NO_SURFACE) { release_gl(); release_qnn(); return FSR4_ERR_INIT_GL; }
 static const EGLint c32[] = { EGL_CONTEXT_MAJOR_VERSION, 3, EGL_CONTEXT_MINOR_VERSION, 2, EGL_NONE };
 static const EGLint c31[] = { EGL_CONTEXT_MAJOR_VERSION, 3, EGL_CONTEXT_MINOR_VERSION, 1, EGL_NONE };
 S.ctx = eglCreateContext(S.dpy, cfg, EGL_NO_CONTEXT, c32);
 if (S.ctx == EGL_NO_CONTEXT) S.ctx = eglCreateContext(S.dpy, cfg, EGL_NO_CONTEXT, c31);
 if (S.ctx == EGL_NO_CONTEXT || eglMakeCurrent(S.dpy, S.surf, S.surf, S.ctx) != EGL_TRUE) { release_gl(); release_qnn(); return FSR4_ERR_INIT_GL; }
 printf("fsr4_service: GL: %s\n", glGetString(GL_RENDERER));
 /* persistent read-maps need GL_EXT_buffer_storage */
 S.have_bstorage = svc_has_ext("GL_EXT_buffer_storage");
 if (S.have_bstorage) {
 pglBufferStorageEXT = (glBufferStorageEXTPROC_t)eglGetProcAddress("glBufferStorageEXT");
 if (!pglBufferStorageEXT) S.have_bstorage = 0;
 }
 printf("fsr4_service: GL_EXT_buffer_storage %s\n",
 S.have_bstorage ? "ON (persistent read maps)": "absent (transient map reads)");
 /* bridge probe: dma-buf import for the NPU bridges */
 S.have_memfd = svc_has_ext("GL_EXT_memory_object_fd") && svc_has_ext("GL_EXT_memory_object");
 if (S.have_memfd) {
 pglCreateMemoryObjectsEXT = (glCreateMemoryObjectsEXT_t)eglGetProcAddress("glCreateMemoryObjectsEXT");
 pglImportMemoryFdEXT = (glImportMemoryFdEXT_t)eglGetProcAddress("glImportMemoryFdEXT");
 pglBufferStorageMemEXT = (glBufferStorageMemEXT_t)eglGetProcAddress("glBufferStorageMemEXT");
 pglMemoryObjectParameterivEXT = (glMemoryObjectParameterivEXT_t)eglGetProcAddress("glMemoryObjectParameterivEXT");
 if (!pglCreateMemoryObjectsEXT || !pglImportMemoryFdEXT || !pglBufferStorageMemEXT || !pglMemoryObjectParameterivEXT) S.have_memfd = 0;
 }
 printf("fsr4_service: GL_EXT_memory_object_fd %s\n",
 S.have_memfd ? "present (dmabuf bridge possible)": "MISSING (copy2 stays)");
 /* AHB probe (FSR4_AHBPROBE=1) — the last bridge route: gralloc
 * AHB -> EGLImage (guaranteed path) + /proc/self/fd scan to find the
 * backing dma-buf for QNN DMA_BUF registration. Diagnostic only. */
 if (getenv("FSR4_AHBPROBE") && atoi(getenv("FSR4_AHBPROBE"))) {
 AHardwareBuffer_Desc abd = {0};
 abd.width = 3840; abd.height = 1080; abd.layers = 1;
 abd.format = AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM;
 abd.usage = AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE | AHARDWAREBUFFER_USAGE_GPU_COLOR_OUTPUT | AHARDWAREBUFFER_USAGE_CPU_READ_OFTEN;
 int fds_before[4096]; int nb = 0;
 DIR* dp = opendir("/proc/self/fd");
 if (dp) { struct dirent* de; while ((de = readdir(dp)) && nb < 4096) { int f = atoi(de->d_name); if (f > 2) fds_before[nb++] = f; } closedir(dp); }
 AHardwareBuffer* ahb = NULL;
 int arc = AHardwareBuffer_allocate(&abd, &ahb);
 printf("fsr4_service: AHBPROBE allocate 3840x1080 RGBA8 rc=%d\n", arc);
 if (arc == 0 && ahb) {
 AHardwareBuffer_Desc gb; AHardwareBuffer_describe(ahb, &gb);
 printf("fsr4_service: AHBPROBE stride=%u (native %u), size would be %lld B\n",
 gb.stride, abd.width, (long long)gb.stride * 1080 * 4);
 dp = opendir("/proc/self/fd");
 if (dp) {
 struct dirent* de; int found = 0;
 while ((de = readdir(dp))) {
 int f = atoi(de->d_name); if (f <= 2) continue;
 int known = 0;
 for (int i = 0; i < nb; i++) if (fds_before[i] == f) { known = 1; break; }
 if (known) continue;
 off_t sz = lseek(f, 0, SEEK_END);
 char pbuf[256]; snprintf(pbuf, sizeof pbuf, "/proc/self/fd/%d", f); ssize_t tl = readlink(pbuf, pbuf, 200);
 if (tl > 0) pbuf[tl] = 0; else pbuf[0] = 0;
 printf("fsr4_service: AHBPROBE NEW fd %d size %lld target %s\n", f, (long long)sz, pbuf);
 found++;
 }
 closedir(dp);
 if (!found) printf("fsr4_service: AHBPROBE no NEW fds visible (handle privatized?)\n");
 }
 /* GL attach probe: EGLImage from the AHB -> texture -> image2D */
 while (glGetError() != GL_NO_ERROR) {}
 void* (*pGetNcb)(const AHardwareBuffer*) = (void* (*)(const AHardwareBuffer*))eglGetProcAddress("eglGetNativeClientBufferANDROID");
 if (pGetNcb && pglEGLImageTargetTexture2DOES) {
 EGLClientBuffer cb = pGetNcb(ahb);
 EGLImageKHR_ img2 = pglCreateImageKHR_(S.dpy, EGL_NO_CONTEXT, 0x3140 /* EGL_NATIVE_BUFFER_ANDROID */, cb, NULL);
 GLuint t; glGenTextures(1, &t);
 glBindTexture(GL_TEXTURE_2D, t);
 glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
 pglEGLImageTargetTexture2DOES(GL_TEXTURE_2D, img2);
 GLenum ge = glGetError();
 glBindTexture(GL_TEXTURE_2D, 0);
 printf("fsr4_service: AHBPROBE EGLImage %s, target-texture GL 0x%x (0 = clean)\n",
 img2 ? "OK": "FAILED", ge);
 } else {
 printf("fsr4_service: AHBPROBE eglGetNativeClientBufferANDROID unavailable\n");
 }
 AHardwareBuffer_release(ahb);
 }
 }
 /* EGL dma-buf TEXTURE import for outB (copy2 bridge, texture path) */
 /* attach the AHB out bufs as GL images — postA imageLoads outB
 * through posta_egl.comp (image unit 4). The probe hang was a NULL
 * pglCreateImageKHR_ call (resolved only under FSR4_EGLIN); resolve
 * everything HERE, before any use. On failure: ahb GL read disabled,
 * copy2 falls back to the mmap'd ptr (correct, just slower). */
 if (S.ahb_in) {
 pglCreateImageKHR_ = (eglCreateImageKHR_t)eglGetProcAddress("eglCreateImageKHR");
 pglEGLImageTargetTexture2DOES = (glEGLImageTargetTexture2DOES_t)eglGetProcAddress("glEGLImageTargetTexture2DOES");
 void* (*pGetNcb)(const AHardwareBuffer*) = (void* (*)(const AHardwareBuffer*))eglGetProcAddress("eglGetNativeClientBufferANDROID");
 if (!pglCreateImageKHR_ || !pglEGLImageTargetTexture2DOES || !pGetNcb) {
 fprintf(stderr, "fsr4_service: AHB GL attach fns unresolved - copy2 mmap fallback\n");
 S.ahb_in = 0;
 } else {
 for (int q = 0; q < 2; q++) {
 while (glGetError() != GL_NO_ERROR) {}
 EGLClientBuffer cb = pGetNcb(S.ahb_out[q]);
 EGLImageKHR_ img = (cb) ? pglCreateImageKHR_(S.dpy, EGL_NO_CONTEXT, 0x3140 /* EGL_NATIVE_BUFFER_ANDROID */, cb, NULL): NULL;
 if (!img) { fprintf(stderr, "fsr4_service: AHB EGLImage slot%d FAILED - copy2 mmap fallback\n", q); S.ahb_in = 0; break; }
 glGenTextures(1, &S.npuimg_tex[q]);
 glBindTexture(GL_TEXTURE_2D, S.npuimg_tex[q]);
 glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
 glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
 pglEGLImageTargetTexture2DOES(GL_TEXTURE_2D, img);
 GLenum ge = glGetError();
 glBindTexture(GL_TEXTURE_2D, 0);
 if (ge != GL_NO_ERROR) { fprintf(stderr, "fsr4_service: AHB target-texture slot%d GL 0x%x - copy2 mmap fallback\n", q, ge); S.ahb_in = 0; break; }
 }
 printf("fsr4_service: AHB GL attach %s\n",
 S.ahb_in ? "OK - copy2 eliminated (postA imageLoads the AHB)": "FAILED - copy2 mmap fallback");
 }
 }
 /* attach the inB AHBs as GL images (image unit 5) — feat
 * imageStores the NPU input directly (features_fused_egl.comp). */
 if (S.ahb_in2 && S.ahb_in) {
 void* (*pGetNcb2)(const AHardwareBuffer*) = (void* (*)(const AHardwareBuffer*))eglGetProcAddress("eglGetNativeClientBufferANDROID");
 if (!pGetNcb2 || !pglCreateImageKHR_ || !pglEGLImageTargetTexture2DOES) {
 fprintf(stderr, "fsr4_service: AHB-in GL attach fns unresolved - rpcmem copy1 stays\n");
 S.ahb_in2 = 0;
 } else {
 for (int q = 0; q < 2; q++) {
 while (glGetError() != GL_NO_ERROR) {}
 EGLClientBuffer cb = pGetNcb2(S.ahb_inkeep[q]);
 EGLImageKHR_ img = (cb) ? pglCreateImageKHR_(S.dpy, EGL_NO_CONTEXT, 0x3140, cb, NULL): NULL;
 if (!img) { fprintf(stderr, "fsr4_service: AHB-in EGLImage slot%d FAILED - copy1 stays\n", q); S.ahb_in2 = 0; break; }
 glGenTextures(1, &S.in_img_tex[q]);
 glBindTexture(GL_TEXTURE_2D, S.in_img_tex[q]);
 glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
 glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
 pglEGLImageTargetTexture2DOES(GL_TEXTURE_2D, img);
 GLenum ge = glGetError();
 glBindTexture(GL_TEXTURE_2D, 0);
 if (ge != GL_NO_ERROR) { fprintf(stderr, "fsr4_service: AHB-in target slot%d GL 0x%x - copy1 stays\n", q, ge); S.ahb_in2 = 0; break; }
 }
 printf("fsr4_service: AHB-in GL attach %s\n",
 S.ahb_in2 ? "OK - copy1 eliminated (feat imageStores the tensor)": "FAILED - copy1 stays");
 }
 } else if (S.ahb_in2 && !S.ahb_in) {
 fprintf(stderr, "fsr4_service: FSR4_AHBIN requires FSR4_AHBCOPY (fn resolution) - disabled\n");
 S.ahb_in2 = 0;
 }
 /* sentinel probe — GPU-view vs mmap-view byte comparison every
 * Nth frame (catches coherent stale frames; see sentinel.comp) */
 if (S.ahb_in && S.sentinel_n > 0) {
 int e5 = 0;
 S.sent_prog = svc_make_prog("sentinel.comp", &e5);
 if (!e5) {
 S.sent_ssbo = make_ssbo(16, NULL, 256, true);
 printf("fsr4_service: sentinel ON (every %d frames)\n", S.sentinel_n);
 } else {
 S.sentinel_n = 0;
 fprintf(stderr, "fsr4_service: sentinel.comp compile failed - off\n");
 }
 }
 S.egl_in = getenv("FSR4_EGLIN") ? (atoi(getenv("FSR4_EGLIN")) & 1): 0;
 if (S.egl_in) {
 const char* eex = eglQueryString(S.dpy, EGL_EXTENSIONS);
 int ehas = (eex && strstr(eex, "EGL_EXT_image_dma_buf_import")) ? 1: 0;
 int ghas = svc_has_ext("GL_OES_EGL_image");
 pglCreateImageKHR_ = (eglCreateImageKHR_t)eglGetProcAddress("eglCreateImageKHR");
 pglEGLImageTargetTexture2DOES = (glEGLImageTargetTexture2DOES_t)eglGetProcAddress("glEGLImageTargetTexture2DOES");
 if (!ehas || !ghas || !pglCreateImageKHR_ || !pglEGLImageTargetTexture2DOES) {
 printf("fsr4_service: EGL dma-buf import unavailable (ext %d, oes %d, fns %d/%d) - copy2 stays\n",
 ehas, ghas, pglCreateImageKHR_ != NULL, pglEGLImageTargetTexture2DOES != NULL);
 S.egl_in = 0;
 } else {
 for (int q = 0; q < 2; q++) {
 while (glGetError() != GL_NO_ERROR) {}
 EGLint attrs[] = {
 EGL_WIDTH, 2 * OS_W, EGL_HEIGHT, OS_H,
 EGL_LINUX_DRM_FOURCC_EXT_, DRM_FORMAT_ABGR8888_,
 EGL_DMA_BUF_PLANE0_FD_EXT_, S.outB[q][0].fd,
 EGL_DMA_BUF_PLANE0_OFFSET_EXT_, 0,
 EGL_DMA_BUF_PLANE0_PITCH_EXT_, 2 * OS_W * 4,
 EGL_NONE };
 EGLImageKHR_ img = pglCreateImageKHR_(S.dpy, EGL_NO_CONTEXT, EGL_LINUX_DMA_BUF_EXT_, NULL, attrs);
 if (!img) { fprintf(stderr, "fsr4_service: EGLImage outB slot%d FAILED - copy2 stays\n", q); S.egl_in = 0; break; }
 glGenTextures(1, &S.npuimg_tex[q]);
 glBindTexture(GL_TEXTURE_2D, S.npuimg_tex[q]);
 glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
 glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
 pglEGLImageTargetTexture2DOES(GL_TEXTURE_2D, img);
 GLenum ge = glGetError();
 glBindTexture(GL_TEXTURE_2D, 0);
 if (ge != GL_NO_ERROR) { fprintf(stderr, "fsr4_service: target-texture outB slot%d FAILED GL 0x%x - copy2 stays\n", q, ge); S.egl_in = 0; break; }
 }
 printf("fsr4_service: EGL dma-buf import %s\n",
 S.egl_in ? "OK - copy2 eliminated (postA imageLoads outB)": "FAILED - copy2 stays");
 }
 }

 int e = FSR4_OK;
 S.pre_prog = svc_make_prog(p->pre_shader, &e);
 if (e) { release_gl(); release_qnn(); return e; }
 S.feat_prog = svc_make_prog(p->feat_shader, &e);
 if (e) { release_gl(); release_qnn(); return e; }
 S.post_prog = svc_make_prog(p->post_shader, &e);
 if (e) { release_gl(); release_qnn(); return e; }
 S.mulr_prog = svc_make_prog("mulr.comp", &e); // mu-law LR precompute
 if (e) { release_gl(); release_qnn(); return e; }
 /* RCAS optional final sharpen (rcas.comp); FSR4_RCAS=sharpness
 * [0,1] (0/unset = off). */
 S.rcas_sharp = 0.0f;
 {
 const char* e2 = getenv("FSR4_RCAS");
 if (e2) S.rcas_sharp = (float)atof(e2);
 }
 if (S.rcas_sharp > 0.0f) {
 /* FSR4_RCASSRC picks the rcas shader file (default rcas.comp) —
 * the LDS-tile keep-or-revert A/B without a rebuild */
 const char* rs = getenv("FSR4_RCASSRC");
 S.rcas_prog = svc_make_prog((rs && *rs) ? rs: "rcas.comp", &e);
 if (e) { release_gl(); release_qnn(); return e; }
 }
 S.expo = 1.0f;
 S.reset = 0;
 /* NPU-thread split — requires RCAS (the postB tail IS rcas
 * reading SSBO 10; no separate postb kernel exists). */
 S.split_mode = 0;
 {
 const char* e2 = getenv("FSR4_SPLIT");
 if (e2 && atoi(e2)) {
 if (S.rcas_sharp > 0.0f) {
 S.split_mode = 1;
 S.post_prog = svc_make_prog((S.egl_in || S.ahb_in) ? "posta_egl.comp": "posta.comp", &e);
 if (e) { release_gl(); release_qnn(); return e; }
 if (sem_init(&S.npu_submit_sem, 0, 0) != 0 || sem_init(&S.npu_done_sem, 0, 0) != 0) {
 release_gl(); release_qnn(); return FSR4_ERR_INIT_MEM;
 }
 if (pthread_create(&S.npu_thr, NULL, npu_thread, NULL) != 0) {
 S.split_mode = 0; /* fallback left split_mode=1 with destroyed semaphores */
 fprintf(stderr, "fsr4_service: npu thread create failed — split off\n");
 sem_destroy(&S.npu_submit_sem); sem_destroy(&S.npu_done_sem);
 S.post_prog = svc_make_prog(p->post_shader, &e);
 if (e) { release_gl(); release_qnn(); return e; }
 }
 memset((void*)&S.tail, 0, sizeof S.tail);
 } else {
 fprintf(stderr, "fsr4_service: FSR4_SPLIT=1 needs FSR4_RCAS>0 — ignored\n");
 }
 }
 }
 S.ul_post_expo = glGetUniformLocation(S.post_prog, "u_expo");
 glUseProgram(S.post_prog);
 glUniform1f(S.ul_post_expo, S.expo);
 glUseProgram(0);
 printf("fsr4_service: rcas=%.2f exposure=1.0 split=%d rawlr=%d\n", S.rcas_sharp, S.split_mode, S.raw_lr);
 /* FSR4_FUSE=1 — replace feat with the fused feat+pre0 kernel
 * (dispatched at LR res, writes pre_out directly; the separate pre0
 * dispatch is skipped in the split path). SPLIT-ONLY: the legacy
 * non-split execute would run pre0 after the fused kernel and clobber
 * pre_out from the stale SSBO 0. Placed BEFORE the name-based uniform
 * queries below so locations resolve against the fused program
 * (uniform names are identical by contract). */
 S.fuse = 0;
 S.raw_lr = (getenv("FSR4_RAWLR") && atoi(getenv("FSR4_RAWLR"))) ? 1: 0;
 {
 const char* e2 = getenv("FSR4_FUSE");
 if (e2 && atoi(e2)) {
 if (S.split_mode && S.raw_lr) { /* v2 shaders read RAW SSBO5 */
 S.fuse = 1;
 S.feat_prog = svc_make_prog(S.ahb_in2 ? "features_fused_egl.comp": "features_fused.comp", &e);
 if (e) { release_gl(); release_qnn(); return e; }
 printf("fsr4_service: FUSE on (features_fused v2, raw LR, mulr deleted)\n");
 S.ul_feat_expo = glGetUniformLocation(S.feat_prog, "u_expo");
 S.ul_feat_bftau = glGetUniformLocation(S.feat_prog, "u_bftau");
 glUseProgram(S.feat_prog);
 glUniform1f(S.ul_feat_expo, S.expo);
 glUniform1f(S.ul_feat_bftau, S.bf_tau);
 glUseProgram(0);
 } else {
 fprintf(stderr, "fsr4_service: FSR4_FUSE=1 needs FSR4_SPLIT=1 and FSR4_RAWLR=1 — ignored\n");
 }
 }
 }

 size_t n = 0; void* d = NULL;
 // feats SSBO 0: zero-filled — the harness's features.raw seed is dead data
 // (feat fully rewrites the buffer before any read).
 // 4 words/texel x 4 B = 16 B/texel = 33,177,600 B (harness-exact size).
 S.bf = make_ssbo(0, NULL, (size_t)OS_W * OS_H * 16, false);
 if (load_file(p->weights_seed, &d, &n) != 0) { release_gl(); release_qnn(); return FSR4_ERR_INIT_SHADER; }
 S.wbuf_ssbo = make_ssbo(2, d, n, false); free(d);
 S.npremap = (getenv("FSR4_NOPREMAP") && atoi(getenv("FSR4_NOPREMAP"))) ? 1: 0;
 if (S.npremap)
 printf("fsr4_service: NOPREMAP=1 (for-the-record run: pre_out transient maps, copy1 via glMapBufferRange)\n");
 if (S.have_bstorage && !S.npremap) {
 S.pre_out_pp[0] = ssbo_create_persistent((size_t)LR_BYTES * 2, &S.pre_map[0]);
 S.pre_out_pp[1] = ssbo_create_persistent((size_t)LR_BYTES * 2, &S.pre_map[1]);
 }
 if (!S.pre_out_pp[0]) S.pre_out_pp[0] = ssbo_create(NULL, (size_t)LR_BYTES * 2, true); // zero-filled (A2); RGBA f32 = LR_BYTES*2
 if (!S.pre_out_pp[1]) S.pre_out_pp[1] = ssbo_create(NULL, (size_t)LR_BYTES * 2, true);
 if (S.have_bstorage) {
 S.post_in = ssbo_create_persistent_write(OS_BYTES, &S.post_in_map);
 if (S.post_in) glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, S.post_in);
 }
 if (!S.post_in) S.post_in = make_ssbo(4, NULL, OS_BYTES, true);
 /* import outB[slot] dma-bufs as per-slot SSBO4 targets — postA
 * reads the NPU output tensor with zero copy (copy2's 16.6MB WC memcpy
 * + ioctls die). glImportMemoryFdEXT CONSUMES the fd on success, so
 * dup() first — the fallback dmabuf_sync() path keeps its valid fd.
 * Order per EXT_memory_object: create -> import -> bind -> storageMem.
 * Any failure => dmabuf_in=0, the copy2 path is untouched. */
 S.dmabuf_in = (getenv("FSR4_DMABUF") && atoi(getenv("FSR4_DMABUF")) && S.have_memfd) ? 1: 0;
 if (S.dmabuf_in) {
 for (int q = 0; q < 2; q++) {
 /* the import 'size' is the ALLOCATION size, not the tensor
 * nbytes — dma-bufs report theirs via lseek(SEEK_END). The bare
 * 0x501 on the first attempt pointed here (rpcmem pads). */
 off_t dbsz = lseek(S.outB[q][0].fd, 0, SEEK_END);
 if (dbsz <= 0) dbsz = (off_t)S.outB[q][0].nbytes;
 printf("fsr4_service: dmabuf slot%d tensor %zu B, dma-buf alloc %lld B\n",
 q, S.outB[q][0].nbytes, (long long)dbsz);
 GLuint mem = 0;
 pglCreateMemoryObjectsEXT(1, &mem);
 GLint ded = 1; /* dma-buf imports are dedicated allocations */
 pglMemoryObjectParameterivEXT(mem, GL_DEDICATED_MEMORY_OBJECT_EXT_, &ded);
 while (glGetError() != GL_NO_ERROR) {}
 int fd2 = dup(S.outB[q][0].fd);
 glGenBuffers(1, &S.post_in_dm[q]);
 pglImportMemoryFdEXT(mem, (long long)dbsz, GL_HANDLE_TYPE_OPAQUE_FD_EXT_, fd2);
 GLenum e1 = glGetError();
 glBindBuffer(GL_SHADER_STORAGE_BUFFER, S.post_in_dm[q]);
 pglBufferStorageMemEXT(GL_SHADER_STORAGE_BUFFER, (long long)S.outB[q][0].nbytes, mem, 0);
 GLenum e2 = glGetError();
 glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
 if (e1 != GL_NO_ERROR || e2 != GL_NO_ERROR) {
 fprintf(stderr, "fsr4_service: dmabuf import slot%d FAILED (import 0x%x, storageMem 0x%x, alloc %lld B) - copy2 fallback\n",
 q, e1, e2, (long long)dbsz);
 S.dmabuf_in = 0;
 break;
 }
 S.post_in_mem[q] = mem;
 }
 printf("fsr4_service: dmabuf import %s\n",
 S.dmabuf_in ? "OK - copy2 eliminated (postA reads outB zero-copy)": "FAILED - copy2 fallback engaged");
 }
 S.lrc_ssbo = make_ssbo(5, NULL, LR_BYTES, true); // per-frame uploads (D2 dynamic)
 S.lrc_pp[0] = S.lrc_ssbo;
 S.lrc_pp[1] = ssbo_create(NULL, LR_BYTES, true); // -pong partner
 S.rep_ssbo = make_ssbo(6, NULL, OS_BYTES, true);
 S.out_ssbo = make_ssbo(7, NULL, OS_BYTES, true);
 S.mv_ssbo = make_ssbo(12, NULL, MV_BYTES, true); // feat mvec SSBO (D2 dynamic)
 S.rpj_ssbo = make_ssbo(9, NULL, OS_BYTES, true); // pre-pass rectified history (mu-law)
 S.hst_ssbo = make_ssbo(10, NULL, OS_BYTES, true); // post writes mu-law history (feat reads it)
 if (S.have_bstorage) {
 S.out8_ssbo = ssbo_create_persistent(OS_BYTES / 2, &S.out8_map);
 if (S.out8_ssbo) glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 11, S.out8_ssbo);
 }
 if (!S.out8_ssbo) S.out8_ssbo = make_ssbo(11, NULL, OS_BYTES / 2, true); // post RGBA8 out (halved readback)
 S.mulr_ssbo = make_ssbo(13, NULL, LR_BYTES, true); // mu-law'd LR (mulr.comp out)
 S.outc2_ssbo = make_ssbo(14, NULL, OS_BYTES, true); // rcas converted fp16 out
 S.dist_ssbo = make_ssbo(15, NULL, (size_t)OS_W * OS_H * 4, true); // compact dist (1 word/px)
 if (!S.bf || !S.wbuf_ssbo || !S.pre_out_pp[0] || !S.pre_out_pp[1] || !S.post_in ||
 !S.lrc_ssbo || !S.lrc_pp[1] || !S.rep_ssbo || !S.out_ssbo || !S.mv_ssbo || !S.rpj_ssbo || !S.hst_ssbo ||
 !S.out8_ssbo || !S.mulr_ssbo || !S.outc2_ssbo || !S.dist_ssbo) {
 fprintf(stderr, "fsr4_service: SSBO creation failure\n");
 release_gl(); release_qnn(); return FSR4_ERR_INIT_GL;
 }
 if (load_file(p->hist_seed, &d, &n) != 0) { release_gl(); release_qnn(); return FSR4_ERR_INIT_SHADER; }
 S.hist_seed_data = d;
 S.hist_tex = make_tex_rgba16f(0, OS_W, OS_H, d);
 if (load_file(p->rec_seed, &d, &n) != 0) { release_gl(); release_qnn(); return FSR4_ERR_INIT_SHADER; }
 /* rec is contractually 8-bit — posta.comp quantizes to the k/255
 * lattice BEFORE imageStore (sigmoid output, [0,1] by construction), so
 * RGBA16F storage was pure bandwidth: 16.6->8.3MB write in postA and half
 * the fetch footprint in feat's bilinear. Convert the f16 seed once here;
 * seed_temporal re-uses the converted bytes. */
 {
 size_t px4 = (size_t)OS_W * OS_H * 4;
 if (n < px4 * 2) { free(d); release_gl(); release_qnn(); return FSR4_ERR_INIT_SHADER; }
 unsigned char* d8 = (unsigned char*)malloc(px4);
 if (!d8) { free(d); release_gl(); release_qnn(); return FSR4_ERR_INIT_GL; }
 const unsigned short* h = (const unsigned short*)d;
 for (size_t i = 0; i < px4; i++) {
 _Float16 f; memcpy(&f, &h[i], 2);
 float x = (float)f;
 d8[i] = (unsigned char)(x <= 0.0f ? 0: x >= 1.0f ? 255: (int)(x * 255.0f + 0.5f));
 }
 free(d); d = d8;
 }
 S.rec_seed_data = d;
 // immutable rec texture (imageStore'd by post_real, sampled by feat);
 // LINEAR/CLAMP params MANDATORY on a 1-level immutable texture
 glGenTextures(1, &S.rec_tex);
 glActiveTexture(GL_TEXTURE1);
 glBindTexture(GL_TEXTURE_2D, S.rec_tex);
 glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, OS_W, OS_H);
 glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, OS_W, OS_H, GL_RGBA, GL_UNSIGNED_BYTE, d);
 glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
 glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
 glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
 glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
 glBindImageTexture(1, S.rec_tex, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8);
 void* zero_lr = calloc(1, LR_BYTES); // lrA seed = zeros; heap, not a 4 MB rodata array
 if (!zero_lr) { release_gl(); release_qnn(); return FSR4_ERR_INIT_GL; }
 S.lrA_tex = make_tex_rgba16f(2, LR_W, LR_H, zero_lr);
 free(zero_lr);

 glUseProgram(S.pre_prog);
 glUniform1ui(glGetUniformLocation(S.pre_prog, "IN_W"), OS_W);
 glUniform1ui(glGetUniformLocation(S.pre_prog, "OUT_W"), LR_W);
 glUniform1ui(glGetUniformLocation(S.pre_prog, "OUT_H"), LR_H);
 glUseProgram(S.post_prog);
 S.blend_floor = 0.0f;
 {
 const char* e = getenv("FSR4_BLENDFLOOR");
 if (e) S.blend_floor = (float)atof(e);
 }
 /* chroma-dist floor decay + legacy lrA upload control */
 S.bf_tau = 0.0f;
 {
 const char* e = getenv("FSR4_BFTAU");
 if (e) S.bf_tau = (float)atof(e);
 }
 S.lra_upload = 0;
 {
 const char* e = getenv("FSR4_LRA");
 if (e) S.lra_upload = atoi(e);
 }
 printf("fsr4_service: bftau=%.3f lra=%d\n", S.bf_tau, S.lra_upload);
 printf("fsr4_service: blend_floor=%.3f (0 = AMD reference; raises the minimum "
 "history weight when the net rejects our 50ms-old history)\n", S.blend_floor);
 glUniform4f(glGetUniformLocation(S.post_prog, "u_params"), 1.0f, 0.25f, -0.125f, S.blend_floor);
 S.jit[0] = 0.25f; S.jit[1] = -0.125f; S.jit_set = 0; // pinned default
 glUniform4f(glGetUniformLocation(S.post_prog, "u_dims"), (float)OS_W, (float)OS_H, (float)LR_W, (float)LR_H);
 glUseProgram(S.feat_prog);
 glUniform1i(glGetUniformLocation(S.feat_prog, "u_hist"), 0);
 glUniform1i(glGetUniformLocation(S.feat_prog, "u_rec"), 1);
 glUniform1i(glGetUniformLocation(S.feat_prog, "u_lr"), 2);
 /* feat's u_jitter needs the same pinned default the
 * post gets — without it a no-tail client ran feat UNJITTERED while post
 * sampled (0.25, -0.125): one stage jittered, the other not */
 glUniform2f(glGetUniformLocation(S.feat_prog, "u_jitter"), 0.25f, -0.125f);
 glUniform1ui(glGetUniformLocation(S.feat_prog, "IN_W"), OS_W);
 glUniform1ui(glGetUniformLocation(S.feat_prog, "IN_H"), OS_H);
 glUniform1ui(glGetUniformLocation(S.feat_prog, "LR_W"), LR_W);
 glUniform1ui(glGetUniformLocation(S.feat_prog, "LR_H"), LR_H);
 /* F7: cache the per-frame uniform locations (8 driver string
 * lookups/frame removed) and set the session-constant ones ONCE */
 S.ul_mulr_expo = glGetUniformLocation(S.mulr_prog, "u_expo");
 S.ul_feat_jit = glGetUniformLocation(S.feat_prog, "u_jitter");
 S.ul_feat_reset = glGetUniformLocation(S.feat_prog, "u_reset");
 S.ul_post_params = glGetUniformLocation(S.post_prog, "u_params");
 S.ul_post_rcas = glGetUniformLocation(S.post_prog, "u_rcas");
 S.ul_post_bftau = glGetUniformLocation(S.post_prog, "u_bftau");
 glUseProgram(S.post_prog);
 glUniform1f(S.ul_post_rcas, S.rcas_sharp > 0.0f ? 1.0f: 0.0f);
 glUniform1f(S.ul_post_bftau, S.bf_tau);
 if (S.rcas_prog) {
 S.ul_rcas_expo = glGetUniformLocation(S.rcas_prog, "u_expo");
 S.ul_rcas_sharp = glGetUniformLocation(S.rcas_prog, "u_sharp");
 S.ul_rcas_out16 = glGetUniformLocation(S.rcas_prog, "u_out16");
 glUseProgram(S.rcas_prog);
 glUniform1f(S.ul_rcas_expo, S.expo);
 glUniform1f(S.ul_rcas_sharp, S.rcas_sharp);
 {
 const char* e = getenv("FSR4_U8OUT");
 int u8 = (e && !atoi(e)) ? 0: 1;
 glUniform1f(S.ul_rcas_out16, u8 ? 0.0f: 1.0f); /* F1 */
 }
 }
 glUseProgram(0);

 // diagnostics
 if (p->debug_dump_dir && *p->debug_dump_dir) {
 snprintf(S.dump_dir, sizeof(S.dump_dir), "%s", p->debug_dump_dir);
 if (p->debug_dump_frames && *p->debug_dump_frames) {
 S.n_dump = parse_dump_frames(p->debug_dump_frames, S.dump_frames, 128);
 if (S.n_dump < 0) { release_gl(); release_qnn(); return FSR4_ERR_BUFFER; }
 }
 }

 // warmup (pinned count; stateless executes on slot-0 zeros, mirrors the
 // adopted harness's 3 hardcoded warm executes before its loop)
 for (unsigned w = 0; w < p->warmup; w++) {
 if (S.split_mode) {
 npu_submit(0);
 sem_wait(&S.npu_done_sem);
 S.npu_inflight--; /* warmup consumed a done token without the decrement — counter left +3, first reconnect wedged the drain forever */
 if (S.npu_err[0]) { release_gl(); release_qnn(); return FSR4_ERR_INIT_QNN; }
 } else if (S.iface.graphExecute(S.graph, S.inT[0], S.nIn, S.outT[0], S.nOut, NULL, NULL) != 0) {
 release_gl(); release_qnn(); return FSR4_ERR_INIT_QNN;
 }
 }
 printf("fsr4_service: warmup done (%u)\n", p->warmup);

 if (seed_temporal() != FSR4_OK) { release_gl(); release_qnn(); return FSR4_ERR_INIT_GL; }
 {
 GLenum e0 = glGetError();
 if (e0 != GL_NO_ERROR)
 fprintf(stderr, "fsr4_service: WARNING init left GL error 0x%x (cleared)\n", e0);
 }
 S.inited = 1;
 return FSR4_OK;
}

static void dump_membuf(const char* name, int fr, const MemBuf* b) {
 char path[600];
 snprintf(path, sizeof(path), "%s/%s_f%d.raw", S.dump_dir, name, fr);
 dmabuf_sync(b->fd, DMABUF_SYNC_RW | DMABUF_SYNC_START);
 FILE* f = fopen(path, "wb");
 if (f) { fwrite(b->ptr, 1, b->nbytes, f); fclose(f); }
 dmabuf_sync(b->fd, DMABUF_SYNC_READ | DMABUF_SYNC_END);
}
static int in_dump_list(int fr) {
 for (int i = 0; i < S.n_dump; i++) if (S.dump_frames[i] == fr) return 1;
 return 0;
}

/* ---- NPU-thread split ----
 * Thread B owns graphExecute (CPU-side FastRPC, no GL). The GL thread's
 * execute(N) FIRST finishes frame N-1's tail (copy2+postA, then rcas+
 * readback AFTER submitting NPU(N)) — the temporal chain postA(N-1) ->
 * feat(N) is preserved on one GL thread, and the NPU's 6.8ms hides behind
 * postA+rcas+readback of the previous frame. RCAS reads SSBO 10 directly
 * (bit-exact vs the old path). Output is one call deferred: the bootstrap
 * execute produces no image (fsr4_frame_ready() == 0). */
static void* npu_thread(void* arg) {
 (void)arg;
 pthread_setname_np(pthread_self(), "fsr4-npu");
 /* every other pipeline thread is pinned (net=5,
 * send=6, GL=7) but this one wasn't — sem wake could land on a busy core
 * shared with the emulated game, a candidate for the up=30.13 outlier
 * frames. cpu4 is unassigned by the daemon's layout. */
 {
 int cpu = 4;
 const char* e = getenv("FSR4_NPU_CPU");
 if (e && *e) cpu = atoi(e);
 cpu_set_t set; CPU_ZERO(&set); CPU_SET(cpu, &set);
 if (sched_setaffinity(0, sizeof set, &set) != 0)
 fprintf(stderr, "fsr4_service: npu pin cpu%d FAILED errno=%d\n", cpu, errno);
 else
 printf("fsr4_service: npu thread pinned cpu%d\n", cpu);
 }
 for (;;) {
 sem_wait(&S.npu_submit_sem);
 if (S.npu_stop) break;
 int p = (int)S.npu_cur_slot;
 double t0 = now_ms();
 int er = S.iface.graphExecute(S.graph, S.inT[p], S.nIn, S.outT[p], S.nOut, NULL, NULL);
 S.npu_wall[p] = now_ms() - t0;
 S.npu_err[p] = er;
 sem_post(&S.npu_done_sem);
 }
 return NULL;
}

static void npu_submit(unsigned slot) {
 S.npu_err[slot] = 0;
 S.npu_cur_slot = slot;
 S.npu_inflight++; /* consumed by execute's wait / invalidate's drain */
 sem_post(&S.npu_submit_sem);
}

static int fsr4_execute_split(const fsr4_frame_in_t* in, fsr4_frame_out_t* out) {
 double t0 = now_ms();
 double t_upld = -1.0, t_nw = -1.0, t_f = -1.0; /* split-bucket stamps */
 GLenum glerr;
 while (glGetError() != GL_NO_ERROR) {}

 /* mvec upload has no dependency on frame N-1's tail
 * (postA reads 4/5/9/15, not 12) — hoist above the NPU wait */
 glBindBuffer(GL_SHADER_STORAGE_BUFFER, S.mv_ssbo);
 glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, (GLsizeiptr)MV_BYTES, in->mvec);

 /* lrc(N) upload ALSO hoisted into the NPU-wait shadow — safe now via
 * the SSBO5 ping-pong: postA(N-1) (dispatched below) still reads binding 5
 * = the partner buffer; binding flips to this one right before feat(N).
 * The partner was last read by postA(N-2), two executes ago — retired. */
 int lslot = (int)(S.frames_done & 1);
 glBindBuffer(GL_SHADER_STORAGE_BUFFER, S.lrc_pp[lslot]);
 if (S.fuse && S.raw_lr) {
 /* v2 shaders decode R11G11B10F in-GPU — raw 2.07MB upload (was
 * 4.15MB fp16) and the mulr bake is deleted (dispatch+barrier gone) */
 glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, (GLsizeiptr)COLOR_RAW_BYTES, in->lr);
 } else {
 glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, (GLsizeiptr)LR_BYTES, in->lr);
 }
 if ((glerr = glGetError()) != GL_NO_ERROR) { fprintf(stderr, "fsr4_service: GL 0x%x upload lrc\n", glerr); return FSR4_ERR_GL_STAGE; }
 t_upld = now_ms();

 /* ---- tail of frame N-1 (chain-critical half first) ---- */
 if (S.tail.valid) {
 int q = (int)S.tail.slot; /* the slot execute N-1 submitted */
 sem_wait(&S.npu_done_sem);
 S.npu_inflight--;
 S.stats.last_npu_ms = S.npu_wall[q];
 if (S.npu_err[q]) return FSR4_ERR_QNN_EXECUTE;
 t_nw = now_ms();
 if (S.dmabuf_in) {
 /* zero-copy bridge: postA reads the imported outB dma-buf
 * directly (SSBO4 retargeted to the tail's slot). begin_cpu_
 * access bracket for the HTP->GPU handoff; the matching END is
 * issued after fence1 drains postA (at the t_f stamp below). */
 dmabuf_sync(S.outB[q][0].fd, DMABUF_SYNC_READ | DMABUF_SYNC_START);
 glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, S.post_in_dm[q]);
 S.dmabuf_pending = 1;
 S.dmabuf_last_fd = S.outB[q][0].fd;
 } else if (S.egl_in || S.ahb_in) {
 /* postA imageLoads outB through the imported/EGL-attached
 * dma-buf texture (image unit 4 = posta_egl.comp's NpuImg).
 * the cpu-access brackets are a CPU protocol — for the
 * GPU-attach path the driver owns coherency, and the bracket's
 * END (a gralloc-path cache pass, up to ms-scale in-game) lands
 * inside the c1 bucket (the in-game c1 doubling). FSR4_AHBSYNC=0
 * skips them; the md5 gate polices staleness. */
 if (S.ahb_sync) dmabuf_sync(S.outB[q][0].fd, DMABUF_SYNC_READ | DMABUF_SYNC_START);
 glBindImageTexture(4, S.npuimg_tex[q], 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA8);
 S.dmabuf_pending = S.ahb_sync;
 S.dmabuf_last_fd = S.outB[q][0].fd;
 S.dmabuf_last_ptr = S.outB[q][0].ptr;
 /* FSR4_SPIN_MS
 * re-inserts the OLD c2 duration as a pure CPU busy-wait —
 * zero memory traffic, read path unchanged. If the in-game
 * fence collapses 14 -> ~9-10 at spin=4.3 with fps unchanged,
 * the displacement theory is proven CAUSALLY (it was the
 * submit DELAY, not the copy, that let the game's GPU burst
 * drain first); if fence stays ~14, the AHB path genuinely
 * costs time under game load and the prediction flips. */
 if (S.spin_ms > 0.0) {
 double st = now_ms();
 while (now_ms() - st < S.spin_ms) { /* spin */ }
 }
 } else {
 /* copy2: outB[q] -> post_in (SSBO 4). WRITE|INVALIDATE (was
 * READ|WRITE) — the memcpy overwrites every byte, so dropping the
 * READ bit lets the driver hand back a write-combined mapping
 * (measurably faster for the 8.29MB bridge on Adreno). */
 dmabuf_sync(S.outB[q][0].fd, DMABUF_SYNC_RW | DMABUF_SYNC_START);
 if (S.post_in_map) {
 /* persistent WRITE map — write-combined memcpy, no per-frame
 * map/unmap pair; COHERENT + the barrier below publish to the GPU */
 memcpy(S.post_in_map, S.outB[q][0].ptr, S.outB[q][0].nbytes);
 dmabuf_sync(S.outB[q][0].fd, DMABUF_SYNC_READ | DMABUF_SYNC_END);
 glMemoryBarrier(GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT_EXT_);
 } else {
 glBindBuffer(GL_SHADER_STORAGE_BUFFER, S.post_in);
 void* m2 = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, (GLsizeiptr)S.outB[q][0].nbytes, GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_RANGE_BIT); /* RANGE, not BUFFER — the unwritten tail stays defined */
 if (!m2) { dmabuf_sync(S.outB[q][0].fd, DMABUF_SYNC_READ | DMABUF_SYNC_END); return FSR4_ERR_GL_STAGE; }
 memcpy(m2, S.outB[q][0].ptr, S.outB[q][0].nbytes);
 dmabuf_sync(S.outB[q][0].fd, DMABUF_SYNC_READ | DMABUF_SYNC_END);
 glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
 }
 } /* end !dmabuf_in copy2 branch */
 if (in_dump_list((int)S.frames_done)) dump_membuf("npuout", (int)S.frames_done, &S.outB[q][0]);
 /* postA(N-1): history + recurrent — ALL_BARRIER covers the rec
 * imageStore -> next feat texture fetch */
 glUseProgram(S.post_prog); // split mode: post_prog = posta.comp
 if (S.jit_set)
 glUniform4f(S.ul_post_params, S.expo, S.jit_prev[0], S.jit_prev[1], S.blend_floor); /* N-1's jitter */
 glDispatchCompute((OS_W + 7) / 8, (OS_H + 7) / 8, 1);
 glMemoryBarrier(GL_ALL_BARRIER_BITS);
 }
 double t_npu = now_ms();

 /* ---- head of frame N ---- */
 int slot = (int)(S.frames_done & 1);
 /* flip binding 5 to the freshly-uploaded partner (postA(N-1) above
 * consumed the old one; feat(N) must read lrc(N)) */
 glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, S.lrc_pp[slot]);

 if (!(S.fuse && S.raw_lr)) {
 glUseProgram(S.mulr_prog);
 glUniform1f(S.ul_mulr_expo, S.expo);
 glDispatchCompute(120, 68, 1);
 glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
 }

 if (!S.first_frame) {
 glUseProgram(S.feat_prog);
 if (S.jit_set)
 glUniform2f(S.ul_feat_jit, S.jit[0], S.jit[1]);
 glUniform1f(S.ul_feat_expo, S.expo); /* was set only at init — silent with autoexp off, wrong when on */
 glUniform1f(S.ul_feat_reset, S.reset ? 1.0f: 0.0f);
 S.reset = 0;
 if (S.fuse) {
 /* fused kernel — LR-res dispatch, writes pre_out (binding 1,
 * per-slot like pre0 did) + rpj + compact dist; the SSBO0 feature
 * write and the separate pre0 dispatch are gone.
 * with FSR4_AHBIN the kernel imageStores the NPU input
 * tensor directly (image unit 5, per-slot) — pre_out unused. */
 if (S.ahb_in2)
 glBindImageTexture(5, S.in_img_tex[slot], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8);
 else
 glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, S.pre_out_pp[slot]);
 glDispatchCompute((LR_W + 7) / 8, (LR_H + 7) / 8, 1);
 glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_BUFFER_UPDATE_BARRIER_BIT);
 } else {
 glDispatchCompute(240, 135, 1);
 glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
 glUseProgram(S.pre_prog);
 glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, S.pre_out_pp[slot]);
 glDispatchCompute((LR_W + 7) / 8, (LR_H + 7) / 8, 1);
 glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);
 }
 if (S.ahb_in2) {
 /* feat imageStored the tensor directly — no map, no 8.29MB
 * memcpy. The fence REMAINS (npu_submit must not race feat's
 * stores); bracketless like the copy2 side (lesson:
 * dmabuf_sync on this path blocks on GPU release). */
 GLsync fs0 = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
 if (fs0) { glClientWaitSync(fs0, GL_SYNC_FLUSH_COMMANDS_BIT, GL_TIMEOUT_IGNORED); glDeleteSync(fs0); }
 } else {
 glBindBuffer(GL_SHADER_STORAGE_BUFFER, S.pre_out_pp[slot]);
 void* m = S.pre_map[slot];
 if (m) {
 /* persistent coherent map — cached read. The spec
 * requires FenceSync+ClientWaitSync for CPU reads of GPU-written
 * coherent storage (barrier orders visibility, not completion);
 * a stale pre_out would feed frame N-2's features to the NPU. */
 glMemoryBarrier(GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT_EXT_);
 GLsync fs0 = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
 if (fs0) { glClientWaitSync(fs0, GL_SYNC_FLUSH_COMMANDS_BIT, GL_TIMEOUT_IGNORED); glDeleteSync(fs0); }
 } else {
 m = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, (GLsizeiptr)S.inB[slot][0].nbytes, GL_MAP_READ_BIT);
 if (!m) return FSR4_ERR_GL_STAGE;
 }
 dmabuf_sync(S.inB[slot][0].fd, DMABUF_SYNC_RW | DMABUF_SYNC_START);
 memcpy(S.inB[slot][0].ptr, m, S.inB[slot][0].nbytes);
 dmabuf_sync(S.inB[slot][0].fd, DMABUF_SYNC_WRITE | DMABUF_SYNC_END);
 if (!S.pre_map[slot]) glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
 }
 t_f = now_ms(); /* fence (or fallback map) drain done */
 if (S.dmabuf_pending) {
 /* postA drained with the fence — end the cpu-access bracket
 * on the imported outB buffer */
 dmabuf_sync(S.dmabuf_last_fd, DMABUF_SYNC_READ | DMABUF_SYNC_END);
 S.dmabuf_pending = 0;
 }
 /* sentinel: every Nth frame compare the GPU's view of the AHB
 * (imageLoad'd bytes, same round-trip as posta_egl) against the
 * mmap'd dma-buf truth. Adds ~1ms on sentinel frames only. */
 if (S.sentinel_n > 0 && S.ahb_in && (S.frames_done % (unsigned)S.sentinel_n) == 0) {
 glUseProgram(S.sent_prog);
 glDispatchCompute(1, 1, 1);
 glMemoryBarrier(GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT_EXT_);
 GLsync fss = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
 if (fss) { glClientWaitSync(fss, GL_SYNC_FLUSH_COMMANDS_BIT, GL_TIMEOUT_IGNORED); glDeleteSync(fss); }
 glBindBuffer(GL_SHADER_STORAGE_BUFFER, S.sent_ssbo);
 uint32_t* sv = (uint32_t*)glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, 256, GL_MAP_READ_BIT);
 if (sv) {
 dmabuf_sync(S.dmabuf_last_fd, DMABUF_SYNC_READ | DMABUF_SYNC_START);
 const unsigned char* mm = (const unsigned char*)S.dmabuf_last_ptr;
 for (uint32_t i = 0; i < 64; i++) {
 uint32_t idx = (i * 2654435761u + 12345u) % ((uint32_t)(2 * OS_W) * (uint32_t)OS_H);
 uint32_t off = idx * 4;
 uint32_t exp = (uint32_t)mm[off] | ((uint32_t)mm[off + 1] << 8) | ((uint32_t)mm[off + 2] << 16) | ((uint32_t)mm[off + 3] << 24);
 S.sent_checked++;
 if (sv[i] != exp) S.sent_mismatch++;
 }
 dmabuf_sync(S.dmabuf_last_fd, DMABUF_SYNC_READ | DMABUF_SYNC_END);
 glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
 if ((S.sent_checked % 6400u) < 64u)
 printf("fsr4_service: sentinel %lu checked, %lu mismatches\n", S.sent_checked, S.sent_mismatch);
 if (S.sent_mismatch > 0 && (S.sent_checked % 640u) < 64u)
 fprintf(stderr, "fsr4_service: SENTINEL MISMATCH (GPU view != DSP truth) - %lu/%lu\n", S.sent_mismatch, S.sent_checked);
 }
 glUseProgram(0);
 }
 /* the copy1 memcpy moved INSIDE the !ahb_in2 else-branch above;
 * only the NPU-input dump remains here (mmap view works for both). */
 if (in_dump_list((int)S.frames_done)) dump_membuf("in", (int)S.frames_done, &S.inB[slot][0]);
 }
 double t_gl = now_ms();

 /* submit NPU(N) — thread B owns it from here */
 npu_submit((unsigned)slot);

 /* ---- tail of frame N-1 (final-color half, off the temporal chain) ---- */
 if (S.tail.valid) {
 if (S.rcas_sharp > 0.0f) {
 glUseProgram(S.rcas_prog);
 glDispatchCompute((OS_W + 7) / 8, (OS_H + 7) / 8, 1);
 }
 /* readback (u8 default) */
 static int u8out = -1;
 if (u8out < 0) {
 const char* e = getenv("FSR4_U8OUT");
 u8out = (e && !atoi(e)) ? 0: 1;
 }
 if (u8out) {
 if (S.out8_map) {
 /* persistent coherent map (cached read). The
 * EXT spec demands FenceSync+ClientWaitSync for CPU reads of
 * GPU-written coherent storage — a barrier alone orders
 * visibility but not completion. The fence costs the stall
 * the old transient map had; the win is the cached mapping. */
 glMemoryBarrier(GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT_EXT_);
 GLsync fs1 = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
 if (fs1) { glClientWaitSync(fs1, GL_SYNC_FLUSH_COMMANDS_BIT, GL_TIMEOUT_IGNORED); glDeleteSync(fs1); }
 memcpy(out->out, S.out8_map, OS_BYTES / 2);
 S.emitted = 1; /* a real frame was produced */
 } else {
 glBindBuffer(GL_SHADER_STORAGE_BUFFER, S.out8_ssbo);
 void* m4 = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, (GLsizeiptr)(OS_BYTES / 2), GL_MAP_READ_BIT);
 if (!m4) return FSR4_ERR_READBACK;
 memcpy(out->out, m4, OS_BYTES / 2);
 glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
 S.emitted = 1; /* a real frame was produced */
 }
 } else {
 glBindBuffer(GL_SHADER_STORAGE_BUFFER, S.rcas_sharp > 0.0f ? S.outc2_ssbo: S.out_ssbo);
 void* m3 = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, (GLsizeiptr)OS_BYTES, GL_MAP_READ_BIT);
 if (!m3) return FSR4_ERR_READBACK;
 memcpy(out->out, m3, OS_BYTES);
 glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
 S.emitted = 1; /* a real frame was produced */
 }
 if (in_dump_list((int)S.frames_done)) {
 char path[600];
 snprintf(path, sizeof path, "%s/rgb_f%d.raw", S.dump_dir, (int)S.frames_done);
 FILE* f = fopen(path, "wb");
 if (f) { fwrite(out->out, 1, u8out ? OS_BYTES / 2: OS_BYTES, f); fclose(f); }
 }
 if ((glerr = glGetError()) != GL_NO_ERROR) {
 fprintf(stderr, "fsr4_service: GL error 0x%x at readback\n", glerr);
 return FSR4_ERR_READBACK;
 }
 }
 double t_end = now_ms();

 S.tail.valid = 1;
 S.tail.slot = (unsigned)slot;
 /* stats: buckets approximate the legacy meanings */
 S.stats.last_upload_ms = t_gl - t_npu; // head incl uploads
 S.stats.last_gl_ms = 0.0;
 S.stats.last_post_ms = (t_end - t_gl) - S.stats.last_npu_ms * 0.0; // tail
 S.stats.last_readback_ms = 0.0;
 S.stats.last_total_ms = t_end - t0;
 /* split decomposition: uploads | NPU residual wait | copy2+postA submit
 * | fence1 GPU drain (postA+feat) | copy1. Negative-guarded stamps cover
 * the bootstrap frame (no tail, no feat). */
 {
 double v_upld = (t_upld >= 0.0) ? t_upld: t0;
 double v_nw = (t_nw >= 0.0) ? t_nw: v_upld;
 double v_tf = (t_f >= 0.0) ? t_f: t_npu;
 S.stats.last_upld_ms = v_upld - t0;
 S.stats.last_nwait_ms = v_nw - v_upld;
 S.stats.last_copy2_ms = t_npu - v_nw;
 S.stats.last_fence_ms = v_tf - t_npu;
 S.stats.last_copy1_ms = t_gl - v_tf;
 }
 S.stats.execute_total_ms += S.stats.last_total_ms;
 if (S.stats.last_total_ms > S.stats.execute_max_ms) S.stats.execute_max_ms = S.stats.last_total_ms;
 if (S.stats.last_total_ms > 100.0) S.stats.frames_over_100ms++;
 S.frames_done++;
 S.stats.frames = S.frames_done;
 S.first_frame = 0;
 return FSR4_OK;
}

int fsr4_frame_ready(void) {
 if (!S.split_mode || !S.inited) return 1;
 /* tail.valid tracks the SUBMIT, not the readback — the old
 * form shipped the bootstrap execute's UNWRITTEN out8 buffer (black
 flash on first connect, stale/junk frame on reconnects). */
 int r = S.emitted;
 S.emitted = 0;
 return r;
}

int fsr4_execute(const fsr4_frame_in_t* in, fsr4_frame_out_t* out) {
 if (!S.inited) return FSR4_ERR_STATE;
 if (!in || !out) return FSR4_ERR_BUFFER;
 if (!in->lr || !in->mvec || !in->reproj || !out->out) return FSR4_ERR_BUFFER;
 if (S.split_mode) return fsr4_execute_split(in, out);
 int slot = (int)(S.frames_done & 1);
 double t0 = now_ms();
 GLenum glerr;

 // uploads (order mirrors the adopted FFX_LIVE block: SSBO5, SSBO6, SSBO12, unit-2 tex)
 while (glGetError() != GL_NO_ERROR) {} // clear any sticky error from init
 glBindBuffer(GL_SHADER_STORAGE_BUFFER, S.lrc_ssbo);
 glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, (GLsizeiptr)LR_BYTES, in->lr);
 if ((glerr = glGetError()) != GL_NO_ERROR) { fprintf(stderr, "fsr4_service: GL 0x%x upload lrc\n", glerr); return FSR4_ERR_GL_STAGE; }
 /* SSBO 6 (daemon-synthesized reproj) is dead weight with the real
 * post (reads SSBO 9) — 16.6MB/frame saved when FSR4_SKIP_SYN is set */
 static int upload_reproj = -1;
 if (upload_reproj < 0) {
 const char* e = getenv("FSR4_SKIP_SYN");
 upload_reproj = (e && atoi(e)) ? 0: 1;
 printf("fsr4_service: reproj upload %s\n", upload_reproj ? "on": "OFF (skip_syn)");
 }
 if (upload_reproj) {
 glBindBuffer(GL_SHADER_STORAGE_BUFFER, S.rep_ssbo);
 glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, (GLsizeiptr)OS_BYTES, in->reproj);
 if ((glerr = glGetError()) != GL_NO_ERROR) { fprintf(stderr, "fsr4_service: GL 0x%x upload reproj\n", glerr); return FSR4_ERR_GL_STAGE; }
 }
 glBindBuffer(GL_SHADER_STORAGE_BUFFER, S.mv_ssbo);
 glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, (GLsizeiptr)MV_BYTES, in->mvec);
 if ((glerr = glGetError()) != GL_NO_ERROR) { fprintf(stderr, "fsr4_service: GL 0x%x upload mvec\n", glerr); return FSR4_ERR_GL_STAGE; }
 glActiveTexture(GL_TEXTURE2);
 if (S.lra_upload) { /* features_real reads SSBO 13 — the bind +
 * 4.1MB texture upload only serve legacy feats */
 glBindTexture(GL_TEXTURE_2D, S.lrA_tex);
 glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, LR_W, LR_H, GL_RGBA, GL_HALF_FLOAT, in->lr);
 }
 glActiveTexture(GL_TEXTURE0);
 if ((glerr = glGetError()) != GL_NO_ERROR) { fprintf(stderr, "fsr4_service: GL 0x%x upload lrA\n", glerr); return FSR4_ERR_GL_STAGE; }
 double t_up = now_ms();

 /* bake mu-law'd LR once (0.5M logs vs post's per-tap 18M).
 * exposure multiplies BEFORE mu-law (SDK mlsr:85-88). */
 glUseProgram(S.mulr_prog);
 glUniform1f(S.ul_mulr_expo, S.expo);
 glDispatchCompute(120, 68, 1);
 glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

 if (!S.first_frame) {
 // feat(n): hist(0) + rec(1, written by post(n-1)) + lrA(2) + mvec SSBO12
 glUseProgram(S.feat_prog);
 if (S.jit_set)
 glUniform2f(S.ul_feat_jit, S.jit[0], S.jit[1]);
 glUniform1f(S.ul_feat_reset, S.reset ? 1.0f: 0.0f);
 S.reset = 0; // one-shot
 glDispatchCompute(240, 135, 1);
 glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
 // pre(n): feats SSBO0 + weights SSBO2 -> pre_out_pp[slot]
 glUseProgram(S.pre_prog);
 glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, S.pre_out_pp[slot]);
 glDispatchCompute((LR_W + 7) / 8, (LR_H + 7) / 8, 1);
 glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT); /* map-read
 visibility; the glFinish was redundant (maps sync implicitly) */
 // copy1: pre_out_pp[slot] -> inB[slot] (dmabuf pattern mirrors the loop)
 glBindBuffer(GL_SHADER_STORAGE_BUFFER, S.pre_out_pp[slot]);
 if (S.pre_map[slot]) {
 /* the legacy path must branch too — a transient READ map
 * on a buffer holding a live persistent mapping is driver-dependent */
 glMemoryBarrier(GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT_EXT_);
 GLsync fs2 = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
 if (fs2) { glClientWaitSync(fs2, GL_SYNC_FLUSH_COMMANDS_BIT, GL_TIMEOUT_IGNORED); glDeleteSync(fs2); }
 dmabuf_sync(S.inB[slot][0].fd, DMABUF_SYNC_RW | DMABUF_SYNC_START);
 memcpy(S.inB[slot][0].ptr, S.pre_map[slot], S.inB[slot][0].nbytes);
 dmabuf_sync(S.inB[slot][0].fd, DMABUF_SYNC_WRITE | DMABUF_SYNC_END);
 } else {
 void* m = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, (GLsizeiptr)S.inB[slot][0].nbytes, GL_MAP_READ_BIT);
 if (!m) return FSR4_ERR_GL_STAGE;
 dmabuf_sync(S.inB[slot][0].fd, DMABUF_SYNC_RW | DMABUF_SYNC_START);
 memcpy(S.inB[slot][0].ptr, m, S.inB[slot][0].nbytes);
 dmabuf_sync(S.inB[slot][0].fd, DMABUF_SYNC_WRITE | DMABUF_SYNC_END);
 glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
 }
 } else {
 glFinish(); // uploads visible before the bootstrap NPU read (defensive; no GL consumer)
 }
 double t_gl = now_ms();

 if (in_dump_list((int)S.frames_done)) dump_membuf("in", (int)S.frames_done, &S.inB[slot][0]);

 // NPU (calling thread, synchronous — contract (e); stateless A4)
 if (S.iface.graphExecute(S.graph, S.inT[slot], S.nIn, S.outT[slot], S.nOut, NULL, NULL) != 0)
 return FSR4_ERR_QNN_EXECUTE;
 double t_npu = now_ms();

 // copy2: outB[slot] -> post_in (SSBO 4); dmabuf read pattern mirrors the loop
 dmabuf_sync(S.outB[slot][0].fd, DMABUF_SYNC_RW | DMABUF_SYNC_START);
 glBindBuffer(GL_SHADER_STORAGE_BUFFER, S.post_in);
 void* m2 = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, (GLsizeiptr)S.outB[slot][0].nbytes, GL_MAP_READ_BIT | GL_MAP_WRITE_BIT);
 if (!m2) { dmabuf_sync(S.outB[slot][0].fd, DMABUF_SYNC_READ | DMABUF_SYNC_END); return FSR4_ERR_GL_STAGE; }
 memcpy(m2, S.outB[slot][0].ptr, S.outB[slot][0].nbytes);
 dmabuf_sync(S.outB[slot][0].fd, DMABUF_SYNC_READ | DMABUF_SYNC_END);
 glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
 if (in_dump_list((int)S.frames_done)) dump_membuf("npuout", (int)S.frames_done, &S.outB[slot][0]);

 // post_real(n): SSBO4 + lrc(5) + reproj(6) -> final RGB (SSBO7) + rec imageStore;
 // the ALL_BARRIER is MANDATORY (imageStore -> feat texture-fetch coherency)
 glUseProgram(S.post_prog);
 if (S.jit_set)
 glUniform4f(S.ul_post_params, S.expo, S.jit[0], S.jit[1], S.blend_floor);
 glDispatchCompute((OS_W + 7) / 8, (OS_H + 7) / 8, 1);
 glMemoryBarrier(GL_ALL_BARRIER_BITS);
 if (S.rcas_sharp > 0.0f) {
 /* sharpen in mu-law domain, convert + divide exposure.
 * (u_expo/u_sharp/u_out16 are session constants — set at init.)
 * no barrier here — the readback map syncs implicitly. */
 glUseProgram(S.rcas_prog);
 glDispatchCompute((OS_W + 7) / 8, (OS_H + 7) / 8, 1);
 }
 /* ALL_BARRIER kept after post: the rec imageStore -> next frame's
 * texture fetch needs it on ES 3.1 (no TEXTURE_FETCH bit); the trailing
 * glFinish was redundant with the readback map's implicit stall. */
 double t_post = now_ms(); // last_post_ms = steps 6-7 (copy2 folded in, per header)

 // readback: post_f -> caller out (16,588,800 B), or the RGBA8
 // mirror (8,294,400 B) when FSR4_U8OUT is on — the daemon then sends it
 // directly (out16_to_u8 deleted from the hot path)
 static int u8out = -1;
 if (u8out < 0) {
 const char* e = getenv("FSR4_U8OUT");
 u8out = (e && !atoi(e)) ? 0: 1; // default ON
 printf("fsr4_service: u8 readback %s\n", u8out ? "ON (8.3MB)": "off (16.6MB)");
 }
 if (u8out) {
 if (S.out8_map) { /* legacy path — branch on the persistent map too */
 glMemoryBarrier(GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT_EXT_);
 GLsync fs3 = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
 if (fs3) { glClientWaitSync(fs3, GL_SYNC_FLUSH_COMMANDS_BIT, GL_TIMEOUT_IGNORED); glDeleteSync(fs3); }
 memcpy(out->out, S.out8_map, OS_BYTES / 2);
 S.emitted = 1;
 } else {
 glBindBuffer(GL_SHADER_STORAGE_BUFFER, S.out8_ssbo);
 void* m4 = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, (GLsizeiptr)(OS_BYTES / 2), GL_MAP_READ_BIT);
 if (!m4) return FSR4_ERR_READBACK;
 memcpy(out->out, m4, OS_BYTES / 2);
 glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
 S.emitted = 1;
 }
 } else {
 /* with RCAS the converted frame lives in SSBO 14 (SSBO 7 holds the
 * mu-law intermediate at this point) */
 glBindBuffer(GL_SHADER_STORAGE_BUFFER, S.rcas_sharp > 0.0f ? S.outc2_ssbo: S.out_ssbo);
 void* m3 = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, (GLsizeiptr)OS_BYTES, GL_MAP_READ_BIT);
 if (!m3) return FSR4_ERR_READBACK;
 memcpy(out->out, m3, OS_BYTES);
 glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
 }
 if (in_dump_list((int)S.frames_done)) {
 char path[600];
 snprintf(path, sizeof path, "%s/rgb_f%d.raw", S.dump_dir, (int)S.frames_done);
 FILE* f = fopen(path, "wb");
 if (f) { fwrite(out->out, 1, u8out ? OS_BYTES / 2: OS_BYTES, f); fclose(f); }
 printf("fsr4_service: dump %s\n", path);
 }
 if ((glerr = glGetError()) != GL_NO_ERROR) {
 fprintf(stderr, "fsr4_service: GL error 0x%x at readback\n", glerr);
 return FSR4_ERR_READBACK;
 }

 /* history for the NEXT feat pass. features_real reads
 * SSBO 10 DIRECTLY now (no per-frame texture round-trip); the legacy
 * map+glTexSubImage2D chain below only runs for the old texture-based
 * shaders, opt-in via FSR4_HIST_TEX=1. */
 {
 static int hist_tex_mode = -1;
 if (hist_tex_mode < 0)
 hist_tex_mode = (getenv("FSR4_HIST_TEX") && atoi(getenv("FSR4_HIST_TEX"))) ? 1: 0;
 if (hist_tex_mode && !getenv("FSR4_NO_HISTUP")) {
 glActiveTexture(GL_TEXTURE0);
 glBindTexture(GL_TEXTURE_2D, S.hist_tex);
 glBindBuffer(GL_SHADER_STORAGE_BUFFER, S.hst_ssbo);
 void* mh = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, (GLsizeiptr)OS_BYTES, GL_MAP_READ_BIT);
 if (mh) {
 glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, OS_W, OS_H, GL_RGBA, GL_HALF_FLOAT, mh);
 glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
 }
 }
 }
 double t_end = now_ms();

 S.stats.last_upload_ms = t_up - t0;
 S.stats.last_gl_ms = t_gl - t_up;
 S.stats.last_npu_ms = t_npu - t_gl;
 S.stats.last_post_ms = t_post - t_npu;
 S.stats.last_readback_ms = t_end - t_post;
 S.stats.last_total_ms = t_end - t0;
 S.stats.execute_total_ms += S.stats.last_total_ms;
 if (S.stats.last_total_ms > S.stats.execute_max_ms) S.stats.execute_max_ms = S.stats.last_total_ms;
 if (S.stats.last_total_ms > 100.0) S.stats.frames_over_100ms++;
 S.frames_done++;
 S.stats.frames = S.frames_done;
 S.first_frame = 0;
 return FSR4_OK;
}

int fsr4_invalidate(void) {
 if (!S.inited) return FSR4_ERR_STATE;
 if (S.split_mode && S.tail.valid) {
 /* drain ONLY the actual in-flight executes. The blind
 * sem_wait wedged forever when an execute-error path returned
 * between consuming the last completion and submitting the next
 * (zero in flight, sem at 0, main thread blocked for good). */
 while (S.npu_inflight > 0) { sem_wait(&S.npu_done_sem); S.npu_inflight--; }
 S.tail.valid = 0;
 S.first_frame = 1;
 S.emitted = 0;
 }
 int e = seed_temporal();
 if (e == FSR4_OK)
 printf("fsr4_service: invalidate (rec+hist reseeded, bootstrap re-armed)\n");
 return e;
}

void fsr4_set_jitter(float jx, float jy) {
 if (!S.inited) return;
 /* the split tail renders frame N-1 but was
 * handed frame N's jitter — the upsampled-current term of EVERY frame
 * sampled at a wrong subpixel offset (up to ~1 LR px) = a standing
 * shimmer contributor since the split pipeline was born. Lag one frame. */
 S.jit_prev[0] = S.jit[0]; S.jit_prev[1] = S.jit[1];
 S.jit[0] = jx; S.jit[1] = jy; S.jit_set = 1;
}

/* per-frame exposure (multiplies LR pre-mu-law, divides at the end)
 * and one-shot cut-reset (feat seeds history with current, rec=0). */
void fsr4_set_exposure(float e) {
 if (!S.inited) return;
 S.expo = (e >= 0.05f && e <= 8.0f) ? e: 1.0f;
}
void fsr4_set_reset(int r) {
 if (!S.inited) return;
 S.reset = r ? 1: 0;
}

void fsr4_get_stats(fsr4_stats_t* out) { if (out) *out = S.stats; }

// ---- teardown (contract order) ----
static void release_gl(void) {
 if (S.dpy != EGL_NO_DISPLAY) {
 if (S.ctx != EGL_NO_CONTEXT) {
 glFinish();
 glDeleteProgram(S.pre_prog); glDeleteProgram(S.feat_prog); glDeleteProgram(S.post_prog);
 glDeleteProgram(S.mulr_prog);
 if (S.rcas_prog) glDeleteProgram(S.rcas_prog);
 glDeleteBuffers(1, &S.bf); glDeleteBuffers(1, &S.wbuf_ssbo);
 glDeleteBuffers(2, S.pre_out_pp);
 glDeleteBuffers(1, &S.post_in); glDeleteBuffers(1, &S.lrc_ssbo); glDeleteBuffers(1, &S.lrc_pp[1]);
 glDeleteBuffers(1, &S.rep_ssbo); glDeleteBuffers(1, &S.out_ssbo);
 glDeleteBuffers(1, &S.rpj_ssbo); glDeleteBuffers(1, &S.hst_ssbo);
 glDeleteBuffers(1, &S.out8_ssbo); glDeleteBuffers(1, &S.mulr_ssbo);
 glDeleteBuffers(1, &S.outc2_ssbo);
 glDeleteBuffers(1, &S.dist_ssbo);
 glDeleteBuffers(1, &S.mv_ssbo);
 glDeleteTextures(1, &S.hist_tex); glDeleteTextures(1, &S.rec_tex);
 glDeleteTextures(1, &S.lrA_tex);
 eglMakeCurrent(S.dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
 eglDestroyContext(S.dpy, S.ctx);
 S.ctx = EGL_NO_CONTEXT;
 }
 if (S.surf != EGL_NO_SURFACE) { eglDestroySurface(S.dpy, S.surf); S.surf = EGL_NO_SURFACE; }
 eglTerminate(S.dpy);
 S.dpy = EGL_NO_DISPLAY;
 }
 free(S.hist_seed_data); S.hist_seed_data = NULL;
 free(S.rec_seed_data); S.rec_seed_data = NULL;
}
static void release_rpcmem(void) {
 for (uint32_t s = 0; s < 2; s++) {
 for (uint32_t i = 0; i < 2; i++) {
 if (S.inB[s][i].ptr && S.p_free && S.inB[s][i].handle) {
 const Qnn_MemHandle_t h = S.inB[s][i].handle;
 S.iface.memDeRegister(&h, 1); S.p_free(S.inB[s][i].ptr); S.inB[s][i].ptr = NULL;
 }
 if (S.outB[s][i].ptr && S.p_free && S.outB[s][i].handle) {
 const Qnn_MemHandle_t h = S.outB[s][i].handle;
 S.iface.memDeRegister(&h, 1); S.p_free(S.outB[s][i].ptr); S.outB[s][i].ptr = NULL;
 }
 }
 }
}
static void release_qnn(void) {
 release_rpcmem();
 if (S.context) S.iface.contextFree(S.context, NULL);
 if (S.device) S.iface.deviceFree(S.device);
 if (S.backend) S.iface.backendFree(S.backend);
 if (S.logh) S.iface.logFree(S.logh);
 S.context = NULL; S.device = NULL; S.backend = NULL; S.logh = NULL;
 free(S.in_info); free(S.out_info); S.in_info = S.out_info = NULL;
 if (S.be_lib) dlclose(S.be_lib);
 if (S.sys_lib) dlclose(S.sys_lib);
 if (S.rpc_lib) dlclose(S.rpc_lib);
 S.be_lib = S.sys_lib = S.rpc_lib = NULL;
}

int fsr4_shutdown(void) {
 if (!S.inited) return FSR4_OK; // no-op before init / after failure / double shutdown
 // partial-teardown diagnostic mode (env-gated): deregister + rpcmem_free,
 // SKIP the QNN context/device/backend/log frees, flush, _exit(0).
 // Gate: outputs already written must md5-match the normal run's.
 if (getenv("FSR4_E4B")) {
 release_rpcmem();
 fflush(NULL);
 fsync(1);
 _exit(0);
 }
 if (S.split_mode) {
 /* stop the NPU thread before teardown */
 S.npu_stop = 1;
 while (S.npu_inflight > 0) { sem_wait(&S.npu_done_sem); S.npu_inflight--; } // counted drain
 sem_post(&S.npu_submit_sem);
 pthread_join(S.npu_thr, NULL);
 sem_destroy(&S.npu_submit_sem); sem_destroy(&S.npu_done_sem);
 }
 release_gl();
 release_qnn();
 memset(&S, 0, sizeof(S));
 printf("fsr4_service: shutdown complete (full teardown order E4a)\n");
 return FSR4_OK;
}

// ================= GE2 driver =================
#ifndef FSR4_NO_MAIN
int main(int argc, char** argv) {
 setvbuf(stdout, NULL, _IONBF, 0);
 if (argc < 16) {
 fprintf(stderr, "usage: %s <backend.so> <system.so> <bin> <pre.comp> <feat.comp> <post.comp> "
 "<weights.raw> <hist.raw> <rec.raw> <lr.raw> <reproj.raw> <mvec.raw> "
 "<frames> <dumpdir> <dumpframes> [invalidate_after]\n", argv[0]);
 return 1;
 }
 fsr4_init_params_t p;
 memset(&p, 0, sizeof(p));
 p.contract_version = FSR4RP6_CONTRACT_VERSION;
 p.backend_so = argv[1]; p.system_so = argv[2]; p.model_bin = argv[3];
 p.pre_shader = argv[4]; p.feat_shader = argv[5]; p.post_shader = argv[6];
 p.weights_seed = argv[7]; p.hist_seed = argv[8]; p.rec_seed = argv[9];
 p.corner.core_corner = 0xA0; p.corner.bus_corner = 0xA0;
 p.slot_ms = 16.667f;
 p.warmup = 3; // mirrors the adopted harness's hardcoded 3 warm executes
 p.debug_dump_dir = argv[14];
 p.debug_dump_frames = argv[15];
 int inv_after = (argc > 16) ? atoi(argv[16]): 0;
 int rc = fsr4_init(&p);
 if (rc != FSR4_OK) { fprintf(stderr, "FATAL init rc=%d\n", rc); return 1; }

 fsr4_frame_in_t fin;
 void* d_lr = NULL; void* d_rp = NULL; void* d_mv = NULL;
 size_t n = 0;
 if (load_file(argv[10], &d_lr, &n) != 0 || n != LR_BYTES) { fprintf(stderr, "FATAL lr\n"); return 1; }
 if (load_file(argv[11], &d_rp, &n) != 0 || n != OS_BYTES) { fprintf(stderr, "FATAL reproj\n"); return 1; }
 if (load_file(argv[12], &d_mv, &n) != 0 || n != MV_BYTES) { fprintf(stderr, "FATAL mvec\n"); return 1; }
 fin.lr = d_lr; fin.reproj = d_rp; fin.mvec = d_mv;
 fsr4_frame_out_t fout; fout.out = malloc(OS_BYTES);

 int frames = atoi(argv[13]);
 for (int fr = 0; fr < frames; fr++) {
 int r = fsr4_execute(&fin, &fout);
 if (r != FSR4_OK) { fprintf(stderr, "FATAL execute fr=%d rc=%d\n", fr, r); return 1; }
 if (inv_after && (fr + 1) == inv_after) {
 r = fsr4_invalidate();
 if (r != FSR4_OK) { fprintf(stderr, "FATAL invalidate rc=%d\n", r); return 1; }
 printf("GE2_INVALIDATED_AT %d\n", fr + 1);
 }
 if ((fr % 100) == 99) {
 fsr4_stats_t st; fsr4_get_stats(&st);
 fprintf(stderr, "stages[%d]: up %.2f gl %.2f npu %.2f post %.2f rb %.2f\n",
 fr + 1, st.last_upload_ms, st.last_gl_ms, st.last_npu_ms,
 st.last_post_ms, st.last_readback_ms);
 }
 }
 printf("GE2_SERVICE_FRAMES_DONE %d\n", frames);
 {
 fsr4_stats_t st; fsr4_get_stats(&st);
 printf("E3_STATS frames=%lu over100=%lu total_ms=%.1f max_ms=%.3f\n",
 st.frames, st.frames_over_100ms, st.execute_total_ms, st.execute_max_ms);
 }
 rc = fsr4_shutdown();
 printf("GE2_SHUTDOWN_RC %d\n", rc);
 free(d_lr); free(d_rp); free(d_mv); free(fout.out);
 return rc == FSR4_OK ? 0: 1;
}
#endif /* FSR4_NO_MAIN */
