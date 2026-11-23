/*
 * SPDX-License-Identifier: MIT
 *
 * Diagnostic for GPU faults in the geometry shader / tessellation emulation. Each emulated draw
 * records the buffers it allocated; when the kernel reports a queue group fault, the most recent
 * records are logged along with how close the fault address is to each buffer, so a fault just
 * past the end of one allocation names that allocation.
 */

#ifndef PANVK_GS_DIAG_H
#define PANVK_GS_DIAG_H

#include <stdbool.h>
#include <stdint.h>

#define PANVK_GS_DIAG_MAX_BUFS 12
#define PANVK_GS_DIAG_MAX_PTRS 96


struct panvk_gs_diag_rec {
   const char *kind;
   uint32_t topology;
   uint32_t vpp;
   uint32_t instances;
   uint32_t records;
   uint32_t invocations;
   uint32_t out_verts;
   uint32_t out_indices;
   uint32_t indirect;
   uint32_t nbufs;
   uint64_t seq;
   /* CPU mappings of the index and position buffers, dumped for the newest draw when a fault
    * arrives. May read stale data if the GPU's writes are not yet visible to the CPU. */
   const uint32_t *dump_idx;
   uint32_t dump_idx_count;
   const float *dump_pos;
   uint32_t dump_pos_count;
   struct {
      const char *name;
      uint64_t addr;
      uint64_t size;
   } buf[PANVK_GS_DIAG_MAX_BUFS];
   /* Pointer parameters handed to the draw's emulation jobs (panvk_gs_diag_ptr/_push). A GPU
    * fault at address 0 means one of them was null when a job needed it. */
   uint32_t nptrs;
   uint32_t nstages;
   struct {
      char name[24];
      uint64_t value;
   } ptr[PANVK_GS_DIAG_MAX_PTRS];
};

static inline void
panvk_gs_diag_add(struct panvk_gs_diag_rec *rec, const char *name, uint64_t addr, uint64_t size)
{
   if (!addr || rec->nbufs >= PANVK_GS_DIAG_MAX_BUFS)
      return;
   rec->buf[rec->nbufs].name = name;
   rec->buf[rec->nbufs].addr = addr;
   rec->buf[rec->nbufs].size = size;
   rec->nbufs++;
}

/* Records one pointer parameter on the record this thread committed last (callers name push-block
 * fields s<N>.<field>, N from panvk_gs_diag_next_stage()). */
void panvk_gs_diag_ptr(const char *name, uint64_t value);
unsigned panvk_gs_diag_next_stage(void);

/* PANVK_GS_SKIP_INSTANCES=<n>: true for an application geometry shader draw with n or more
 * instances, which the caller then drops (counted as gs_skipped, the first few logged). Unset:
 * always false. */
bool panvk_gs_diag_skip_instances(uint32_t instances, uint32_t records);

/* PANVK_GS_NORAST_INSTANCES=<n>: a bisection aid. True for an application geometry shader draw
 * with n or more instances; the caller then runs the vertex and geometry compute jobs but skips
 * the rasterization pass. If a GPU fault goes away, it was in the rasterization pass. */
bool panvk_gs_diag_norast_instances(uint32_t instances);

/* PANVK_GS_VS_ONLY_INSTANCES=<n>: like the above, but the geometry shader compute job is skipped
 * too, so only the vertex compute job runs. */
bool panvk_gs_diag_vsonly_instances(uint32_t instances);

/* An application geometry shader was compiled: logs what it reads and writes, including the
 * built-in outputs (layer, viewport index) that the emulation has no tiler path for. */
void panvk_gs_diag_gs_shader(uint32_t in_verts, uint32_t out_verts, uint32_t max_verts,
                             uint32_t invocations, uint64_t outputs_written, bool writes_layer,
                             bool writes_viewport, bool writes_clip, bool writes_psiz);

/* A shader compile finished: logs compiles slower than 50 ms, and every 100th, with a running
 * total, to tell a slow-compile stall from a GPU stall. */
void panvk_gs_diag_compile(uint32_t shader_count, uint32_t first_stage, uint64_t elapsed_ns);

/* Breakdown of one compile: NIR instruction counts before the generic post-processing and
 * before the backend, time spent in each, the binary size, and the backend's per-phase times
 * (translate, opt, valhall, late, iter-sched, pressure-sched, regalloc, pack; 8 values). Logged for slow ones only. */
void panvk_gs_diag_compile_phases(uint32_t stage, uint32_t instrs_in, uint32_t instrs_backend,
                                  uint64_t postprocess_ns, uint64_t backend_ns, uint32_t bin_bytes,
                                  const uint64_t *phase_ns, const uint32_t *counters);

/* vkAllocateMemory refused because of PANVK_HEAP_LIMIT_MIB. Logged (first 20, then powers of
 * two) so a game dying right after shows up as the cause. */
void panvk_gs_diag_heap_refused(uint64_t size, uint64_t live, uint64_t limit);

/* A large vkAllocateMemory (64 MiB or more) succeeded: size, live total afterwards, memory
 * type, dedicated or not. Logged for the first 100, then every 10th. */
void panvk_gs_diag_big_alloc(uint64_t size, uint64_t live_after, uint32_t type, bool dedicated);

/* A block-compression image the driver emulates (hidden second plane, transcoded after copies):
 * logs the first ones, and a count of all, to the stats file. */
void panvk_gs_diag_bc_image(uint32_t vk_format, uint32_t carrier, uint32_t w, uint32_t h,
                            uint32_t mips, uint32_t layers, bool linear);

/* Stores a copy of rec in the ring and installs the fault hook on first use. */
void panvk_gs_diag_commit(const struct panvk_gs_diag_rec *rec);

/*
 * Opt-in draw statistics: when PANVK_STATS is set to a file path, a line with the per-second rate
 * of each kind of draw is appended to that file every two seconds. Without it the counters cost
 * one relaxed atomic add per draw.
 */
enum panvk_stat {
   PANVK_STAT_DRAW,
   PANVK_STAT_TESS,
   PANVK_STAT_GS_APP,
   PANVK_STAT_XFB,
   PANVK_STAT_POLYGON,
   PANVK_STAT_SUBMIT,
   PANVK_STAT_FRAME, /* vkQueuePresentKHR calls */
   /* Cumulative nanoseconds of CPU time, summed over all threads. */
   PANVK_STAT_NS_DRAW,        /* recording draws (vkCmdDraw*) */
   PANVK_STAT_NS_SUBMIT,      /* whole vkQueueSubmit on the kbase path */
   PANVK_STAT_NS_SUBMIT_WAIT, /* of which: CPU waits on semaphores/fences before submitting */
   PANVK_STAT_NS_BC_REFRESH,  /* of which: refreshing emulated BC linear images */
   PANVK_STAT_NS_GPU_BUSY,    /* wall time with a submission outstanding (see below) */
   PANVK_STAT_EMU_BYTES,      /* bytes of buffers allocated for GS/tessellation emulation */
   PANVK_STAT_EMU_PLANNED,    /* bytes a tessellation draw intended to allocate, before allocating */
   PANVK_STAT_WLS_BYTES,      /* bytes of workgroup-local scratch requested by compute dispatches */
   PANVK_STAT_TLS_BYTES,      /* bytes of thread stack (TLS) allocated per command buffer */
   PANVK_STAT_GS_SKIPPED,     /* GS draws dropped by PANVK_GS_SKIP_INSTANCES */
   PANVK_STAT_PRIVBO_NEW,     /* bytes of driver-internal BOs created (pools, shaders, ...) */
   PANVK_STAT_PRIVBO_FREED,   /* ... and destroyed */
   PANVK_STAT_VKMEM_NEW,      /* bytes from vkAllocateMemory */
   PANVK_STAT_VKMEM_FREED,    /* ... and vkFreeMemory */
   PANVK_STAT_TESS_REUSED,    /* bytes carved from the PANVK_TESS_REUSE scratch buffer */
   PANVK_STAT_TESS_REUSE_MISS, /* tess draws that did not fit it and allocated per draw */
   PANVK_STAT_COUNT,
};

void panvk_stat_count(enum panvk_stat stat);
void panvk_stat_add(enum panvk_stat stat, uint64_t n);

/*
 * Timing for the same file. panvk_stat_t0() returns 0 (one relaxed atomic load) when PANVK_STATS
 * is not set, otherwise a timestamp; panvk_stat_t1() adds the time elapsed since t0 to a
 * PANVK_STAT_NS_* counter and does nothing when t0 is 0. The per-line output reports each as
 * milliseconds of CPU time per second of wall time, so 1000 means one thread busy throughout.
 */
uint64_t panvk_stat_t0(void);
void panvk_stat_t1(enum panvk_stat stat, uint64_t t0);

/*
 * GPU-busy estimate: call with the panvk_stat_t0() taken just before a submission was handed to
 * the kernel, when that submission completes (completions arrive in order). Each submission is
 * counted busy from the later of its own submit time and the previous completion until its own
 * completion, so the sum is the wall time with at least one submission outstanding. It is an
 * upper bound on real GPU activity: time a submission spends waiting on its dependencies counts.
 * Reported as gpu_busy_pct. Does nothing when submit_ns is 0.
 */
void panvk_stat_gpu_done(uint64_t submit_ns);

/*
 * Called by a tessellation draw before it allocates its buffers, with the total it is about to
 * allocate. Counts it (emu_plan_mib_s), logs the first draws above 16 MiB to the stats file with
 * the sizes that produced the total, and returns true when PANVK_EMU_MAX_MIB=<n> is set and the
 * total exceeds it: the caller should then skip the draw instead of allocating. Unset: never true.
 */
bool panvk_gs_diag_plan(const char *kind, uint64_t bytes, uint32_t nr_patches, uint32_t cap,
                        uint32_t invocations, uint32_t out_verts, uint32_t out_indices);

/* Called for each compute dispatch that needs workgroup-local scratch (WLS): counts the bytes
 * (wls_mib_s) and logs the first few large ones. */
void panvk_gs_diag_wls(uint64_t bytes, uint32_t wls_size, uint32_t instances, uint32_t core_range,
                       bool indirect);

/* Called when a command buffer allocates its thread stack: counts the bytes (tls_mib_s) and logs
 * the first few. */
void panvk_gs_diag_tls(uint64_t bytes, uint32_t tls_size, uint32_t threads_per_core,
                       uint32_t core_range);

/* Logs each distinct (kind, format, size) once; kind 0 = colour render target, 1 = depth render
 * target, 2 = storage image view. */
void panvk_gs_diag_rt(unsigned kind, unsigned vk_format, unsigned width, unsigned height);

/* Logs each distinct (vk_format, blend equation, write mask, path) once, as the blend descriptor
 * of a draw is built. path: 0 = fixed function or opaque, 1 = blend shader. The factors and
 * functions are the pipe_blend_func / pipe_blendfactor values (PIPE_BLEND_ADD = 0, PIPE_BLENDFACTOR_ONE
 * = 1, _SRC_COLOR = 2, _SRC_ALPHA = 3, _DST_ALPHA = 4, _DST_COLOR = 5, ..., _ZERO = 0x11). */
void panvk_gs_diag_blend(unsigned vk_format, unsigned blend_enable, unsigned rgb_func,
                         unsigned rgb_src, unsigned rgb_dst, unsigned a_func, unsigned a_src,
                         unsigned a_dst, unsigned color_mask, unsigned path);

#endif
