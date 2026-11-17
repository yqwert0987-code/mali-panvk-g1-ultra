/*
 * SPDX-License-Identifier: MIT
 */

#include "panvk_gs_diag.h"

#include <inttypes.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "kmod/kbase_kmod.h"
#include "util/log.h"
#include "util/os_misc.h"
#include "util/os_time.h"

#define RING_SIZE 32
/* How many of the newest records are listed in full when a fault arrives. */
#define LISTED_RECS 6
/* A buffer is reported as a likely culprit when the fault lies within this many bytes of it. */
#define NEAR_BYTES (1u << 20)

static pthread_mutex_t diag_lock = PTHREAD_MUTEX_INITIALIZER;
/* The sequence number this thread's last committed record got. */
static __thread uint64_t last_seq = ~0ull;
static struct panvk_gs_diag_rec ring[RING_SIZE];
static uint64_t next_seq;
static int hook_installed;

static FILE *stats_file;
static void stat_add(enum panvk_stat stat, uint64_t n);

/* Logs to Mesa's log and, when PANVK_STATS is set, also to the stats file as a "# " line. The
 * stats file is opened in append mode, so unlike MESA_LOG_FILE (reopened and truncated by every
 * process that creates a Vulkan instance, and a launcher starts many) it keeps the dump of a
 * process that then dies. */
static void __attribute__((format(printf, 1, 2)))
diag_logf(const char *fmt, ...)
{
   char buf[512];
   va_list ap;

   va_start(ap, fmt);
   vsnprintf(buf, sizeof(buf), fmt, ap);
   va_end(ap);

   mesa_loge("%s", buf);
   if (stats_file) {
      fprintf(stats_file, "# %s\n", buf);
      fflush(stats_file);
   }
}

static void
dump_on_fault(uint64_t sideband, uint32_t status, uint32_t kind, uint32_t queue)
{
   static const char *const kinds[] = {"group fault", "queue fault", "progress timeout",
                                       "tiler heap out of memory"};
   pthread_mutex_lock(&diag_lock);

   diag_logf("gs-diag: %s (type %u, queue %u), status 0x%08x sideband 0x%016" PRIx64 ", %" PRIu64
             " emulated draws recorded",
             kind < 4 ? kinds[kind] : "unknown", kind, queue, status, sideband, next_seq);

   const uint64_t total = next_seq < RING_SIZE ? next_seq : RING_SIZE;
   for (uint64_t n = 0; n < total; n++) {
      const struct panvk_gs_diag_rec *r = &ring[(next_seq - 1 - n) % RING_SIZE];
      const int listed = n < LISTED_RECS;

      if (listed)
         diag_logf("gs-diag: draw #%" PRIu64 " (%" PRIu64 " back) %s topo=%u vpp=%u inst=%u "
                   "recs=%u invocations=%u out_verts=%u out_indices=%u indirect=%u",
                   r->seq, n, r->kind, r->topology, r->vpp, r->instances, r->records,
                   r->invocations, r->out_verts, r->out_indices, r->indirect);

      for (uint32_t i = 0; i < r->nbufs; i++) {
         const uint64_t start = r->buf[i].addr;
         const uint64_t end = start + r->buf[i].size;
         const int inside = sideband >= start && sideband < end;
         const int near = sideband + NEAR_BYTES >= start && sideband <= end + NEAR_BYTES;

         if (!listed && !near)
            continue;

         if (inside)
            diag_logf("gs-diag:   %-10s 0x%012" PRIx64 " size %" PRIu64 "  FAULT INSIDE (+%" PRIu64
                      ")",
                      r->buf[i].name, start, r->buf[i].size, sideband - start);
         else if (sideband >= end && near)
            diag_logf("gs-diag:   %-10s 0x%012" PRIx64 " size %" PRIu64
                      "  FAULT PAST END by %" PRIu64,
                      r->buf[i].name, start, r->buf[i].size, sideband - end);
         else if (sideband < start && near)
            diag_logf("gs-diag:   %-10s 0x%012" PRIx64 " size %" PRIu64
                      "  FAULT BEFORE START by %" PRIu64,
                      r->buf[i].name, start, r->buf[i].size, start - sideband);
         else
            diag_logf("gs-diag:   %-10s 0x%012" PRIx64 " size %" PRIu64, r->buf[i].name, start,
                      r->buf[i].size);
      }
   }

   /* Pointer parameters that were 0, per listed draw. Many are legitimately 0 (an index buffer
    * on a non-indexed draw); the one a job dereferenced anywhere is not. */
   for (uint64_t n = 0; n < (total < LISTED_RECS ? total : LISTED_RECS); n++) {
      const struct panvk_gs_diag_rec *r = &ring[(next_seq - 1 - n) % RING_SIZE];
      char line[400];
      int len = 0;
      for (uint32_t i = 0; i < r->nptrs && len < (int)sizeof(line) - 30; i++)
         if (!r->ptr[i].value)
            len += snprintf(line + len, sizeof(line) - len, "%s ", r->ptr[i].name);
      diag_logf("gs-diag: draw #%" PRIu64 " (%" PRIu64 " back): %u pointers, zero: %s", r->seq, n,
                r->nptrs, len ? line : "(none)");
   }

   /* The newest draw's index and position buffers, as the CPU sees them. */
   if (total) {
      const struct panvk_gs_diag_rec *r = &ring[(next_seq - 1) % RING_SIZE];
      const uint32_t nidx = r->dump_idx_count < 96 ? r->dump_idx_count : 96;
      const uint32_t npos = r->dump_pos_count < 24 ? r->dump_pos_count : 24;

      if (r->dump_idx) {
         diag_logf("gs-diag: newest draw out_idx (%u of %u words):", nidx, r->dump_idx_count);
         for (uint32_t i = 0; i < nidx; i += 12) {
            char line[12 * 12 + 1];
            int len = 0;
            for (uint32_t j = i; j < i + 12 && j < nidx; j++)
               len += snprintf(line + len, sizeof(line) - len, "%x ", r->dump_idx[j]);
            diag_logf("gs-diag:   [%3u] %s", i, line);
         }
      }

      if (r->dump_pos) {
         diag_logf("gs-diag: newest draw out_pos (%u of %u vertices):", npos, r->dump_pos_count);
         for (uint32_t i = 0; i < npos; i++)
            diag_logf("gs-diag:   v%-2u %g %g %g %g", i, r->dump_pos[i * 4], r->dump_pos[i * 4 + 1],
                      r->dump_pos[i * 4 + 2], r->dump_pos[i * 4 + 3]);
      }
   }

   pthread_mutex_unlock(&diag_lock);
}

void
panvk_gs_diag_ptr(const char *name, uint64_t value)
{
   pthread_mutex_lock(&diag_lock);
   struct panvk_gs_diag_rec *r = &ring[last_seq % RING_SIZE];
   if (last_seq != ~0ull && r->seq == last_seq && r->nptrs < PANVK_GS_DIAG_MAX_PTRS) {
      snprintf(r->ptr[r->nptrs].name, sizeof(r->ptr[0].name), "%s", name);
      r->ptr[r->nptrs].value = value;
      r->nptrs++;
   }
   pthread_mutex_unlock(&diag_lock);
}

bool
panvk_gs_diag_skip_instances(uint32_t instances, uint32_t records)
{
   static int init;
   static uint32_t limit;
   static unsigned logged;

   if (!init) {
      const char *e = os_get_option("PANVK_GS_SKIP_INSTANCES");
      limit = e ? (uint32_t)strtoul(e, NULL, 0) : 0;
      init = 1;
   }
   if (!limit || instances < limit)
      return false;

   stat_add(PANVK_STAT_GS_SKIPPED, 1);
   if (stats_file && logged < 5) {
      logged++;
      diag_logf("gs-diag: skipping geometry shader draw: instances=%u records=%u "
                "(PANVK_GS_SKIP_INSTANCES=%u)",
                instances, records, limit);
   }
   return true;
}

bool
panvk_gs_diag_norast_instances(uint32_t instances)
{
   static int init;
   static uint32_t limit;
   static unsigned logged;

   if (!init) {
      const char *e = os_get_option("PANVK_GS_NORAST_INSTANCES");
      limit = e ? (uint32_t)strtoul(e, NULL, 0) : 0;
      init = 1;
   }
   if (!limit || instances < limit)
      return false;

   if (stats_file && logged < 5) {
      logged++;
      diag_logf("gs-diag: geometry shader draw runs its compute jobs only: instances=%u "
                "(PANVK_GS_NORAST_INSTANCES=%u)",
                instances, limit);
   }
   return true;
}

bool
panvk_gs_diag_vsonly_instances(uint32_t instances)
{
   static int init;
   static uint32_t limit;
   static unsigned logged;

   if (!init) {
      const char *e = os_get_option("PANVK_GS_VS_ONLY_INSTANCES");
      limit = e ? (uint32_t)strtoul(e, NULL, 0) : 0;
      init = 1;
   }
   if (!limit || instances < limit)
      return false;

   if (stats_file && logged < 5) {
      logged++;
      diag_logf("gs-diag: geometry shader draw runs its vertex job only: instances=%u "
                "(PANVK_GS_VS_ONLY_INSTANCES=%u)",
                instances, limit);
   }
   return true;
}

unsigned
panvk_gs_diag_next_stage(void)
{
   unsigned stage = 0;
   pthread_mutex_lock(&diag_lock);
   struct panvk_gs_diag_rec *r = &ring[last_seq % RING_SIZE];
   if (last_seq != ~0ull && r->seq == last_seq)
      stage = r->nstages++;
   pthread_mutex_unlock(&diag_lock);
   return stage;
}

void
panvk_gs_diag_commit(const struct panvk_gs_diag_rec *rec)
{
   pthread_mutex_lock(&diag_lock);

   if (!hook_installed) {
      pan_kmod_kbase_fault_hook = dump_on_fault;
      hook_installed = 1;
   }

   struct panvk_gs_diag_rec *slot = &ring[next_seq % RING_SIZE];
   *slot = *rec;
   slot->seq = next_seq++;
   last_seq = slot->seq;

   pthread_mutex_unlock(&diag_lock);

   uint64_t bytes = 0;
   for (uint32_t i = 0; i < rec->nbufs; i++)
      bytes += rec->buf[i].size;
   stat_add(PANVK_STAT_EMU_BYTES, bytes);
}

#define STATS_INTERVAL_NS 2000000000ull

static uint64_t stat_counts[PANVK_STAT_COUNT];
static uint64_t stat_prev[PANVK_STAT_COUNT];
static uint64_t stat_last_ns;
static uint64_t stat_start_ns;
/* 0: not looked at yet, 1: off, 2: on. */
static int stats_mode;
static pthread_mutex_t stats_lock = PTHREAD_MUTEX_INITIALIZER;

static void
stats_init(void)
{
   pthread_mutex_lock(&stats_lock);
   if (!stats_mode) {
      const char *path = os_get_option("PANVK_STATS");
      if (path && path[0])
         stats_file = fopen(path, "a");
      if (stats_file) {
         stat_start_ns = stat_last_ns = os_time_get_nano();
         stats_mode = 2;
         /* Says which settings produced the lines that follow. */
         const char *cap = os_get_option("PANVK_TESS_CAP");
         const char *share = os_get_option("PANVK_TLS_SHARE");
         const char *skip = os_get_option("PANVK_GS_SKIP_INSTANCES");
         const char *norast = os_get_option("PANVK_GS_NORAST_INSTANCES");
         const char *vsonly = os_get_option("PANVK_GS_VS_ONLY_INSTANCES");
         const char *chunks = os_get_option("PANVK_TILER_MAX_CHUNKS");
         const char *level = os_get_option("PANVK_TESS_MAX_LEVEL");
         const char *clear = os_get_option("PANVK_EMU_CLEAR");
         const char *inflight = os_get_option("PANVK_TILER_IN_FLIGHT");
         const char *nobc = os_get_option("PANVK_NO_BC_EMU");
         const char *bigreuse = os_get_option("PANVK_DESC_BIG_REUSE");
         const char *heaplim = os_get_option("PANVK_HEAP_LIMIT_MIB");
         const char *bcview = os_get_option("PANVK_BC_EMU_VIEW");
         /* rev 7: thread stack shared by default, fault dump lists pointers. */
         fprintf(stats_file,
                 "# panvk stats rev 7, pid %d, PANVK_TESS_CAP=%s PANVK_TLS_SHARE=%s "
                 "PANVK_GS_SKIP_INSTANCES=%s PANVK_GS_NORAST_INSTANCES=%s PANVK_GS_VS_ONLY_INSTANCES=%s PANVK_TILER_MAX_CHUNKS=%s PANVK_TESS_MAX_LEVEL=%s PANVK_EMU_CLEAR=%s PANVK_TILER_IN_FLIGHT=%s PANVK_NO_BC_EMU=%s PANVK_DESC_BIG_REUSE=%s PANVK_HEAP_LIMIT_MIB=%s PANVK_BC_EMU_VIEW=%s\n",
                 (int)getpid(), cap ? cap : "(unset)", share ? share : "(unset=on)",
                 skip ? skip : "(unset)", norast ? norast : "(unset)", vsonly ? vsonly : "(unset)",
                 chunks ? chunks : "(unset)",
                 level ? level : "(unset)", clear ? clear : "(unset)",
                 inflight ? inflight : "(unset)", nobc ? nobc : "(unset)",
                 bigreuse ? bigreuse : "(unset)", heaplim ? heaplim : "(unset)",
                 bcview ? bcview : "(unset)");
         fflush(stats_file);
      } else {
         stats_mode = 1;
      }
   }
   pthread_mutex_unlock(&stats_lock);
}

/* Resident set size of this process and the system's available memory, in MiB, for the stats
 * line: a process killed for memory dies without a word in any log of its own. */
static double
read_rss_mib(void)
{
   unsigned long size = 0, res = 0;
   FILE *f = fopen("/proc/self/statm", "r");
   if (!f)
      return 0.0;
   if (fscanf(f, "%lu %lu", &size, &res) != 2)
      res = 0;
   fclose(f);
   return (double)res * (double)sysconf(_SC_PAGESIZE) / (1024.0 * 1024.0);
}

static double
read_memavail_mib(void)
{
   char line[128];
   double mib = 0.0;
   FILE *f = fopen("/proc/meminfo", "r");
   if (!f)
      return 0.0;
   while (fgets(line, sizeof(line), f)) {
      unsigned long kb;
      if (sscanf(line, "MemAvailable: %lu kB", &kb) == 1) {
         mib = kb / 1024.0;
         break;
      }
   }
   fclose(f);
   return mib;
}

static void
stat_add(enum panvk_stat stat, uint64_t n)
{
   __atomic_fetch_add(&stat_counts[stat], n, __ATOMIC_RELAXED);

   if (__atomic_load_n(&stats_mode, __ATOMIC_RELAXED) == 0)
      stats_init();
   if (__atomic_load_n(&stats_mode, __ATOMIC_RELAXED) != 2)
      return;

   const uint64_t now = os_time_get_nano();
   if (now - __atomic_load_n(&stat_last_ns, __ATOMIC_RELAXED) < STATS_INTERVAL_NS)
      return;
   if (pthread_mutex_trylock(&stats_lock))
      return;

   const double secs = (now - stat_last_ns) / 1e9;
   uint64_t d[PANVK_STAT_COUNT];
   for (int i = 0; i < PANVK_STAT_COUNT; i++) {
      const uint64_t c = __atomic_load_n(&stat_counts[i], __ATOMIC_RELAXED);
      d[i] = c - stat_prev[i];
      stat_prev[i] = c;
   }
   fprintf(stats_file,
           "t=%.0fs per_second: draws=%.0f tess=%.0f gs=%.0f xfb=%.0f polygon_mode=%.0f"
           " submits=%.0f cpu_ms_per_s: draw=%.0f submit=%.0f submit_wait=%.0f bc_refresh=%.0f"
           " us_per_draw=%.2f gpu_busy_pct=%.0f emu_mib_per_s=%.1f rss_mib=%.0f"
           " memavail_mib=%.0f privbo_new_mib_s=%.1f privbo_free_mib_s=%.1f vkmem_new_mib_s=%.1f"
           " vkmem_free_mib_s=%.1f emu_plan_mib_s=%.1f wls_mib_s=%.1f tls_mib_s=%.1f gs_skipped=%.0f\n",
           (now - stat_start_ns) / 1e9, d[PANVK_STAT_DRAW] / secs, d[PANVK_STAT_TESS] / secs,
           d[PANVK_STAT_GS_APP] / secs, d[PANVK_STAT_XFB] / secs,
           d[PANVK_STAT_POLYGON] / secs, d[PANVK_STAT_SUBMIT] / secs,
           d[PANVK_STAT_NS_DRAW] / 1e6 / secs, d[PANVK_STAT_NS_SUBMIT] / 1e6 / secs,
           d[PANVK_STAT_NS_SUBMIT_WAIT] / 1e6 / secs, d[PANVK_STAT_NS_BC_REFRESH] / 1e6 / secs,
           d[PANVK_STAT_DRAW] ? d[PANVK_STAT_NS_DRAW] / 1e3 / d[PANVK_STAT_DRAW] : 0.0,
           d[PANVK_STAT_NS_GPU_BUSY] / 1e9 / secs * 100.0,
           d[PANVK_STAT_EMU_BYTES] / (1024.0 * 1024.0) / secs, read_rss_mib(),
           read_memavail_mib(), d[PANVK_STAT_PRIVBO_NEW] / 1048576.0 / secs,
           d[PANVK_STAT_PRIVBO_FREED] / 1048576.0 / secs,
           d[PANVK_STAT_VKMEM_NEW] / 1048576.0 / secs,
           d[PANVK_STAT_VKMEM_FREED] / 1048576.0 / secs,
           d[PANVK_STAT_EMU_PLANNED] / 1048576.0 / secs,
           d[PANVK_STAT_WLS_BYTES] / 1048576.0 / secs,
           d[PANVK_STAT_TLS_BYTES] / 1048576.0 / secs,
           d[PANVK_STAT_GS_SKIPPED] / secs);
   fflush(stats_file);
   __atomic_store_n(&stat_last_ns, now, __ATOMIC_RELAXED);

   pthread_mutex_unlock(&stats_lock);
}

void
panvk_stat_count(enum panvk_stat stat)
{
   stat_add(stat, 1);
}

bool
panvk_gs_diag_plan(const char *kind, uint64_t bytes, uint32_t nr_patches, uint32_t cap,
                   uint32_t invocations, uint32_t out_verts, uint32_t out_indices)
{
   static int limit_init;
   static uint64_t limit_bytes;
   static unsigned logged;

   if (!limit_init) {
      const char *e = os_get_option("PANVK_EMU_MAX_MIB");
      limit_bytes = e ? strtoull(e, NULL, 0) << 20 : 0;
      limit_init = 1;
   }

   stat_add(PANVK_STAT_EMU_PLANNED, bytes);

   const bool skip = limit_bytes && bytes > limit_bytes;
   if (stats_file && (skip || bytes > (16u << 20)) && logged < 24) {
      logged++;
      diag_logf("gs-diag: %s plans %.1f MiB: patches=%u cap=%u gs_invocations=%u out_verts=%u "
                "out_indices=%u%s",
                kind, bytes / 1048576.0, nr_patches, cap, invocations, out_verts, out_indices,
                skip ? " SKIPPED (PANVK_EMU_MAX_MIB)" : "");
   }
   return skip;
}

void
panvk_gs_diag_wls(uint64_t bytes, uint32_t wls_size, uint32_t instances, uint32_t core_range,
                  bool indirect)
{
   static unsigned logged;

   stat_add(PANVK_STAT_WLS_BYTES, bytes);
   if (stats_file && bytes > (8u << 20) && logged < 16) {
      logged++;
      diag_logf("gs-diag: dispatch wants %.1f MiB of WLS: wls_size=%u instances=%u core_id_range=%u "
                "indirect=%d",
                bytes / 1048576.0, wls_size, instances, core_range, (int)indirect);
   }
}

void
panvk_gs_diag_tls(uint64_t bytes, uint32_t tls_size, uint32_t threads_per_core, uint32_t core_range)
{
   static unsigned logged;

   stat_add(PANVK_STAT_TLS_BYTES, bytes);
   if (stats_file && logged < 12) {
      logged++;
      diag_logf("gs-diag: command buffer stack %.1f MiB: tls_size=%u threads_per_core=%u "
                "core_id_range=%u",
                bytes / 1048576.0, tls_size, threads_per_core, core_range);
   }
}

void
panvk_stat_add(enum panvk_stat stat, uint64_t n)
{
   stat_add(stat, n);
}

uint64_t
panvk_stat_t0(void)
{
   if (__atomic_load_n(&stats_mode, __ATOMIC_RELAXED) == 0)
      stats_init();
   if (__atomic_load_n(&stats_mode, __ATOMIC_RELAXED) != 2)
      return 0;
   return os_time_get_nano();
}

void
panvk_stat_t1(enum panvk_stat stat, uint64_t t0)
{
   if (!t0)
      return;
   stat_add(stat, os_time_get_nano() - t0);
}

static uint64_t gpu_last_done_ns;
static pthread_mutex_t gpu_lock = PTHREAD_MUTEX_INITIALIZER;

void
panvk_stat_gpu_done(uint64_t submit_ns)
{
   if (!submit_ns)
      return;

   const uint64_t now = os_time_get_nano();
   pthread_mutex_lock(&gpu_lock);
   const uint64_t start = submit_ns > gpu_last_done_ns ? submit_ns : gpu_last_done_ns;
   const uint64_t busy = now > start ? now - start : 0;
   gpu_last_done_ns = now;
   pthread_mutex_unlock(&gpu_lock);

   stat_add(PANVK_STAT_NS_GPU_BUSY, busy);
}

void
panvk_gs_diag_bc_image(uint32_t vk_format, uint32_t carrier, uint32_t w, uint32_t h,
                       uint32_t mips, uint32_t layers, bool linear)
{
   static unsigned total;
   unsigned n = __atomic_add_fetch(&total, 1, __ATOMIC_RELAXED);

   if (__atomic_load_n(&stats_mode, __ATOMIC_RELAXED) == 0)
      stats_init();
   if (stats_file && (n <= 60 || (n & (n - 1)) == 0))
      diag_logf("gs-diag: bc-emu image #%u: vk_format=%u carrier=%u %ux%u mips=%u layers=%u%s",
                n, vk_format, carrier, w, h, mips, layers, linear ? " linear" : "");
}

void
panvk_gs_diag_gs_shader(uint32_t in_verts, uint32_t out_verts, uint32_t max_verts,
                        uint32_t invocations, uint64_t outputs_written, bool writes_layer,
                        bool writes_viewport, bool writes_clip, bool writes_psiz)
{
   if (__atomic_load_n(&stats_mode, __ATOMIC_RELAXED) == 0)
      stats_init();
   if (stats_file)
      diag_logf("gs-diag: geometry shader compiled: in_verts=%u out_verts=%u max_vertices=%u "
                "invocations=%u outputs_written=0x%llx layer=%d viewport=%d clip_cull=%d "
                "point_size=%d",
                in_verts, out_verts, max_verts, invocations, (unsigned long long)outputs_written,
                writes_layer, writes_viewport, writes_clip, writes_psiz);
}

void
panvk_gs_diag_compile(uint32_t shader_count, uint32_t first_stage, uint64_t elapsed_ns)
{
   static uint64_t total_ns;
   static unsigned total_n;
   uint64_t tot = __atomic_add_fetch(&total_ns, elapsed_ns, __ATOMIC_RELAXED);
   unsigned n = __atomic_add_fetch(&total_n, 1, __ATOMIC_RELAXED);

   if (__atomic_load_n(&stats_mode, __ATOMIC_RELAXED) == 0)
      stats_init();
   /* Only slow compiles and every 100th are logged, to keep the file small. */
   if (stats_file && (elapsed_ns >= 50000000ull || n % 100 == 0))
      diag_logf("gs-diag: shader compile #%u: %u shader(s), first stage %u, took %llu ms "
                "(total so far %llu ms)",
                n, shader_count, first_stage, (unsigned long long)(elapsed_ns / 1000000),
                (unsigned long long)(tot / 1000000));
}

void
panvk_gs_diag_compile_phases(uint32_t stage, uint32_t instrs_in, uint32_t instrs_backend,
                             uint64_t postprocess_ns, uint64_t backend_ns, uint32_t bin_bytes,
                             const uint64_t *phase_ns, const uint32_t *counters)
{
   static const char *const names[] = {"translate", "opt",          "valhall",       "late",
                                       "iter_sched", "pressure_sched", "ra_other",     "pack",
                                       "ra_ssa_spill", "ra_interference", "ra_solve", "ra_spill"};
   char buf[400];
   size_t len = 0;

   if (postprocess_ns + backend_ns < 200000000ull)
      return;
   if (__atomic_load_n(&stats_mode, __ATOMIC_RELAXED) == 0)
      stats_init();
   if (!stats_file)
      return;

   for (unsigned i = 0; i < ARRAY_SIZE(names) && len < sizeof(buf); i++)
      len += snprintf(buf + len, sizeof(buf) - len, " %s=%llu", names[i],
                      (unsigned long long)(phase_ns[i] / 1000000));
   diag_logf("gs-diag: slow compile detail: stage=%u nir_instrs=%u -> %u postprocess=%llu ms "
             "backend=%llu ms binary=%u bytes; backend phases ms:%s; ra attempts=%u "
             "spill_iters=%u nodes=%u spill_bytes=%u",
             stage, instrs_in, instrs_backend, (unsigned long long)(postprocess_ns / 1000000),
             (unsigned long long)(backend_ns / 1000000), bin_bytes, buf, counters[0], counters[1],
             counters[2], counters[3]);
}

void
panvk_gs_diag_heap_refused(uint64_t size, uint64_t live, uint64_t limit)
{
   static unsigned count;
   unsigned n = __atomic_add_fetch(&count, 1, __ATOMIC_RELAXED);

   if (n > 20 && (n & (n - 1)))
      return;
   if (__atomic_load_n(&stats_mode, __ATOMIC_RELAXED) == 0)
      stats_init();
   if (stats_file)
      diag_logf("gs-diag: vkAllocateMemory refused #%u: %llu MiB requested, %llu MiB live, "
                "limit %llu MiB (PANVK_HEAP_LIMIT_MIB)",
                n, (unsigned long long)(size >> 20), (unsigned long long)(live >> 20),
                (unsigned long long)(limit >> 20));
}

void
panvk_gs_diag_big_alloc(uint64_t size, uint64_t live_after, uint32_t type, bool dedicated)
{
   static unsigned count;

   if (size < (64ull << 20))
      return;
   unsigned n = __atomic_add_fetch(&count, 1, __ATOMIC_RELAXED);
   if (n > 100 && n % 10)
      return;
   if (__atomic_load_n(&stats_mode, __ATOMIC_RELAXED) == 0)
      stats_init();
   if (stats_file)
      diag_logf("gs-diag: big allocation #%u: %llu MiB type %u%s, %llu MiB live after", n,
                (unsigned long long)(size >> 20), type, dedicated ? " dedicated" : "",
                (unsigned long long)(live_after >> 20));
}
