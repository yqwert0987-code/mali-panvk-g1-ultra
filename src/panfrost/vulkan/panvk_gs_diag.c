/*
 * SPDX-License-Identifier: MIT
 */

#include "panvk_gs_diag.h"

#include <inttypes.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>

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
static struct panvk_gs_diag_rec ring[RING_SIZE];
static uint64_t next_seq;
static int hook_installed;

static void
dump_on_fault(uint64_t sideband, uint32_t status)
{
   pthread_mutex_lock(&diag_lock);

   mesa_loge("gs-diag: fault status 0x%08x sideband 0x%016" PRIx64 ", %" PRIu64
             " emulated draws recorded",
             status, sideband, next_seq);

   const uint64_t total = next_seq < RING_SIZE ? next_seq : RING_SIZE;
   for (uint64_t n = 0; n < total; n++) {
      const struct panvk_gs_diag_rec *r = &ring[(next_seq - 1 - n) % RING_SIZE];
      const int listed = n < LISTED_RECS;

      if (listed)
         mesa_loge("gs-diag: draw #%" PRIu64 " (%" PRIu64 " back) %s topo=%u vpp=%u inst=%u "
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
            mesa_loge("gs-diag:   %-10s 0x%012" PRIx64 " size %" PRIu64 "  FAULT INSIDE (+%" PRIu64
                      ")",
                      r->buf[i].name, start, r->buf[i].size, sideband - start);
         else if (sideband >= end && near)
            mesa_loge("gs-diag:   %-10s 0x%012" PRIx64 " size %" PRIu64
                      "  FAULT PAST END by %" PRIu64,
                      r->buf[i].name, start, r->buf[i].size, sideband - end);
         else if (sideband < start && near)
            mesa_loge("gs-diag:   %-10s 0x%012" PRIx64 " size %" PRIu64
                      "  FAULT BEFORE START by %" PRIu64,
                      r->buf[i].name, start, r->buf[i].size, start - sideband);
         else
            mesa_loge("gs-diag:   %-10s 0x%012" PRIx64 " size %" PRIu64, r->buf[i].name, start,
                      r->buf[i].size);
      }
   }

   /* The newest draw's index and position buffers, as the CPU sees them. */
   if (total) {
      const struct panvk_gs_diag_rec *r = &ring[(next_seq - 1) % RING_SIZE];
      const uint32_t nidx = r->dump_idx_count < 96 ? r->dump_idx_count : 96;
      const uint32_t npos = r->dump_pos_count < 24 ? r->dump_pos_count : 24;

      if (r->dump_idx) {
         mesa_loge("gs-diag: newest draw out_idx (%u of %u words):", nidx, r->dump_idx_count);
         for (uint32_t i = 0; i < nidx; i += 12) {
            char line[12 * 12 + 1];
            int len = 0;
            for (uint32_t j = i; j < i + 12 && j < nidx; j++)
               len += snprintf(line + len, sizeof(line) - len, "%x ", r->dump_idx[j]);
            mesa_loge("gs-diag:   [%3u] %s", i, line);
         }
      }

      if (r->dump_pos) {
         mesa_loge("gs-diag: newest draw out_pos (%u of %u vertices):", npos, r->dump_pos_count);
         for (uint32_t i = 0; i < npos; i++)
            mesa_loge("gs-diag:   v%-2u %g %g %g %g", i, r->dump_pos[i * 4], r->dump_pos[i * 4 + 1],
                      r->dump_pos[i * 4 + 2], r->dump_pos[i * 4 + 3]);
      }
   }

   pthread_mutex_unlock(&diag_lock);
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

   pthread_mutex_unlock(&diag_lock);
}

#define STATS_INTERVAL_NS 2000000000ull

static uint64_t stat_counts[PANVK_STAT_COUNT];
static uint64_t stat_prev[PANVK_STAT_COUNT];
static uint64_t stat_last_ns;
static uint64_t stat_start_ns;
/* 0: not looked at yet, 1: off, 2: on. */
static int stats_mode;
static FILE *stats_file;
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
      } else {
         stats_mode = 1;
      }
   }
   pthread_mutex_unlock(&stats_lock);
}

void
panvk_stat_count(enum panvk_stat stat)
{
   __atomic_fetch_add(&stat_counts[stat], 1, __ATOMIC_RELAXED);

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
           "t=%.0fs per_second: draws=%.0f tess=%.0f gs=%.0f xfb=%.0f polygon_mode=%.0f\n",
           (now - stat_start_ns) / 1e9, d[PANVK_STAT_DRAW] / secs, d[PANVK_STAT_TESS] / secs,
           d[PANVK_STAT_GS_APP] / secs, d[PANVK_STAT_XFB] / secs,
           d[PANVK_STAT_POLYGON] / secs);
   fflush(stats_file);
   __atomic_store_n(&stat_last_ns, now, __ATOMIC_RELAXED);

   pthread_mutex_unlock(&stats_lock);
}
