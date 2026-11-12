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
