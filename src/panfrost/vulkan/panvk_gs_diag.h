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

#include <stdint.h>

#define PANVK_GS_DIAG_MAX_BUFS 12

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
   PANVK_STAT_COUNT,
};

void panvk_stat_count(enum panvk_stat stat);

#endif
