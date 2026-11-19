/*
 * Copyright (C) 2021 Collabora, Ltd.
 * Copyright (C) 2021 Alyssa Rosenzweig
 * SPDX-License-Identifier: MIT
 */

#include "bi_builder.h"
#include "compiler.h"
#include "valhall.h"

#include <stdio.h>
#include <stdlib.h>

/* Diagnostic: PAN_BI_SKIP bits 512 (small int -> f32 fusion), 1024 (fabsneg fusion) and
 * 2048 (DISCARD+FCMP fusion) disable the three parts of the forward pass individually. */
static bool
bi_fwd_skip(unsigned bit)
{
   static unsigned mask;
   static bool init;
   if (!init) {
      const char *e = getenv("PAN_BI_SKIP");
      mask = e ? (unsigned)strtoul(e, NULL, 0) : 0;
      init = true;
   }
   return mask & bit;
}

/*
 * Due to a Bifrost encoding restriction, some instructions cannot have an abs
 * modifier on both sources. Check if adding a fabs modifier to a given source
 * of a binary instruction would cause this restriction to be hit.
 */
static bool
bi_would_impact_abs(unsigned arch, bi_instr *I, bi_index repl, unsigned s)
{
   return (arch <= 8) && I->src[1 - s].abs &&
          bi_is_word_equiv(I->src[1 - s], repl);
}

static bool
bi_takes_fabs(unsigned arch, bi_instr *I, bi_index repl, unsigned s)
{
   switch (I->op) {
   case BI_OPCODE_FCMP_V2F16:
   case BI_OPCODE_FMAX_V2F16:
   case BI_OPCODE_FMIN_V2F16:
      return !bi_would_impact_abs(arch, I, repl, s);
   case BI_OPCODE_FADD_V2F16:
      /*
       * For FADD.v2f16, the FMA pipe has the abs encoding hazard,
       * while the FADD pipe cannot encode a clamp. Either case in
       * isolation can be worked around in the scheduler, but both
       * together is impossible to encode. Avoid the hazard.
       */
      return !(I->clamp && bi_would_impact_abs(arch, I, repl, s));
   case BI_OPCODE_V2F32_TO_V2F16:
      /* TODO: Needs both match or lower */
      return false;
   case BI_OPCODE_FLOG_TABLE_F32:
      /* TODO: Need to check mode */
      return false;
   default:
      return bi_get_opcode_props(I)->abs & BITFIELD_BIT(s);
   }
}

static bool
bi_takes_fneg(unsigned arch, bi_instr *I, unsigned s)
{
   switch (I->op) {
   case BI_OPCODE_CUBE_SSEL:
   case BI_OPCODE_CUBE_TSEL:
   case BI_OPCODE_CUBEFACE:
      /* TODO: Bifrost encoding restriction: need to match or lower */
      return arch >= 9;
   case BI_OPCODE_FREXPE_F32:
   case BI_OPCODE_FREXPE_V2F16:
   case BI_OPCODE_FLOG_TABLE_F32:
      /* TODO: Need to check mode */
      return false;
   default:
      return bi_get_opcode_props(I)->neg & BITFIELD_BIT(s);
   }
}

static bool
bi_is_fabsneg(enum bi_opcode op, enum bi_size size)
{
   return (size == BI_SIZE_32 && op == BI_OPCODE_FABSNEG_F32) ||
          (size == BI_SIZE_16 && op == BI_OPCODE_FABSNEG_V2F16);
}

static enum bi_swizzle
bi_compose_swizzle_16(enum bi_swizzle a, enum bi_swizzle b)
{
   assert(a <= BI_SWIZZLE_H11);
   assert(b <= BI_SWIZZLE_H11);

   bool al = (a & BI_SWIZZLE_H10);
   bool ar = (a & BI_SWIZZLE_H01);
   bool bl = (b & BI_SWIZZLE_H10);
   bool br = (b & BI_SWIZZLE_H01);

   return ((al ? br : bl) ? BI_SWIZZLE_H10 : 0) |
          ((ar ? br : bl) ? BI_SWIZZLE_H01 : 0);
}

/* Like bi_replace_index, but composes instead of overwrites */

static inline bi_index
bi_compose_float_index(bi_index old, bi_index repl)
{
   /* abs(-x) = abs(+x) so ignore repl.neg if old.abs is set, otherwise
    * -(-x) = x but -(+x) = +(-x) so need to exclusive-or the negates */
   repl.neg = old.neg ^ (repl.neg && !old.abs);

   /* +/- abs(+/- abs(x)) = +/- abs(x), etc so just or the two */
   repl.abs |= old.abs;

   /* Use the old swizzle to select from the replacement swizzle */
   repl.swizzle = bi_compose_swizzle_16(old.swizzle, repl.swizzle);

   return repl;
}

/* DISCARD.b32(FCMP.f(x, y)) --> DISCARD.f(x, y) */

static inline bool
bi_fuse_discard_fcmp(bi_context *ctx, bi_instr *I, bi_instr *mod)
{
   if (!mod)
      return false;
   if (I->op != BI_OPCODE_DISCARD_B32)
      return false;
   if (mod->op != BI_OPCODE_FCMP_F32 && mod->op != BI_OPCODE_FCMP_V2F16)
      return false;
   if (mod->cmpf >= BI_CMPF_GTLT)
      return false;

   /* result_type doesn't matter */

   /* .abs and .neg modifiers allowed on Valhall DISCARD but not Bifrost */
   bool absneg = mod->src[0].neg || mod->src[0].abs;
   absneg |= mod->src[1].neg || mod->src[1].abs;

   if (ctx->arch <= 8 && absneg)
      return false;

   enum bi_swizzle r = I->src[0].swizzle;

   bi_builder b = bi_init_builder(ctx, bi_before_instr(I));
   I = bi_discard_f32(&b, mod->src[0], mod->src[1], mod->cmpf);

   if (mod->op == BI_OPCODE_FCMP_V2F16) {
      I->src[0].swizzle = bi_compose_swizzle_16(r, I->src[0].swizzle);
      I->src[1].swizzle = bi_compose_swizzle_16(r, I->src[1].swizzle);
   }

   return true;
}

/*
 * S32_TO_F32(S8_TO_S32(x)) -> S8_TO_F32 and friends. Round modes don't matter
 * because all 8-bit and 16-bit integers may be represented exactly as fp32.
 */
struct {
   enum bi_opcode inner;
   enum bi_opcode outer;
   enum bi_opcode small_replacement;
   enum bi_opcode wide_replacement;
} bi_small_int_patterns[] = {
   {BI_OPCODE_S8_TO_S32, BI_OPCODE_S32_TO_F32, BI_OPCODE_S8_TO_F32, BI_OPCODE_S32_TO_F32},
   {BI_OPCODE_U8_TO_U32, BI_OPCODE_U32_TO_F32, BI_OPCODE_U8_TO_F32, BI_OPCODE_U32_TO_F32},
   {BI_OPCODE_U8_TO_U32, BI_OPCODE_S32_TO_F32, BI_OPCODE_U8_TO_F32, BI_OPCODE_U32_TO_F32},
   {BI_OPCODE_S16_TO_S32, BI_OPCODE_S32_TO_F32, BI_OPCODE_S16_TO_F32, BI_OPCODE_S32_TO_F32},
   {BI_OPCODE_U16_TO_U32, BI_OPCODE_U32_TO_F32, BI_OPCODE_U16_TO_F32, BI_OPCODE_U32_TO_F32},
   {BI_OPCODE_U16_TO_U32, BI_OPCODE_S32_TO_F32, BI_OPCODE_U16_TO_F32, BI_OPCODE_U32_TO_F32},
};

static inline void
bi_fuse_small_int_to_f32(bi_context *ctx, bi_instr *I, bi_instr *mod)
{
   for (unsigned i = 0; i < ARRAY_SIZE(bi_small_int_patterns); ++i) {
      if (I->op != bi_small_int_patterns[i].outer)
         continue;
      if (mod->op != bi_small_int_patterns[i].inner)
         continue;

      assert(I->src[0].swizzle == BI_SWIZZLE_H01);
      I->src[0] = mod->src[0];
      I->round = BI_ROUND_NONE;

      /* On v11, All small int instructions are gone but we now can widen instead */
      if (ctx->arch >= 11)
         bi_set_opcode(I, bi_small_int_patterns[i].wide_replacement);
      else
         bi_set_opcode(I, bi_small_int_patterns[i].small_replacement);
   }
}

/* Diagnostic: PAN_BI_NOFUSE=<mask> stops the fabsneg fusion into whole groups of consumers:
 * 1 FROUND, 2 FRCP/FRSQ, 4 FMA_RSCALE/FADD_LSCALE/FADD_RSCALE, 8 FMIN/FMAX/FCMP,
 * 16 FABSNEG chains, 32 FADD, 64 FMA. */
static bool
bi_nofuse_group(const bi_instr *I)
{
   static unsigned mask;
   static bool init;
   if (!init) {
      const char *e = getenv("PAN_BI_NOFUSE");
      mask = e ? (unsigned)strtoul(e, NULL, 0) : 0;
      init = true;
   }
   if (!mask)
      return false;

   switch (I->op) {
   case BI_OPCODE_FROUND_F32:
   case BI_OPCODE_FROUND_V2F16:
      return mask & 1;
   case BI_OPCODE_FRCP_F32:
   case BI_OPCODE_FRCP_F16:
   case BI_OPCODE_FRSQ_F32:
   case BI_OPCODE_FRSQ_F16:
      return mask & 2;
   case BI_OPCODE_FMA_RSCALE_F32:
   case BI_OPCODE_FMA_RSCALE_V2F16:
   case BI_OPCODE_FADD_RSCALE_F32:
   case BI_OPCODE_FADD_LSCALE_F32:
      return mask & 4;
   case BI_OPCODE_FMIN_F32:
   case BI_OPCODE_FMAX_F32:
   case BI_OPCODE_FCMP_F32:
   case BI_OPCODE_FMIN_V2F16:
   case BI_OPCODE_FMAX_V2F16:
   case BI_OPCODE_FCMP_V2F16:
      return mask & 8;
   case BI_OPCODE_FABSNEG_F32:
   case BI_OPCODE_FABSNEG_V2F16:
      return mask & 16;
   case BI_OPCODE_FADD_F32:
   case BI_OPCODE_FADD_V2F16:
      return mask & 32;
   case BI_OPCODE_FMA_F32:
   case BI_OPCODE_FMA_V2F16:
      return mask & 64;
   default:
      return false;
   }
}

/* Diagnostic: PAN_BI_NOFUSE_FADD=<mask> narrows PAN_BI_NOFUSE=32 down: blocks the fusion into
 * FADD.f32 when 1 the source is src0, 2 the source is src1, 4 the folded modifier is neg only,
 * 8 abs only, 16 neg+abs, 32 the other FADD source is the same value as the one being folded,
 * 64 the other FADD source is already modified (neg or abs). Any set bit that matches blocks. */
static bool
bi_nofuse_fadd(const bi_instr *I, const bi_instr *mod, unsigned s)
{
   static unsigned mask;
   static bool init;
   if (!init) {
      const char *e = getenv("PAN_BI_NOFUSE_FADD");
      mask = e ? (unsigned)strtoul(e, NULL, 0) : 0;
      init = true;
   }
   if (!mask || I->op != BI_OPCODE_FADD_F32)
      return false;

   const bi_index r = mod->src[0];
   const bi_index other = I->src[1 - s];
   if ((mask & 1) && s == 0)
      return true;
   if ((mask & 2) && s == 1)
      return true;
   if ((mask & 4) && r.neg && !r.abs)
      return true;
   if ((mask & 8) && r.abs && !r.neg)
      return true;
   if ((mask & 16) && r.abs && r.neg)
      return true;
   if ((mask & 32) && bi_is_word_equiv(other, r))
      return true;
   if ((mask & 64) && (other.neg || other.abs))
      return true;
   return false;
}

/* Diagnostic: PAN_BI_FUSELOG=<path> appends one line per distinct fabsneg fusion (consumer opcode,
 * source index, modifiers, swizzles, shader stage). */
static void
bi_fuse_log(bi_context *ctx, bi_instr *I, bi_instr *mod, unsigned s)
{
   static const char *path;
   static bool init;
   static unsigned long long seen[512];
   static unsigned nseen;
   if (!init) {
      path = getenv("PAN_BI_FUSELOG");
      init = true;
   }
   if (!path)
      return;

   const bi_index o = I->src[s], r = mod->src[0];
   unsigned long long key = ((unsigned long long)I->op << 40) | ((unsigned long long)s << 36) |
                            ((unsigned long long)mod->op << 24) | (o.neg << 23) | (o.abs << 22) |
                            (r.neg << 21) | (r.abs << 20) | ((unsigned)o.swizzle << 12) |
                            ((unsigned)r.swizzle << 4) | (unsigned)ctx->stage;
   for (unsigned i = 0; i < nseen; i++)
      if (seen[i] == key)
         return;
   if (nseen >= ARRAY_SIZE(seen))
      return;
   seen[nseen++] = key;

   FILE *f = fopen(path, "a");
   if (!f)
      return;
   fprintf(f, "fuse: use=%s src%u size=%u  mod=%s  use_mod(neg=%d abs=%d swz=%d)  mod_src(neg=%d abs=%d swz=%d)  stage=%d arch=%u\n",
           bi_get_opcode_props(I)->name, s, bi_get_opcode_props(I)->size,
           bi_get_opcode_props(mod)->name, o.neg, o.abs, o.swizzle, r.neg, r.abs, r.swizzle,
           ctx->stage, ctx->arch);
   fclose(f);
}

void
bi_opt_mod_prop_forward(bi_context *ctx)
{
   bi_instr **lut = calloc(sizeof(bi_instr *), ctx->ssa_alloc);

   bi_foreach_instr_global_safe(ctx, I) {
      /* Try fusing FCMP into DISCARD.b32, building a new DISCARD.f32
       * instruction. As this is the only optimization DISCARD is
       * involved in, this shortcircuits other processing.
       */
      if (I->op == BI_OPCODE_DISCARD_B32) {
         if (!bi_fwd_skip(2048) && bi_is_ssa(I->src[0]) &&
             bi_fuse_discard_fcmp(ctx, I, lut[I->src[0].value])) {
            bi_remove_instruction(I);
         }

         continue;
      }

      bi_foreach_dest(I, d) {
         lut[I->dest[d].value] = I;
      }

      bi_foreach_ssa_src(I, s) {
         bi_instr *mod = lut[I->src[s].value];

         if (!mod)
            continue;

         unsigned size = bi_get_opcode_props(I)->size;

         if (!bi_fwd_skip(512))
            bi_fuse_small_int_to_f32(ctx, I, mod);

         if (!bi_fwd_skip(1024) && bi_is_fabsneg(mod->op, size)) {
            if (mod->src[0].abs && !bi_takes_fabs(ctx->arch, I, mod->src[0], s))
               continue;

            if (mod->src[0].neg && !bi_takes_fneg(ctx->arch, I, s))
               continue;

            if (bi_nofuse_group(I))
               continue;
            if (bi_nofuse_fadd(I, mod, s))
               continue;
            bi_fuse_log(ctx, I, mod, s);
            I->src[s] = bi_compose_float_index(I->src[s], mod->src[0]);
         }
      }
   }

   free(lut);
}

/* RSCALE has restrictions on how the clamp may be used, only used for
 * specialized transcendental sequences that set the clamp explicitly anyway */

static bool
bi_takes_clamp(bi_instr *I)
{
   switch (I->op) {
   case BI_OPCODE_FMA_RSCALE_F32:
   case BI_OPCODE_FMA_RSCALE_V2F16:
   case BI_OPCODE_FADD_RSCALE_F32:
      return false;
   case BI_OPCODE_FADD_V2F16:
      /* Encoding restriction */
      return !(I->src[0].abs && I->src[1].abs &&
               bi_is_word_equiv(I->src[0], I->src[1]));
   default:
      return bi_get_opcode_props(I)->clamp;
   }
}

static bool
bi_is_fclamp(enum bi_opcode op, enum bi_size size)
{
   return (size == BI_SIZE_32 && op == BI_OPCODE_FCLAMP_F32) ||
          (size == BI_SIZE_16 && op == BI_OPCODE_FCLAMP_V2F16);
}

static bool
bi_takes_f2f16_dest(unsigned arch, bi_instr *I, enum bi_round round)
{
   return arch >= 9 && va_op_dest_modifier_does_convert(I->op) &&
          I->round == round;
}

static bool
bi_optimizer_clamp(unsigned arch, bi_instr *I, bi_instr *use)
{
   if (!bi_is_fclamp(use->op, bi_get_opcode_props(I)->size))
      return false;
   if (!bi_takes_clamp(I))
      return false;

   /* swizzle != H01 converts in valhall, check if it's available */
   if (use->dest[0].swizzle != BI_SWIZZLE_H01 &&
       !bi_takes_f2f16_dest(arch, I, use->round))
      return false;

   /* Clamps are bitfields (clamp_m1_1/clamp_0_inf) so composition is OR */
   I->clamp |= use->clamp;
   I->dest[0] = use->dest[0];
   return true;
}

static bool
bi_is_f2f16(bi_instr *I)
{
   return I->op == BI_OPCODE_FADD_F32 &&
          bi_is_value_equiv(I->src[1], bi_negzero()) &&
          !I->src[0].abs && !I->src[0].neg &&
          I->src[0].swizzle == BI_SWIZZLE_NONE &&
          I->clamp == BI_CLAMP_NONE &&
          (I->dest->swizzle == BI_SWIZZLE_H0 ||
           I->dest->swizzle == BI_SWIZZLE_H1);
}

static bool
bi_takes_f2f16(bi_instr *I)
{
   return va_op_dest_modifier_does_convert(I->op) &&
          I->dest->swizzle == BI_SWIZZLE_H01;
}

static bool
bi_optimizer_f2f16(unsigned arch, bi_instr *I, bi_instr *use)
{
   /* Only available since Valhall */
   if (arch < 9 || !bi_is_f2f16(use) || !bi_takes_f2f16(I))
      return false;
   if (I->round != use->round)
      return false;
   I->dest[0] = use->dest[0];
   return true;
}

static enum bi_opcode
bi_sized_mux_op(unsigned size)
{
   switch (size) {
   case 8:
      return BI_OPCODE_MUX_V4I8;
   case 16:
      return BI_OPCODE_MUX_V2I16;
   case 32:
      return BI_OPCODE_MUX_I32;
   default:
      UNREACHABLE("invalid size");
   }
}

static bool
bi_is_fixed_mux(bi_instr *I, unsigned size, bi_index v1)
{
   return I->op == bi_sized_mux_op(size) &&
          bi_is_value_equiv(I->src[0], bi_zero()) &&
          bi_is_value_equiv(I->src[1], v1);
}

static bool
bi_takes_int_result_type(enum bi_opcode op)
{
   switch (op) {
   case BI_OPCODE_ICMP_I32:
   case BI_OPCODE_ICMP_S32:
   case BI_OPCODE_ICMP_U32:
   case BI_OPCODE_ICMP_V2I16:
   case BI_OPCODE_ICMP_V2S16:
   case BI_OPCODE_ICMP_V2U16:
   case BI_OPCODE_ICMP_V4I8:
   case BI_OPCODE_ICMP_V4S8:
   case BI_OPCODE_ICMP_V4U8:
   case BI_OPCODE_FCMP_F32:
   case BI_OPCODE_FCMP_V2F16:
      return true;
   default:
      return false;
   }
}

static bool
bi_takes_float_result_type(enum bi_opcode op)
{
   return (op == BI_OPCODE_FCMP_F32) || (op == BI_OPCODE_FCMP_V2F16);
}

/* CMP+MUX -> CMP with result type */
static bool
bi_optimizer_result_type(bi_instr *I, bi_instr *mux)
{
   if (bi_get_opcode_props(I)->size != bi_get_opcode_props(mux)->size)
      return false;

   enum bi_result_type result_type;

   if (bi_is_fixed_mux(mux, 32, bi_imm_f32(1.0)) ||
       bi_is_fixed_mux(mux, 16, bi_imm_f16(1.0))) {

      if (!bi_takes_float_result_type(I->op))
         return false;

      result_type = BI_RESULT_TYPE_F1;
   } else if (bi_is_fixed_mux(mux, 32, bi_imm_u32(1)) ||
              bi_is_fixed_mux(mux, 16, bi_imm_u16(1)) ||
              bi_is_fixed_mux(mux, 8, bi_imm_u8(1))) {

      if (!bi_takes_int_result_type(I->op))
         return false;

      result_type = BI_RESULT_TYPE_I1;
   } else {
      return false;
   }

   /* The MUX condition may have a non-identity swizzle (e.g. H10 = swap).
    * Compose that swizzle into the CMP sources so the fused CMP.f1/i1 compares
    * the right lane pairs.
    */
   enum bi_swizzle swizzle = mux->src[2].swizzle;
   if (swizzle != BI_SWIZZLE_H01) {
      if (bi_get_opcode_props(I)->size != BI_SIZE_16)
         return false;
      /* Check if swizzles are composable */
      if (swizzle > BI_SWIZZLE_H11)
         return false;

      I->src[0].swizzle = bi_compose_swizzle_16(swizzle, I->src[0].swizzle);
      I->src[1].swizzle = bi_compose_swizzle_16(swizzle, I->src[1].swizzle);
   }
   I->result_type = result_type;
   I->dest[0] = mux->dest[0];

   return true;
}

static bool
bi_is_var_tex(bi_instr *var, bi_instr *tex)
{
   return (var->op == BI_OPCODE_LD_VAR_IMM) &&
          (tex->op == BI_OPCODE_TEXS_2D_F16 ||
           tex->op == BI_OPCODE_TEXS_2D_F32) &&
          (var->register_format == BI_REGISTER_FORMAT_F32) &&
          ((var->sample == BI_SAMPLE_CENTER &&
            var->update == BI_UPDATE_STORE) ||
           (var->sample == BI_SAMPLE_NONE &&
            var->update == BI_UPDATE_RETRIEVE)) &&
          (tex->texture_index == tex->sampler_index) &&
          (tex->texture_index < 4) && (var->index < 8);
}

static bool
bi_optimizer_var_tex(bi_context *ctx, bi_instr *var, bi_instr *tex)
{
   if (!bi_is_var_tex(var, tex))
      return false;

   /* Construct the corresponding VAR_TEX intruction */
   bi_builder b = bi_init_builder(ctx, bi_after_instr(var));

   bi_instr *I = bi_var_tex_f32_to(&b, tex->dest[0], tex->lod_mode, var->sample,
                                   var->update, tex->texture_index, var->index);
   I->skip = tex->skip;

   if (tex->op == BI_OPCODE_TEXS_2D_F16)
      bi_set_opcode(I, BI_OPCODE_VAR_TEX_F16);

   /* Dead code elimination will clean up for us */
   return true;
}

void
bi_opt_mod_prop_backward(bi_context *ctx)
{
   unsigned count = ctx->ssa_alloc;
   bi_instr **uses = calloc(count, sizeof(*uses));
   BITSET_WORD *multiple = BITSET_CALLOC(count);

   bi_foreach_block_rev(ctx, block) {
      /* Watch out for PHI instructions in loops!
       * PHI sources are logically read at the end of the predecessor,
       * so process our source in successor phis first
       */
      bi_foreach_successor(block, succ) {
         unsigned s = bi_predecessor_index(succ, block);
         bi_foreach_instr_in_block(succ, phi) {
            /* the PHIs are all at the start of the block, so stop
             * when we see a non-PHI
             */
            if (phi->op != BI_OPCODE_PHI)
               break;
            if (phi->src[s].type == BI_INDEX_NORMAL)
               bi_record_use(uses, multiple, phi, s);
         }
      }
      /* now go through the instructions backwards */
      bi_foreach_instr_in_block_rev(block, I) {
         /* skip PHIs, they are handled specially */
         if (I->op == BI_OPCODE_PHI)
            continue;

         /* record uses */
         bi_foreach_ssa_src(I, s) {
            bi_record_use(uses, multiple, I, s);
         }

         /* check for definitions */
         if (I->nr_dests != 1)
            continue;

         bi_instr *use = uses[I->dest[0].value];

         if (!use || BITSET_TEST(multiple, I->dest[0].value))
            continue;

         /* Destination has a single use, try to propagate */
         bool propagated =
            bi_optimizer_clamp(ctx->arch, I, use) ||
            bi_optimizer_f2f16(ctx->arch, I, use) ||
            bi_optimizer_result_type(I, use);

         if (!propagated && I->op == BI_OPCODE_LD_VAR_IMM &&
             use->op == BI_OPCODE_SPLIT_I32) {
            /* Need to see through the split in a
             * ld_var_imm/split/var_tex  sequence
             */
            bi_instr *tex = uses[use->dest[0].value];

            if (!tex || BITSET_TEST(multiple, use->dest[0].value))
               continue;

            use = tex;
            propagated = bi_optimizer_var_tex(ctx, I, use);
         }

         if (propagated) {
            bi_remove_instruction(use);
            continue;
         }
      }
   }

   free(uses);
   free(multiple);
}

/*
 * Lower pseudo instructions that exist to simplify the optimizer. Returns the
 * replacement instruction, or NULL if no replacement is needed.
 */
static bool
bi_lower_opt_instruction_helper(bi_builder *b, bi_instr *I)
{
   bi_instr *repl;

   switch (I->op) {
   case BI_OPCODE_FABSNEG_F32:
   case BI_OPCODE_FCLAMP_F32:
      repl = bi_fadd_f32_to(b, I->dest[0], I->src[0], bi_negzero());
      repl->clamp = I->clamp;
      repl->round = I->round;
      return true;

   case BI_OPCODE_FABSNEG_V2F16:
   case BI_OPCODE_FCLAMP_V2F16:
      repl = bi_fadd_v2f16_to(b, I->dest[0], I->src[0], bi_negzero());
      repl->clamp = I->clamp;
      return true;

   case BI_OPCODE_DISCARD_B32: {
      bi_index src1 = bi_zero();
      /* Bifrost can only encode this if either swizzles are NONE or if both are
       * h0|h1
       */
      if (I->src[0].swizzle != BI_SWIZZLE_NONE) {
         assert(I->src[0].swizzle == BI_SWIZZLE_H0 ||
                I->src[0].swizzle == BI_SWIZZLE_H1);
         src1 = bi_half(src1, false);
      }

      bi_discard_f32(b, I->src[0], src1, BI_CMPF_NE);
      return true;
   }

   default:
      return false;
   }
}

void
bi_lower_opt_instructions(bi_context *ctx)
{
   bi_foreach_instr_global_safe(ctx, I) {
      bi_builder b = bi_init_builder(ctx, bi_before_instr(I));

      if (bi_lower_opt_instruction_helper(&b, I))
         bi_remove_instruction(I);
   }
}
