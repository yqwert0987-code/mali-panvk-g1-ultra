# Sekiro on PanVK (Mali-G1-Ultra, kbase CSF): investigation notes

Setup: OPPO PLG110 (MT6993, Mali-G1-Ultra MC12, arch v14), Bannerlator/Winlator, FEX/Box64EC.
Game: Sekiro (DX11). Renderer used so far: wined3d (OpenGL) on Zink; the DXVK run was never
confirmed to be drawing (the Wine log of that run still showed wined3d lines).

## What works now
- The game starts, runs the cutscene and gameplay at 10-20 fps for about two minutes.
- The out-of-memory kill after the cutscene is gone: command-buffer thread stacks are shared
  (default on, PANVK_TLS_SHARE=0 turns it off).

## Opt-in switches (none changes a default)
| Variable | Effect |
|---|---|
| PANVK_STATS=<file> | per-second stats and fault dumps, appended to the file |
| PANVK_TESS_MAX_LEVEL=<1..64> | tessellator lowers levels above this; far smaller buffers and meshes |
| PANVK_TESS_CAP=<n> | caps tessellation output per draw (drops primitives) |
| PANVK_GS_SKIP_INSTANCES=<n> | drops app geometry shader draws with n or more instances |
| PANVK_GS_VS_ONLY_INSTANCES=<n> | runs only the vertex job of those draws (no hang) |
| PANVK_GS_NORAST_INSTANCES=<n> | runs vertex + GS jobs, skips rasterization (hangs) |
| PANVK_TILER_MAX_CHUNKS=<64..4096> | tiler heap cap, 2 MiB chunks (1024 = memory starvation) |
| PANVK_TILER_IN_FLIGHT=<1..65535> | render passes in flight on the kbase heap |
| PANVK_EMU_CLEAR=1 | zero emulation index/position buffers on allocation |
| PANVK_EMU_MAX_MIB=<n> | skip tessellation draws planning more than n MiB |
| PANVK_NO_BC_EMU=1 | block-compression emulation off |
| PANVK_WLS_REUSE=1 | untested workgroup-memory reuse |

## Findings
1. 64-instance geometry shader draws (inst=64, recs=4, invocations=128, out_verts=384) fault or
   hang in the GS compute job. Vertex job alone is fine; the rasterization pass is not the cause.
2. Tiler heap out of memory (kind 3, queue 0) ends most runs after ~120-290 s in a heavy
   tessellation scene. Raising PANVK_TILER_MAX_CHUNKS delays it (64 -> ~140 s, 256 -> ~290 s) and
   1024 starves the phone. It still happens with tiny tessellation output (MAX_LEVEL=8), so the
   geometry volume is not the whole story.
3. PANVK_TILER_IN_FLIGHT=2 turned the heap fault into a GPU group fault (status 0x008602c0,
   sideband 0x0000212200000a00, a non-null unmapped address) on both a 64-instance GS draw and a
   tessellation draw. Not understood.
4. Assets render with wrong colours (purple/blue/black) from the moment the game loads; UI is
   fine. Same with BC emulation on or off. The BC emulation is selected from a texture-feature
   register decoded with v6/v7 bit positions (panvk_bc_emu.c), which may be wrong for v14: not
   verified.
5. The logos fade correctly under the second renderer setting; the OpenGL path fades wrongly.
6. Memory: the game plus emulation leaves 0.3-1.5 GB free late in a run; the last runs die
   there too.

## Next steps if this is picked up again
- Confirm DXVK is really the renderer (DXVK_HUD=version,devinfo, Wine log without wined3d_ lines).
- Read the v14 texture-feature register and compare with TF_* bit positions.
- Decode the group-fault address of finding 3 (it may be two 32-bit values glued together).
- Find why the GS compute job faults at 64 instances (layered output / gl_Layer is unsupported).

Run settings that gave the longest run so far: PANVK_TESS_MAX_LEVEL=8,
PANVK_GS_VS_ONLY_INSTANCES=64, PANVK_TILER_MAX_CHUNKS=256.

## No-rebuild toggles worth trying for the wrong colours / vanishing ground (untested)
- PANVK_DEBUG=noafbc: no AFBC compression on images. A deferred renderer writes its G-buffer to
  offscreen targets and samples them in the lighting pass; broken AFBC on v14 would colour every
  3D asset wrongly while the UI (drawn straight to the swapchain) stays fine, and a broken depth
  buffer would make the ground vanish and pop back.
- PANVK_DEBUG=no_crc, PANVK_DEBUG=linear: other layout / tile-CRC suspects.
- PANVK_NO_MEDIUMP=1: keeps all shader maths 32-bit (the driver runs RelaxedPrecision in 16 bits).

## Shader compile time (measured)
Sekiro spent ~92 s compiling shaders before gameplay. Per-phase timing (stats file "slow compile
detail" lines) showed ~99% of the time in register allocation: shaders needing ~100 spills ran
~100 full interference passes (an 11412-instruction compute shader: 13 s, compiled 5-7 times).
Spilling up to 8 values per failed attempt (PAN_RA_SPILL_BATCH, default 8; 1 = old
behaviour) cut total compile time to ~37 s and that shader to ~2.6-3.2 s, spill memory +4%
(1840 -> 1920 bytes). Remaining: the same shader is still compiled repeatedly (no cache hit).
