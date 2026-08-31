"""O1: query-row tiling for the gfx950 Gluon fp8 MQA logits kernel.

Stock kernel: grid=(seq_len,), num_warps=1 -> one wave per query token, so the
whole indexer K stream is re-read once per query row (16384x amplification).

This version: grid=(cdiv(seq_len, RPB),), still num_warps=1, but each wave owns
ROWS_PER_BLOCK consecutive query rows. The KV tile is async-copied into LDS once
and consumed RPB times out of registers -> global/L2 K traffic divided by RPB.

Design choice: RPB rows are iterated with a trace-time `static_range` over a
tuple of per-row Q fragments, rather than widening the MFMA M axis to
RPB*NUM_HEADS. Reason: the accumulator stays [NUM_HEADS, BLOCK_KV] (16 VGPR)
and only one is live at a time, versus RPB*16 VGPR for a widened M axis. It also
leaves `_weighted_sum_fma_fold`'s head-reduction layout algebra untouched.

Causal handling: rows in a block have different windows, so the tile loop walks
the union [min_r start_r, max_r end_r) and every store is masked per row. With
the production pattern ke[i] = prefix + i + 1 the union costs at most RPB-1
extra keys per row (<= 7 of ~57344 at RPB=8).

Tuples are read-only inside the runtime KV loop (Triton carries only scalars
across `tl.range`); per-tile addressing is derived from a scalar tile offset.
"""

import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton._gluon_kernels.gfx950.attention.fp8_mqa_logits import (
    MQAAsyncKVLoader,
    _load_kv_scales_block,
    _mqa_dot,
    _store_logits_block,
    _weighted_sum_fma_fold,
    relu_f32,
)

_PROP_NAN_NONE = gl.constexpr(tl.PropagateNan.NONE)


@gluon.jit
def relu_f32_fast(x):
    """O2: plain fmaxnum -> single v_max_f32, no NaN-propagation fixup."""
    return gl.maximum(x, 0.0, propagate_nan=_PROP_NAN_NONE)


@gluon.jit
def _score_rows(
    mfma_qs,
    w_blocks,
    logits_ptrs,
    starts,
    ends,
    mfma_k,
    kv_scales,
    tile_off,
    tile_start,
    store_arange,
    stride_logits_k,
    ROWS_PER_BLOCK: gl.constexpr,
    NUM_HEADS: gl.constexpr,
    BLOCK_KV: gl.constexpr,
    mfma_layout: gl.constexpr,
    NUM_CHAINS: gl.constexpr,
    USE_BUFFER_STORE: gl.constexpr,
    FAST_RELU: gl.constexpr,
    CHECK_START: gl.constexpr,
    STORE_CACHE: gl.constexpr,
    OUT_BF16: gl.constexpr,
):
    """One KV tile x ROWS_PER_BLOCK query rows, reusing `mfma_k` from registers."""
    pos = tile_start + tile_off + store_arange
    store_offsets = (tile_off + store_arange) * stride_logits_k
    for r in gl.static_range(ROWS_PER_BLOCK):
        scores = _mqa_dot(mfma_qs[r], mfma_k, NUM_HEADS, BLOCK_KV, mfma_layout)
        if FAST_RELU:
            scores = relu_f32_fast(scores)
        else:
            scores = relu_f32(scores)
        scores = _weighted_sum_fma_fold(
            scores, w_blocks[r], NUM_HEADS, BLOCK_KV, mfma_layout, NUM_CHAINS
        )
        scores = scores * kv_scales
        # Rows share one tile window, so each row needs its own causal mask.
        # ends[r] == 0 also encodes "row past seq_len" -> stores nothing.
        if CHECK_START:
            mask = (pos < ends[r]) & (pos >= starts[r])
        else:
            mask = pos < ends[r]
        if OUT_BF16:  # O7: halve the logits write stream
            scores = scores.to(gl.bfloat16)
        if STORE_CACHE == "":
            _store_logits_block(
                logits_ptrs[r], store_offsets, scores, USE_BUFFER_STORE, mask=mask
            )
        else:
            # O3: `.cs` marks the line evict-first so the 6.4 GB logits stream
            # does not push the reused indexer K out of L2 / Infinity Cache.
            gl.amd.cdna4.buffer_store(
                scores, ptr=logits_ptrs[r], offsets=store_offsets, mask=mask,
                cache=STORE_CACHE,
            )


@gluon.jit
def mqa_logits_loop_rpb(
    kv_loader,
    mfma_qs,
    w_blocks,
    logits_ptrs,
    starts,
    ends,
    kv_scales_ptr,
    tile_start,
    tile_end,
    num_full_tiles,
    ROWS_PER_BLOCK: gl.constexpr,
    NUM_HEADS: gl.constexpr,
    BLOCK_KV: gl.constexpr,
    stride_logits_k,
    mfma_layout: gl.constexpr,
    dot_b_layout: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    NUM_CHAINS: gl.constexpr,
    USE_BUFFER_LOAD: gl.constexpr,
    USE_BUFFER_STORE: gl.constexpr,
    FAST_RELU: gl.constexpr,
    UNIFORM_START: gl.constexpr,
    STORE_CACHE: gl.constexpr,
    OUT_BF16: gl.constexpr,
):
    store_arange = gl.arange(0, BLOCK_KV, layout=gl.SliceLayout(0, mfma_layout))
    relative_end: gl.int32 = tile_end - tile_start

    kv_loader.load_to_shared(
        tile_start, buffer_id=0, USE_BUFFER_LOAD=USE_BUFFER_LOAD, masked=True
    )
    kv_loader.load_to_shared(
        tile_start + BLOCK_KV, buffer_id=1, USE_BUFFER_LOAD=USE_BUFFER_LOAD,
        masked=True,
    )

    buf_cur: gl.int32 = 0
    for i in tl.range(0, num_full_tiles - 2):
        tile_off = i * BLOCK_KV
        kv_scales = _load_kv_scales_block(
            kv_scales_ptr, tile_off, BLOCK_KV, mfma_layout, USE_BUFFER_LOAD,
            relative_end,
        )
        mfma_k = kv_loader.load_from_shared(
            wait_count=1, target_layout=dot_b_layout, buffer_id=buf_cur
        )
        kv_loader.load_to_shared(
            tile_start + tile_off + 2 * BLOCK_KV,
            buffer_id=buf_cur,
            USE_BUFFER_LOAD=USE_BUFFER_LOAD,
        )
        _score_rows(
            mfma_qs, w_blocks, logits_ptrs, starts, ends, mfma_k, kv_scales,
            tile_off, tile_start, store_arange, stride_logits_k,
            ROWS_PER_BLOCK, NUM_HEADS, BLOCK_KV, mfma_layout, NUM_CHAINS,
            USE_BUFFER_STORE, FAST_RELU, CHECK_START=not UNIFORM_START,
            STORE_CACHE=STORE_CACHE, OUT_BF16=OUT_BF16,
        )
        buf_cur = 1 - buf_cur

    tail0 = gl.maximum(num_full_tiles - 2, 0)

    if num_full_tiles > 1:
        tile_off = tail0 * BLOCK_KV
        kv_scales = _load_kv_scales_block(
            kv_scales_ptr, tile_off, BLOCK_KV, mfma_layout, USE_BUFFER_LOAD,
            relative_end,
        )
        mfma_k = kv_loader.load_from_shared(
            wait_count=1, target_layout=dot_b_layout, buffer_id=buf_cur
        )
        kv_loader.load_to_shared(
            tile_start + num_full_tiles * BLOCK_KV,
            buffer_id=buf_cur,
            USE_BUFFER_LOAD=USE_BUFFER_LOAD,
            masked=True,
        )
        _score_rows(
            mfma_qs, w_blocks, logits_ptrs, starts, ends, mfma_k, kv_scales,
            tile_off, tile_start, store_arange, stride_logits_k,
            ROWS_PER_BLOCK, NUM_HEADS, BLOCK_KV, mfma_layout, NUM_CHAINS,
            USE_BUFFER_STORE, FAST_RELU, CHECK_START=not UNIFORM_START,
            STORE_CACHE=STORE_CACHE, OUT_BF16=OUT_BF16,
        )
        buf_cur = 1 - buf_cur

    # Last full tile. num_full_tiles == 0 still consumes prologue buffer 0 here
    # (matching the stock peel order), so clamp rather than go negative.
    last_full = gl.maximum(num_full_tiles - 1, 0)
    tile_off = last_full * BLOCK_KV
    kv_scales = _load_kv_scales_block(
        kv_scales_ptr, tile_off, BLOCK_KV, mfma_layout, USE_BUFFER_LOAD,
        relative_end, masked=True,
    )
    mfma_k = kv_loader.load_from_shared(
        wait_count=1, target_layout=dot_b_layout, buffer_id=buf_cur
    )
    _score_rows(
        mfma_qs, w_blocks, logits_ptrs, starts, ends, mfma_k, kv_scales,
        tile_off, tile_start, store_arange, stride_logits_k, ROWS_PER_BLOCK,
        NUM_HEADS, BLOCK_KV, mfma_layout, NUM_CHAINS, USE_BUFFER_STORE,
        FAST_RELU, CHECK_START=True, STORE_CACHE=STORE_CACHE, OUT_BF16=OUT_BF16,
    )
    buf_cur = 1 - buf_cur

    # Partial tail
    tile_off = (last_full + 1) * BLOCK_KV
    kv_scales = _load_kv_scales_block(
        kv_scales_ptr, tile_off, BLOCK_KV, mfma_layout, USE_BUFFER_LOAD,
        relative_end, masked=True,
    )
    mfma_k = kv_loader.load_from_shared(
        wait_count=0, target_layout=dot_b_layout, buffer_id=buf_cur
    )
    _score_rows(
        mfma_qs, w_blocks, logits_ptrs, starts, ends, mfma_k, kv_scales,
        tile_off, tile_start, store_arange, stride_logits_k, ROWS_PER_BLOCK,
        NUM_HEADS, BLOCK_KV, mfma_layout, NUM_CHAINS, USE_BUFFER_STORE,
        FAST_RELU, CHECK_START=True, STORE_CACHE=STORE_CACHE, OUT_BF16=OUT_BF16,
    )


@gluon.jit
def _gluon_fp8_mqa_logits_kernel_rpb(
    Q_ptr,  # fp8e4m3 [seq_len, NUM_HEADS, HEAD_SIZE]
    KV_ptr,  # fp8e4m3 [seq_len_kv, HEAD_SIZE]
    kv_scales_ptr,  # fp32   [seq_len_kv]
    weights_ptr,  # fp32   [seq_len, NUM_HEADS]
    cu_start_ptr,  # int32  [seq_len]
    cu_end_ptr,  # int32  [seq_len]
    logits_ptr,  # fp32   [seq_len, seq_len_kv]
    seq_len: gl.int32,
    seq_len_kv: gl.int32,
    NUM_HEADS: gl.constexpr,
    HEAD_SIZE: gl.constexpr,
    stride_q_s: gl.int32,
    stride_q_h: gl.constexpr,
    stride_q_d: gl.constexpr,
    stride_kv_s: gl.int32,
    stride_kv_d: gl.constexpr,
    stride_w_s: gl.int32,
    stride_w_h: gl.constexpr,
    stride_logits_s: gl.int32,
    stride_logits_k: gl.int32,
    BLOCK_KV: gl.constexpr,
    ROWS_PER_BLOCK: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    NUM_CHAINS: gl.constexpr,
    USE_BUFFER_LOAD: gl.constexpr,
    USE_BUFFER_STORE: gl.constexpr,
    USE_PADDED_SHARED_LAYOUT: gl.constexpr,
    FAST_RELU: gl.constexpr,
    UNIFORM_START: gl.constexpr,
    STORE_CACHE: gl.constexpr,
    OUT_BF16: gl.constexpr,
):
    gl.static_assert(NUM_BUFFERS == 2, "NUM_BUFFERS must be 2 (double buffering)")

    NUM_WARPS: gl.constexpr = 1
    # Reverse order so the heaviest row-blocks (largest end_ind) launch first.
    block_id = gl.num_programs(0) - gl.program_id(axis=0) - 1
    r0 = block_id * ROWS_PER_BLOCK

    if not USE_BUFFER_LOAD:
        stride_kv_s = stride_kv_s.to(gl.int64)
    # The row base is always computed in 64-bit: seq_len * seq_len_kv overflows
    # int32 well before the tensor itself gets large. In-row offsets stay 32-bit
    # (<= seq_len_kv elements), which is what lets USE_BUFFER_STORE stay on even
    # when the whole logits tensor exceeds the 2 GiB buffer-descriptor cap.
    stride_logits_s_64 = stride_logits_s.to(gl.int64)

    WARP_SIZE: gl.constexpr = 64
    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[32, 32, 64],
        transposed=False,
        warps_per_cta=[1, NUM_WARPS],
    )
    K_WIDTH: gl.constexpr = 16
    dot_a_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout, k_width=K_WIDTH
    )
    dot_b_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout, k_width=K_WIDTH
    )
    Q_INNER: gl.constexpr = HEAD_SIZE // 16
    layout_q: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 16],
        threads_per_warp=[WARP_SIZE // Q_INNER, Q_INNER],
        warps_per_cta=[NUM_WARPS, 1],
        order=[1, 0],
    )

    q_head_off = gl.arange(0, NUM_HEADS, layout=gl.SliceLayout(1, layout_q))[:, None]
    q_dim_off = gl.arange(0, HEAD_SIZE, layout=gl.SliceLayout(0, layout_q))[None, :]
    w_head_off = gl.arange(0, NUM_HEADS, layout=gl.SliceLayout(1, mfma_layout))[:, None]

    mfma_qs = ()
    w_blocks = ()
    logits_ptrs = ()
    starts = ()
    ends = ()
    tile_start = seq_len_kv
    tile_end: gl.int32 = 0

    for r in gl.static_range(ROWS_PER_BLOCK):
        row = r0 + r
        in_range = row < seq_len
        # Clamp the address so out-of-range rows still form a legal descriptor;
        # their end==0 makes every store mask false.
        row_c = gl.minimum(row, seq_len - 1)
        s_r = gl.maximum(gl.load(cu_start_ptr + row_c), 0)
        e_r = gl.minimum(gl.load(cu_end_ptr + row_c), seq_len_kv)
        s_r = gl.where(in_range, s_r, 0)
        e_r = gl.where(in_range, e_r, 0)
        tile_start = gl.where(e_r > s_r, gl.minimum(tile_start, s_r), tile_start)
        tile_end = gl.maximum(tile_end, e_r)

        q = gl.amd.cdna4.buffer_load(
            ptr=Q_ptr,
            offsets=row_c * stride_q_s
            + q_head_off * stride_q_h
            + q_dim_off * stride_q_d,
            cache=".cg",
        )
        w = gl.amd.cdna4.buffer_load(
            ptr=weights_ptr,
            offsets=row_c * stride_w_s + w_head_off * stride_w_h,
            cache=".cg",
        )
        mfma_qs = mfma_qs + (gl.convert_layout(q, dot_a_layout),)
        w_blocks = w_blocks + (w,)
        starts = starts + (s_r,)
        ends = ends + (e_r,)
        logits_ptrs = logits_ptrs + (logits_ptr + row_c * stride_logits_s_64,)

    tile_start = gl.minimum(tile_start, tile_end)

    base_ptrs = ()
    for r in gl.static_range(ROWS_PER_BLOCK):
        base_ptrs = base_ptrs + (logits_ptrs[r] + tile_start * stride_logits_k,)

    kv_loader = MQAAsyncKVLoader.initialize(
        KV_ptr, seq_len_kv, stride_kv_s, stride_kv_d, BLOCK_KV, HEAD_SIZE,
        NUM_WARPS, WARP_SIZE, NUM_BUFFERS, USE_PADDED_SHARED_LAYOUT,
    )

    num_full_tiles = (tile_end - tile_start) // BLOCK_KV

    mqa_logits_loop_rpb(
        kv_loader, mfma_qs, w_blocks, base_ptrs, starts, ends,
        kv_scales_ptr + tile_start, tile_start, tile_end, num_full_tiles,
        ROWS_PER_BLOCK, NUM_HEADS, BLOCK_KV, stride_logits_k, mfma_layout,
        dot_b_layout, NUM_BUFFERS, NUM_CHAINS, USE_BUFFER_LOAD,
        USE_BUFFER_STORE, FAST_RELU, UNIFORM_START, STORE_CACHE, OUT_BF16,
    )
