# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import functools
import logging

import torch
from torch import Tensor

from ..jit.core import compile_ops

logger = logging.getLogger("aiter")


@compile_ops(
    "module_gemm_a16w16_asm",
    fc_name="gemm_a16w16_asm",
    ffi_type="ctypes",
)
def _gemm_a16w16_asm(
    A: Tensor,
    B: Tensor,
    out: Tensor,
    semaphore: Tensor,
    bias: Tensor | None = None,
    splitK: int | None = None,
    kernelName: str | None = None,
    bpreshuffle: bool = False,
) -> None: ...


# Semaphore workspace shape for ASM SplitK kernels.
# The kernel indexes into a flat array of size rows*cols; candidates whose
# grid (gdx*gdy) exceeds this limit must be skipped to avoid out-of-bounds writes.
_SEMA_SHAPE = (16, 64)
ASM_SPLITK_MAX_GRID = _SEMA_SHAPE[0] * _SEMA_SHAPE[1]

# Ring of workspaces handed to captured launches, one slot per recorded launch.
# Each slot is 4 KiB, so the ring is a fixed 256 KiB per device no matter how
# many launches a process records.
_CAPTURE_RING_SIZE = 64


@functools.lru_cache(maxsize=64)
def _get_semaphore_workspace_keyed(device: torch.device, stream_id: int) -> Tensor:
    return torch.zeros(_SEMA_SHAPE, dtype=torch.uint32, device=device)


@functools.lru_cache(maxsize=None)
def _get_capture_ring(device_index: int) -> tuple[Tensor, ...]:
    device = torch.device("cuda", device_index)
    return tuple(
        torch.zeros(_SEMA_SHAPE, dtype=torch.uint32, device=device)
        for _ in range(_CAPTURE_RING_SIZE)
    )


_capture_ring_cursor: dict[int, int] = {}


def _next_captured_workspace(device_index: int) -> Tensor:
    ring = _get_capture_ring(device_index)
    cursor = _capture_ring_cursor.get(device_index, 0)
    _capture_ring_cursor[device_index] = cursor + 1
    if cursor == _CAPTURE_RING_SIZE:
        logger.warning(
            "More than %d split-K launches captured on device %d; ring slots "
            "are now shared between recorded launches. That is only unsafe if "
            "two graphs sharing a slot replay concurrently.",
            _CAPTURE_RING_SIZE,
            device_index,
        )
    workspace = ring[cursor % _CAPTURE_RING_SIZE]
    # Recorded as a graph node, so every replay re-establishes counter == 0.
    workspace.zero_()
    return workspace


def get_semaphore_workspace(device: torch.device) -> Tensor:
    """Return a zero-initialized semaphore workspace for one split-K launch.

    SplitK a16w16 ASM kernels use an atomic-counter protocol where the last
    workgroup performs the reduction phase. The counter must read zero when a
    launch starts, and the kernel resets it once the reduction completes.
    Concurrent launches must not share a counter, or the counts get mixed and
    the reduction phase never fires (deadlock).

    Eager launches reuse a per-(device, stream) workspace and rely on that
    self-reset. A captured launch cannot: the zero-fill happens once at
    allocation, outside the graph, so replay never restores the entry state and
    a counter left dirty stays dirty. Captured launches therefore draw a ring
    slot whose zero_() is recorded inside the graph. Taking a distinct slot per
    recorded launch also keeps the separation that the stream key provides in
    the eager case, since a graph replays on whatever stream the caller uses.

    The ring is primed here on the eager path so its buffers come from the
    regular allocator. Allocating under capture would place them in the graph's
    private pool, which cannot be released while any recorded graph may replay.
    """
    index = torch.cuda.current_device() if device.index is None else device.index
    if torch.cuda.is_current_stream_capturing():
        return _next_captured_workspace(index)

    # Prime the ring while allocating outside a graph private pool is still safe.
    _get_capture_ring(index)
    stream = torch.cuda.current_stream(device)
    return _get_semaphore_workspace_keyed(device, stream.cuda_stream)


def gemm_a16w16_asm(
    A: Tensor,
    B: Tensor,
    out: Tensor,
    bias: Tensor | None = None,
    splitK: int | None = None,
    kernelName: str | None = None,
    bpreshuffle: bool = False,
):
    if splitK is None or splitK > 1:
        sema = get_semaphore_workspace(out.device)
    else:
        sema = torch.empty((0,), dtype=torch.uint32, device=out.device)

    _gemm_a16w16_asm(A, B, out, sema, bias, splitK, kernelName, bpreshuffle)
    return out
