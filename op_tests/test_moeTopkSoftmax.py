# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# ruff: noqa: N999  camelCase module name kept: renaming would break the test
# selection lists and FILE_TIMES bookkeeping that reference this path.

import argparse

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.test_common import (
    benchmark,
    checkAllclose,
    perftest,
    run_perftest,
)

torch.set_default_device("cuda")
torch.set_printoptions(sci_mode=False)


@perftest(num_iters=2, num_warmup=1)
def test_nofuse(
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
):
    gating_output = torch.nn.functional.softmax(
        gating_output.float(),
        dim=-1,
    )
    topk_weights, topk_ids = gating_output.topk(
        k=topk,
        dim=-1,
        largest=True,
        sorted=True,
    )

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    return topk_weights, topk_ids.to(dtypes.i32)


@perftest()
def test_fuse(
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
):
    # hidden_states = torch.empty(gating_output.shape, dtype=dtypes.fp32, device=gating_output.device)
    # from aiter.fused_moe import fused_topk
    # return fused_topk(hidden_states, gating_output, topk, renormalize)

    M, _expert = gating_output.shape
    topk_weights = torch.empty_strided(
        (M, topk), (topk + 10, 1), dtype=dtypes.fp32, device=gating_output.device
    )
    topk_ids = torch.empty_strided(
        (M, topk), (topk + 10, 1), dtype=dtypes.i32, device=gating_output.device
    )
    token_expert_indicies = torch.empty_strided(
        (M, topk), (topk + 10, 1), dtype=dtypes.i32, device=gating_output.device
    )
    aiter.topk_softmax(
        topk_weights,
        topk_ids,
        token_expert_indicies,
        gating_output,
        renormalize,
    )
    return topk_weights, topk_ids


@perftest()
def test_asm(
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
):
    M, _expert = gating_output.shape
    topk_weights = torch.empty_strided(
        (M, topk), (topk + 10, 1), dtype=dtypes.fp32, device=gating_output.device
    )
    topk_ids = torch.empty_strided(
        (M, topk), (topk + 10, 1), dtype=dtypes.i32, device=gating_output.device
    )
    token_expert_indicies = torch.empty_strided(
        (M, topk), (topk + 10, 1), dtype=dtypes.i32, device=gating_output.device
    )
    aiter.topk_softmax_asm(
        topk_weights,
        topk_ids,
        token_expert_indicies,
        gating_output,
        renormalize,
    )
    del token_expert_indicies  # Not used. Will be used in the future.
    return topk_weights, topk_ids


@benchmark()
def test_topk_softmax(dtype, token, E, topk, renormalize=True):
    gating_output = torch.randn((token, E + 10), dtype=dtype, device="cuda")
    # making gating_output as strided tensor for testing
    gating_output = gating_output[:, :E]
    (topk_weights_a, topk_ids_a), _avg_a = test_nofuse(gating_output, topk, renormalize)
    id_ref, _ref = torch.sort(topk_ids_a)
    w_ref = topk_weights_a.gather(1, _ref)

    func_dict = {"hip": test_fuse, "asm": test_asm}
    ret = {}
    for tag, func in func_dict.items():
        if tag == "asm" and not (
            (E, topk)
            in [(128, 4), (128, 6), (128, 8), (256, 6), (256, 8), (384, 8), (640, 8)]
            and dtype in [dtypes.bf16, dtypes.fp32]
            and get_gfx() in ["gfx942", "gfx950"]
        ):
            continue
        gating_output = gating_output.contiguous() if tag == "asm" else gating_output
        (topk_weights, topk_ids), us = func(gating_output, topk, renormalize)
        topk_ids = topk_ids.to(dtypes.i32)
        id, _ref = torch.sort(topk_ids)
        weight = topk_weights.gather(1, _ref)
        ret[f"{tag} err"] = checkAllclose(w_ref, weight, msg=f"{tag} topk_weights")
        checkAllclose(id_ref, id, msg=f"{tag} topk_ids")
        ret[f"{tag} us"] = us
    return ret


# this function test a value/index pair, like the output of a topk function
# w.r.t a target dim
def check_topk_softmax_allclose(
    ref_val,
    ref_idx,
    tar_val,
    tar_idx,
    scores,
    bias,
    target_dim=-1,  # last dim by default
    target_dim_len=-1,  # the dim could be larger than ref/tar val dim length. if -1, then same size as
    sort_before_compare=True,  # this is useful when we don't care about the absolute position of the val/idx
    rtol=1e-2,
    atol=1e-2,
    tol_err_ratio=0.05,
    msg="",
    printNum=8,
    printLog=True,
):
    from aiter import logger

    # first let's sort the index in case
    if sort_before_compare:
        # NOTE: need add bias before sorting
        _, _r_sorted_idx = torch.sort(
            ref_val
            + bias.repeat(ref_val.shape[0], 1).gather(-1, ref_idx.to(dtype=torch.int64))
        )
        _, _t_sorted_idx = torch.sort(
            tar_val
            + bias.repeat(ref_val.shape[0], 1).gather(-1, tar_idx.to(dtype=torch.int64))
        )
        r_val = ref_val.gather(target_dim, _r_sorted_idx)
        t_val = tar_val.gather(target_dim, _t_sorted_idx)
        r_idx = ref_idx.gather(target_dim, _r_sorted_idx)
        t_idx = tar_idx.gather(target_dim, _t_sorted_idx)
    else:
        r_val = ref_val
        t_val = tar_val
        r_idx = ref_idx
        t_idx = tar_idx

    if target_dim_len < 0:
        target_dim_len = ref_val.shape[target_dim]

    assert target_dim_len >= ref_val.shape[target_dim]

    original_shape = list(ref_val.shape)
    original_shape[target_dim] = target_dim_len

    is_close_v = torch.isclose(r_val, t_val, rtol=rtol, atol=atol)
    is_close_i = torch.isclose(r_idx, t_idx)  # use high resolution for index

    scores_for_choice = scores.view(original_shape)
    if bias is not None:
        scores_for_choice = scores_for_choice + bias.unsqueeze(0)

    if is_close_v.all():
        if printLog:
            logger.info(
                f"{msg}[check_topk_softmax_allclose/value {atol=} {rtol=} \033[32mpassed~\033[0m]"
            )

        if is_close_i.all():
            if printLog:
                logger.info(
                    f"{msg}[check_topk_softmax_allclose/index \033[32mpassed~\033[0m]"
                )
            return 0
        else:
            # this case there must be some duplicate value, and due to compare order, index maybe different
            mask = ~(is_close_i)
            mismatch_r = scores_for_choice.gather(-1, r_idx.to(dtype=torch.int64))[mask]
            mismatch_t = scores_for_choice.gather(-1, t_idx.to(dtype=torch.int64))[mask]

            # if index mismatch, the the index pointed value must be the same
            # below we are checking such case
            is_close_dup_i = torch.isclose(mismatch_r, mismatch_t, rtol=rtol, atol=atol)

            if not is_close_dup_i.all():
                # this check should contain same index mask bool tensor, otherwise something wrong
                num = mask.sum()
                printNum = min(printNum, num)
                percent = (num / r_val.numel()).item()
                logger.info(
                    f"""{msg}[check_topk_softmax_allclose/index \033[32mfailed~\033[0m]"""
                )
                for i_row in range(r_idx.shape[0]):
                    for i_col in range(r_idx.shape[1]):
                        if r_idx[i_row, i_col] != t_idx[i_row, i_col]:
                            sr = scores_for_choice[i_row, r_idx[i_row, i_col]]
                            st = scores_for_choice[i_row, t_idx[i_row, i_col]]
                            torch.isclose(sr, st, rtol=rtol, atol=atol)
                            logger.info(
                                f"{msg} [{i_row}x{i_col}], r:{r_idx[i_row, i_col]}->{sr}, t:{t_idx[i_row, i_col]}->{st}"
                            )
                return 1

            else:
                if printLog:
                    logger.info(
                        f"{msg}[check_topk_softmax_allclose/index(duplicated) \033[32mpassed~\033[0m]"
                    )
                return 0

    else:
        mask = ~is_close_v
        num = mask.sum()
        printNum = min(printNum, num)
        percent = (num / r_val.numel()).item()
        if not printLog:
            return percent
        r_msked = r_val[mask]
        t_msked = t_val[mask]
        delta = (r_msked - t_msked).abs()
        if percent > tol_err_ratio:
            logger.info(
                f"""{msg}[check_topk_softmax_allclose.value {atol=} {rtol=} \033[31mfailed!\033[0m]
    ref  : {r_msked[:printNum]}
    tar  : {t_msked[:printNum]}
    delta:
           {delta[:printNum]}"""
            )
        return percent


@aiter.test_common.benchmark()
def test_biased_grouped_topk(
    token,
    expert,
    group,
    topk,
    topk_group,
    need_renorm,
    dtype,
    scale_factor=1.0,
    num_iters=101,
    num_warmup=2,
    gating_output=None,
):
    ret = {}
    if gating_output is None:
        gating_output = torch.randn((token, expert), dtype=dtype)
    correction_bias = torch.randn((expert,), dtype=dtype)

    (w_ref, id_ref, score_ref), us_ref = run_perftest(
        aiter.biased_grouped_topk_torch,
        gating_output,
        correction_bias,
        topk,
        need_renorm,
        group,
        topk_group,
        True,  # return score
        num_iters=2,
        num_warmup=1,
    )
    w_ref = w_ref * scale_factor
    w_aiter = torch.empty_strided((token, topk), (topk + 10, 1), dtype=dtypes.fp32)
    id_aiter = torch.empty_strided((token, topk), (topk + 10, 1), dtype=dtypes.i32)
    _, us_aiter = run_perftest(
        aiter.biased_grouped_topk_hip,
        gating_output,
        correction_bias,
        w_aiter,
        id_aiter,
        group,
        topk_group,
        need_renorm,
        scale_factor,
        num_iters=num_iters,
        num_warmup=num_warmup,
    )

    # use a special function to check result. The HIP topk may using sort algorithm
    # ... which will make the result order unpredictable
    err = check_topk_softmax_allclose(
        w_ref,
        id_ref,
        w_aiter,
        id_aiter,
        score_ref,
        correction_bias,
        target_dim_len=expert,
        msg=f"[golden vs aiter]:{us_ref:>8.2f} us vs {us_aiter:>8.2f} us......",
    )
    id_ref, _ref = torch.sort(id_ref)
    id_aiter, _aiter = torch.sort(id_aiter)
    w_ref = w_ref.gather(1, _ref)
    w_aiter = w_aiter.gather(1, _aiter)
    # print(f'  {id_ref=}')
    # print(f'{id_aiter=}')
    # print(f'  {w_ref=}')
    # print(f'{w_aiter=}')
    # err = checkAllclose(w_ref, w_aiter, msg="topk_weights [golden vs aiter]")
    # checkAllclose(
    #     id_ref,
    #     id_aiter,
    #     msg=f"topk_ids     [golden vs aiter]:{us_ref:>8.2f} us vs {us_aiter:>8.2f} us......",
    # )
    ret["us_aiter"] = us_aiter
    ret["err_aiter"] = err
    # return {"err": err, "us": us_aiter}

    if expert // group <= 32:
        w_sglang = torch.empty_strided((token, topk), (topk + 10, 1), dtype=dtypes.fp32)
        id_sglang = torch.empty_strided((token, topk), (topk + 10, 1), dtype=dtypes.i32)
        _, us_sglang = run_perftest(
            aiter.moe_fused_gate,
            gating_output,
            correction_bias,
            w_sglang,
            id_sglang,
            group,
            topk_group,
            topk,
            0,
            scale_factor,
        )

        w_sglang = _[0]
        id_sglang = _[1]

        id_sglang, _sglang = torch.sort(id_sglang)
        w_sglang = w_sglang.gather(1, _sglang)
        ret["us_sglang"] = us_sglang

        # print(f"{w_ref=}")
        # print(f"{w_sglang=}")
        # print(f"{id_ref=}")
        # print(f"{id_sglang=}")

        err = checkAllclose(w_ref, w_sglang, msg="topk_weights [golden vs sglang]")
        checkAllclose(
            id_ref,
            id_sglang,
            msg=f"topk_ids     [aiter vs sglang]:{us_aiter:>8.2f} us vs {us_sglang:>8.2f} us......",
        )
        ret["err_sglang"] = err
    return ret


@benchmark()
def test_grouped_topk(
    token,
    expert,
    group,
    topk,
    topk_group,
    need_renorm,
    dtype,
    scale_factor=1.0,
    scoring_func="softmax",
):
    gating_output = torch.randn((token, expert), dtype=dtype)

    (w_ref, id_ref), us_ref = run_perftest(
        aiter.grouped_topk_torch,
        gating_output,
        topk,
        need_renorm,
        group,
        topk_group,
        scoring_func,
        num_iters=2,
        num_warmup=1,
    )
    w_ref = w_ref * scale_factor
    w_aiter = torch.empty_strided((token, topk), (topk + 10, 1), dtype=dtypes.fp32)
    id_aiter = torch.empty_strided((token, topk), (topk + 10, 1), dtype=dtypes.i32)
    is_softmax = scoring_func == "softmax"
    _, us_aiter = run_perftest(
        aiter.grouped_topk,
        gating_output,
        w_aiter,
        id_aiter,
        group,
        topk_group,
        need_renorm,
        is_softmax,
        scale_factor,
    )
    id_ref, _ref = torch.sort(id_ref)
    id_aiter, _aiter = torch.sort(id_aiter)
    err = checkAllclose(
        w_ref.gather(1, _ref),
        w_aiter.gather(1, _aiter),
        msg="topk_weights [golden vs aiter]",
    )
    checkAllclose(
        id_ref,
        id_aiter,
        msg=f"topk_ids     [golden vs aiter]:{us_ref:>8.2f} us vs {us_aiter:>8.2f} us......",
    )

    return {"err": err, "us": us_aiter}


def check_grouped_topk_pairing(
    gating_output,
    correction_bias,
    topk_weights,
    topk_ids,
    need_renorm,
    scale_factor,
    msg,
):
    """Assert a grouped-topk result without assuming any emission order.

    Order is not part of this op's contract: the torch golden selects with
    sorted=False, biased_grouped_topk dispatches to moe_fused_gate above
    cu_num * 212 tokens and that kernel emits a different order again, and
    topk_weights holds the pre-bias sigmoid while selection uses the biased
    score, so topk_weights is not sorted on any path. So check the two things
    that are guaranteed:

      1. the selected set is a valid top-k, i.e. its sorted biased scores equal
         torch's. This tolerates ties, where several experts are equally
         correct answers, without tolerating a wrong pick.
      2. each weight is the one belonging to the id sitting beside it, which is
         what a permuted emission order could plausibly break.
    """
    scores = gating_output.to(dtypes.fp32).sigmoid()
    biased = (
        scores if correction_bias is None else scores + correction_bias.to(dtypes.fp32)
    )
    topk = topk_ids.shape[1]

    assert bool((topk_ids >= 0).all()), f"{msg}: unfilled slot"
    assert bool((topk_ids < gating_output.shape[1]).all()), (
        f"{msg}: expert id out of range"
    )
    srt = torch.sort(topk_ids, dim=-1).values
    assert bool((srt[:, 1:] != srt[:, :-1]).all()), f"{msg}: duplicate expert in a row"

    ref_biased = torch.topk(biased, topk, dim=-1).values
    got_biased = torch.sort(
        biased.gather(-1, topk_ids.long()), -1, descending=True
    ).values
    checkAllclose(ref_biased, got_biased, msg=f"{msg} selected biased scores")

    want = scores.gather(-1, topk_ids.long())
    if need_renorm:
        want = want / want.sum(-1, keepdim=True)
    checkAllclose(want * scale_factor, topk_weights, msg=f"{msg} weight/id pairing")


def test_grouped_topk_radix_envelope():
    """Cover the dispatch envelope of the ungrouped radix path.

    grouped_topk_radix_kernel takes over for ungrouped sigmoid routers, and it
    holds RADIX_SLOTS * WARP_SIZE * vec_size scores in registers. vec_size
    follows num_experts % 4, so the expert ceiling is 1024 for the vec4 path but
    512 for vec2 and 256 for vec1. Each width is exercised on both sides of its
    own ceiling, because a count picked to select a narrow width can otherwise
    quietly fall back to grouped_topk_kernel and test nothing.
    """
    shapes = [
        # experts, vec_size, inside its own ceiling
        (896, 4, True),  # Kimi-K3
        (1024, 4, True),  # vec4 ceiling exactly
        (1028, 4, False),  # one vector past it, falls back
        (384, 4, True),  # Kimi-K2.5
        (510, 2, True),  # vec2, inside 512
        (514, 2, False),  # vec2, past 512
        (898, 2, False),  # vec2, well past 512
        (255, 1, True),  # vec1, inside 256
        (257, 1, False),  # vec1, past 256
        (897, 1, False),  # vec1, well past 256
    ]
    fails = 0
    for expert, vec_size, inside in shapes:
        assert (4 if expert % 4 == 0 else (2 if expert % 4 == 2 else 1)) == vec_size
        for topk in [8, 16, 32]:
            if topk > expert:
                continue
            for dtype in [dtypes.fp32, dtypes.bf16]:
                for need_renorm, scale_factor in [
                    (True, 1.0),
                    (False, 2.5),
                    (True, 2.827),
                ]:
                    for token in [1, 3, 64, 1024]:
                        tag = (
                            f"{expert}e/vec{vec_size}/top{topk}/{token}t/"
                            f"{'renorm' if need_renorm else 'raw'}/{dtype}"
                        )
                        gating_output = torch.randn((token, expert), dtype=dtype)
                        correction_bias = torch.randn((expert,), dtype=dtype)
                        w = torch.empty_strided(
                            (token, topk), (topk + 10, 1), dtype=dtypes.fp32
                        )
                        i = torch.empty_strided(
                            (token, topk), (topk + 10, 1), dtype=dtypes.i32
                        )
                        i.fill_(-1)
                        aiter.biased_grouped_topk_hip(
                            gating_output,
                            correction_bias,
                            w,
                            i,
                            1,
                            1,
                            need_renorm,
                            scale_factor,
                        )
                        try:
                            check_grouped_topk_pairing(
                                gating_output,
                                correction_bias,
                                w,
                                i,
                                need_renorm,
                                scale_factor,
                                f"biased {tag}",
                            )
                        except AssertionError as e:
                            aiter.logger.error("%s", e)
                            fails += 1

                        # The non-biased entry point instantiates the other side
                        # of the isBiased template and is otherwise untested here.
                        w.zero_()
                        i.fill_(-1)
                        aiter.grouped_topk(
                            gating_output, w, i, 1, 1, need_renorm, False, scale_factor
                        )
                        try:
                            check_grouped_topk_pairing(
                                gating_output,
                                None,
                                w,
                                i,
                                need_renorm,
                                scale_factor,
                                f"plain {tag}",
                            )
                        except AssertionError as e:
                            aiter.logger.error("%s", e)
                            fails += 1
    assert fails == 0, f"{fails} grouped-topk envelope checks failed"
    aiter.logger.info("grouped_topk radix envelope: all shapes correct")


def test_grouped_topk_ties():
    """Ties force the radix select's non-exact path.

    When the boundary bucket holds more equal keys than there are slots left,
    the kernel cannot take the whole bucket and has to take a rank-limited
    subset of it. Random logits essentially never reach that path, so drive it
    with a tiny set of distinct values.
    """
    fails = 0
    for expert, topk in [(896, 16), (1024, 32), (510, 16), (255, 16), (384, 8)]:
        for levels in [2, 3, 5]:
            for token in [1, 64]:
                tag = f"{expert}e/top{topk}/{levels} distinct logits/{token}t"
                gating_output = (
                    torch.randint(0, levels, (token, expert)).to(dtypes.fp32)
                    - levels // 2
                )
                correction_bias = torch.zeros((expert,), dtype=dtypes.fp32)
                w = torch.empty_strided(
                    (token, topk), (topk + 10, 1), dtype=dtypes.fp32
                )
                i = torch.empty_strided((token, topk), (topk + 10, 1), dtype=dtypes.i32)
                i.fill_(-1)
                aiter.biased_grouped_topk_hip(
                    gating_output, correction_bias, w, i, 1, 1, True, 1.0
                )
                try:
                    check_grouped_topk_pairing(
                        gating_output, correction_bias, w, i, True, 1.0, tag
                    )
                except AssertionError as e:
                    aiter.logger.error("%s", e)
                    fails += 1
    assert fails == 0, f"{fails} grouped-topk tie checks failed"
    aiter.logger.info("grouped_topk ties: all shapes correct")


@perftest(num_iters=2, num_warmup=1)
def test_nofuse_shared(
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_shared_experts: int,
):
    """Reference implementation with shared experts using PyTorch."""
    num_routing_experts = gating_output.size(-1) - num_shared_experts

    # Split routing and shared expert logits
    routing_logits = gating_output[:, :num_routing_experts]
    shared_logits = gating_output[:, num_routing_experts:]

    # Process routing experts with softmax + topk
    routing_probs = torch.nn.functional.softmax(routing_logits.float(), dim=-1)
    topk_weights_routing, topk_indices = routing_probs.topk(
        k=topk, dim=-1, largest=True, sorted=True
    )

    if renormalize:
        topk_weights_routing = topk_weights_routing / topk_weights_routing.sum(
            dim=-1, keepdim=True
        )

    # Process shared experts with sigmoid
    shared_weights = torch.sigmoid(shared_logits.float())

    # Concatenate routing and shared weights
    topk_weights = torch.cat([topk_weights_routing, shared_weights], dim=-1)

    return topk_weights, topk_indices.to(dtypes.i32)


@perftest()
def test_fuse_shared(
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_shared_experts: int,
):
    """AITER kernel with shared experts."""
    M = gating_output.shape[0]
    total_output_size = topk + num_shared_experts

    topk_weights = torch.empty_strided(
        (M, total_output_size),
        (total_output_size + 10, 1),
        dtype=dtypes.fp32,
        device=gating_output.device,
    )
    topk_ids = torch.empty_strided(
        (M, topk), (topk + 10, 1), dtype=dtypes.i32, device=gating_output.device
    )
    token_expert_indicies = torch.empty_strided(
        (M, topk), (topk + 10, 1), dtype=dtypes.i32, device=gating_output.device
    )

    aiter.topk_softmax(
        topk_weights,
        topk_ids,
        token_expert_indicies,
        gating_output,
        renormalize,
        num_shared_experts,
        "sigmoid",
    )

    return topk_weights, topk_ids


@benchmark()
def test_topk_softmax_shared_experts(
    dtype, token, num_routing_experts, num_shared_experts, topk, renormalize=True
):
    """Test topk_softmax with shared expert sigmoid scoring."""
    num_experts = num_routing_experts + num_shared_experts
    gating_output = torch.randn((token, num_experts + 10), dtype=dtype, device="cuda")
    gating_output = gating_output[:, :num_experts]

    # Reference (PyTorch)
    (topk_weights_ref, topk_ids_ref), us_ref = test_nofuse_shared(
        gating_output, topk, renormalize, num_shared_experts
    )

    # AITER kernel
    (topk_weights_aiter, topk_ids_aiter), us_aiter = test_fuse_shared(
        gating_output, topk, renormalize, num_shared_experts
    )

    ret = {}

    # Split routing and shared weights for comparison
    routing_weights_ref = topk_weights_ref[:, :topk]
    routing_weights_aiter = topk_weights_aiter[:, :topk]

    # Check routing weights (sort by indices first for fair comparison)
    id_ref_sorted, _ref = torch.sort(topk_ids_ref)
    id_aiter_sorted, _aiter = torch.sort(topk_ids_aiter)
    w_ref_sorted = routing_weights_ref.gather(1, _ref)
    w_aiter_sorted = routing_weights_aiter.gather(1, _aiter)

    ret["routing_err"] = checkAllclose(
        w_ref_sorted, w_aiter_sorted, msg="routing_weights [ref vs aiter]"
    )
    checkAllclose(id_ref_sorted, id_aiter_sorted, msg="routing_ids [ref vs aiter]")

    # Check shared weights
    if num_shared_experts > 0:
        shared_weights_ref = topk_weights_ref[:, topk:]
        shared_weights_aiter = topk_weights_aiter[:, topk:]
        ret["shared_err"] = checkAllclose(
            shared_weights_ref,
            shared_weights_aiter,
            msg=f"shared_weights [ref vs aiter]: {us_ref:>8.2f} us vs {us_aiter:>8.2f} us",
        )

    ret["us_ref"] = us_ref
    ret["us_aiter"] = us_aiter
    ret["speedup"] = us_ref / us_aiter if us_aiter > 0 else 0

    return ret


parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="config input of test",
)
parser.add_argument(
    "-d",
    "--dtype",
    type=dtypes.str2Dtype,
    choices=[dtypes.d_dtypes["fp32"], dtypes.d_dtypes["bf16"]],
    nargs="*",
    metavar="{fp32, bf16}",
    default=[dtypes.d_dtypes["fp32"], dtypes.d_dtypes["bf16"]],
    help="""Data type.
    e.g.: -d bf16""",
)
parser.add_argument(
    "-e",
    "--expert",
    type=int,
    nargs="*",
    default=[128, 256],
    help="""Number of experts.
    e.g.: -e 64""",
)
parser.add_argument(
    "-t",
    "--token",
    type=int,
    # choices=l_token,
    nargs="*",
    default=[
        1,
        2,
        5,
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        1024,
        2048,
        4096,
        10000,
        16384,
        65536,
        163840,
    ],
    help="""Number of tokens.
    e.g.: -t 64""",
)
parser.add_argument(
    "-k",
    type=int,
    default=8,
    help="""Number of topk.
    e.g.: -k 8""",
)
parser.add_argument(
    "-i",
    "--iters",
    type=int,
    default=101,
    help="""Number of timed iterations per measurement (grouped/biased topk).
    Raise this to reduce per-measurement noise.
    e.g.: -i 500""",
)
parser.add_argument(
    "-w",
    "--warmup",
    type=int,
    default=2,
    help="""Number of warmup iterations before timing (grouped/biased topk).
    Raise this to stabilize GPU clocks for latency-bound (small-token) shapes.
    e.g.: -w 50""",
)

args = parser.parse_args()

# Correctness of the ungrouped radix dispatch envelope. Run first and let it
# raise: unlike the benchmark tables below, a failure here is a real bug rather
# than a tie-ordering difference.
test_grouped_topk_radix_envelope()
test_grouped_topk_ties()

df = []
for dtype in args.dtype:
    for e in args.expert:
        for m in args.token:
            ret = test_topk_softmax(dtype, m, e, args.k)
            df.append(ret)
df = pd.DataFrame(df)
df_md = df.to_markdown(index=False)
aiter.logger.info("moeTopkSoftmax summary (markdown):\n%s", df_md)

df = []
for token in args.token:
    # DeepSeek-R1
    topk = 8
    group = 8
    topk_group = 4
    expert = 256
    dtype = dtypes.bf16
    need_renorm = True
    ret = test_biased_grouped_topk(
        token,
        expert,
        group,
        topk,
        topk_group,
        need_renorm,
        dtype,
        num_iters=args.iters,
        num_warmup=args.warmup,
    )
    df.append(ret)
df = pd.DataFrame(df)
df_md = df.to_markdown(index=False)
aiter.logger.info("moeTopkSoftmax_biased_grouped_topk summary (markdown):\n%s", df_md)

df = []
for token in args.token:
    # Kimi-K2.5 shapes
    topk = 8
    group = 1
    topk_group = 1
    expert = 384
    dtype = dtypes.bf16
    need_renorm = True
    ret = test_biased_grouped_topk(
        token,
        expert,
        group,
        topk,
        topk_group,
        need_renorm,
        dtype,
        scale_factor=2.827,
        num_iters=args.iters,
        num_warmup=args.warmup,
    )
    df.append(ret)
df = pd.DataFrame(df)
df_md = df.to_markdown(index=False)
aiter.logger.info(
    "moeTopkSoftmax_biased_grouped_topk_kimi_k25 summary (markdown):\n%s", df_md
)

df = []
for token in args.token:
    # Kimi-K3 fused MoE-front router: logits are a row-strided slice of the
    # fused [gate_up | experts | routed] buffer, not a standalone tensor
    gate_up_width, num_experts, routed_width = 1536, 896, 3584
    fused_front_width = gate_up_width + num_experts + routed_width
    backing = torch.randn((token, fused_front_width), dtype=dtypes.bf16)
    gating_output = backing[:, gate_up_width : gate_up_width + num_experts]
    # stride(0) is the thing under test. Do not also assert non-contiguity:
    # at token=1 the row stride is unreachable, so the slice is contiguous.
    assert gating_output.stride(0) == fused_front_width
    ret = test_biased_grouped_topk(
        token,
        num_experts,
        1,  # group
        16,  # topk
        1,  # topk_group
        True,  # need_renorm
        dtypes.bf16,
        gating_output=gating_output,
        num_iters=args.iters,
        num_warmup=args.warmup,
    )
    df.append(ret)
df = pd.DataFrame(df)
df_md = df.to_markdown(index=False)
aiter.logger.info(
    "moeTopkSoftmax_biased_grouped_topk_kimi_k3_strided summary (markdown):\n%s",
    df_md,
)

df = []
for token in args.token:
    for scoring_func in ["softmax", "sigmoid"]:
        # DeepSeek-R1
        topk = 8
        group = 8
        topk_group = 4
        expert = 256
        dtype = dtypes.bf16
        need_renorm = True
        ret = test_grouped_topk(
            token,
            expert,
            group,
            topk,
            topk_group,
            need_renorm,
            dtype,
            scale_factor=1.5,
            scoring_func=scoring_func,
        )
        df.append(ret)
df = pd.DataFrame(df)
df_md = df.to_markdown(index=False)
aiter.logger.info("moeTopkSoftmax_grouped_topk summary (markdown):\n%s", df_md)

# Test shared expert sigmoid scoring
aiter.logger.info("\n" + "=" * 70)
aiter.logger.info("Testing topk_softmax with shared expert sigmoid scoring")
aiter.logger.info("=" * 70)

df = []
# Test configurations: (num_routing_experts, num_shared_experts, topk, dtype, renormalize)
shared_expert_configs = [
    (8, 2, 2, dtypes.bf16, False),
    (16, 2, 4, dtypes.bf16, True),
    (32, 4, 8, dtypes.fp32, False),
    (64, 4, 16, dtypes.fp32, True),
    (8, 0, 2, dtypes.bf16, False),  # No shared experts (backward compatibility)
    (16, 0, 4, dtypes.bf16, True),  # No shared experts (backward compatibility)
]

for token in [128, 256, 512, 1024]:
    for num_routing, num_shared, topk, dtype, renorm in shared_expert_configs:
        ret = test_topk_softmax_shared_experts(
            dtype, token, num_routing, num_shared, topk, renorm
        )
        ret["token"] = token
        ret["routing_experts"] = num_routing
        ret["shared_experts"] = num_shared
        ret["topk"] = topk
        ret["dtype"] = str(dtype)
        ret["renorm"] = renorm
        df.append(ret)

df = pd.DataFrame(df)
df_md = df.to_markdown(index=False)
aiter.logger.info("moeTopkSoftmax_shared_experts summary (markdown):\n%s", df_md)
