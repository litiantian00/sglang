"""Triton kernels for DeepEP V2 (ElasticBuffer) decode-path format conversion.

These kernels convert V2 non-expand dispatch output (2D flat, unsorted) to
3D masked-slab format [E_local, max_m, H] required by DeepGEMM's masked
grouped GEMM, and reverse the operation after GEMM for V2 combine.

All kernels are CUDA-Graph-safe: they read routing metadata from GPU tensors
(no host sync) and operate on pre-allocated, fixed-shape buffers.

Performance critical notes:
- NO slab zeroing: masked GEMM skips padding rows via masked_m, so
  uninitialized padding values are never read.
- cleanup + slot_map build fused into one kernel to minimize launch overhead.
- slot_map reused by both scatter and reverse-scatter (no redundant atomic ops).
"""
from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _cleanup_and_build_slot_map_kernel(
    topk_idx_ptr,
    psum_per_rank_ptr,
    expert_counters_ptr,
    slot_map_ptr,
    local_expert_ids_ptr,
    num_scaleup_ranks,
    rank_expert_offset,
    num_experts,
    N_padded,
    E_local,
    max_m,
    topk: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    token_id = (pid * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
    valid = token_id < N_padded

    num_recv = tl.load(psum_per_rank_ptr + num_scaleup_ranks - 1).to(tl.int64)

    for k in tl.range(topk):
        idx_offset = token_id * topk + k
        raw_val = tl.load(topk_idx_ptr + idx_offset, mask=valid, other=-1).to(tl.int64)

        global_eid = raw_val + rank_expert_offset
        is_recv = (token_id < num_recv) & valid
        is_valid_expert = (raw_val >= 0) & (global_eid < num_experts) & is_recv

        tl.store(
            topk_idx_ptr + idx_offset,
            tl.where(is_valid_expert, global_eid, tl.full([], -1, tl.int64)),
            mask=valid,
        )

        local_eid = raw_val
        is_local = is_valid_expert & (local_eid >= 0) & (local_eid < E_local)

        slot = tl.where(
            is_local,
            tl.atomic_add(expert_counters_ptr + local_eid, 1, mask=is_local),
            -1,
        ).to(tl.int64)
        safe_slot = tl.where(slot < max_m, slot, tl.full([], -1, tl.int64))

        tl.store(slot_map_ptr + idx_offset, safe_slot, mask=valid)
        tl.store(
            local_expert_ids_ptr + idx_offset,
            tl.where(is_local, local_eid, tl.full([], -1, tl.int64)),
            mask=valid,
        )


@triton.jit
def _extract_masked_m_kernel(
    expert_counters_ptr,
    masked_m_ptr,
    max_m,
):
    eid = tl.program_id(0)
    count = tl.load(expert_counters_ptr + eid)
    clamped = tl.minimum(count, max_m)
    tl.store(masked_m_ptr + eid, clamped.to(tl.int32))

@torch.no_grad()
def cleanup_and_build_slot_map(
    recv_topk_idx: torch.Tensor,
    psum_per_rank: torch.Tensor,
    expert_counters: torch.Tensor,
    masked_m: torch.Tensor,
    slot_map: torch.Tensor,
    local_expert_ids: torch.Tensor,
    E_local: int,
    max_m: int,
    rank_expert_offset: int,
    num_experts: int,
) -> None:
    N_padded = recv_topk_idx.shape[0]
    topk = recv_topk_idx.shape[1]

    expert_counters.zero_()

    _cleanup_and_build_slot_map_kernel[(triton.cdiv(N_padded, 256),)](
        recv_topk_idx,
        psum_per_rank,
        expert_counters,
        slot_map,
        local_expert_ids,
        psum_per_rank.shape[0],
        rank_expert_offset,
        num_experts,
        N_padded, E_local, max_m,
        topk=topk,
        BLOCK=256,
    )

    _extract_masked_m_kernel[(E_local,)](
        expert_counters,
        masked_m,
        max_m,
    )


@triton.jit
def _scatter_by_slotmap_kernel(
    recv_x_ptr,
    slot_map_ptr,
    local_expert_ids_ptr,
    output_x_ptr,
    N_padded,
    max_m,
    topk: tl.constexpr,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    token_id = tl.program_id(0).to(tl.int64)
    if token_id >= N_padded:
        return

    h_block = tl.program_id(1)
    h_start = (h_block * BLOCK_H).to(tl.int64)
    h_offs = h_start + tl.arange(0, BLOCK_H).to(tl.int64)
    h_mask = h_offs < H

    x_data = tl.load(recv_x_ptr + token_id * H + h_offs, mask=h_mask, other=0.0)

    for k in tl.range(topk):
        idx_offset = token_id * topk + k
        slot = tl.load(slot_map_ptr + idx_offset).to(tl.int64)
        local_eid = tl.load(local_expert_ids_ptr + idx_offset).to(tl.int64)

        if slot >= 0:
            out_offset = local_eid * max_m * H + slot * H + h_offs
            tl.store(output_x_ptr + out_offset, x_data, mask=h_mask)


@triton.jit
def _scatter_scale_by_slotmap_kernel(
    recv_sf_ptr,
    slot_map_ptr,
    local_expert_ids_ptr,
    output_sf_ptr,
    N_padded,
    max_m,
    topk: tl.constexpr,
    sf_dim: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    token_id = tl.program_id(0).to(tl.int64)
    if token_id >= N_padded:
        return

    s_offs = tl.arange(0, BLOCK_S).to(tl.int64)
    s_mask = s_offs < sf_dim

    sf_data = tl.load(recv_sf_ptr + token_id * sf_dim + s_offs, mask=s_mask, other=0.0)

    for k in tl.range(topk):
        idx_offset = token_id * topk + k
        slot = tl.load(slot_map_ptr + idx_offset).to(tl.int64)
        local_eid = tl.load(local_expert_ids_ptr + idx_offset).to(tl.int64)

        if slot >= 0:
            out_offset = local_eid * max_m * sf_dim + slot * sf_dim + s_offs
            tl.store(output_sf_ptr + out_offset, sf_data, mask=s_mask)


@torch.no_grad()
def scatter_2d_to_3d(
    recv_x: torch.Tensor,
    recv_sf: Optional[torch.Tensor],
    slot_map: torch.Tensor,
    local_expert_ids: torch.Tensor,
    output_x: torch.Tensor,
    output_sf: Optional[torch.Tensor],
    max_m: int,
) -> None:
    N_padded = recv_x.shape[0]
    H = recv_x.shape[1]
    topk = slot_map.shape[1]

    BLOCK_H = triton.next_power_of_2(min(H, 1024))
    grid_h = triton.cdiv(H, BLOCK_H)

    _scatter_by_slotmap_kernel[(N_padded, grid_h)](
        recv_x,
        slot_map,
        local_expert_ids,
        output_x,
        N_padded, max_m,
        topk=topk,
        H=H,
        BLOCK_H=BLOCK_H,
    )

    if recv_sf is not None and output_sf is not None:
        sf_dim = recv_sf.shape[1]
        BLOCK_S = triton.next_power_of_2(sf_dim)
        _scatter_scale_by_slotmap_kernel[(N_padded,)](
            recv_sf,
            slot_map,
            local_expert_ids,
            output_sf,
            N_padded, max_m,
            topk=topk,
            sf_dim=sf_dim,
            BLOCK_S=BLOCK_S,
        )


@triton.jit
def _reverse_scatter_by_slotmap_kernel(
    slab_ptr,
    slot_map_ptr,
    local_expert_ids_ptr,
    topk_weights_ptr,
    output_ptr,
    N_padded,
    max_m,
    topk: tl.constexpr,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    token_id = tl.program_id(0).to(tl.int64)
    if token_id >= N_padded:
        return

    h_block = tl.program_id(1)
    h_start = (h_block * BLOCK_H).to(tl.int64)
    h_offs = h_start + tl.arange(0, BLOCK_H).to(tl.int64)
    h_mask = h_offs < H

    accum = tl.zeros([BLOCK_H], dtype=tl.float32)

    for k in tl.range(topk):
        idx_offset = token_id * topk + k
        slot = tl.load(slot_map_ptr + idx_offset).to(tl.int64)
        local_eid = tl.load(local_expert_ids_ptr + idx_offset).to(tl.int64)

        if slot >= 0:
            weight = tl.load(topk_weights_ptr + token_id * topk + k).to(tl.float32)
            slab_data = tl.load(
                slab_ptr + local_eid * max_m * H + slot * H + h_offs,
                mask=h_mask,
                other=0.0,
            ).to(tl.float32)
            accum += weight * slab_data

    tl.store(
        output_ptr + token_id * H + h_offs,
        accum.to(output_ptr.dtype.element_ty),
        mask=h_mask,
    )


@torch.no_grad()
def reverse_scatter_3d_to_2d(
    slab: torch.Tensor,
    slot_map: torch.Tensor,
    local_expert_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    output: torch.Tensor,
    max_m: int,
) -> None:
    N_padded = slot_map.shape[0]
    H = slab.shape[2]
    topk = slot_map.shape[1]

    BLOCK_H = triton.next_power_of_2(min(H, 1024))
    grid_h = triton.cdiv(H, BLOCK_H)

    _reverse_scatter_by_slotmap_kernel[(N_padded, grid_h)](
        slab,
        slot_map,
        local_expert_ids,
        topk_weights,
        output,
        N_padded, max_m,
        topk=topk,
        H=H,
        BLOCK_H=BLOCK_H,
    )
