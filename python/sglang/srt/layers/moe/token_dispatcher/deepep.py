from __future__ import annotations

import logging
import os
from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, NamedTuple, Optional, Tuple, Union

from sglang.srt.distributed.parallel_state import get_tp_group
from sglang.srt.environ import envs
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.dp_attention import get_is_extend_in_batch
from sglang.srt.layers.moe.token_dispatcher.base import (
    BaseDispatcher,
    BaseDispatcherConfig,
    CombineInput,
    CombineInputFormat,
    DispatcherBaseHooks,
    DispatchOutput,
    DispatchOutputFormat,
)
from sglang.srt.layers.moe.topk import TopKOutput
from sglang.srt.layers.moe.utils import (
    DeepEPMode,
    get_deepep_config,
    get_moe_runner_backend,
    is_tbo_enabled,
)
from sglang.srt.utils import (
    get_bool_env_var,
    is_blackwell,
    is_hip,
    is_npu,
    load_json_config,
)

_is_npu = is_npu()

if TYPE_CHECKING:
    from sglang.srt.batch_overlap.single_batch_overlap import CombineOverlapArgs

try:
    if _is_npu and envs.SGLANG_ZBAL_LOCAL_MEM_SIZE.get() > 0:
        from zbal.zbal.deepep_adaptor import Config
        from zbal.zbal_buffer import Buffer
    else:
        from deep_ep import Buffer, Config

    if not _is_npu:
        from sglang.srt.layers.quantization.fp8_kernel import (
            sglang_per_token_group_quant_fp8,
        )

    use_deepep = True
except ImportError:
    use_deepep = False

# DeepEP V2 introduces `ElasticBuffer` alongside the legacy `Buffer`
#(deepseek-ai/DeepEP#605, merged 2026-04-29). On V2 both classes are
# exported from `deep_ep.__init__`, so the existing `from deep_ep import
# Buffer` surface above continues to work unchanged — `ElasticBuffer` is
# an additional, MoE-shape ctor with auto-QP sizing that callers may
# opt into. The probe below is orthogonal to `use_deepep` and does not
# affect the default code path. V2 usage is further gated on
# `SGLANG_DEEPEP_USE_V2=1`. Mirrors the `HAVE_DEEP_EP_V2` probe shape
# already used in NVIDIA/Megatron-LM's `fused_a2a.py`.
try:
    from deep_ep import ElasticBuffer

    have_deepep_v2 = True
except ImportError:
    ElasticBuffer = None
    have_deepep_v2 = False

import inspect

_dispatch_supports_3d_sf = (
    have_deepep_v2
    and 'use_3d_sf_layout' in inspect.signature(ElasticBuffer.dispatch).parameters
)

from enum import Enum, IntEnum, auto

import torch
import torch.distributed as dist

_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and is_hip()

logger = logging.getLogger(__name__)


def _copy_rows_by_index(
    dst: torch.Tensor,
    dst_idx: torch.Tensor,
    src: torch.Tensor,
    src_idx: torch.Tensor,
) -> None:
    """dst[dst_idx] = src[src_idx], row-wise, supporting Float8_e4m3fn.

    torch's index_copy_/index_select are not implemented for Float8_e4m3fn on CUDA,
    so for such dtypes we operate on a uint8 byte view (pure data movement, dtype
    semantics irrelevant). Both dst and src must be 2D and share the same dtype/width.
    """
    if src_idx.numel() == 0:
        return
    if dst.dtype == torch.float8_e4m3fn:
        dst.view(torch.uint8).index_copy_(
            0, dst_idx, src.view(torch.uint8).index_select(0, src_idx)
        )
    else:
        dst.index_copy_(0, dst_idx, src.index_select(0, src_idx))


def _deepep_precompile_tp_barrier() -> None:
    # DeepEP's all-to-all operation has a much shorter timeout compared to torch.distributed,
    # so if different ranks compile at different speeds, it may quickly trigger a timeout.
    # To avoid this, we use torch.distributed's barrier during the compile stage.
    # We apply this barrier only in the compile stage to prevent extra all-reduce overhead at runtime.
    if envs.SGLANG_IN_DEEPGEMM_PRECOMPILE_STAGE.get():
        get_tp_group().barrier()


class DeepEPPDispatchHooks(DispatcherBaseHooks):
    def __call__(self, dispatcher: BaseDispatcher):
        for hook_fun in self.hook_dict.values():
            hook_fun(dispatcher)


class DeepEPNormalDispatchOutput(NamedTuple):
    """DeepEP normal dispatch output."""

    hidden_states: torch.Tensor
    hidden_states_scale: Optional[torch.Tensor]
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    num_recv_tokens_per_expert: List[int]

    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.DEEPEP_NORMAL


class DeepEPLLDispatchOutput(NamedTuple):
    """DeepEP low latency dispatch output."""

    hidden_states: torch.Tensor
    hidden_states_scale: Optional[torch.Tensor]
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    masked_m: torch.Tensor
    expected_m: int

    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.DEEPEP_LL


assert isinstance(DeepEPNormalDispatchOutput, DispatchOutput)
assert isinstance(DeepEPLLDispatchOutput, DispatchOutput)


class DeepEPNormalCombineInput(NamedTuple):
    """DeepEP normal combine input."""

    hidden_states: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor

    @property
    def format(self) -> CombineInputFormat:
        return CombineInputFormat.DEEPEP_NORMAL


class DeepEPLLCombineInput(NamedTuple):
    """DeepEP low latency combine input."""

    hidden_states: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor

    @property
    def format(self) -> CombineInputFormat:
        return CombineInputFormat.DEEPEP_LL


assert isinstance(DeepEPNormalCombineInput, CombineInput)
assert isinstance(DeepEPLLCombineInput, CombineInput)


class DeepEPV2ExpandDispatchOutput(NamedTuple):
    """V2 ElasticBuffer dispatch output for eager/prefill path (do_expand=True).

    hidden_states is [num_expanded_tokens, H] — already expert-sorted by the V2
    copy-epilogue kernel.  No Python-side format conversion needed.
    """

    hidden_states: torch.Tensor
    hidden_states_scale: Optional[torch.Tensor]
    psum_num_recv_tokens_per_expert: torch.Tensor
    num_recv_tokens_per_expert_list: List[int]

    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.DEEPEP_V2_EXPAND


class DeepEPV2ExpandCombineInput(NamedTuple):
    """V2 expand combine input — 2D expert-sorted, consumed directly by V2 combine."""

    hidden_states: torch.Tensor
    topk_ids: Optional[torch.Tensor] = None
    topk_weights: Optional[torch.Tensor] = None

    @property
    def format(self) -> CombineInputFormat:
        return CombineInputFormat.DEEPEP_V2_EXPAND


assert isinstance(DeepEPV2ExpandDispatchOutput, DispatchOutput)
assert isinstance(DeepEPV2ExpandCombineInput, CombineInput)


class DeepEPV2NormalDispatchOutput(NamedTuple):
    """V2 ElasticBuffer dispatch output for prefill (do_expand=False).

    Same layout as DeepEPNormalDispatchOutput but uses a GPU psum tensor
    instead of a CPU list, avoiding do_cpu_sync overhead.
    num_total_alloc is the worst-case allocation size from dispatch,
    needed by combine to match the expected tensor size.
    """

    hidden_states: torch.Tensor
    hidden_states_scale: Optional[torch.Tensor]
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    psum_num_recv_tokens_per_expert: torch.Tensor
    num_total_alloc: int

    @property
    def format(self) -> DispatchOutputFormat:
        return DispatchOutputFormat.DEEPEP_V2_NORMAL


assert isinstance(DeepEPV2NormalDispatchOutput, DispatchOutput)


class DeepEPDispatchMode(IntEnum):
    NORMAL = auto()
    LOW_LATENCY = auto()


class DeepEPBuffer:
    _buffer = None
    _dispatch_mode: Optional[DeepEPDispatchMode] = None
    _hidden_size: Optional[int] = None
    _num_max_dispatch_tokens_per_rank: Optional[int] = None
    _num_experts: Optional[int] = None

    @classmethod
    def get_deepep_buffer(
        cls,
        group: dist.ProcessGroup,
        hidden_size: int,
        param_bytes: int,
        deepep_mode: DeepEPMode,
        num_max_dispatch_tokens_per_rank: int = -1,
        num_experts: int = -1,
        num_topk: int = 0,
    ):
        if cls._buffer is not None:
            return cls._buffer

        cls._hidden_size = hidden_size
        cls._num_max_dispatch_tokens_per_rank = num_max_dispatch_tokens_per_rank
        cls._num_experts = num_experts

        if have_deepep_v2 and get_bool_env_var("SGLANG_DEEPEP_USE_V2", default="false"):
            cls._buffer = cls._build_v2_buffer(
                group,
                hidden_size,
                deepep_mode,
                num_max_dispatch_tokens_per_rank,
                num_experts,
                num_topk,
            )
            return cls._buffer

        num_nvl_bytes, num_rdma_bytes = 0, 0
        if deepep_mode.enable_normal():
            hidden_bytes = hidden_size * param_bytes
            for config in (
                DeepEPConfig.get_instance().normal_dispatch_config
                or Buffer.get_dispatch_config(group.size()),
                DeepEPConfig.get_instance().normal_combine_config
                or Buffer.get_combine_config(group.size()),
            ):
                num_nvl_bytes = max(
                    config.get_nvl_buffer_size_hint(hidden_bytes, group.size()),
                    num_nvl_bytes,
                )
                num_rdma_bytes = max(
                    config.get_rdma_buffer_size_hint(hidden_bytes, group.size()),
                    num_rdma_bytes,
                )
        if deepep_mode.enable_low_latency():
            assert num_max_dispatch_tokens_per_rank != -1
            assert num_experts != -1 and num_experts % group.size() == 0
            num_rdma_bytes = max(
                Buffer.get_low_latency_rdma_size_hint(
                    num_max_dispatch_tokens_per_rank,
                    hidden_size,
                    group.size(),
                    num_experts,
                ),
                num_rdma_bytes,
            )

        # We should calculate num_qps_per_rank consistently with DeepEP's test script logic:
        if deepep_mode == DeepEPMode.NORMAL:
            # refer: https://github.com/deepseek-ai/DeepEP/blob/main/tests/test_internode.py#L235
            num_qps_per_rank = DeepEPConfig.get_instance().num_sms
        elif deepep_mode == DeepEPMode.LOW_LATENCY:
            # refer: https://github.com/deepseek-ai/DeepEP/blob/main/tests/test_low_latency.py#L176
            num_qps_per_rank = num_experts // group.size()
        elif deepep_mode == DeepEPMode.AUTO:
            # low-latency and normal mode all need run
            # refer: https://github.com/deepseek-ai/DeepEP/blob/main/tests/test_internode.py#L235
            num_qps_per_rank = max(
                DeepEPConfig.get_instance().num_sms, num_experts // group.size()
            )
        else:
            raise NotImplementedError

        if not _is_npu:
            total_num_sms = torch.cuda.get_device_properties(
                device="cuda"
            ).multi_processor_count
            if (
                (deepep_mode != DeepEPMode.LOW_LATENCY)
                and not is_tbo_enabled()
                and (DeepEPConfig.get_instance().num_sms < total_num_sms // 2)
            ):
                logger.warning(
                    f"Only use {DeepEPConfig.get_instance().num_sms} SMs for DeepEP communication. "
                    f"This may result in highly suboptimal performance. "
                    f"Consider using --deepep-config to change the behavior."
                )

        cls._buffer = Buffer(
            group,
            num_nvl_bytes,
            num_rdma_bytes,
            low_latency_mode=deepep_mode.enable_low_latency(),
            num_qps_per_rank=num_qps_per_rank,
            # TODO can be false when unneeded
            allow_mnnvl=True,
        )
        return cls._buffer

    @classmethod
    def _build_v2_buffer(
        cls,
        group: dist.ProcessGroup,
        hidden_size: int,
        deepep_mode: DeepEPMode,
        num_max_dispatch_tokens_per_rank: int,
        num_experts: int,
        num_topk: int,
    ):
        if num_max_dispatch_tokens_per_rank <= 0:
            num_max_dispatch_tokens_per_rank = (
                envs.SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK.get()
            )
        if deepep_mode.enable_low_latency():
            assert num_experts != -1 and num_experts % group.size() == 0

        backend = get_moe_runner_backend()
        need_bf16_dispatch = backend.is_cutlass() or (
            backend.is_deep_gemm() and envs.SGLANG_DEEPEP_BF16_DISPATCH.get()
        )
        use_fp8 = deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM and not need_bf16_dispatch
        allow_multiple_reduction = int(os.environ.get("EP_MULTIPLE_REDUCTION", "1")) != 0

        logger.info(
            "SGLANG_DEEPEP_USE_V2=1: constructing deep_ep.ElasticBuffer "
            "(num_max_tokens_per_rank=%d, hidden=%d, num_topk=%d, use_fp8_dispatch=%s, multiple_reduction=%d).",
            num_max_dispatch_tokens_per_rank,
            hidden_size,
            num_topk,
            use_fp8,
            allow_multiple_reduction,
        )

        return ElasticBuffer(
            group=group,
            num_max_tokens_per_rank=num_max_dispatch_tokens_per_rank,
            hidden=hidden_size,
            num_topk=num_topk,
            use_fp8_dispatch=use_fp8,
            allow_multiple_reduction=allow_multiple_reduction,
        )

    @classmethod
    def clean_buffer(cls):
        # DeepEP V2's `ElasticBuffer` does not expose `low_latency_mode`
        # or `clean_low_latency_buffer` — low-latency cleanup is handled
        # internally via `EPHandle` lifetime. Fall through for V2.
        if not hasattr(cls._buffer, "clean_low_latency_buffer"):
            return
        if not cls._buffer.low_latency_mode:
            return
        cls._buffer.clean_low_latency_buffer(
            cls._num_max_dispatch_tokens_per_rank,
            cls._hidden_size,
            cls._num_experts,
        )

    @classmethod
    def set_dispatch_mode_as_normal(cls):
        cls._dispatch_mode = DeepEPDispatchMode.NORMAL

    @classmethod
    def set_dispatch_mode_as_low_latency(cls):
        if cls._dispatch_mode == DeepEPDispatchMode.NORMAL:
            cls.clean_buffer()
        cls._dispatch_mode = DeepEPDispatchMode.LOW_LATENCY

    @classmethod
    def set_dispatch_mode(cls, mode: DeepEPMode):
        if mode.is_low_latency():
            cls.set_dispatch_mode_as_low_latency()
        elif mode.is_normal():
            cls.set_dispatch_mode_as_normal()
        else:
            raise Exception("unsupported mode")


class DeepEPConfig(BaseDispatcherConfig):
    _instance = None

    def __init__(self):
        config_str = get_deepep_config()
        if config_str:
            config_parsed = load_json_config(config_str)
            if torch.distributed.get_rank() == 0:
                logger.info(f"Use DeepEP Config: {config_parsed}")
            config_dispatch = config_parsed["normal_dispatch"]
            config_combine = config_parsed["normal_combine"]

            self.normal_dispatch_config = Config(**config_dispatch)
            self.normal_combine_config = Config(**config_combine)

            assert config_dispatch["num_sms"] == config_combine["num_sms"]
            self.num_sms = config_dispatch["num_sms"]
        else:
            self.normal_dispatch_config = None
            self.normal_combine_config = None
            self.num_sms = Buffer.num_sms

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = DeepEPConfig()
        return cls._instance


class _DeepEPDispatcherImplBase:
    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
        router_topk: int,
        permute_fusion: bool,
        num_experts: int,
        num_local_experts: int,
        hidden_size: int,
        params_dtype: torch.dtype,
        deepep_mode: DeepEPMode,
    ):
        if not use_deepep:
            raise ImportError(
                "DeepEP is not installed. Please install DeepEP package from "
                "https://github.com/deepseek-ai/deepep."
            )

        self.group = group
        self.router_topk = router_topk
        self.permute_fusion = permute_fusion
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.hidden_size = hidden_size
        self.params_dtype = params_dtype
        self.deepep_mode = deepep_mode

        self.params_bytes = 2
        # A large value will lead to large memory occupation, thus users should change it accordingly
        self.num_max_dispatch_tokens_per_rank = (
            envs.SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK.get()
        )
        # DeepEP internode_ll dispatch uses FINISHED_SUM_TAG=1024
        # and the logic requires num-tokens-sent-from-one-rank-to-another-rank less than it
        if not have_deepep_v2 and not get_bool_env_var("SGLANG_DEEPEP_USE_V2", default="false"):
            assert self.num_max_dispatch_tokens_per_rank <= 1024

        self.handle = None

        self.quant_config: Optional[dict] = None

        self.overlap_args: Optional[CombineOverlapArgs] = None
        self.meta_overlap_args: Optional[dict] = None

    def dispatch_a(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ):
        raise NotImplementedError

    def dispatch_b(self, *args, **kwargs):
        raise NotImplementedError

    def combine_a(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        raise NotImplementedError

    def combine_b(self, *args, **kwargs):
        raise NotImplementedError

    def _get_buffer(self):
        raise NotImplementedError

    def set_quant_config(self, quant_config: dict) -> None:
        self.quant_config = quant_config

    def set_overlap_args(
        self, combine_overlap_args: CombineOverlapArgs, meta_overlap_args: dict
    ) -> None:
        self.overlap_args = combine_overlap_args
        self.meta_overlap_args = meta_overlap_args

    def clear_overlap_args(self) -> None:
        self.overlap_args = None
        self.meta_overlap_args = None


class _DeepEPDispatcherImplNormal(_DeepEPDispatcherImplBase):
    def __init__(self, async_finish: bool, **kwargs):
        super().__init__(**kwargs)

        self.async_finish = async_finish
        self.src2dst = None
        self.quant_config = {}

    def dispatch_a(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ):
        topk_weights, topk_ids = topk_output.topk_weights, topk_output.topk_ids
        topk_ids = topk_ids.to(torch.int64)
        backend = get_moe_runner_backend()
        # BF16 dispatch is needed when:
        #   - cutlass backend (uses different kernel)
        #   - deep_gemm backend with SGLANG_DEEPEP_BF16_DISPATCH enabled
        need_bf16_dispatch = backend.is_cutlass() or (
            backend.is_deep_gemm() and envs.SGLANG_DEEPEP_BF16_DISPATCH.get()
        )
        if deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM and not need_bf16_dispatch:
            # TODO hard code 128 block quant,use fp8 communication
            hidden_states = sglang_per_token_group_quant_fp8(
                hidden_states,
                128,
                column_major_scales=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
                scale_tma_aligned=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
                scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
            )
        previous_event = Buffer.capture() if self.async_finish else None
        return hidden_states, topk_ids, topk_weights, previous_event

    def dispatch_b(self, hidden_states, topk_ids, topk_weights, previous_event):
        (
            hidden_states,
            topk_ids,
            topk_weights,
            num_recv_tokens_per_expert,
            event,
        ) = self._dispatch_core(hidden_states, topk_ids, topk_weights, previous_event)
        event.current_stream_wait() if self.async_finish else ()

        if isinstance(hidden_states, tuple):
            hidden_states, hidden_states_scale = hidden_states
        else:
            hidden_states_scale = None

        return DeepEPNormalDispatchOutput(
            hidden_states,
            hidden_states_scale,
            topk_ids,
            topk_weights,
            num_recv_tokens_per_expert,
        )

    def _dispatch_core(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        previous_event,
    ):
        buffer = self._get_buffer()
        (
            num_tokens_per_rank,
            num_tokens_per_rdma_rank,
            num_tokens_per_expert,
            is_token_in_rank,
            previous_event,
        ) = buffer.get_dispatch_layout(
            topk_ids,
            self.num_experts,
            previous_event=previous_event,
            async_finish=self.async_finish,
            allocate_on_comm_stream=previous_event is not None,
        )
        # FIXME: `handle` should be transmitted with tokens from dispatch to combine.
        # However, doing this would incur an unknown synchronization error, but keeping
        # `handle` as a member variable works.

        _deepep_precompile_tp_barrier()
        (
            recv_x,
            recv_topk_ids,
            recv_topk_weights,
            num_recv_tokens_per_expert,
            self.handle,
            event,
        ) = buffer.dispatch(
            x,
            topk_idx=topk_ids,
            topk_weights=topk_weights,
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            previous_event=previous_event,
            async_finish=self.async_finish,
            allocate_on_comm_stream=(previous_event is not None) and self.async_finish,
            expert_alignment=128 if deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM else 1,
            config=DeepEPConfig.get_instance().normal_dispatch_config,
        )
        get_global_expert_distribution_recorder().on_deepep_dispatch_normal(
            num_recv_tokens_per_expert,
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            num_tokens_per_expert=num_tokens_per_expert,
        )

        return (
            recv_x,
            recv_topk_ids,
            recv_topk_weights,
            num_recv_tokens_per_expert,
            event,
        )

    def combine_a(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ):

        if deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM or _use_aiter or _is_npu:
            output = hidden_states
        else:
            raise NotImplementedError()  # triton runner was supported but it's temporarily disabled

        previous_event = Buffer.capture() if self.async_finish else None
        return output, previous_event

    def combine_b(self, output, previous_event):
        hidden_states, event = self._combine_core(output, previous_event)
        event.current_stream_wait() if self.async_finish else ()
        self.handle = None
        self.src2dst = None
        return hidden_states

    def _combine_core(self, x: torch.Tensor, previous_event):
        buffer = self._get_buffer()
        _deepep_precompile_tp_barrier()
        combined_x, _, event = buffer.combine(
            x,
            self.handle,
            async_finish=self.async_finish,
            previous_event=previous_event,
            allocate_on_comm_stream=previous_event is not None,
            config=DeepEPConfig.get_instance().normal_combine_config,
        )
        return combined_x, event

    def _get_buffer(self):
        DeepEPBuffer.set_dispatch_mode_as_normal()

        return DeepEPBuffer.get_deepep_buffer(
            self.group,
            self.hidden_size,
            self.params_bytes,
            self.deepep_mode,
            self.num_max_dispatch_tokens_per_rank,
            self.num_experts,
        )


class _DeepEPDispatcherImplLowLatency(_DeepEPDispatcherImplBase):
    def __init__(self, return_recv_hook: bool, **kwargs):
        super().__init__(**kwargs)

        """
        num_max_dispatch_tokens_per_rank: the actual batch size in the decoding engine should be less than 256
        https://github.com/deepseek-ai/DeepEP?tab=readme-ov-file#example-use-in-inference-decoding
        """
        self.return_recv_hook = return_recv_hook
        self.device_module = torch.get_device_module()
        self.quant_config = {}

    def dispatch_a(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ):
        buffer = self._get_buffer()
        topk_weights, topk_ids = topk_output.topk_weights, topk_output.topk_ids
        topk_ids = topk_ids.to(torch.int64)
        expected_m = (
            hidden_states.shape[0] * buffer.group_size * topk_ids.shape[1]
            + self.num_experts
        ) // self.num_experts
        hidden_states, masked_m, event, hook = self._dispatch_core(
            hidden_states,
            topk_ids,
        )
        return (
            hidden_states,
            topk_ids,
            topk_weights,
            masked_m,
            expected_m,
            event,
            hook,
        )

    def dispatch_b(
        self,
        hidden_states,
        topk_ids,
        topk_weights,
        masked_m,
        expected_m,
        event,
        hook,
    ):
        hook() if self.return_recv_hook else event.current_stream_wait()

        get_global_expert_distribution_recorder().on_deepep_dispatch_low_latency(
            masked_m
        )

        if isinstance(hidden_states, tuple):
            hidden_states, hidden_states_scale = hidden_states
        else:
            hidden_states_scale = None

        deepep_output = DeepEPLLDispatchOutput(
            hidden_states,
            hidden_states_scale,
            topk_ids,
            topk_weights,
            masked_m,
            expected_m,
        )
        return deepep_output

    def _dispatch_core(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
    ):
        use_nvfp4 = use_fp8 = False
        input_global_scale = self.quant_config.get("input_global_scale", None)
        bf16_dispatch = self.quant_config.get("bf16_dispatch", False)
        if input_global_scale is not None:
            use_nvfp4 = True
        else:
            backend = get_moe_runner_backend()
            # BF16 dispatch is needed when:
            #   - quant_config requests BF16 dispatch explicitly
            #   - flashinfer_cutedsl: kernel quantizes to NVFP4 internally
            #   - NPU with SGLANG_DEEPEP_BF16_DISPATCH: INT8 input + BF16 weight GMM not supported
            #   - deep_gemm with SGLANG_DEEPEP_BF16_DISPATCH: user requests BF16 dispatch
            need_bf16_dispatch = (
                bf16_dispatch
                or backend.is_flashinfer_cutedsl()
                or (_is_npu and envs.SGLANG_DEEPEP_BF16_DISPATCH.get())
                or (backend.is_deep_gemm() and envs.SGLANG_DEEPEP_BF16_DISPATCH.get())
            )
            if not need_bf16_dispatch:
                use_fp8 = True

        # round_scale / use_ue8m0 are FP8-DeepGEMM specific; they cause DeepEP
        # to return int32-packed UE8M0 scales that don't feed the flashinfer
        # cutedsl kernel.
        fp8_deepgemm_scale_opts = (
            dict(
                round_scale=deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
                and deep_gemm_wrapper.DEEPGEMM_BLACKWELL,
                use_ue8m0=deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
                and deep_gemm_wrapper.DEEPGEMM_BLACKWELL,
            )
            if use_fp8
            else dict()
        )

        buffer = self._get_buffer()
        _deepep_precompile_tp_barrier()
        packed_recv_hidden, self.packed_recv_count, self.handle, event, hook = (
            buffer.low_latency_dispatch(
                hidden_states,
                topk_ids,
                self.num_max_dispatch_tokens_per_rank,
                self.num_experts,
                use_fp8=use_fp8,
                **(dict(use_nvfp4=True) if use_nvfp4 else dict()),
                **(
                    dict(x_global_scale=input_global_scale)
                    if input_global_scale is not None
                    else dict()
                ),
                async_finish=not self.return_recv_hook,
                return_recv_hook=self.return_recv_hook,
                **fp8_deepgemm_scale_opts,
            )
        )
        return packed_recv_hidden, self.packed_recv_count, event, hook

    def combine_a(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        hidden_states, event, hook = self._combine_core(
            hidden_states,
            topk_ids,
            topk_weights,
        )
        return hidden_states, event, hook

    def combine_b(self, hidden_states, event, hook):
        overlap_args = self.overlap_args
        if overlap_args is not None:
            overlap_args.stream.wait_stream(self.device_module.current_stream())

        hook() if self.return_recv_hook else event.current_stream_wait()

        if overlap_args is not None:
            self.device_module.current_stream().wait_stream(overlap_args.stream)

        return hidden_states

    def _combine_core(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        buffer = self._get_buffer()
        overlap_args = self.overlap_args
        meta_overlap_args = self.meta_overlap_args

        ctx = nullcontext()
        if overlap_args is not None:
            overlap_args.stream.wait_event(overlap_args.wait_event)
            ctx = torch.cuda.stream(overlap_args.stream)

            if is_blackwell():
                overlap_args_dict = dict(
                    overlap=overlap_args.overlap,
                    src_signals=overlap_args.signal,
                    src_signal_expect_value=overlap_args.threshold,
                )
            else:
                overlap_args_dict = dict(
                    overlap=overlap_args.overlap,
                    packed_recv_count=self.packed_recv_count,
                    comp_signal=overlap_args.signal,
                    block_m=meta_overlap_args["block_m"],
                    threshold=meta_overlap_args["threshold"],
                    num_sms=overlap_args.num_sms,
                )
        else:
            overlap_args_dict = {}

        with ctx:
            _deepep_precompile_tp_barrier()
            combined_hidden_states, event, hook = buffer.low_latency_combine(
                x=hidden_states,
                topk_idx=topk_ids,
                topk_weights=topk_weights,
                handle=self.handle,
                async_finish=not self.return_recv_hook,
                return_recv_hook=self.return_recv_hook,
                **overlap_args_dict,
            )

        self.packed_recv_count = self.handle = None
        return combined_hidden_states, event, hook

    def _get_buffer(self):
        DeepEPBuffer.set_dispatch_mode_as_low_latency()
        return DeepEPBuffer.get_deepep_buffer(
            self.group,
            self.hidden_size,
            self.params_bytes,
            self.deepep_mode,
            self.num_max_dispatch_tokens_per_rank,
            self.num_experts,
        )

class _DeepEPDispatcherImplV2(_DeepEPDispatcherImplBase):
    """DeepEP V2 (ElasticBuffer) dispatcher implementation.
    """

    _shared_grid_hidden = None
    _shared_combine_compact_buf = None

    def __init__(self, async_finish: bool, **kwargs):
        super().__init__(**kwargs)
        self.async_finish = async_finish
        self.quant_config = {}
        self.num_comm_sms = 0
        self._dispatch_ctx = {}
        self.device_module = torch.get_device_module()
        self._grid_sf = None
        self._expand_dst_idx = None
        self._expand_src_idx = None
        R = self.group.size()
        T_max = self.num_max_dispatch_tokens_per_rank
        E_local = self.num_local_experts
        H = self.hidden_size
        total_slots = E_local * R * T_max
        if _DeepEPDispatcherImplV2._shared_grid_hidden is None:
            _DeepEPDispatcherImplV2._shared_grid_hidden = torch.empty(
                (total_slots, H), dtype=torch.float8_e4m3fn, device="cuda"
            )
            _DeepEPDispatcherImplV2._shared_combine_compact_buf = torch.zeros(
                (total_slots, H), dtype=torch.bfloat16, device="cuda"
            )

    def dispatch_a(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ):
        topk_weights, topk_ids = topk_output.topk_weights, topk_output.topk_ids
        topk_ids = topk_ids.to(torch.int64)

        is_extend = get_is_extend_in_batch()

        all_zero_mask = (topk_ids == 0).all(dim=1)
        topk_ids = topk_ids.masked_fill(all_zero_mask.unsqueeze(1), -1)
        topk_weights = topk_weights.masked_fill(all_zero_mask.unsqueeze(1), 0.0)

        backend = get_moe_runner_backend()
        need_bf16_dispatch = backend.is_cutlass() or (
            backend.is_deep_gemm() and envs.SGLANG_DEEPEP_BF16_DISPATCH.get()
        )
        if deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM and not need_bf16_dispatch:
            hidden_states = sglang_per_token_group_quant_fp8(
                hidden_states,
                128,
                column_major_scales=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
                scale_tma_aligned=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
                scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
            )

        E_local = self.num_local_experts
        R = self.group.size()
        T_max = self.num_max_dispatch_tokens_per_rank
        alignment = 128 if deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM else 1

        if is_extend:
            do_expand = False
            expert_alignment = alignment
            do_cpu_sync = True
            self._prefill_sync_mode = True
        else:
            do_expand = True
            expert_alignment = R * T_max
            do_cpu_sync = False
            self._prefill_sync_mode = False

        previous_event = ElasticBuffer.capture() if (self.async_finish and not do_cpu_sync) else None
        recv_x, recv_topk_ids, recv_topk_weights, event = self._dispatch_core(
            hidden_states, topk_ids, topk_weights, previous_event,
            do_expand=do_expand,
            expert_alignment=expert_alignment,
            do_cpu_sync=do_cpu_sync,
        )

        return (topk_ids, topk_weights, is_extend, expert_alignment,
                recv_x, recv_topk_ids, recv_topk_weights, event)

    def dispatch_b(self, topk_ids, topk_weights, is_extend, expert_alignment,
                   recv_x, recv_topk_ids, recv_topk_weights, event):
        if event is not None and event.event is not None:
            event.current_stream_wait()

        if isinstance(recv_x, tuple):
            recv_hidden, recv_sf = recv_x
        else:
            recv_hidden, recv_sf = recv_x, None

        if is_extend:
            self._dispatch_ctx = {}
            self._expand_dst_idx = None
            self._expand_src_idx = None

            v2_output = DeepEPNormalDispatchOutput(
                hidden_states=recv_hidden,
                hidden_states_scale=recv_sf,
                topk_ids=recv_topk_ids,
                topk_weights=recv_topk_weights,
                num_recv_tokens_per_expert=self.handle.num_recv_tokens_per_expert_list,
            )

            return v2_output

        E_local = self.num_local_experts
        R = self.group.size()
        T_max = self.num_max_dispatch_tokens_per_rank
        num_topk = topk_ids.shape[1]
        max_m = expert_alignment

        H = recv_hidden.shape[-1]

        # V2 expand layout is COMPACT + per-expert alignment, NOT a fixed
        # [E_local, max_m] grid. Data for expert e lives at:
        #   start_e = align(psum_padded[e], max_m); count_e = psum_padded[e+1] - start_e
        # where psum_padded = [0] + psum. A naive view(E_local, max_m, H) reads the
        # wrong offsets. We scatter the compact rows into a preallocated fixed grid
        # so that grid[e, :count_e] == expert e's real tokens (padding is garbage but
        # unused: DeepGEMM masked GEMM only reads the first masked_m[e] rows).
        psum = self.handle.psum_num_recv_tokens_per_expert  # inclusive, aligned prefix sum
        psum_padded = torch.cat(
            [psum.new_zeros(1), psum]
        )  # [E_local + 1], psum_padded[0] = 0

        # start_e = align(psum_padded[e], max_m), count_e = psum_padded[e+1] - start_e
        starts = ((psum_padded[:-1] + max_m - 1) // max_m) * max_m  # [E_local]
        masked_m = (psum_padded[1:] - starts).to(torch.int32)  # [E_local]

        # Scatter compact rows into the fixed grid using pure GPU ops (no .item(),
        # no boolean indexing, no Python loops) for cuda graph compatibility.
        total_slots = E_local * max_m
        grid_hidden = _DeepEPDispatcherImplV2._shared_grid_hidden

        # Build full [E_local, max_m] index grids on GPU (fixed shape, no .item())
        row_offsets = torch.arange(max_m, device=recv_hidden.device).unsqueeze(0)  # [1, max_m]
        src_all = starts.unsqueeze(1) + row_offsets      # [E_local, max_m] compact src indices
        dst_all = (torch.arange(E_local, device=recv_hidden.device) * max_m).unsqueeze(1) + row_offsets  # [E_local, max_m] grid dst indices
        valid_mask = row_offsets < masked_m.unsqueeze(1)  # [E_local, max_m] bool

        # Clamp invalid src indices to 0 (they will be overwritten or ignored by GEMM)
        src_flat = src_all.clamp(max=recv_hidden.shape[0] - 1).view(-1)  # [E_local*max_m]
        dst_flat = dst_all.view(-1)                                       # [E_local*max_m]
        valid_flat = valid_mask.view(-1)                                  # [E_local*max_m]

        # Copy ALL rows (including invalid → garbage, but masked GEMM ignores them)
        _copy_rows_by_index(grid_hidden, dst_flat, recv_hidden, src_flat)
        slab_3d = grid_hidden.view(E_local, max_m, H)

        # Save the VALID grid<->compact row maps for combine inverse gather.
        # torch.where returns fixed-shape when given full args (no dynamic indexing).
        # For combine we only need valid indices, but we must build them without .item().
        # Store the full maps + mask; combine_a will filter using the mask.
        # Save the grid<->compact row maps for combine inverse gather.
        # For invalid rows (where valid_mask=False), set src to dst (identity mapping)
        # so index_copy_ in combine won't have duplicate targets.
        safe_src = torch.where(valid_mask.view(-1), src_all.view(-1), dst_flat)
        self._expand_dst_all = dst_flat
        self._expand_src_all = safe_src
        self._expand_valid_mask = valid_mask.view(-1)
        self._expand_compact_rows = psum_padded[-1]

        slab_sf_3d = None
        if recv_sf is not None:
            if recv_sf.dim() == 3:
                # SF already in fixed [E_local, tokens_per_expert, sf_dim] grid
                # (use_3d_sf_layout=True). Its per-expert row layout matches the
                # scattered hidden grid (expert e's data in rows [0, count_e)),
                # so it can be consumed directly.
                slab_sf_3d = recv_sf
            else:
                # 2D compact SF: scatter into a fixed grid the same way as hidden.
                sf_dim = recv_sf.shape[-1]
                if (
                    getattr(self, "_grid_sf", None) is None
                    or self._grid_sf.shape != (total_slots, sf_dim)
                    or self._grid_sf.dtype != recv_sf.dtype
                ):
                    self._grid_sf = recv_sf.new_empty((total_slots, sf_dim))
                grid_sf = self._grid_sf
                sf_src = src_all.clamp(max=recv_sf.shape[0] - 1).view(-1)
                grid_sf.index_copy_(0, dst_flat, recv_sf.index_select(0, sf_src))
                slab_sf_3d = grid_sf.view(E_local, max_m, sf_dim)
        expected_m = (topk_ids.shape[0] * R * num_topk + self.num_experts) // self.num_experts
        self._dispatch_ctx = {}
        self._num_expanded_tokens = psum_padded[-1]

        return DeepEPLLDispatchOutput(
            hidden_states=slab_3d,
            hidden_states_scale=slab_sf_3d,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            masked_m=masked_m,
            expected_m=expected_m,
        )

    def _dispatch_core(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        previous_event,
        do_expand: bool = True,
        expert_alignment: Optional[int] = None,
        do_cpu_sync: bool = False,
    ):
        """Execute the ElasticBuffer all-to-all dispatch communication.

        do_expand controls the output layout:
          False (prefill): deduplicated 2D, do_cpu_sync=True for CPU token counts
          True (decode): expanded 2D sorted by expert, do_cpu_sync=False
        """
        buffer = self._get_buffer()
        _deepep_precompile_tp_barrier()

        if expert_alignment is None:
            expert_alignment = 128 if deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM else 1

        use_async = self.async_finish and not do_cpu_sync

        extra_kwargs = {}
        if _dispatch_supports_3d_sf:
            extra_kwargs['use_3d_sf_layout'] = do_expand

        # logger.info(
        #     f"[V2 dispatch] extra_kwargs={extra_kwargs}, "
        #     f"_dispatch_supports_3d_sf={_dispatch_supports_3d_sf}, "
        #     f"do_expand={do_expand}, do_cpu_sync={do_cpu_sync}, "
        #     f"use_async={use_async}, expert_alignment={expert_alignment}, "
        #     f"num_sms={self.num_comm_sms}"
        # )

        (
            recv_x,
            recv_topk_ids,
            recv_topk_weights,
            self.handle,
            event,
        ) = buffer.dispatch(
            x,
            topk_idx=topk_ids,
            topk_weights=topk_weights,
            num_experts=self.num_experts,
            num_max_tokens_per_rank=self.num_max_dispatch_tokens_per_rank,
            expert_alignment=expert_alignment,
            num_sms=self.num_comm_sms,
            previous_event=previous_event if use_async else None,
            async_with_compute_stream=use_async,
            allocate_on_comm_stream=(previous_event is not None) and use_async,
            do_expand=do_expand,
            do_cpu_sync=do_cpu_sync,
            use_tma_aligned_col_major_sf=(
                deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
                and (deep_gemm_wrapper.DEEPGEMM_BLACKWELL or do_expand)
            ),
            **extra_kwargs,
        )

        return (
            recv_x,
            recv_topk_ids,
            recv_topk_weights,
            event,
        )

    def combine_a(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        if deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM or _use_aiter or _is_npu:
            output = hidden_states
        else:
            raise NotImplementedError()

        if output.dim() == 3:
            output = output.view(-1, output.shape[-1])

        expand_dst_all = getattr(self, "_expand_dst_all", None)
        if expand_dst_all is not None:
            src_all = self._expand_src_all
            dst_expanded = expand_dst_all.unsqueeze(1).expand_as(output)
            src_expanded = src_all.unsqueeze(1).expand_as(output)
            compact = _DeepEPDispatcherImplV2._shared_combine_compact_buf
            compact.zero_()
            compact.scatter_(0, src_expanded, output.gather(0, dst_expanded))
            output = compact
            self._expand_dst_all = None
            self._expand_src_all = None
            self._expand_valid_mask = None

        self._combine_topk_weights = topk_weights
        self._combine_use_async = self.async_finish and not getattr(self, '_prefill_sync_mode', False)
        previous_event = ElasticBuffer.capture() if self._combine_use_async else None
        return output, previous_event

    def combine_b(self, output, previous_event):
        """Phase 2 of combine: execute V2 ElasticBuffer combine communication."""
        overlap_args = self.overlap_args

        if overlap_args is not None:
            overlap_args.stream.wait_event(overlap_args.wait_event)
            with torch.cuda.stream(overlap_args.stream):
                hidden_states, event = self._combine_core(output, previous_event)
                if event is not None and event.event is not None:
                    event.current_stream_wait()
        else:
            hidden_states, event = self._combine_core(output, previous_event)
            if event is not None and event.event is not None:
                event.current_stream_wait()

        self.handle = None
        self._dispatch_ctx = {}

        if overlap_args is not None:
            self.device_module.current_stream().wait_stream(overlap_args.stream)

        return hidden_states

    def _combine_core(self, x: torch.Tensor, previous_event):
        buffer = self._get_buffer()
        _deepep_precompile_tp_barrier()

        topk_weights = getattr(self, '_combine_topk_weights', None)
        if self.handle is not None and self.handle.do_expand:
            topk_weights = None

        use_async = getattr(self, '_combine_use_async', self.async_finish)

        combined_x, _, event = buffer.combine(
            x,
            self.handle,
            topk_weights=topk_weights,
            num_sms=self.num_comm_sms,
            previous_event=previous_event,
            async_with_compute_stream=use_async,
            allocate_on_comm_stream=previous_event is not None,
        )
        self._combine_topk_weights = None
        return combined_x, event

    def _get_buffer(self):
        buffer = DeepEPBuffer.get_deepep_buffer(
            self.group,
            self.hidden_size,
            self.params_bytes,
            self.deepep_mode,
            self.num_max_dispatch_tokens_per_rank,
            self.num_experts,
            num_topk=self.router_topk,
        )
        if self.num_comm_sms == 0:
            self.num_comm_sms = buffer.get_theoretical_num_sms(
                self.num_experts, self.router_topk
            )
            # logger.info(
            #     "DeepEP V2 num_comm_sms=%d (num_experts=%d, router_topk=%d)",
            #     self.num_comm_sms, self.num_experts, self.router_topk,
            # )
        return buffer

@dataclass
class _Stage(Enum):
    INITIAL = auto()
    AFTER_DISPATCH_A = auto()
    AFTER_DISPATCH_B = auto()
    AFTER_COMBINE_A = auto()


class DeepEPDispatcher(BaseDispatcher):
    @property
    def _is_v2(self) -> bool:
        return have_deepep_v2 and get_bool_env_var("SGLANG_DEEPEP_USE_V2", default="false")

    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
        router_topk: int,
        permute_fusion: bool = False,
        num_experts: int = None,
        num_local_experts: int = None,
        hidden_size: int = None,
        params_dtype: torch.dtype = None,
        deepep_mode: DeepEPMode = DeepEPMode.AUTO,
        async_finish: bool = False,
        return_recv_hook: bool = False,
    ):
        super().__init__()

        self.deepep_mode = deepep_mode

        common_kwargs = dict(
            group=group,
            router_topk=router_topk,
            permute_fusion=permute_fusion,
            num_experts=num_experts,
            num_local_experts=num_local_experts,
            hidden_size=hidden_size,
            params_dtype=params_dtype,
            deepep_mode=deepep_mode,
        )

        if self._is_v2:
            self._normal_dispatcher = _DeepEPDispatcherImplV2(
                async_finish=async_finish,
                **common_kwargs,
            )
        else:
            if self.deepep_mode.enable_low_latency():
                self._low_latency_dispatcher = _DeepEPDispatcherImplLowLatency(
                    return_recv_hook=return_recv_hook,
                    **common_kwargs,
                )
            if self.deepep_mode.enable_normal():
                self._normal_dispatcher = _DeepEPDispatcherImplNormal(
                    async_finish=async_finish,
                    **common_kwargs,
                )

        self._stage = _Stage.INITIAL
        self._deepep_dispatch_hooks = DeepEPPDispatchHooks()

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ) -> DispatchOutput:
        self.dispatch_a(hidden_states, topk_output)
        if self._deepep_dispatch_hooks is not None:
            self._deepep_dispatch_hooks(self)
        ret = self.dispatch_b()
        return ret

    def dispatch_a(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ):
        self._update_stage(_Stage.INITIAL, _Stage.AFTER_DISPATCH_A)
        inner_state = self._get_impl().dispatch_a(
            hidden_states=hidden_states,
            topk_output=topk_output,
        )
        self._dispatch_intermediate_state = inner_state

    def dispatch_b(self):
        self._update_stage(_Stage.AFTER_DISPATCH_A, _Stage.AFTER_DISPATCH_B)
        inner_state = self._dispatch_intermediate_state
        del self._dispatch_intermediate_state
        return self._get_impl().dispatch_b(*inner_state)

    def combine(
        self,
        combine_input: CombineInput,
    ) -> torch.Tensor:
        self.combine_a(combine_input)
        ret = self.combine_b()
        return ret

    def combine_a(
        self,
        combine_input: CombineInput,
    ):
        hidden_states, topk_ids, topk_weights = combine_input
        self._update_stage(_Stage.AFTER_DISPATCH_B, _Stage.AFTER_COMBINE_A)
        inner_state = self._get_impl().combine_a(
            hidden_states=hidden_states,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        )
        self._combine_intermediate_state = inner_state

    def combine_b(self):
        self._update_stage(_Stage.AFTER_COMBINE_A, _Stage.INITIAL)
        inner_state = self._combine_intermediate_state
        del self._combine_intermediate_state
        return self._get_impl().combine_b(*inner_state)

    def _get_impl(self) -> _DeepEPDispatcherImplBase:
        if self._is_v2:
            return self._normal_dispatcher
        is_extend_in_batch = get_is_extend_in_batch()
        resolved_deepep_mode = self.deepep_mode.resolve(is_extend_in_batch)
        if resolved_deepep_mode == DeepEPMode.NORMAL:
            return self._normal_dispatcher
        elif resolved_deepep_mode == DeepEPMode.LOW_LATENCY:
            return self._low_latency_dispatcher
        else:
            raise ValueError(f"Invalid deepep_mode: {self.deepep_mode}")

    def _update_stage(self, old_stage, new_stage):
        assert self._stage == old_stage
        self._stage = new_stage

    def set_quant_config(self, quant_config: dict):
        super().set_quant_config(quant_config)
        if self._is_v2:
            self._normal_dispatcher.set_quant_config(quant_config)
            return
        if self.deepep_mode.enable_low_latency():
            self._low_latency_dispatcher.set_quant_config(quant_config)
        if self.deepep_mode.enable_normal():
            self._normal_dispatcher.set_quant_config(quant_config)

    def set_overlap_args(
        self, combine_overlap_args: CombineOverlapArgs, meta_overlap_args: dict
    ):
        super().set_overlap_args(combine_overlap_args, meta_overlap_args)
        if self._is_v2:
            self._normal_dispatcher.set_overlap_args(
                combine_overlap_args, meta_overlap_args
            )
            return
        if self.deepep_mode.enable_low_latency():
            self._low_latency_dispatcher.set_overlap_args(
                combine_overlap_args, meta_overlap_args
            )
        if self.deepep_mode.enable_normal():
            self._normal_dispatcher.set_overlap_args(
                combine_overlap_args, meta_overlap_args
            )

    def clear_overlap_args(self):
        super().clear_overlap_args()
        if self._is_v2:
            self._normal_dispatcher.clear_overlap_args()
            return
        if self.deepep_mode.enable_low_latency():
            self._low_latency_dispatcher.clear_overlap_args()
        if self.deepep_mode.enable_normal():
            self._normal_dispatcher.clear_overlap_args()

    def register_deepep_dispatch_hook(self, hook):
        return self._deepep_dispatch_hooks.register_hook(hook)
