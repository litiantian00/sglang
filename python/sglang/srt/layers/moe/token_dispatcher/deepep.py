from __future__ import annotations

import logging
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

from enum import Enum, IntEnum, auto

import torch
import torch.distributed as dist

_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and is_hip()

logger = logging.getLogger(__name__)


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

        logger.info(
            "SGLANG_DEEPEP_USE_V2=1: constructing deep_ep.ElasticBuffer "
            "(num_max_tokens_per_rank=%d, hidden=%d, num_topk=%d, use_fp8_dispatch=%s).",
            num_max_dispatch_tokens_per_rank,
            hidden_size,
            num_topk,
            use_fp8,
        )
        return ElasticBuffer(
            group=group,
            num_max_tokens_per_rank=num_max_dispatch_tokens_per_rank,
            hidden=hidden_size,
            num_topk=num_topk,
            use_fp8_dispatch=use_fp8,
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

def _align_up_val(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _psum_to_starts_counts(psum: torch.Tensor, alignment: int):
    psum_i64 = psum.to(torch.int64)
    psum_shifted = torch.zeros_like(psum_i64)
    psum_shifted[1:] = psum_i64[:-1]
    starts = (psum_shifted + alignment - 1) // alignment * alignment
    counts = psum_i64 - starts
    return starts, counts


def _convert_v2_expand_to_v1(
    recv_x: torch.Tensor,
    recv_sf: Optional[torch.Tensor],
    handle,
    num_max_dispatch_tokens_per_rank: int,
    num_ranks: int,
    num_experts: int,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, List[int], List[int]]:
    E_local = num_experts // num_ranks
    T_max = num_max_dispatch_tokens_per_rank
    R = num_ranks
    H = recv_x.shape[-1]
    alignment = handle.expert_alignment

    counts_list = handle.num_recv_tokens_per_expert_list
    starts_list = []
    offset = 0
    for e in range(E_local):
        starts_list.append(offset)
        offset += counts_list[e]
        offset = _align_up_val(offset, alignment)

    counts_tensor = torch.tensor(counts_list, dtype=torch.int32, device=recv_x.device)

    packed_x = recv_x.new_empty((E_local, R * T_max, H))
    for e in range(E_local):
        c = counts_list[e]
        s = starts_list[e]
        if c > 0:
            packed_x[e, :c, :].copy_(recv_x[s:s + c, :], non_blocking=True)
        if c < R * T_max:
            packed_x[e, c:, :].zero_()

    packed_sf = None
    if recv_sf is not None:
        num_sf = recv_sf.shape[1]
        packed_sf_raw = recv_sf.new_zeros((E_local, num_sf, R * T_max))
        for e in range(E_local):
            c = counts_list[e]
            s = starts_list[e]
            if c > 0:
                packed_sf_raw[e, :, :c] = recv_sf[s:s + c, :].T
        packed_sf = packed_sf_raw.permute(0, 2, 1)

    return packed_x, packed_sf, counts_tensor, starts_list, counts_list


def _convert_v2_nonexpand_to_v1(
    recv_x: torch.Tensor,
    recv_sf: Optional[torch.Tensor],
    recv_topk_idx: torch.Tensor,
    handle,
    num_max_dispatch_tokens_per_rank: int,
    num_ranks: int,
    num_experts: int,
    local_rank: int,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    E_local = num_experts // num_ranks
    T_max = num_max_dispatch_tokens_per_rank
    R = num_ranks
    H = recv_x.shape[-1]
    num_topk = recv_topk_idx.shape[1]
    num_recv_tokens = handle.num_recv_tokens
    local_expert_start = local_rank * E_local

    valid_topk = recv_topk_idx[:num_recv_tokens]
    token_indices = torch.arange(num_recv_tokens, device=recv_x.device, dtype=torch.int64).unsqueeze(1).expand(-1, num_topk).reshape(-1)
    expert_global = valid_topk.reshape(-1)

    valid_mask = (
        (expert_global >= local_expert_start) &
        (expert_global < local_expert_start + E_local) &
        (expert_global >= 0)
    )
    valid_token_idx = token_indices[valid_mask]
    valid_expert_local = (expert_global[valid_mask] - local_expert_start).to(torch.int32)

    sort_key = valid_expert_local.to(torch.int64) * num_recv_tokens + valid_token_idx
    sorted_indices = torch.argsort(sort_key)
    valid_token_idx = valid_token_idx[sorted_indices]
    valid_expert_local = valid_expert_local[sorted_indices]

    expert_counts = torch.bincount(valid_expert_local, minlength=E_local).to(torch.int32)
    expert_offsets = torch.zeros(E_local + 1, dtype=torch.int64, device=recv_x.device)
    expert_offsets[1:] = torch.cumsum(expert_counts.to(torch.int64), dim=0)
    global_prefix = torch.arange(len(valid_token_idx), dtype=torch.int64, device=recv_x.device)
    slot_in_expert = global_prefix - expert_offsets[valid_expert_local.long()]

    packed_x = torch.zeros((E_local, R * T_max, H), dtype=recv_x.dtype, device=recv_x.device)
    packed_x[valid_expert_local.long(), slot_in_expert.long()] = recv_x[valid_token_idx.long()]

    packed_sf = None
    if recv_sf is not None:
        num_sf = recv_sf.shape[1]
        packed_sf_raw = torch.zeros((E_local, num_sf, R * T_max), dtype=recv_sf.dtype, device=recv_sf.device)
        expanded_sf = recv_sf[valid_token_idx.long()]
        for sf_ch in range(num_sf):
            packed_sf_raw[valid_expert_local.long(), sf_ch, slot_in_expert.long()] = expanded_sf[:, sf_ch]
        packed_sf = packed_sf_raw.permute(0, 2, 1)

    return packed_x, packed_sf, expert_counts, valid_expert_local.long(), slot_in_expert.long(), valid_token_idx.long()


def _convert_v2_dispatch_to_v1_ll_format(
    recv_x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    recv_topk_ids: Optional[torch.Tensor],
    handle,
    original_topk_ids: torch.Tensor,
    original_topk_weights: torch.Tensor,
    num_max_dispatch_tokens_per_rank: int,
    num_ranks: int,
    num_experts: int,
    local_rank: int,
    dispatch_ctx: dict,
) -> "DeepEPLLDispatchOutput":
    if isinstance(recv_x, tuple):
        recv_data, recv_sf = recv_x
    else:
        recv_data, recv_sf = recv_x, None

    cached_buffers = dispatch_ctx.get('_cached_buffers')
    if cached_buffers is None:
        cached_buffers = {}
        dispatch_ctx['_cached_buffers'] = cached_buffers

    if handle.do_expand:
        packed_x, packed_sf, masked_m, starts_list, counts_list = _convert_v2_expand_to_v1(
            recv_data, recv_sf, handle,
            num_max_dispatch_tokens_per_rank, num_ranks, num_experts,
        )
        dispatch_ctx['starts_list'] = starts_list
        dispatch_ctx['counts_list'] = counts_list
        dispatch_ctx['recv_x_shape'] = recv_data.shape
    else:
        packed_x, packed_sf, masked_m, valid_expert_local, slot_in_expert, valid_token_idx = _convert_v2_nonexpand_to_v1(
            recv_data, recv_sf, recv_topk_ids, handle,
            num_max_dispatch_tokens_per_rank, num_ranks, num_experts, local_rank,
        )
        dispatch_ctx['recv_x_shape'] = recv_data.shape
        dispatch_ctx['valid_expert_local'] = valid_expert_local
        dispatch_ctx['slot_in_expert'] = slot_in_expert
        dispatch_ctx['valid_token_idx'] = valid_token_idx

    num_tokens = original_topk_ids.shape[0]
    num_topk = original_topk_ids.shape[1]
    expected_m = (num_tokens * num_ranks * num_topk + num_experts) // num_experts

    return DeepEPLLDispatchOutput(
        hidden_states=packed_x,
        hidden_states_scale=packed_sf,
        topk_ids=original_topk_ids,
        topk_weights=original_topk_weights,
        masked_m=masked_m,
        expected_m=expected_m,
    )


def _reverse_scatter_3d_to_v2_flat(
    hidden_states_3d: torch.Tensor,
    dispatch_ctx: dict,
    handle,
) -> torch.Tensor:
    if handle.do_expand:
        starts_list = dispatch_ctx['starts_list']
        counts_list = dispatch_ctx['counts_list']
        recv_x_shape = dispatch_ctx['recv_x_shape']
        E_local = hidden_states_3d.shape[0]
        H = hidden_states_3d.shape[2]

        flat = hidden_states_3d.new_zeros((recv_x_shape[0], H))
        for e in range(E_local):
            c = counts_list[e]
            s = starts_list[e]
            if c > 0:
                flat[s:s + c, :].copy_(hidden_states_3d[e, :c, :], non_blocking=True)
        return flat
    else:
        recv_x_shape = dispatch_ctx['recv_x_shape']
        valid_expert_local = dispatch_ctx['valid_expert_local']
        slot_in_expert = dispatch_ctx['slot_in_expert']
        valid_token_idx = dispatch_ctx['valid_token_idx']
        H = hidden_states_3d.shape[2]
        num_recv_tokens = recv_x_shape[0]

        gathered = hidden_states_3d[valid_expert_local, slot_in_expert]
        flat = torch.zeros(num_recv_tokens, H, dtype=hidden_states_3d.dtype, device=hidden_states_3d.device)
        flat.scatter_add_(0, valid_token_idx.unsqueeze(1).expand(-1, H), gathered)
        return flat


class _DeepEPDispatcherImplV2(_DeepEPDispatcherImplBase):
    """DeepEP V2 (ElasticBuffer) dispatcher implementation.
    """

    def __init__(self, async_finish: bool, **kwargs):
        super().__init__(**kwargs)
        self.async_finish = async_finish
        self.quant_config = {}
        self.num_comm_sms = 0
        # do_expand controls whether ElasticBuffer performs C++ side expert-sort
        # (expand) during dispatch. Controlled by env SGLANG_DEEPEP_V2_DO_EXPAND.
        self.do_expand = get_bool_env_var("SGLANG_DEEPEP_V2_DO_EXPAND", default="true")
        self._dispatch_ctx = {}

    def dispatch_a(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
    ):
        """Phase 1 of dispatch: quantize hidden_states and deduplicate topk_ids.

        Deduplication: if a token is routed to the same expert multiple times
        (duplicate entries in topk_ids), only the first occurrence is kept;
        later duplicates are masked to -1 with weight 0.
        """
        topk_weights, topk_ids = topk_output.topk_weights, topk_output.topk_ids
        topk_ids = topk_ids.to(torch.int64)

        num_topk = topk_ids.shape[1]
        if num_topk > 1:
            pair_eq = (topk_ids.unsqueeze(2) == topk_ids.unsqueeze(1))
            is_later_dup = pair_eq.triu(diagonal=1).any(dim=2)
            topk_ids = topk_ids.masked_fill(is_later_dup, -1)
            topk_weights = topk_weights.masked_fill(is_later_dup, 0.0)

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

        previous_event = ElasticBuffer.capture() if self.async_finish else None
        return hidden_states, topk_ids, topk_weights, previous_event

    def dispatch_b(self, hidden_states, topk_ids, topk_weights, previous_event):
        """Phase 2 of dispatch: all-to-all communication and output formatting.

        Three possible return paths:
        - is_capturing=True: placeholder DeepEPLLDispatchOutput for CUDA Graph capture
        - use_expand_now=True: DeepEPV2ExpandDispatchOutput for prefill (zero-copy, contiguous GEMM)
        - otherwise: DeepEPLLDispatchOutput via format conversion (decode warmup, masked GEMM)
        """
        original_topk_ids = topk_ids
        original_topk_weights = topk_weights
        is_capturing = torch.cuda.is_current_stream_capturing()
        is_extend = get_is_extend_in_batch()
        # Prefill/extend uses the optimized V2 expand path: do_expand=True in buffer.dispatch
        # produces expert-sorted data directly, avoiding Python-side format conversion.
        # Decode uses do_expand=False to stay compatible with masked GEMM / CUDA Graph.
        use_expand_now = self.do_expand and is_extend and not is_capturing

        (
            recv_x,
            recv_topk_ids,
            recv_topk_weights,
            event,
        ) = self._dispatch_core(hidden_states, topk_ids, topk_weights, previous_event,
                                is_capturing=is_capturing,
                                use_expand=use_expand_now)
        event.current_stream_wait() if self.async_finish else ()

        if isinstance(recv_x, tuple):
            recv_hidden, recv_sf = recv_x
        else:
            recv_hidden, recv_sf = recv_x, None

        if is_capturing:
            # CUDA Graph capture path: construct a placeholder DeepEPLLDispatchOutput
            # with masked_m=0 so the masked GEMM kernel is recorded but does not
            # compute real tokens. Replay will update masked_m with actual values.
            E_local = self.num_local_experts
            R = self.group.size()
            T_max = self.num_max_dispatch_tokens_per_rank
            N = recv_hidden.shape[0]
            H = recv_hidden.shape[-1]
            M = max(N // E_local, 1) if E_local > 0 else R * T_max
            packed_x = recv_hidden[:E_local * M].view(E_local, M, H)
            packed_sf = recv_sf[:E_local * M].view(E_local, M, recv_sf.shape[-1]) if recv_sf is not None else None
            masked_m = torch.zeros(E_local, dtype=torch.int32, device=recv_hidden.device)
            num_topk = original_topk_ids.shape[1]
            expected_m = (original_topk_ids.shape[0] * R * num_topk + self.num_experts) // self.num_experts
            self._dispatch_ctx = {'capture_mode': True}
            return DeepEPLLDispatchOutput(
                hidden_states=packed_x,
                hidden_states_scale=packed_sf,
                topk_ids=original_topk_ids,
                topk_weights=original_topk_weights,
                masked_m=masked_m,
                expected_m=expected_m,
            )

        if use_expand_now:
            # Prefill/extend optimized path: V2 C++ expand kernel already produced
            # expert-sorted flat data. Return DeepEPV2ExpandDispatchOutput which
            # downstream feeds into psum_to_m_indices (1 triton kernel) → contiguous
            # GEMM. No Python format conversion, no data copy, no padding waste.
            self._dispatch_ctx = {}
            return DeepEPV2ExpandDispatchOutput(
                hidden_states=recv_hidden,
                hidden_states_scale=recv_sf,
                psum_num_recv_tokens_per_expert=self.handle.psum_num_recv_tokens_per_expert,
                num_recv_tokens_per_expert_list=self.handle.num_recv_tokens_per_expert_list,
            )

        # Decode non-capture path (warmup or eager without CUDA Graph):
        # Convert V2 dispatch output to V1 LL format [E, R*T_max, H] + masked_m,
        # so the same masked GEMM kernel shape is used as in CUDA Graph capture.
        self._dispatch_ctx = {}
        return _convert_v2_dispatch_to_v1_ll_format(
            recv_x=recv_x,
            recv_topk_ids=recv_topk_ids,
            handle=self.handle,
            original_topk_ids=original_topk_ids,
            original_topk_weights=original_topk_weights,
            num_max_dispatch_tokens_per_rank=self.num_max_dispatch_tokens_per_rank,
            num_ranks=self.group.size(),
            num_experts=self.num_experts,
            local_rank=self.group.rank(),
            dispatch_ctx=self._dispatch_ctx,
        )

    def _dispatch_core(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        previous_event,
        is_capturing: bool = False,
        use_expand: bool = False,
    ):
        """Execute the ElasticBuffer all-to-all dispatch communication.

        Args:
            use_expand: if True, V2 C++ kernel sorts tokens by expert in-place
                (zero-copy expand). The resulting recv_x is already expert-sorted.
            is_capturing: if True, we are inside CUDA Graph capture_begin/end.

        do_expand and do_cpu_sync logic:
            - use_expand=True (prefill): do_expand=True, do_cpu_sync=False
              (GPU psum tensor is sufficient; no CPU sync needed)
            - use_expand=False, is_capturing=True (capture): do_expand=False, do_cpu_sync=False
              (placeholder path, no real data needed on CPU)
            - use_expand=False, is_capturing=False (decode warmup): do_expand=False, do_cpu_sync=True
              (_convert_v2_nonexpand_to_v1 needs handle.num_recv_tokens from CPU sync)
        """
        buffer = self._get_buffer()
        _deepep_precompile_tp_barrier()

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
            expert_alignment=128 if deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM else 1,
            num_sms=self.num_comm_sms,
            previous_event=previous_event,
            async_with_compute_stream=self.async_finish,
            allocate_on_comm_stream=(previous_event is not None) and self.async_finish,
            do_expand=use_expand,
            do_cpu_sync=not is_capturing and not use_expand,
            use_tma_aligned_col_major_sf=(
                deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
                and deep_gemm_wrapper.DEEPGEMM_BLACKWELL
            ),
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
        """Phase 1 of combine: prepare GEMM output for V2 combine communication.

        Three cases based on output dimensionality:
        - 2D tensor: V2 expand path output, already flat expert-sorted → pass through
        - 3D tensor + capture_mode: CUDA Graph path [E, M, H] → simple reshape to 2D
        - 3D tensor + normal: decode warmup path [E, M, H] → reverse scatter to flat
        """
        if deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM or _use_aiter or _is_npu:
            output = hidden_states
        else:
            raise NotImplementedError()

        if output.dim() == 3:
            if getattr(self, '_dispatch_ctx', {}).get('capture_mode', False):
                # CUDA Graph capture/replay: packed [E, M, H] → flat [E*M, H]
                E, M, H = output.shape
                output = output.reshape(E * M, H)
            else:
                # Decode warmup: reverse the scatter done in _convert_v2_dispatch_to_v1_ll_format
                output = _reverse_scatter_3d_to_v2_flat(output, self._dispatch_ctx, self.handle)

        previous_event = ElasticBuffer.capture() if self.async_finish else None
        return output, previous_event

    def combine_b(self, output, previous_event):
        """Phase 2 of combine: execute V2 ElasticBuffer combine communication."""
        hidden_states, event = self._combine_core(output, previous_event)
        event.current_stream_wait() if self.async_finish else ()
        self.handle = None
        self._dispatch_ctx = {}
        return hidden_states

    def _combine_core(self, x: torch.Tensor, previous_event):
        buffer = self._get_buffer()
        _deepep_precompile_tp_barrier()

        combined_x, _, event = buffer.combine(
            x,
            self.handle,
            num_sms=self.num_comm_sms,
            previous_event=previous_event,
            async_with_compute_stream=self.async_finish,
            allocate_on_comm_stream=previous_event is not None,
        )
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
                # async_finish=async_finish,
                async_finish=False,
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
