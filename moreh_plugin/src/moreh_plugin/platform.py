"""Platform plugin: an AMD ROCm SRTPlatform.

SGLang does not ship an explicit ``ROCmPlatform`` subclass — its existing
ROCm support is realised through ``is_hip()`` branches inside the
generic CUDA-shaped code paths (``MultiPlatformOp.forward_cuda``, the
attention backends, the quantization layers, …).  This class plugs into
that existing surface by:

  * Declaring ``_enum = PlatformEnum.ROCM`` so every
    ``current_platform.is_rocm()`` / ``is_cuda_alike()`` check in the
    codebase resolves correctly.
  * Setting ``device_type = "cuda"`` because ROCm builds of PyTorch expose
    the ROCm runtime under the CUDA API surface (``torch.cuda.*``).
  * Returning ``"cuda"`` from ``get_dispatch_key_name()`` so
    ``MultiPlatformOp`` picks the existing ``forward_cuda`` methods (which
    already contain the ``is_hip()``-gated ROCm paths).
  * Implementing the ``[Active]`` ``DeviceMixin`` methods via
    ``torch.cuda.*`` — those calls transparently dispatch to HIP on a
    ROCm PyTorch build, so we get parity with the in-tree behaviour for
    free.

Override the demo hooks (``apply_server_args_defaults``,
``get_default_attention_backend``, capability flags) to taste — they are
the obvious customisation seams for a downstream ROCm flavour.
"""

from __future__ import annotations

from typing import Optional

import torch

from sglang.srt.platforms.device_mixin import PlatformEnum
from sglang.srt.platforms.interface import SRTPlatform

from moreh_plugin import logger


class MorehPlatform(SRTPlatform):
    """ROCm platform exposed as an OOT plugin.

    Inherits the generic SRTPlatform interface and reuses SGLang's existing
    ``is_hip()``-driven ROCm code paths by setting the platform enum and
    dispatch key correctly. Override the methods below to plug Moreh-
    specific kernels / KV pools / graph runners in.
    """

    # ------------------------------------------------------------------
    # Class-level identity (read by DeviceMixin queries)
    # ------------------------------------------------------------------
    _enum: PlatformEnum = PlatformEnum.ROCM
    device_name: str = "rocm"
    # ROCm PyTorch exposes HIP under the "cuda" torch.device type.
    device_type: str = "cuda"

    # Conservative capability set — replace with ROCm-tuned values.
    supported_quantization = ["fp8", "awq", "gptq"]

    # ------------------------------------------------------------------
    # Configuration lifecycle
    # ------------------------------------------------------------------

    def apply_server_args_defaults(self, server_args) -> None:
        logger.info("[moreh-plugin] MorehPlatform.apply_server_args_defaults()")
        # Example override: prefer AITER on ROCm if the user didn't pick one.
        # if getattr(server_args, "attention_backend", None) is None:
        #     server_args.attention_backend = "aiter"

    # ------------------------------------------------------------------
    # Subsystem factory methods
    # ------------------------------------------------------------------

    # def get_default_attention_backend(self) -> str:
    #     # AITER is AMD's tuned attention backend on ROCm.
    #     return "aiter"

    def get_dispatch_key_name(self) -> str:
        # Pick the existing forward_cuda paths (which already gate ROCm
        # specifics with is_hip()).  Returning "hip" would *miss* those.
        return "cuda"

    # ------------------------------------------------------------------
    # Capability flags
    # ------------------------------------------------------------------

    def supports_fp8(self) -> bool:
        # MI300-class hardware supports FP8; finer-grained checks
        # (e4m3fnuz vs e4m3fn) are handled by is_hip()-gated branches.
        return True

    def support_cuda_graph(self) -> bool:
        # HIP graphs are exposed via the torch CUDA-graph API on ROCm.
        return True

    def is_pin_memory_available(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # DeviceMixin [Active] methods — implemented via torch.cuda.*
    # which transparently dispatches to HIP on ROCm PyTorch builds.
    # ------------------------------------------------------------------

    def get_device_total_memory(self, device_id: int = 0) -> int:
        return torch.cuda.get_device_properties(device_id).total_memory

    def get_current_memory_usage(
        self, device: Optional["torch.device"] = None
    ) -> float:
        torch.cuda.reset_peak_memory_stats(device)
        return torch.cuda.max_memory_allocated(device)

    # ------------------------------------------------------------------
    # DeviceMixin [Planned] methods — wired up for parity even though
    # core sglang still uses hardcoded torch.cuda calls.  When the core
    # is migrated, these are what will actually fire on ROCm.
    # ------------------------------------------------------------------

    def get_device(self, local_rank: int) -> "torch.device":
        return torch.device(f"cuda:{local_rank}")

    def set_device(self, device: "torch.device") -> None:
        torch.cuda.set_device(device)

    def get_device_name(self, device_id: int = 0) -> str:
        return torch.cuda.get_device_name(device_id)

    def get_device_capability(self, device_id: int = 0):
        from sglang.srt.platforms.device_mixin import DeviceCapability

        major, minor = torch.cuda.get_device_capability(device_id)
        return DeviceCapability(major, minor)

    def empty_cache(self) -> None:
        torch.cuda.empty_cache()

    def synchronize(self) -> None:
        torch.cuda.synchronize()

    def get_available_memory(self, device_id: int = 0) -> tuple[int, int]:
        free, total = torch.cuda.mem_get_info(device_id)
        return free, total

    def get_torch_distributed_backend_str(self) -> str:
        # NCCL is built against RCCL on ROCm and is the right backend str.
        return "nccl"

    # ------------------------------------------------------------------
    # Subsystem factories left to inherit (NotImplementedError stubs).
    #
    # A production Moreh plugin would override these to return ROCm-tuned
    # classes (paged allocators, graph runners, KV pools).
    # ------------------------------------------------------------------


_PLATFORM_QUALNAME = "moreh_plugin.platform:MorehPlatform"


def activate() -> str | None:
    """Entry point: return the qualname so SGLang can resolve the class.

    Returns ``None`` if no ROCm-capable hardware is detected, which tells
    SGLang to skip this plugin (and fall back to whatever platform is
    actually present on this machine).
    """
    if torch.version.hip is None:
        logger.info(
            "[moreh-plugin] activate() skipped: torch was not built against HIP"
        )
        return None

    logger.info(
        "[moreh-plugin] activate() called — selecting %s", _PLATFORM_QUALNAME
    )
    return _PLATFORM_QUALNAME
