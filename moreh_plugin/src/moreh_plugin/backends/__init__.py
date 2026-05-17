"""Typed-registry examples: custom attention backend + custom sampler.

These don't go through HookRegistry — instead they call the dedicated
registration functions sglang exposes for backends that are meant to be
*extensible by design*:

  - register_attention_backend("name") — selectable via --attention-backend
  - register_sampler_backend("name", factory) — selectable via --sampling-backend

After this plugin loads, you can run:

  python -m sglang.launch_server ... \
      --attention-backend moreh --sampling-backend moreh
"""

from __future__ import annotations

from moreh_plugin import logger


def install() -> None:
    """Entry point: register backends in the typed registries."""
    _register_attention_backend()
    _register_sampler_backend()


# ----------------------------------------------------------------------
# Attention backend
# ----------------------------------------------------------------------

def _register_attention_backend() -> None:
    from sglang.srt.layers.attention.attention_registry import (
        register_attention_backend,
    )

    @register_attention_backend("moreh")
    def _create_moreh_backend(runner):
        logger.info(
            "[moreh-plugin] attention backend 'moreh' created for "
            "model %s",
            getattr(runner, "model_config", None)
            and runner.model_config.model_path,
        )
        return _build_attn_backend_cls()(runner)


def _build_attn_backend_cls():
    """Build a dummy AttentionBackend subclass lazily.

    All methods are no-ops that return zero tensors. Plug a real ROCm-tuned
    attention kernel (AITER / FlashAttention-ROCm / Triton) in here.
    """
    import torch
    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend

    class MorehAttnBackend(AttentionBackend):
        def __init__(self, runner):
            super().__init__()
            self.runner = runner

        def init_forward_metadata(self, forward_batch):
            logger.debug("[moreh-plugin] moreh: init_forward_metadata")

        def forward_extend(self, q, k, v, layer, forward_batch, **kw):
            logger.debug("[moreh-plugin] moreh: forward_extend (no-op)")
            return torch.zeros_like(q)

        def forward_decode(self, q, k, v, layer, forward_batch, **kw):
            logger.debug("[moreh-plugin] moreh: forward_decode (no-op)")
            return torch.zeros_like(q)

    return MorehAttnBackend


# ----------------------------------------------------------------------
# Sampler backend
# ----------------------------------------------------------------------

def _register_sampler_backend() -> None:
    from sglang.srt.layers.sampler import Sampler, register_sampler_backend

    class MorehSampler(Sampler):
        """Trivial sampler that logs once, then defers to the base Sampler."""

        _logged_once = False

        def forward(self, *args, **kwargs):
            if not MorehSampler._logged_once:
                logger.info("[moreh-plugin] sampler backend 'moreh' active")
                MorehSampler._logged_once = True
            return super().forward(*args, **kwargs)

    register_sampler_backend("moreh", lambda: MorehSampler())
