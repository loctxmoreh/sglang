"""HookRegistry examples: AROUND, BEFORE, and class REPLACE.

The hooks here are deliberately log-only — they prove the wiring works
without changing behaviour. Swap the function bodies for real logic.
"""

from __future__ import annotations

import time

from moreh_plugin import logger


def install() -> None:
    """Entry point: register all hooks in this plugin.

    Note: we import HookRegistry / HookType *inside* the function so that
    merely importing this module (e.g. from tests) does not pull sglang
    in eagerly.
    """
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    logger.info("[moreh-plugin] installing hooks: AROUND / BEFORE / REPLACE")

    # ------------------------------------------------------------------
    # Example 1: AROUND hook
    #
    # Wrap Scheduler.event_loop_normal so we can time it / inject logging
    # without modifying sglang. The signature for AROUND is:
    #   fn(original_fn, *args, **kwargs) -> result
    # ------------------------------------------------------------------
    HookRegistry.register(
        "sglang.srt.managers.scheduler.Scheduler.event_loop_normal",
        _around_scheduler_event_loop,
        HookType.AROUND,
    )

    # ------------------------------------------------------------------
    # Example 2: BEFORE hook
    #
    # Mutate ServerArgs after __post_init__ runs. BEFORE hooks may return
    #   (args, kwargs)  -> to replace the call args, or
    #   None            -> to leave them alone (this example just logs).
    # ------------------------------------------------------------------
    HookRegistry.register(
        "sglang.srt.server_args.ServerArgs.__post_init__",
        _before_server_args_post_init,
        HookType.BEFORE,
    )

    # ------------------------------------------------------------------
    # Example 3: class REPLACE
    #
    # Swap sglang's RMSNorm class for a logging subclass. Because the
    # parent attribute is a class (not a method), only HookType.REPLACE is
    # legal here; AROUND/BEFORE/AFTER on a class target would raise.
    #
    # We build the subclass lazily inside _replacement_rmsnorm() so the
    # import only happens when SGLang actually applies the hook.
    # ------------------------------------------------------------------
    HookRegistry.register(
        "sglang.srt.layers.layernorm.RMSNorm",
        _replacement_rmsnorm(),
        HookType.REPLACE,
    )


# ----------------------------------------------------------------------
# Hook bodies
# ----------------------------------------------------------------------

def _around_scheduler_event_loop(original_fn, *args, **kwargs):
    """AROUND: time the scheduler loop and log start/stop."""
    logger.info("[moreh-plugin] AROUND Scheduler.event_loop_normal: entering")
    t0 = time.perf_counter()
    try:
        return original_fn(*args, **kwargs)
    finally:
        dt = time.perf_counter() - t0
        logger.info(
            "[moreh-plugin] AROUND Scheduler.event_loop_normal: exited after %.2fs",
            dt,
        )


def _before_server_args_post_init(*args, **kwargs):
    """BEFORE: peek at ServerArgs as they get finalised.

    Returning None means "leave args/kwargs alone". A real plugin could
    return ``(new_args, new_kwargs)`` to substitute them, e.g. to force
    ``attention_backend='aiter'`` whenever this plugin is active.
    """
    # args[0] is `self` (the ServerArgs instance) because this hooks an
    # instance method (__post_init__).
    server_args = args[0] if args else None
    backend = getattr(server_args, "attention_backend", None)
    logger.info(
        "[moreh-plugin] BEFORE ServerArgs.__post_init__: attention_backend=%r",
        backend,
    )
    return None


def _replacement_rmsnorm():
    """Build a subclass of RMSNorm that just logs on init.

    Wrapped in a function so the sglang import is deferred to hook-apply
    time — the module that contains RMSNorm pulls in torch / sgl_kernel.
    """
    from sglang.srt.layers.layernorm import RMSNorm as _OrigRMSNorm

    class MorehRMSNorm(_OrigRMSNorm):
        _logged_once = False

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if not MorehRMSNorm._logged_once:
                logger.info(
                    "[moreh-plugin] REPLACE RMSNorm: first instance constructed "
                    "(hidden_size=%s) ; Replacement is live",
                    kwargs.get("hidden_size") or (args[0] if args else "?"),
                )
                MorehRMSNorm._logged_once = True

    return MorehRMSNorm
