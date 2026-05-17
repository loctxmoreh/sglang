"""Smoke test: install this package (`pip install -e ./moreh_plugin`),
then run `pytest moreh_plugin/tests/`.

We do NOT spin up a real engine here — instead we just call SGLang's
plugin loader directly and verify that:
  1. our entry points are discovered, and
  2. the typed registries (attention backend, sampler backend, chat
     template) end up populated.

The hook-application step is also exercised, which will fail-fast if any
of our hook targets are misspelled.
"""

from __future__ import annotations

import importlib.metadata as md

import pytest


def _eps(group: str) -> dict[str, str]:
    return {ep.name: ep.value for ep in md.entry_points(group=group)}


def test_entry_points_declared():
    platforms = _eps("sglang.srt.platforms")
    plugins = _eps("sglang.srt.plugins")
    assert platforms.get("moreh_platform") == "moreh_plugin.platform:activate"
    assert "moreh_hooks" in plugins
    assert "moreh_backends" in plugins
    assert "moreh_templates" in plugins


def test_general_plugins_execute():
    """Drive sglang's plugin loader and verify side effects."""
    pytest.importorskip("sglang")

    # Reset any prior state so the test is idempotent across reruns.
    from sglang.srt.plugins import load_plugins
    from sglang.srt.plugins.hook_registry import HookRegistry

    HookRegistry.reset()
    import sglang.srt.plugins as plugins_mod
    plugins_mod._plugins_loaded = False

    load_plugins()

    # Attention backend registry
    from sglang.srt.layers.attention.attention_registry import ATTENTION_BACKENDS
    assert "moreh" in ATTENTION_BACKENDS

    # Sampler backend registry
    from sglang.srt.layers.sampler import _CUSTOM_SAMPLER_FACTORIES
    assert "moreh" in _CUSTOM_SAMPLER_FACTORIES

    # Chat template registry
    from sglang.srt.parser.conversation import chat_templates
    assert "moreh-chat" in chat_templates


def test_hooks_applied():
    """Verify HookRegistry actually patched the targets we declared."""
    pytest.importorskip("sglang")

    from sglang.srt.plugins import load_plugins
    from sglang.srt.plugins.hook_registry import HookRegistry

    HookRegistry.reset()
    import sglang.srt.plugins as plugins_mod
    plugins_mod._plugins_loaded = False
    load_plugins()

    patched = HookRegistry._patched
    assert "sglang.srt.managers.scheduler.Scheduler.event_loop_normal" in patched
    assert "sglang.srt.server_args.ServerArgs.__post_init__" in patched
    assert "sglang.srt.layers.layernorm.RMSNorm" in patched

    # And RMSNorm should now be our MorehRMSNorm subclass.
    from sglang.srt.layers.layernorm import RMSNorm
    assert RMSNorm.__name__ == "MorehRMSNorm"
