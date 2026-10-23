"""Hermes 0.19 compatibility contracts for the OneBot plugin.

Covers the base-class surfaces the plugin must match:

- ``is_connected`` is backed by ``_running``; only ``_mark_connected()`` /
  ``_mark_disconnected()`` update it (gateway/platforms/base.py:2789-2800,
  2883-2886).
- ``send_image_file(chat_id, image_path, caption, reply_to, metadata, **kwargs)``
  is the native local-image entry point (base.py:3584).
- ``send(chat_id, content, reply_to, metadata) -> SendResult`` is the only text
  entry point (base.py:3008).

These tests run without Hermes too: the plugin module degrades to stand-ins and
the fallback connection-state path is plain Python.
"""
from __future__ import annotations

import inspect
import os
import sys
from types import SimpleNamespace

import pytest

_HERMES_AGENT_DIR = os.environ.get("HERMES_AGENT_DIR", "/home/alec/.hermes/hermes-agent")
if _HERMES_AGENT_DIR not in sys.path:
    sys.path.insert(0, _HERMES_AGENT_DIR)

from onebot_adapter.hermes_plugin import adapter as plugin_mod  # noqa: E402

try:
    from gateway.platforms.base import BasePlatformAdapter
    _HERMES_OK = True
except Exception:
    BasePlatformAdapter = None  # type: ignore[assignment]
    _HERMES_OK = False

requires_hermes = pytest.mark.skipif(not _HERMES_OK, reason="Hermes base not available")


def test_connection_state_falls_back_to_running_without_markers():
    """Older hosts / test doubles lack the marker helpers; ``_running`` still updates."""
    dummy = SimpleNamespace()
    plugin_mod.OneBotAdapter._set_connection_state(dummy, True)
    assert dummy._is_connected is True
    assert dummy._running is True

    plugin_mod.OneBotAdapter._set_connection_state(dummy, False)
    assert dummy._is_connected is False
    assert dummy._running is False


def test_connection_state_prefers_marker_helpers():
    """When the 0.19 markers exist, delegate to them instead of touching _running."""
    dummy = SimpleNamespace()
    calls: list[str] = []

    def _mark_connected() -> None:
        calls.append("connected")
        dummy._running = True

    def _mark_disconnected() -> None:
        calls.append("disconnected")
        dummy._running = False

    dummy._mark_connected = _mark_connected
    dummy._mark_disconnected = _mark_disconnected

    plugin_mod.OneBotAdapter._set_connection_state(dummy, True)
    assert calls == ["connected"]
    assert dummy._running is True

    plugin_mod.OneBotAdapter._set_connection_state(dummy, False)
    assert calls == ["connected", "disconnected"]
    assert dummy._running is False


def test_send_image_file_signature_matches_019_shape():
    params = inspect.signature(plugin_mod.OneBotAdapter.send_image_file).parameters
    assert list(params)[:6] == ["self", "chat_id", "image_path", "caption", "reply_to", "metadata"]
    assert params["caption"].default is None
    assert params["reply_to"].default is None
    assert params["metadata"].default is None
    assert any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


@requires_hermes
def test_send_image_file_signature_matches_base():
    base_params = inspect.signature(BasePlatformAdapter.send_image_file).parameters
    our_params = inspect.signature(plugin_mod.OneBotAdapter.send_image_file).parameters
    assert list(base_params) == list(our_params)


@requires_hermes
def test_send_signature_matches_019_abstract_method():
    base_params = inspect.signature(BasePlatformAdapter.send).parameters
    our_params = inspect.signature(plugin_mod.OneBotAdapter.send).parameters
    assert list(base_params) == list(our_params)
