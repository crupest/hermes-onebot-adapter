"""Tests for the OneBotAdapter plugin WS client protocol handling.

These test the protocol logic without a real Hermes gateway or adapter service.
We mock the WS layer and verify frame construction, media buffering, and result
correlation.
"""
from __future__ import annotations

import asyncio
import json
import os

# Skip entire module if Hermes base isn't importable (standalone CI without Hermes)
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

_HERMES_AGENT_DIR = os.environ.get("HERMES_AGENT_DIR", "/home/alec/.hermes/hermes-agent")
sys.path.insert(0, _HERMES_AGENT_DIR)

try:
    from gateway.config import Platform  # noqa: F401
    from gateway.platforms.base import BasePlatformAdapter, MessageType, SendResult  # noqa: F401
    _HERMES_OK = True
except Exception:
    _HERMES_OK = False

pytestmark = pytest.mark.skipif(not _HERMES_OK, reason="Hermes base not available")

if _HERMES_OK:
    from onebot_adapter.hermes_plugin.adapter import OneBotAdapter

    # Try to register the "onebot" platform so Platform("onebot") works.
    # In a real gateway this happens during plugin load; in tests we may be
    # running without the full registry, so guard against failure.
    try:
        from unittest.mock import MagicMock as _MagicMock

        from onebot_adapter.hermes_plugin.adapter import register as _ob_register
        _ctx = _MagicMock()
        _ob_register(_ctx)
    except Exception:
        pass


class FakeWS:
    """Minimal WS mock that records sent frames and can trigger result futures."""

    def __init__(self):
        self.closed = False
        self.sent: list[dict | bytes] = []
        self._handler = None

    async def send_json(self, msg: dict):
        self.sent.append(msg)
        # If this is a send/api_call, auto-resolve with a success result
        if msg.get("type") in ("send", "api_call"):
            req_id = msg.get("req_id", "")
            # Schedule the result on the next loop iteration
            def _resolve():
                fut = self._adapter._futures.pop(req_id, None)
                if fut and not fut.done():
                    fut.set_result({"success": True, "message_id": "mock_mid", "data": {"ok": True}})
            loop = asyncio.get_event_loop()
            loop.call_soon(_resolve)

    async def close(self):
        self.closed = True


def _make_adapter():
    config = MagicMock()
    config.extra = {"adapter_url": "ws://127.0.0.1:18810/hermes", "adapter_token": "tok"}
    adapter = OneBotAdapter.__new__(OneBotAdapter)
    # Manual init without super().__init__ (which needs Platform)
    adapter._adapter_url = "ws://127.0.0.1:18810/hermes"
    adapter._adapter_token = "tok"
    adapter._session = None
    adapter._ws = None
    adapter._recv_task = None
    adapter._futures = {}
    adapter._send_semaphore = asyncio.Semaphore(2)  # match _MAX_INFLIGHT_SENDS
    adapter._push_tasks = set()
    adapter._onebot_connected = False
    adapter._self_id = ""
    adapter._file_upload_timeout = 600.0
    adapter._is_connected = True
    adapter.MAX_MESSAGE_LENGTH = 4500
    # Platform("onebot") requires the platform to be registered; use a mock
    # fallback if registration didn't succeed (tests don't exercise platform enum).
    try:
        adapter.platform = Platform("onebot")
    except (ValueError, TypeError):
        adapter.platform = MagicMock()
    return adapter


async def test_send_text_constructs_frame():
    adapter = _make_adapter()
    fake_ws = FakeWS()
    fake_ws._adapter = adapter
    adapter._ws = fake_ws
    result = await adapter._request("send_text", chat_id="100", content="hello")
    assert result["success"] is True
    assert result["message_id"] == "mock_mid"
    sent = fake_ws.sent[0]
    assert sent["type"] == "send"
    assert sent["action"] == "send_text"
    assert sent["chat_id"] == "100"
    assert sent["content"] == "hello"


async def test_send_image_via_path():
    """send_image with a local path sends a single JSON send frame."""
    adapter = _make_adapter()
    fake_ws = FakeWS()
    fake_ws._adapter = adapter
    adapter._ws = fake_ws
    result = await adapter.send_image("group:42", "/tmp/img.jpg")
    assert result.success is True
    sent = fake_ws.sent[0]
    assert sent["type"] == "send"
    assert sent["action"] == "send_image"
    assert sent["image_url"] == "/tmp/img.jpg"


async def test_api_call_constructs_frame():
    adapter = _make_adapter()
    fake_ws = FakeWS()
    fake_ws._adapter = adapter
    adapter._ws = fake_ws
    result = await adapter._api_call("get_group_info", {"group_id": 42})
    assert result["success"] is True
    sent = fake_ws.sent[0]
    assert sent["type"] == "api_call"
    assert sent["action"] == "get_group_info"
    assert sent["params"] == {"group_id": 42}


async def test_rpc_send_failure_cleans_pending_future():
    adapter = _make_adapter()
    fake_ws = FakeWS()
    fake_ws._adapter = adapter
    fake_ws.send_json = AsyncMock(side_effect=ConnectionError("write failed"))
    adapter._ws = fake_ws

    with pytest.raises(ConnectionError, match="write failed"):
        await adapter._api_call("get_group_info", {"group_id": 42})
    assert adapter._futures == {}


def test_upload_rpcs_use_long_result_timeout():
    from onebot_adapter.hermes_plugin.adapter import _result_timeout

    assert _result_timeout("api_call", "upload_group_file") == 640.0
    assert _result_timeout("api_call", "upload_group_file", 480.0) == 520.0
    assert _result_timeout("api_call", "upload_private_file") == 630.0
    assert _result_timeout("api_call", "upload_private_file", 480.0) == 510.0
    assert _result_timeout("api_call", "create_flash_task") == 630.0
    assert _result_timeout("api_call", "create_flash_task", 480.0) == 510.0
    assert _result_timeout("send", "send_document") == 640.0
    assert _result_timeout("api_call", "get_group_info") == 30.0
    assert _result_timeout("send", "send_text") == 30.0


def test_result_retryable_override_is_preserved():
    from onebot_adapter.hermes_plugin.adapter import _result_to_send_result

    result = _result_to_send_result({
        "success": False,
        "error": "upload outcome unknown",
        "retryable": False,
    })
    assert result.success is False
    assert result.retryable is False


async def test_handle_text_ready_frame():
    adapter = _make_adapter()
    await adapter._handle_text(json.dumps({
        "type": "ready",
        "onebot_connected": True,
        "adapter_version": "0.1.0",
        "self_id": "123456",
        "file_upload_timeout": 480.0,
    }))
    assert adapter._onebot_connected is True
    assert adapter._self_id == "123456"
    assert adapter._file_upload_timeout == 480.0


async def test_handle_text_result_resolves_future():
    adapter = _make_adapter()
    loop = asyncio.get_event_loop()
    fut = loop.create_future()
    adapter._futures["r1"] = fut
    await adapter._handle_text(json.dumps({"type": "result", "req_id": "r1", "success": True, "message_id": "99"}))
    assert fut.done()
    assert fut.result()["message_id"] == "99"
    assert "r1" not in adapter._futures


async def test_handle_text_result_wakes_future_owner_loop_thread_safely():
    adapter = _make_adapter()
    fut = MagicMock()
    owner_loop = MagicMock()
    fut.get_loop.return_value = owner_loop
    adapter._futures["r1"] = fut

    frame = {"type": "result", "req_id": "r1", "success": True}
    await adapter._handle_text(json.dumps(frame))

    fut.set_result.assert_not_called()
    owner_loop.call_soon_threadsafe.assert_called_once()
    callback, callback_fut, callback_data = owner_loop.call_soon_threadsafe.call_args.args
    assert callback_fut is fut
    assert callback_data == frame
    fut.done.return_value = False
    callback(callback_fut, callback_data)
    fut.set_result.assert_called_once_with(frame)


async def test_request_timeout():
    adapter = _make_adapter()
    fake_ws = FakeWS()
    # Don't auto-resolve — override send_json to just record
    async def _silent_send(msg):
        fake_ws.sent.append(msg)
    fake_ws.send_json = _silent_send
    fake_ws.closed = False
    adapter._ws = fake_ws
    # Use a very short timeout
    import onebot_adapter.hermes_plugin.adapter as mod
    old = mod._RESULT_TIMEOUT
    mod._RESULT_TIMEOUT = 0.1
    try:
        result = await adapter._request("send_text", chat_id="100", content="hi")
    finally:
        mod._RESULT_TIMEOUT = old
    assert result["success"] is False
    assert "timeout" in result["error"]


async def test_request_ws_closed():
    adapter = _make_adapter()
    adapter._ws = None
    result = await adapter._request("send_text", chat_id="100", content="hi")
    assert result["success"] is False


# ── Hermes 0.19 compatibility ─────────────────────────────────────────────


def test_connection_state_drives_base_is_connected():
    """0.19 ``is_connected`` reads ``_running``; markers must update it."""
    adapter = _make_adapter()
    adapter._fatal_error_message = "previous failure"
    adapter._fatal_error_code = "connect_failed"
    adapter._fatal_error_retryable = True
    adapter._write_runtime_status_safe = MagicMock()

    adapter._set_connection_state(True)
    assert adapter._is_connected is True
    assert adapter._running is True
    assert adapter.is_connected is True
    # 0.19 _mark_connected() clears a fatal error so reconnects recover.
    assert adapter._fatal_error_message is None
    assert adapter._fatal_error_code is None

    adapter._set_connection_state(False)
    assert adapter._is_connected is False
    assert adapter._running is False
    assert adapter.is_connected is False


async def test_connect_and_disconnect_toggle_base_is_connected(monkeypatch):
    """connect()/disconnect() must drive 0.19's ``is_connected`` property."""
    import onebot_adapter.hermes_plugin.adapter as plugin_mod

    adapter = _make_adapter()
    adapter._stop_event = asyncio.Event()
    adapter._watchdog_task = None
    adapter._recv_task = None
    adapter._event_tasks = set()
    adapter._write_runtime_status_safe = MagicMock()
    adapter._ws_connect = AsyncMock()
    adapter._receive_loop = AsyncMock()
    adapter._watchdog = AsyncMock()

    class _FakeSession:
        closed = False

        async def close(self):
            self.closed = True

    monkeypatch.setattr(plugin_mod.aiohttp, "ClientSession", lambda: _FakeSession())

    assert await adapter.connect() is True
    assert adapter.is_connected is True
    await adapter.disconnect()
    assert adapter.is_connected is False


async def test_send_image_file_sends_native_image_frame():
    """0.19 routes local images through send_image_file; it must not fall back to text."""
    adapter = _make_adapter()
    fake_ws = FakeWS()
    fake_ws._adapter = adapter
    adapter._ws = fake_ws
    result = await adapter.send_image_file(
        "group:42", "/tmp/img file.jpg", caption="**cap**", reply_to="7",
    )
    assert result.success is True
    sent = fake_ws.sent[0]
    assert sent["type"] == "send"
    assert sent["action"] == "send_image"
    assert sent["chat_id"] == "group:42"
    assert sent["image_url"] == "/tmp/img file.jpg"
    assert sent["caption"] == "cap"
    assert sent["reply_to"] == "7"


async def test_send_image_file_preserves_non_retryable_failure():
    adapter = _make_adapter()
    fake_ws = FakeWS()
    fake_ws._adapter = adapter

    async def _fail(msg):
        fake_ws.sent.append(msg)
        fut = adapter._futures.pop(msg["req_id"])
        fut.set_result({"success": False, "error": "blocked", "retryable": False})

    fake_ws.send_json = _fail
    adapter._ws = fake_ws
    result = await adapter.send_image_file("group:42", "/tmp/x.jpg")
    assert result.success is False
    assert result.retryable is False
    assert result.error == "blocked"


async def test_send_multiple_images_routes_file_uri_to_native_send():
    """0.19's send_multiple_images(file://…) must reach our native image sender."""
    adapter = _make_adapter()
    fake_ws = FakeWS()
    fake_ws._adapter = adapter
    adapter._ws = fake_ws
    await adapter.send_multiple_images("group:42", [("file:///tmp/a%20b.jpg", "")])
    sent = fake_ws.sent[0]
    assert sent["action"] == "send_image"
    assert sent["image_url"] == "/tmp/a b.jpg"


async def test_message_event_uses_source_for_identity(monkeypatch):
    """0.19 MessageEvent has no top-level chat_id/user_id; identity is on source."""
    import onebot_adapter.hermes_plugin.adapter as plugin_mod

    adapter = _make_adapter()
    adapter._media_delivery_mode = "passthrough"
    adapter._completed_deliveries = {}
    adapter._event_tasks = set()
    seen = []
    captured = {}

    real_message_event = plugin_mod.MessageEvent

    def _recording_message_event(**kwargs):
        captured.update(kwargs)
        return real_message_event(**kwargs)

    monkeypatch.setattr(plugin_mod, "MessageEvent", _recording_message_event)

    async def _capture(event, msg_ctx, delivery_ids=None):
        seen.append(event)

    adapter._dispatch_event = _capture
    await adapter._handle_event({
        "event": "message",
        "chat_id": "group:42",
        "chat_type": "group",
        "chat_name": "Test Group",
        "user_id": "123",
        "user_name": "Alice",
        "text": "hello",
        "message_id": "77",
        "timestamp": 1700000000,
        "delivery_ids": ["d1"],
    })
    await asyncio.sleep(0)

    assert "source" in captured
    for unsupported in ("chat_id", "user_id", "user_name", "is_group", "platform"):
        assert unsupported not in captured

    assert len(seen) == 1
    event = seen[0]
    assert event.source.chat_id == "group:42"
    assert event.source.chat_type == "group"
    assert event.source.user_id == "123"
    assert event.source.user_name == "Alice"


async def test_send_returns_sendresult_and_respects_retryable():
    adapter = _make_adapter()
    fake_ws = FakeWS()
    fake_ws._adapter = adapter
    adapter._ws = fake_ws
    result = await adapter.send("100", "hello")
    assert isinstance(result, SendResult)
    assert result.success is True
    assert fake_ws.sent[0]["action"] == "send_text"

    failing = FakeWS()
    failing._adapter = adapter
    adapter._ws = failing

    async def _fail(msg):
        failing.sent.append(msg)
        fut = adapter._futures.pop(msg["req_id"])
        fut.set_result({"success": False, "error": "nope", "retryable": False})

    failing.send_json = _fail
    result = await adapter.send("100", "hello")
    assert result.success is False
    assert result.retryable is False
    assert result.error == "nope"
