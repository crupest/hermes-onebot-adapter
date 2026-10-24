"""Outbound window renewal and reset handling in the Hermes relay."""
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

from onebot_adapter.config import AdapterConfig
from onebot_adapter.relay.hermes_ws import HermesRelayServer
from onebot_adapter.relay.protocol import (
    NormalizedEvent,
    api_call_message,
    send_message,
)
from onebot_adapter.window import NoticeBuffer, WindowStore


class FakeClock:
    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _make_relay(**cfg_overrides: Any):
    clock = FakeClock()
    store = WindowStore(300.0, clock=clock)
    buffer = NoticeBuffer(clock=clock)
    mock_api = MagicMock()
    mock_api.send_group_msg = AsyncMock(return_value={"message_id": "g1"})
    mock_api.send_private_msg = AsyncMock(return_value={"message_id": "p1"})
    mock_api.upload_group_file = AsyncMock(return_value=None)
    mock_api.upload_private_file = AsyncMock(return_value=None)
    mock_api.call = AsyncMock(return_value={"data": {"message_id": "api1"}})
    defaults: dict[str, Any] = {
        "hermes_ws_token": "testtoken",
        "hermes_ws_path": "/hermes",
        "message_show_group_id": False,
    }
    defaults.update(cfg_overrides)
    cfg = AdapterConfig(**defaults)
    relay = HermesRelayServer(
        cfg,
        mock_api,
        adapter_version="0.1.0-test",
        onebot_connected_fn=lambda: True,
        window_store=store,
        notice_buffer=buffer,
    )
    fake_ws = MagicMock()
    fake_ws.send_json = AsyncMock()
    relay._clients.add(fake_ws)
    return relay, mock_api, store, buffer, clock, fake_ws


def _event(text: str = "hello", *, chat_id: str = "group:42") -> NormalizedEvent:
    return NormalizedEvent(
        message_id="1",
        chat_id=chat_id,
        chat_type="group" if chat_id.startswith("group:") else "dm",
        user_id="100",
        user_name="Alice",
        text=text,
    )


async def test_successful_group_send_renews_window():
    relay, _, store, _, _, ws = _make_relay()
    assert store.active("42") is False
    await relay._handle_send(ws, send_message("send_text", "r1", "group:42", content="hi"))
    assert store.active("42") is True
    result = ws.send_json.await_args.args[0]
    assert result["success"] is True


async def test_failed_group_send_does_not_renew_window():
    relay, api, store, _, _, ws = _make_relay()
    api.send_group_msg.side_effect = RuntimeError("onebot down")
    await relay._handle_send(ws, send_message("send_text", "r1", "group:42", content="hi"))
    assert store.active("42") is False
    result = ws.send_json.await_args.args[0]
    assert result["success"] is False


async def test_outbound_filter_drop_does_not_renew_window():
    relay, api, store, _, _, ws = _make_relay(
        outbound_filter_enabled=True, outbound_filter_patterns=["forbidden"],
    )
    await relay._handle_send(ws, send_message("send_text", "r1", "group:42", content="forbidden text"))
    assert store.active("42") is False
    api.send_group_msg.assert_not_awaited()
    result = ws.send_json.await_args.args[0]
    assert result["data"] == {"filtered": True}


async def test_private_send_does_not_open_any_group_window():
    relay, _, store, _, _, ws = _make_relay()
    await relay._handle_send(ws, send_message("send_text", "r1", "100", content="dm"))
    assert len(store) == 0


async def test_successful_send_document_renews_window():
    relay, api, store, _, _, ws = _make_relay()
    await relay._handle_send(
        ws,
        send_message("send_document", "r1", "group:42", file_path="/tmp/a.txt", filename="a.txt"),
    )
    api.upload_group_file.assert_awaited_once()
    assert store.active("42") is True


async def test_send_document_touches_inside_upload_success_branch():
    """The window is renewed by _send_document itself, not by its caller."""
    relay, api, store, _, _, _ = _make_relay()
    await relay._send_document(
        {"file_path": "/tmp/a.txt", "filename": "a.txt"},
        "r1", "group:42", True, 42,
    )
    api.upload_group_file.assert_awaited_once()
    assert store.active("42") is True


async def test_send_document_caption_failure_keeps_window_open():
    """Upload success renews the window even if the caption send then fails."""
    relay, api, store, _, _, ws = _make_relay()
    api.send_group_msg.side_effect = RuntimeError("caption failed")
    await relay._handle_send(
        ws,
        send_message(
            "send_document", "r1", "group:42",
            file_path="/tmp/a.txt", filename="a.txt", caption="caption",
        ),
    )
    assert store.active("42") is True
    result = ws.send_json.await_args.args[0]
    assert result["success"] is False


async def test_failed_send_document_does_not_renew_window():
    relay, api, store, _, _, ws = _make_relay()
    api.upload_group_file.side_effect = RuntimeError("upload failed")
    await relay._handle_send(
        ws,
        send_message("send_document", "r1", "group:42", file_path="/tmp/a.txt", filename="a.txt"),
    )
    assert store.active("42") is False
    result = ws.send_json.await_args.args[0]
    assert result["success"] is False


async def test_api_call_send_group_msg_renews_window():
    relay, api, store, _, _, ws = _make_relay()
    await relay._handle_api_call(
        ws,
        api_call_message(
            "send_group_msg", "r1",
            {"group_id": 42, "message": [{"type": "text", "data": {"text": "hi"}}]},
        ),
    )
    api.call.assert_awaited_once()
    assert store.active("42") is True


async def test_api_call_failure_does_not_renew_window():
    relay, api, store, _, _, ws = _make_relay()
    api.call.side_effect = RuntimeError("boom")
    await relay._handle_api_call(
        ws,
        api_call_message(
            "send_group_msg", "r1",
            {"group_id": 42, "message": [{"type": "text", "data": {"text": "hi"}}]},
        ),
    )
    assert store.active("42") is False


async def test_api_call_filtered_send_does_not_renew_window():
    relay, api, store, _, _, ws = _make_relay(
        outbound_filter_enabled=True, outbound_filter_patterns=["forbidden"],
    )
    await relay._handle_api_call(
        ws,
        api_call_message(
            "send_group_msg", "r1",
            {"group_id": 42, "message": [{"type": "text", "data": {"text": "forbidden"}}]},
        ),
    )
    api.call.assert_not_awaited()
    assert store.active("42") is False


async def test_api_call_emoji_reaction_renews_window_via_chat_id():
    relay, api, store, _, _, ws = _make_relay()
    await relay._handle_api_call(
        ws,
        api_call_message(
            "set_msg_emoji_like", "r1",
            {"message_id": "77", "emoji_id": "128076"},
            chat_id="group:42",
        ),
    )
    api.call.assert_awaited_once()
    assert store.active("42") is True


async def test_api_call_emoji_reaction_without_chat_id_does_not_renew_window():
    relay, _, store, _, _, ws = _make_relay()
    await relay._handle_api_call(
        ws,
        api_call_message("set_msg_emoji_like", "r1", {"message_id": "77", "emoji_id": "128076"}),
    )
    assert len(store) == 0


async def test_api_call_emoji_reaction_in_dm_does_not_open_group_window():
    relay, _, store, _, _, ws = _make_relay()
    await relay._handle_api_call(
        ws,
        api_call_message(
            "set_msg_emoji_like", "r1",
            {"message_id": "77", "emoji_id": "128076"},
            chat_id="100",
        ),
    )
    assert len(store) == 0


async def test_failed_emoji_reaction_does_not_renew_window():
    relay, api, store, _, _, ws = _make_relay()
    api.call.side_effect = RuntimeError("boom")
    await relay._handle_api_call(
        ws,
        api_call_message(
            "set_msg_emoji_like", "r1",
            {"message_id": "77", "emoji_id": "128076"},
            chat_id="group:42",
        ),
    )
    assert store.active("42") is False


async def test_inbound_event_without_send_does_not_renew_window():
    relay, _, store, _, clock, _ = _make_relay()
    store.touch("42")
    clock.advance(100)
    await relay.push_event(_event("agent will answer NO_REPLY"))
    assert store.active("42") is True  # untouched by the inbound event
    clock.advance(201)
    assert store.active("42") is False


async def test_adapter_local_reject_does_not_renew_window():
    relay, api, store, _, _, _ = _make_relay()
    sent = await relay.send_direct_message("group:42", "⛔ rejected")
    assert sent is True
    api.send_group_msg.assert_awaited_once()
    assert store.active("42") is False


async def test_reset_clears_window_and_notice_buffer():
    relay, _, store, buffer, _, _ = _make_relay()
    store.touch("42")
    buffer.add("42", "系统通知")
    await relay.push_event(_event("/reset"))
    assert store.active("42") is False
    assert buffer.pending("42") == []


async def test_new_clears_window_and_notice_buffer():
    relay, _, store, buffer, _, _ = _make_relay()
    store.touch("42")
    buffer.add("42", "系统通知")
    await relay.push_event(_event("/new"))
    assert store.active("42") is False
    assert buffer.pending("42") == []


async def test_reset_confirmation_send_does_not_reopen_window():
    """The /reset reply must not immediately renew the window it just closed."""
    relay, _, store, _, _, _ = _make_relay()
    await relay.push_event(_event("/reset"))
    ws = MagicMock()
    ws.send_json = AsyncMock()
    await relay._handle_send(ws, send_message("send_text", "r1", "group:42", content="会话已重置"))
    assert store.active("42") is False


async def test_window_touch_resumes_after_reset_quiet_period():
    relay, _, store, _, _, _ = _make_relay()
    await relay.push_event(_event("/reset"))
    # Simulate the quiet period having elapsed.
    relay._window_quiet_until["42"] = 0.0
    ws = MagicMock()
    ws.send_json = AsyncMock()
    await relay._handle_send(ws, send_message("send_text", "r1", "group:42", content="hi"))
    assert store.active("42") is True


async def test_reset_for_other_group_does_not_clear_this_window():
    relay, _, store, _, _, _ = _make_relay()
    store.touch("42")
    await relay.push_event(_event("/reset", chat_id="group:43"))
    assert store.active("42") is True


async def test_relay_without_window_store_is_a_noop():
    cfg = AdapterConfig(hermes_ws_token="t", hermes_ws_path="/hermes")
    api = MagicMock()
    api.send_group_msg = AsyncMock(return_value={"message_id": "g1"})
    relay = HermesRelayServer(cfg, api, adapter_version="test", onebot_connected_fn=lambda: True)
    ws = MagicMock()
    ws.send_json = AsyncMock()
    await relay._handle_send(ws, send_message("send_text", "r1", "group:42", content="hi"))
    assert relay.window_store is None
    assert relay.notice_buffer is None
