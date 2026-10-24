"""Plugin ``api_call`` frames carry the originating chat id."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

from onebot_adapter.hermes_plugin import adapter as plugin_adapter


class _FakeWs:
    """Resolves the pending ``_rpc`` future as soon as the frame is sent."""

    def __init__(self, adapter: plugin_adapter.OneBotAdapter) -> None:
        self.closed = False
        self.sent: list[dict] = []
        self._adapter = adapter

    async def send_json(self, msg: dict) -> None:
        self.sent.append(msg)
        self._adapter._futures[msg["req_id"]].set_result({"success": True, "data": {}})


def _make_adapter() -> plugin_adapter.OneBotAdapter:
    adapter = plugin_adapter.OneBotAdapter.__new__(plugin_adapter.OneBotAdapter)
    adapter._rpc = AsyncMock(return_value={"success": True})
    return adapter


def _make_rpc_adapter() -> plugin_adapter.OneBotAdapter:
    adapter = plugin_adapter.OneBotAdapter.__new__(plugin_adapter.OneBotAdapter)
    adapter._send_semaphore = asyncio.Semaphore(1)
    adapter._futures = {}
    adapter._file_upload_timeout = 600
    adapter._ws = _FakeWs(adapter)
    return adapter


def _set_context(is_admin: bool, group_id: str, user_id: str, is_global_admin: bool):
    return plugin_adapter._msg_context.set((is_admin, group_id, user_id, is_global_admin))


async def test_api_call_passes_group_chat_id():
    adapter = _make_adapter()
    token = _set_context(False, "42", "100", False)
    try:
        await adapter._api_call("set_msg_emoji_like", {"message_id": 7})
    finally:
        plugin_adapter._msg_context.reset(token)
    adapter._rpc.assert_awaited_once()
    assert adapter._rpc.await_args.kwargs["chat_id"] == "group:42"


async def test_api_call_passes_dm_chat_id():
    adapter = _make_adapter()
    token = _set_context(False, "", "100", False)
    try:
        await adapter._api_call("set_msg_emoji_like", {"message_id": 7})
    finally:
        plugin_adapter._msg_context.reset(token)
    assert adapter._rpc.await_args.kwargs["chat_id"] == "100"


async def test_api_call_without_message_context_has_empty_chat_id():
    adapter = _make_adapter()
    await adapter._api_call("get_login_info", {})
    assert adapter._rpc.await_args.kwargs["chat_id"] == ""


async def test_rpc_api_call_frame_carries_chat_id():
    adapter = _make_rpc_adapter()
    result = await adapter._rpc(
        "api_call", "set_msg_emoji_like", {"message_id": 7}, chat_id="group:42",
    )
    assert result == {"success": True, "data": {}}
    frame = adapter._ws.sent[0]
    assert frame["type"] == "api_call"
    assert frame["params"] == {"message_id": 7}
    assert frame["chat_id"] == "group:42"


async def test_rpc_api_call_frame_omits_empty_chat_id():
    adapter = _make_rpc_adapter()
    await adapter._rpc("api_call", "get_login_info", {}, chat_id="")
    assert "chat_id" not in adapter._ws.sent[0]
