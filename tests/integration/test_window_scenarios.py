"""Adapter-level integration scenarios (MockOneBot -> AdapterService -> fake plugin).

These exercise the *whole* adapter pipeline without QQ or NapCat: a fake
OneBot client dials ``/onebot`` and injects real OneBot v11 frames, while a
fake Hermes plugin dials ``/hermes`` and receives ``NormalizedEvent`` frames.
Outbound ``send_group_msg`` calls are observed on the MockOneBot socket.

The active-window contract (fixed by the adapter design):

* ``active_window_enabled`` / ``active_window_seconds`` on ``AdapterConfig``
  (per-group overrides via ``GroupConfig``);
* @bot or keyword trigger opens the window; non-@ group messages are forwarded
  only while the window is open, otherwise dropped with reason ``trigger``;
* a successful outbound ``send_group_msg`` refreshes the window;
* group notices never become events — while the window is open they are
  buffered and prepended as ``[系统] ...`` lines to the next accepted
  ordinary group message;
* ``/reset`` and ``/new`` close the window and discard buffered notices.
"""
from __future__ import annotations

import asyncio
from dataclasses import fields

import pytest

from onebot_adapter.config import AdapterConfig

GROUP_ID = 424242
USER_ID = 10001
OTHER_USER_ID = 10002
SELF_ID = "10000"

pytestmark = pytest.mark.integration


def _notice_overrides() -> dict[str, object]:
    """Enable the legacy notice switches when they still exist.

    The window contract makes notice forwarding unconditional, but the
    pre-window config keeps ``notify_poke_enabled`` / ``notify_member_change_enabled``
    as gates.  Turn them on when present so the test proves notices are
    *buffered*, not merely disabled.
    """
    names = {field.name for field in fields(AdapterConfig)}
    overrides: dict[str, object] = {}
    if "notify_poke_enabled" in names:
        overrides["notify_poke_enabled"] = True
    if "notify_member_change_enabled" in names:
        overrides["notify_member_change_enabled"] = True
    return overrides


async def _open_window(adapter) -> tuple[object, object]:
    """Connect both peers and open the group window with an @bot message."""
    mock = await adapter.new_mock()
    plugin = await adapter.new_plugin()
    await mock.send_group_message(GROUP_ID, USER_ID, text="在吗", at=True)
    event = await plugin.wait_event(1)
    assert event["chat_id"] == f"group:{GROUP_ID}"
    assert "在吗" in event["text"]
    return mock, plugin


# ── Trigger / reply ─────────────────────────────────────────────────────


async def test_at_trigger_delivers_event_and_reply(adapter):
    """@bot 触发事件，Hermes 回复经适配器变为 send_group_msg。"""
    mock, plugin = await _open_window(adapter)
    assert mock.has_api_call("get_group_info")

    result = await plugin.send_text(f"group:{GROUP_ID}", "收到")
    assert result is not None
    assert result["success"] is True

    send = await mock.wait_for_sends(1)
    assert send.action == "send_group_msg"
    assert send.group_id == GROUP_ID
    assert send.text == "收到"


async def test_non_at_outside_window_is_dropped(adapter):
    """窗口未打开时的非 @ 群消息静默丢弃：无事件、无 API、无出站。"""
    mock = await adapter.new_mock()
    plugin = await adapter.new_plugin()

    await mock.send_group_message(GROUP_ID, USER_ID, text="路过看看")
    await asyncio.sleep(0.5)

    assert plugin.events == []
    assert mock.sends == []
    assert mock.api_calls == []


async def test_non_at_inside_window_is_delivered(adapter_factory):
    """窗口内（@ 触发后）的非 @ 消息照常转发。"""
    adapter = await adapter_factory(active_window_enabled=True, active_window_seconds=30.0)
    mock, plugin = await _open_window(adapter)

    await mock.send_group_message(GROUP_ID, OTHER_USER_ID, text="接着说")
    event = await plugin.wait_event(2)
    assert "接着说" in event["text"]
    assert event["user_id"] == str(OTHER_USER_ID)


async def test_no_reply_produces_no_outbound(adapter):
    """插件保持沉默（NO_REPLY）时适配器不发任何消息。"""
    mock, plugin = await _open_window(adapter)
    await asyncio.sleep(0.5)
    assert mock.sends == []


# ── Notices become context, never events ────────────────────────────────


async def test_notices_are_buffered_into_next_window_message(adapter_factory):
    """入群/戳一戳/退群不产生事件，只作为 [系统] 前缀挂到下一条普通消息。"""
    adapter = await adapter_factory(
        active_window_enabled=True,
        active_window_seconds=30.0,
        **_notice_overrides(),
    )
    mock, plugin = await _open_window(adapter)

    await mock.send_member_join(GROUP_ID, OTHER_USER_ID)
    await asyncio.sleep(0.05)
    await mock.send_poke_notice(GROUP_ID, USER_ID, target_id=SELF_ID)
    await asyncio.sleep(0.05)
    await mock.send_member_leave(GROUP_ID, OTHER_USER_ID)
    await asyncio.sleep(0.4)

    assert len(plugin.events) == 1  # only the initial @ message
    assert all("[系统]" not in text for text in plugin.event_texts)

    await mock.send_group_message(GROUP_ID, OTHER_USER_ID, text="继续聊")
    event = await plugin.wait_event(2)
    text = event["text"]
    assert "[系统]" in text
    assert "加入了群聊" in text
    assert "戳了戳你" in text
    assert "退出了群聊" in text
    assert text.index("[系统]") < text.index("继续聊")
    assert len(plugin.events) == 2


# ── Session reset ───────────────────────────────────────────────────────


async def test_reset_clears_window_and_notice_buffer(adapter_factory):
    """/reset 关闭当前群窗口并丢弃缓存的通知。"""
    adapter = await adapter_factory(
        active_window_enabled=True,
        active_window_seconds=60.0,
        **_notice_overrides(),
    )
    mock, plugin = await _open_window(adapter)

    await mock.send_poke_notice(GROUP_ID, USER_ID, target_id=SELF_ID)
    await asyncio.sleep(0.2)

    await mock.send_group_message(GROUP_ID, USER_ID, text="/reset")
    reset_event = await plugin.wait_event(2)
    assert reset_event["text"] == "/reset"

    await mock.send_group_message(GROUP_ID, USER_ID, text="还在吗")
    await asyncio.sleep(0.5)

    assert len(plugin.events) == 2
    assert mock.sends == []


# ── Window renewal by outbound bot activity ─────────────────────────────


async def test_successful_send_renews_window(adapter_factory):
    """成功 send_group_msg（cron 主动推送）续期窗口；否则已过期。"""
    adapter = await adapter_factory(active_window_enabled=True, active_window_seconds=2.0)
    mock, plugin = await _open_window(adapter)

    # Wait past the original window (opened by the @ message).
    await asyncio.sleep(1.2)
    result = await plugin.send_text(f"group:{GROUP_ID}", "定时播报")
    assert result is not None
    assert result["success"] is True
    await mock.wait_for_sends(1)

    # Past the original TTL, but inside the window refreshed by the send.
    await asyncio.sleep(1.2)
    await mock.send_group_message(GROUP_ID, OTHER_USER_ID, text="续期后")
    event = await plugin.wait_event(2)
    assert "续期后" in event["text"]


async def test_window_expires_without_activity(adapter_factory):
    """窗口到期后非 @ 消息再次被丢弃。"""
    adapter = await adapter_factory(active_window_enabled=True, active_window_seconds=0.6)
    mock, plugin = await _open_window(adapter)

    await asyncio.sleep(1.0)
    await mock.send_group_message(GROUP_ID, USER_ID, text="过期后")
    await asyncio.sleep(0.5)

    assert len(plugin.events) == 1
    assert mock.sends == []
