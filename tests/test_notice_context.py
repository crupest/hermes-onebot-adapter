"""System notices are cached as context and never forwarded to Hermes."""
from __future__ import annotations

from typing import Any

from onebot_adapter.config import AdapterConfig
from onebot_adapter.onebot.parser import parse_event
from onebot_adapter.relay.protocol import NormalizedEvent
from onebot_adapter.window import NoticeBuffer, WindowStore, bind_window_context


class FakeClock:
    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _config(**overrides: Any) -> AdapterConfig:
    defaults: dict[str, Any] = {
        "group_require_mention": True,
        "message_show_group_id": False,
        "notify_poke_enabled": True,
        "notify_member_change_enabled": True,
    }
    defaults.update(overrides)
    return AdapterConfig(**defaults)


def _notice_event(
    notice_type: str,
    *,
    sub_type: str = "",
    user_id: int = 100,
    target_id: int | None = 999,
    group_id: int | None = 42,
) -> dict[str, Any]:
    event: dict[str, Any] = {
        "post_type": "notice",
        "notice_type": notice_type,
        "user_id": user_id,
        "time": 1_700_000_000,
    }
    if sub_type:
        event["sub_type"] = sub_type
    if target_id is not None:
        event["target_id"] = target_id
    if group_id is not None:
        event["group_id"] = group_id
    return event


def _group_message(text: str, *, mention: bool = False, user_id: int = 100) -> dict[str, Any]:
    segments: list[dict] = []
    if mention:
        segments.append({"type": "at", "data": {"qq": "999"}})
    segments.append({"type": "text", "data": {"text": text}})
    return {
        "post_type": "message",
        "message_type": "group",
        "group_id": 42,
        "user_id": user_id,
        "message_id": 7,
        "real_seq": 7,
        "time": 1_700_000_000,
        "sender": {"card": "Alice", "nickname": "Alice", "user_id": user_id},
        "message": segments,
    }


async def _parse_notice(event: dict[str, Any], cfg: AdapterConfig, store, buffer):
    return await parse_event(
        event,
        self_id="999",
        group_require_mention=True,
        config=cfg,
        window_store=store,
        notice_buffer=buffer,
    )


async def _parse_message(event: dict[str, Any], cfg: AdapterConfig, store, buffer):
    return await parse_event(
        event,
        self_id="999",
        group_require_mention=True,
        config=cfg,
        window_store=store,
        notice_buffer=buffer,
    )


def _make_context(**cfg_overrides: Any):
    clock = FakeClock()
    store = WindowStore(300.0, clock=clock)
    buffer = NoticeBuffer(clock=clock)
    cfg = _config(**cfg_overrides)
    return cfg, store, buffer, clock


async def test_poke_in_active_window_is_buffered_not_forwarded():
    cfg, store, buffer, _ = _make_context()
    store.touch("42")
    result = await _parse_notice(
        _notice_event("notify", sub_type="poke"), cfg, store, buffer,
    )
    assert result is None
    assert buffer.pending("42") == ["用户 100(100) 戳了戳你"]


async def test_poke_outside_window_is_discarded():
    cfg, store, buffer, _ = _make_context()
    result = await _parse_notice(
        _notice_event("notify", sub_type="poke"), cfg, store, buffer,
    )
    assert result is None
    assert buffer.pending("42") == []


async def test_notice_does_not_renew_window():
    cfg, store, buffer, clock = _make_context()
    store.touch("42")
    clock.advance(100)
    await _parse_notice(_notice_event("notify", sub_type="poke"), cfg, store, buffer)
    clock.advance(199)
    assert store.active("42") is True
    clock.advance(2)  # 301s after the original touch
    assert store.active("42") is False


async def test_next_message_gets_notice_prefix_and_drains_buffer():
    cfg, store, buffer, _ = _make_context()
    store.touch("42")
    await _parse_notice(_notice_event("notify", sub_type="poke"), cfg, store, buffer)
    result = await _parse_message(_group_message("hello"), cfg, store, buffer)
    assert isinstance(result, NormalizedEvent)
    assert result.text == "[系统] 用户 100(100) 戳了戳你\n[Alice(100)#7]: hello"
    assert buffer.pending("42") == []


async def test_multiple_notices_are_joined_with_newlines():
    cfg, store, buffer, _ = _make_context()
    store.touch("42")
    await _parse_notice(_notice_event("notify", sub_type="poke", user_id=100), cfg, store, buffer)
    await _parse_notice(
        _notice_event("group_increase", sub_type="approve", user_id=200), cfg, store, buffer,
    )
    result = await _parse_message(_group_message("hello"), cfg, store, buffer)
    assert isinstance(result, NormalizedEvent)
    lines = result.text.splitlines()
    assert lines[0] == "[系统] 用户 100(100) 戳了戳你"
    assert lines[1] == "[系统] 用户 200(200) 加入了群聊"
    assert lines[2] == "[Alice(100)#7]: hello"


async def test_notice_older_than_window_ttl_is_not_injected():
    cfg, store, buffer, clock = _make_context()
    store.touch("42")
    await _parse_notice(_notice_event("notify", sub_type="poke"), cfg, store, buffer)
    clock.advance(301)
    result = await _parse_message(_group_message("hi", mention=True), cfg, store, buffer)
    assert isinstance(result, NormalizedEvent)
    assert "[系统]" not in result.text
    assert result.text == "[Alice(100)#7]: hi"


async def test_reset_discards_buffered_notices():
    cfg, store, buffer, _ = _make_context()
    store.touch("42")
    await _parse_notice(_notice_event("notify", sub_type="poke"), cfg, store, buffer)
    reset = await _parse_message(
        {
            "post_type": "message", "message_type": "group", "group_id": 42,
            "user_id": 100, "message_id": 8, "real_seq": 8, "time": 1_700_000_000,
            "sender": {"card": "Alice", "nickname": "Alice", "user_id": 100},
            "message": [
                {"type": "at", "data": {"qq": "999"}},
                {"type": "text", "data": {"text": "/reset"}},
            ],
        },
        cfg, store, buffer,
    )
    assert isinstance(reset, NormalizedEvent)
    assert buffer.pending("42") == []
    nxt = await _parse_message(_group_message("hi", mention=True), cfg, store, buffer)
    assert isinstance(nxt, NormalizedEvent)
    assert "[系统]" not in nxt.text


async def test_member_leave_notice_uses_kick_wording():
    cfg, store, buffer, _ = _make_context()
    store.touch("42")
    await _parse_notice(
        _notice_event("group_decrease", sub_type="kick", user_id=200), cfg, store, buffer,
    )
    assert buffer.pending("42") == ["用户 200(200) 被管理员移出了群聊"]


async def test_dm_poke_is_not_buffered():
    cfg, store, buffer, _ = _make_context(dm_policy="allow")
    store.touch("42")
    result = await _parse_notice(
        _notice_event("notify", sub_type="poke", group_id=None), cfg, store, buffer,
    )
    assert result is None
    assert len(buffer) == 0


async def test_disabled_window_does_not_buffer_notices():
    cfg, store, buffer, _ = _make_context(active_window_enabled=False)
    store.touch("42")
    await _parse_notice(_notice_event("notify", sub_type="poke"), cfg, store, buffer)
    assert len(buffer) == 0


async def test_disabled_notice_flags_discard_events():
    cfg, store, buffer, _ = _make_context(
        notify_poke_enabled=False, notify_member_change_enabled=False,
    )
    store.touch("42")
    await _parse_notice(_notice_event("notify", sub_type="poke"), cfg, store, buffer)
    await _parse_notice(_notice_event("group_increase", sub_type="approve"), cfg, store, buffer)
    assert len(buffer) == 0


async def test_notice_buffer_limits_apply_before_injection():
    cfg, store, buffer, _ = _make_context()
    store.touch("42")
    for i in range(25):
        await _parse_notice(
            _notice_event("group_increase", sub_type="approve", user_id=200 + i),
            cfg, store, buffer,
        )
    result = await _parse_message(_group_message("hello"), cfg, store, buffer)
    assert isinstance(result, NormalizedEvent)
    notice_lines = [line for line in result.text.splitlines() if line.startswith("[系统]")]
    assert len(notice_lines) == 20
    assert notice_lines[-1] == "[系统] 用户 224(224) 加入了群聊"


async def test_config_bound_stores_are_used_without_explicit_arguments():
    cfg, store, buffer, _ = _make_context()
    bind_window_context(cfg, store, buffer)
    store.touch("42")
    await parse_event(
        _notice_event("notify", sub_type="poke"),
        self_id="999",
        group_require_mention=True,
        config=cfg,
    )
    result = await parse_event(
        _group_message("hello"),
        self_id="999",
        group_require_mention=True,
        config=cfg,
    )
    assert isinstance(result, NormalizedEvent)
    assert result.text.startswith("[系统] ")


def test_poke_as_trigger_is_reserved_and_defaults_to_false():
    cfg = AdapterConfig()
    assert cfg.poke_as_trigger is False
    assert cfg.resolve_poke_as_trigger("42") is False
    enabled = AdapterConfig(groups={
        "42": {"group_id": "42", "poke_as_trigger": True},
    })
    assert enabled.resolve_poke_as_trigger("42") is True
