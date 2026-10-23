"""Active-window gating in the OneBot parser."""
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

from onebot_adapter.config import AdapterConfig, GroupConfig
from onebot_adapter.onebot.parser import parse_event
from onebot_adapter.relay.protocol import DroppedEvent, NormalizedEvent
from onebot_adapter.window import NoticeBuffer, WindowStore, bind_window_context


class FakeClock:
    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _group_event(
    text: str = "",
    *,
    mention: bool = False,
    user_id: int = 100,
    group_id: int = 42,
    message_id: int = 1,
    segments: list[dict] | None = None,
) -> dict[str, Any]:
    if segments is None:
        segments = []
        if mention:
            segments.append({"type": "at", "data": {"qq": "999"}})
        segments.append({"type": "text", "data": {"text": text}})
    return {
        "post_type": "message",
        "message_type": "group",
        "group_id": group_id,
        "user_id": user_id,
        "message_id": message_id,
        "time": 1_700_000_000,
        "real_seq": message_id,
        "sender": {"card": "Alice", "nickname": "Alice", "user_id": user_id},
        "message": segments,
    }


def _config(**overrides: Any) -> AdapterConfig:
    defaults: dict[str, Any] = {
        "group_require_mention": True,
        "message_show_group_id": False,
    }
    defaults.update(overrides)
    return AdapterConfig(**defaults)


async def _parse(
    event: dict[str, Any],
    cfg: AdapterConfig,
    store: WindowStore,
    buffer: NoticeBuffer | None = None,
    **kwargs: Any,
):
    return await parse_event(
        event,
        self_id="999",
        group_require_mention=cfg.group_require_mention,
        config=cfg,
        window_store=store,
        notice_buffer=buffer,
        **kwargs,
    )


async def test_group_mention_opens_window_and_marks_is_mention():
    clock = FakeClock()
    store = WindowStore(300.0, clock=clock)
    cfg = _config()
    result = await _parse(_group_event("hi bot", mention=True), cfg, store)
    assert isinstance(result, NormalizedEvent)
    assert result.is_mention is True
    assert result.text == "[Alice(100)#1]: hi bot"
    assert store.active("42") is True


async def test_group_keyword_opens_window_without_mention():
    clock = FakeClock()
    store = WindowStore(300.0, clock=clock)
    cfg = _config(group_trigger_keywords=["#bot"])
    result = await _parse(_group_event("#bot hello"), cfg, store)
    assert isinstance(result, NormalizedEvent)
    assert result.is_mention is False
    assert store.active("42") is True


async def test_non_mention_outside_window_is_dropped_as_trigger():
    store = WindowStore(300.0, clock=FakeClock())
    cfg = _config()
    result = await _parse(_group_event("just chatting"), cfg, store)
    assert isinstance(result, DroppedEvent)
    assert result.reason == "trigger"
    assert store.active("42") is False


async def test_non_mention_inside_window_is_forwarded():
    store = WindowStore(300.0, clock=FakeClock())
    store.touch("42")
    cfg = _config()
    result = await _parse(_group_event("inside window"), cfg, store)
    assert isinstance(result, NormalizedEvent)
    assert result.is_mention is False
    assert "inside window" in result.text


async def test_passive_message_does_not_slide_window():
    clock = FakeClock()
    store = WindowStore(300.0, clock=clock)
    store.touch("42")
    cfg = _config()
    clock.advance(200)
    result = await _parse(_group_event("passive"), cfg, store)
    assert isinstance(result, NormalizedEvent)
    clock.advance(101)  # 301s after the original touch
    assert store.active("42") is False


async def test_explicit_trigger_refreshes_window():
    clock = FakeClock()
    store = WindowStore(300.0, clock=clock)
    store.touch("42")
    cfg = _config()
    clock.advance(200)
    result = await _parse(_group_event("hi", mention=True), cfg, store)
    assert isinstance(result, NormalizedEvent)
    clock.advance(200)
    assert store.active("42") is True


async def test_window_disabled_drops_non_mention_even_if_store_has_entry():
    store = WindowStore(300.0, clock=FakeClock())
    store.touch("42")
    cfg = _config(active_window_enabled=False)
    result = await _parse(_group_event("no trigger"), cfg, store)
    assert isinstance(result, DroppedEvent)
    assert result.reason == "trigger"


async def test_window_check_happens_before_media_context_api_calls():
    store = WindowStore(300.0, clock=FakeClock())
    cfg = _config()
    api = MagicMock()
    api.get_msg = AsyncMock(return_value={"message": []})
    api.get_forward_msg = AsyncMock(return_value={"messages": []})
    segments = [
        {"type": "reply", "data": {"id": "55"}},
        {"type": "image", "data": {"url": "https://example.com/x.png", "file": "x.png"}},
        {"type": "text", "data": {"text": "look at this"}},
    ]
    result = await _parse(_group_event(segments=segments), cfg, store, api=api)
    assert isinstance(result, DroppedEvent)
    assert result.reason == "trigger"
    api.get_msg.assert_not_awaited()
    api.get_forward_msg.assert_not_awaited()


async def test_reset_clears_window_and_notice_buffer():
    store = WindowStore(300.0, clock=FakeClock())
    buffer = NoticeBuffer()
    store.touch("42")
    buffer.add("42", "用户 A(1) 加入了群聊")
    cfg = _config()
    result = await _parse(_group_event("", mention=True, segments=[
        {"type": "at", "data": {"qq": "999"}},
        {"type": "text", "data": {"text": "/reset"}},
    ]), cfg, store, buffer)
    assert isinstance(result, NormalizedEvent)
    assert result.text == "/reset"
    assert store.active("42") is False
    assert buffer.pending("42") == []


async def test_new_with_arguments_clears_window():
    store = WindowStore(300.0, clock=FakeClock())
    buffer = NoticeBuffer()
    store.touch("42")
    buffer.add("42", "notice")
    cfg = _config()
    result = await _parse(_group_event("", mention=True, segments=[
        {"type": "at", "data": {"qq": "999"}},
        {"type": "text", "data": {"text": "/new keep-memory"}},
    ]), cfg, store, buffer)
    assert isinstance(result, NormalizedEvent)
    assert store.active("42") is False
    assert buffer.pending("42") == []


async def test_unknown_reset_prefix_does_not_clear_window():
    store = WindowStore(300.0, clock=FakeClock())
    store.touch("42")
    cfg = _config()
    result = await _parse(_group_event("", mention=True, segments=[
        {"type": "at", "data": {"qq": "999"}},
        {"type": "text", "data": {"text": "/resetfoo"}},
    ]), cfg, store)
    assert isinstance(result, NormalizedEvent)
    assert store.active("42") is True


async def test_member_filter_runs_after_window_and_does_not_touch():
    clock = FakeClock()
    store = WindowStore(300.0, clock=clock)
    cfg = _config(groups={"42": GroupConfig(
        group_id="42", group_user_filter_mode="blacklist", group_user_list=["100"],
    ).to_dict()})
    result = await _parse(_group_event("hi", mention=True), cfg, store)
    assert isinstance(result, DroppedEvent)
    assert result.reason == "user_filter"
    # The rejected message did not open the window.
    assert store.active("42") is False


async def test_disabled_group_is_filtered_before_window():
    store = WindowStore(300.0, clock=FakeClock())
    store.touch("42")
    cfg = _config(groups={"42": GroupConfig(group_id="42", enabled=False).to_dict()})
    result = await _parse(_group_event("hi", mention=True), cfg, store)
    assert isinstance(result, DroppedEvent)
    assert result.reason == "user_filter"


async def test_mention_first_only_mid_mention_passes_via_window_but_is_not_mention():
    clock = FakeClock()
    store = WindowStore(300.0, clock=clock)
    store.touch("42")
    cfg = _config(group_mention_first_only=True)
    segments = [
        {"type": "text", "data": {"text": "hi "}},
        {"type": "at", "data": {"qq": "999"}},
    ]
    result = await _parse(_group_event(segments=segments), cfg, store)
    assert isinstance(result, NormalizedEvent)
    assert result.is_mention is False
    clock.advance(301)
    assert store.active("42") is False


async def test_store_can_be_bound_to_config_instead_of_passed_explicitly():
    store = WindowStore(300.0, clock=FakeClock())
    store.touch("42")
    cfg = _config()
    bind_window_context(cfg, store)
    result = await parse_event(
        _group_event("bound config"),
        self_id="999",
        group_require_mention=True,
        config=cfg,
    )
    assert isinstance(result, NormalizedEvent)
    assert result.is_mention is False


async def test_pass_all_group_touches_window_without_trigger():
    store = WindowStore(300.0, clock=FakeClock())
    cfg = _config(group_require_mention=False)
    result = await _parse(_group_event("anything"), cfg, store)
    assert isinstance(result, NormalizedEvent)
    assert store.active("42") is True


async def test_protocol_serializes_is_mention():
    event = NormalizedEvent(
        message_id="1", chat_id="group:42", chat_type="group",
        user_id="100", user_name="Alice", text="hi", is_mention=True,
    )
    assert event.to_dict()["is_mention"] is True
    assert NormalizedEvent(
        message_id="1", chat_id="group:42", chat_type="group",
        user_id="100", user_name="Alice", text="hi",
    ).to_dict()["is_mention"] is False
