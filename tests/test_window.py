"""Tests for the per-group active window and notice context buffer."""
from __future__ import annotations

from onebot_adapter.window import (
    DEFAULT_ACTIVE_WINDOW_SECONDS,
    NOTICE_MAX_CHARS,
    NOTICE_MAX_ENTRIES,
    NoticeBuffer,
    WindowStore,
)


class FakeClock:
    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


# ── WindowStore ──────────────────────────────────────────────────────────


def test_window_store_default_ttl():
    store = WindowStore()
    assert store.ttl_seconds == DEFAULT_ACTIVE_WINDOW_SECONDS == 300.0
    assert len(store) == 0


def test_window_touch_and_active():
    clock = FakeClock()
    store = WindowStore(300.0, clock=clock)
    assert store.active("42") is False
    store.touch("42")
    assert store.active("42") is True
    assert "42" in store
    assert len(store) == 1


def test_window_expires_after_ttl():
    clock = FakeClock()
    store = WindowStore(300.0, clock=clock)
    store.touch("42")
    clock.advance(299.9)
    assert store.active("42") is True
    clock.advance(0.1)
    assert store.active("42") is False


def test_window_is_per_group():
    clock = FakeClock()
    store = WindowStore(60.0, clock=clock)
    store.touch("42")
    clock.advance(30)
    assert store.active("42") is True
    assert store.active("43") is False


def test_window_clear_one_and_all():
    clock = FakeClock()
    store = WindowStore(300.0, clock=clock)
    store.touch("42")
    store.touch("43")
    store.clear("42")
    assert store.active("42") is False
    assert store.active("43") is True
    store.clear()
    assert store.active("43") is False
    assert len(store) == 0


def test_window_call_level_ttl_override():
    clock = FakeClock()
    store = WindowStore(300.0, clock=clock)
    store.touch("42")
    clock.advance(100)
    assert store.active("42") is True
    assert store.active("42", ttl_seconds=100) is False
    assert store.active("42", ttl_seconds=0) is False
    assert store.active("42", ttl_seconds=101) is True


def test_window_prune_removes_only_expired():
    clock = FakeClock()
    store = WindowStore(100.0, clock=clock)
    store.touch("42")
    clock.advance(50)
    store.touch("43")
    clock.advance(60)  # 110s since 42, 60s since 43
    removed = store.prune()
    assert removed == 1
    assert store.active("42") is False
    assert store.active("43") is True


def test_window_set_ttl_prunes_existing():
    clock = FakeClock()
    store = WindowStore(300.0, clock=clock)
    store.touch("42")
    clock.advance(100)
    store.set_ttl(60.0)
    assert store.ttl_seconds == 60.0
    assert store.active("42") is False


def test_window_ttl_zero_never_active():
    clock = FakeClock()
    store = WindowStore(0.0, clock=clock)
    store.touch("42")
    assert store.active("42") is False


# ── NoticeBuffer ─────────────────────────────────────────────────────────


def test_notice_buffer_add_and_drain_formatting():
    clock = FakeClock()
    buffer = NoticeBuffer(clock=clock)
    buffer.add("42", "用户 A(1) 加入了群聊")
    buffer.add("42", "用户 B(2) 戳了戳你")
    lines = buffer.drain("42")
    assert lines == [
        "[系统] 用户 A(1) 加入了群聊",
        "[系统] 用户 B(2) 戳了戳你",
    ]
    assert buffer.drain("42") == []
    assert buffer.pending("42") == []


def test_notice_buffer_is_per_group():
    buffer = NoticeBuffer()
    buffer.add("42", "a")
    assert buffer.pending("43") == []
    assert len(buffer.drain("43")) == 0
    assert buffer.pending("42") == ["a"]


def test_notice_buffer_drops_entries_older_than_ttl():
    clock = FakeClock()
    buffer = NoticeBuffer(clock=clock)
    buffer.add("42", "old")
    clock.advance(10)
    buffer.add("42", "new")
    lines = buffer.drain("42", ttl_seconds=5)
    assert lines == ["[系统] new"]


def test_notice_buffer_drain_without_ttl_keeps_all():
    clock = FakeClock()
    buffer = NoticeBuffer(clock=clock)
    buffer.add("42", "old")
    clock.advance(10_000)
    assert buffer.drain("42") == ["[系统] old"]


def test_notice_buffer_clear_one_and_all():
    buffer = NoticeBuffer()
    buffer.add("42", "a")
    buffer.add("43", "b")
    buffer.clear("42")
    assert buffer.pending("42") == []
    assert buffer.pending("43") == ["b"]
    buffer.clear()
    assert len(buffer) == 0


def test_notice_buffer_entry_count_cap():
    buffer = NoticeBuffer()
    for i in range(NOTICE_MAX_ENTRIES + 5):
        buffer.add("42", f"n{i}")
    pending = buffer.pending("42")
    assert len(pending) == NOTICE_MAX_ENTRIES == 20
    assert pending[0] == "n5"
    assert pending[-1] == f"n{NOTICE_MAX_ENTRIES + 4}"


def test_notice_buffer_total_char_cap():
    buffer = NoticeBuffer(max_chars=10)
    buffer.add("42", "aaaa")
    buffer.add("42", "bbbbbbbb")  # total 12 -> oldest evicted
    assert buffer.pending("42") == ["bbbbbbbb"]
    buffer.add("42", "x" * 50)
    assert buffer.pending("42") == ["x" * 10]
    assert NOTICE_MAX_CHARS == 2000


def test_notice_buffer_prune():
    clock = FakeClock()
    buffer = NoticeBuffer(clock=clock)
    buffer.add("42", "old")
    clock.advance(100)
    buffer.add("42", "new")
    removed = buffer.prune(50)
    assert removed == 1
    assert buffer.pending("42") == ["new"]
    assert buffer.prune(0) == 1
    assert buffer.pending("42") == []


def test_notice_buffer_ignores_empty_text():
    buffer = NoticeBuffer()
    buffer.add("42", "")
    buffer.add("42", "   ")
    assert len(buffer) == 0


# ── Config serialization / hot-reload wiring ─────────────────────────────


def test_config_defaults_and_serialization_round_trip():
    from onebot_adapter.config import AdapterConfig, GroupConfig

    cfg = AdapterConfig()
    data = cfg.to_dict()
    assert data["active_window_enabled"] is True
    assert data["active_window_seconds"] == 300.0
    assert data["poke_as_trigger"] is False
    restored = AdapterConfig.from_dict(data)
    assert restored.resolve_active_window_enabled("42") is True
    assert restored.resolve_active_window_seconds("42") == 300.0

    grouped = AdapterConfig(groups={
        "42": GroupConfig(
            group_id="42", active_window_enabled=False, active_window_seconds=30.0,
            poke_as_trigger=True,
        ).to_dict(),
    })
    assert grouped.resolve_active_window_enabled("42") is False
    assert grouped.resolve_active_window_seconds("42") == 30.0
    assert grouped.resolve_poke_as_trigger("42") is True
    # Other groups still follow the global defaults.
    assert grouped.resolve_active_window_enabled("43") is True
    assert grouped.resolve_active_window_seconds("43") == 300.0


async def test_config_hot_reload_updates_ttl_and_rebinds_store():
    from onebot_adapter.app import AdapterService
    from onebot_adapter.config import AdapterConfig, ConfigStore
    from onebot_adapter.window import NoticeBuffer, WindowStore, window_store_for

    cfg = AdapterConfig(
        onebot_ws_token="t", hermes_ws_token="h", webui_token="w",
    )
    service = AdapterService(ConfigStore(cfg))
    store = WindowStore(300.0)
    buffer = NoticeBuffer()
    service._window_store = store
    service._notice_buffer = buffer
    new_cfg = cfg.with_overrides(active_window_seconds=30.0)
    await service._apply_config_change(cfg, new_cfg)
    assert store.ttl_seconds == 30.0
    assert window_store_for(new_cfg) is store


async def test_disabling_window_on_reload_clears_state():
    from onebot_adapter.app import AdapterService
    from onebot_adapter.config import AdapterConfig, ConfigStore
    from onebot_adapter.window import NoticeBuffer, WindowStore

    cfg = AdapterConfig(onebot_ws_token="t", hermes_ws_token="h", webui_token="w")
    service = AdapterService(ConfigStore(cfg))
    store = WindowStore(300.0)
    buffer = NoticeBuffer()
    store.touch("42")
    buffer.add("42", "notice")
    service._window_store = store
    service._notice_buffer = buffer
    new_cfg = cfg.with_overrides(active_window_enabled=False)
    await service._apply_config_change(cfg, new_cfg)
    assert store.active("42") is False
    assert len(buffer) == 0
