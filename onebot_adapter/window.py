"""Per-group active window and notice context buffer.

The adapter service owns both pieces of state because it is the only process
that sees inbound group messages *and* outbound bot messages:

* :class:`WindowStore` tracks, per group, when the bot last participated
  (an explicit @/keyword trigger or a successful outbound send).  While the
  window is open, ordinary non-@ group messages are forwarded to Hermes so a
  shared group session can keep the conversational context; outside it they
  are dropped before any media/context API call.
* :class:`NoticeBuffer` keeps short-lived system notices (join/leave/poke) as
  text context.  Notices never wake the agent; the next accepted ordinary
  group message drains them and prepends ``[系统] ...`` lines.

Both stores are pure in-memory state keyed by the bare group id (the same key
used by ``HermesRelayServer`` and the history queue) and use
``time.monotonic()`` so wall-clock jumps cannot keep a window open.
"""
from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

DEFAULT_ACTIVE_WINDOW_SECONDS = 300.0
NOTICE_MAX_ENTRIES = 20
NOTICE_MAX_CHARS = 2000
NOTICE_LINE_PREFIX = "[系统] "

# Attributes used to hand the shared stores to the parser.  They are attached
# to AdapterConfig at service startup (and on every hot-reload) so components
# that only receive a config object (the OneBot parser pipeline) can reach the
# service-wide stores without rebuilding the transport chain.
WINDOW_STORE_ATTR = "_onebot_window_store"
NOTICE_BUFFER_ATTR = "_onebot_notice_buffer"


class WindowStore:
    """In-memory per-group activity window.

    ``touch`` starts/refreshes a group's window; ``active`` reports whether it
    is still open.  The TTL can be overridden per call (``ttl_seconds=...``) so
    per-group configuration is honoured without mutating the global default.
    """

    def __init__(
        self,
        ttl_seconds: float = DEFAULT_ACTIVE_WINDOW_SECONDS,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._clock = clock
        self._ttl_seconds = float(ttl_seconds)
        self._last_touch: dict[str, float] = {}

    @property
    def ttl_seconds(self) -> float:
        return self._ttl_seconds

    def set_ttl(self, ttl_seconds: float) -> None:
        """Update the default TTL and immediately drop windows that expired."""
        self._ttl_seconds = float(ttl_seconds)
        self.prune()

    def touch(self, group_id: str, *, ttl_seconds: float | None = None) -> None:
        """Open or refresh the window for *group_id* (TTL is not needed here)."""
        del ttl_seconds  # accepted for symmetry with active()/prune()
        self._last_touch[str(group_id)] = self._clock()

    def active(self, group_id: str, *, ttl_seconds: float | None = None) -> bool:
        """Whether *group_id*'s window is currently open."""
        last = self._last_touch.get(str(group_id))
        if last is None:
            return False
        ttl = self._ttl_seconds if ttl_seconds is None else float(ttl_seconds)
        if ttl <= 0:
            return False
        return (self._clock() - last) < ttl

    def clear(self, group_id: str | None = None) -> None:
        """Close one group's window, or every window when *group_id* is None."""
        if group_id is None:
            self._last_touch.clear()
        else:
            self._last_touch.pop(str(group_id), None)

    def prune(self, now: float | None = None, *, ttl_seconds: float | None = None) -> int:
        """Drop expired windows and return how many entries were removed."""
        current = self._clock() if now is None else now
        ttl = self._ttl_seconds if ttl_seconds is None else float(ttl_seconds)
        expired = [
            gid for gid, last in self._last_touch.items()
            if ttl <= 0 or (current - last) >= ttl
        ]
        for gid in expired:
            self._last_touch.pop(gid, None)
        return len(expired)

    def __len__(self) -> int:
        return len(self._last_touch)

    def __contains__(self, group_id: object) -> bool:
        return str(group_id) in self._last_touch


@dataclass(frozen=True)
class NoticeEntry:
    """One buffered system notice with its monotonic arrival timestamp."""

    text: str
    timestamp: float


class NoticeBuffer:
    """Bounded per-group buffer for system notices awaiting the next message.

    Entries are capped per group by both count (``max_entries``) and total
    characters (``max_chars``); adding beyond either cap evicts the oldest
    notices first.  ``drain`` returns rendered ``[系统] ...`` lines, newest
    last, and silently discards entries older than ``ttl_seconds``.
    """

    def __init__(
        self,
        *,
        max_entries: int = NOTICE_MAX_ENTRIES,
        max_chars: int = NOTICE_MAX_CHARS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._clock = clock
        self._max_entries = max(1, int(max_entries))
        self._max_chars = max(1, int(max_chars))
        self._entries: dict[str, deque[NoticeEntry]] = {}

    @property
    def max_entries(self) -> int:
        return self._max_entries

    @property
    def max_chars(self) -> int:
        return self._max_chars

    def add(
        self,
        group_id: str,
        text: str,
        *,
        timestamp: float | None = None,
    ) -> None:
        """Append one notice body (already without the ``[系统]`` prefix)."""
        body = (text or "").strip()
        if not body:
            return
        if len(body) > self._max_chars:
            body = body[: self._max_chars]
        gid = str(group_id)
        entries = self._entries.setdefault(gid, deque())
        entries.append(NoticeEntry(body, self._clock() if timestamp is None else timestamp))
        while len(entries) > self._max_entries:
            entries.popleft()
        while sum(len(entry.text) for entry in entries) > self._max_chars:
            entries.popleft()
        if not entries:
            self._entries.pop(gid, None)

    def drain(
        self,
        group_id: str,
        *,
        ttl_seconds: float | None = None,
        now: float | None = None,
    ) -> list[str]:
        """Remove and return the group's pending lines as ``[系统] ...``.

        Entries older than *ttl_seconds* are dropped without being returned.
        """
        gid = str(group_id)
        entries = self._entries.pop(gid, ())
        if not entries:
            return []
        current = self._clock() if now is None else now
        lines: list[str] = []
        for entry in entries:
            if ttl_seconds is not None and ttl_seconds > 0 and (current - entry.timestamp) >= ttl_seconds:
                continue
            lines.append(f"{NOTICE_LINE_PREFIX}{entry.text}")
        return lines

    def clear(self, group_id: str | None = None) -> None:
        """Discard one group's notices, or all notices when *group_id* is None."""
        if group_id is None:
            self._entries.clear()
        else:
            self._entries.pop(str(group_id), None)

    def prune(self, ttl_seconds: float, *, now: float | None = None) -> int:
        """Drop expired notices (across all groups) and return the count."""
        if ttl_seconds <= 0:
            removed = sum(len(entries) for entries in self._entries.values())
            self._entries.clear()
            return removed
        current = self._clock() if now is None else now
        removed = 0
        for gid, entries in list(self._entries.items()):
            while entries and (current - entries[0].timestamp) >= ttl_seconds:
                entries.popleft()
                removed += 1
            if not entries:
                self._entries.pop(gid, None)
        return removed

    def pending(self, group_id: str) -> list[str]:
        """Return a copy of the group's pending notice bodies (no rendering)."""
        return [entry.text for entry in self._entries.get(str(group_id), ())]

    def __len__(self) -> int:
        return sum(len(entries) for entries in self._entries.values())


def bind_window_context(
    config: object | None,
    window_store: WindowStore | None,
    notice_buffer: NoticeBuffer | None = None,
) -> None:
    """Attach the service-wide stores to *config* for the parser pipeline.

    ``AdapterConfig`` instances are recreated on hot-reload (``dataclasses.
    replace`` does not copy non-field attributes), so the service must call
    this again whenever it hands a fresh config to the transport handlers.
    """
    if config is None:
        return
    if window_store is None:
        try:
            delattr(config, WINDOW_STORE_ATTR)
        except AttributeError:
            pass
    else:
        setattr(config, WINDOW_STORE_ATTR, window_store)
    if notice_buffer is None:
        try:
            delattr(config, NOTICE_BUFFER_ATTR)
        except AttributeError:
            pass
    else:
        setattr(config, NOTICE_BUFFER_ATTR, notice_buffer)


def window_store_for(config: object | None) -> WindowStore | None:
    """Return the WindowStore bound to *config*, if the service wired one."""
    store = getattr(config, WINDOW_STORE_ATTR, None)
    return store if isinstance(store, WindowStore) else None


def notice_buffer_for(config: object | None) -> NoticeBuffer | None:
    """Return the NoticeBuffer bound to *config*, if the service wired one."""
    buffer = getattr(config, NOTICE_BUFFER_ATTR, None)
    return buffer if isinstance(buffer, NoticeBuffer) else None
