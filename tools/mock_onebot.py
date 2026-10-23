"""MockOneBot — a OneBot v11 client for adapter-level integration tests.

In reverse mode the adapter does not dial anything: MockOneBot connects to
``/onebot?access_token=<token>`` (the same socket NapCat would use) and

* injects OneBot events — group/private messages, join/leave notices, poke
  notices;
* answers the adapter's API calls on that same WebSocket, echoing the request
  ``echo`` back as ``{"status": "ok", "retcode": 0, "data": ..., "echo": ...}``;
* records every ``send_group_msg`` / ``send_private_msg`` request (segments
  and timing) so tests can assert on outbound behaviour.

The module is both a library (used by ``tests/integration``) and a CLI::

    python tools/mock_onebot.py --adapter-url ws://127.0.0.1:18800/onebot \\
        --token X --scenario scenarios/group_at_reply.yaml --report out.json

Scenario YAML::

    name: example
    steps:
      - send_event: {type: group_message, group_id: 42, user_id: 100,
                     text: hi, at: true}
      - sleep: 0.3
      - expect_api:
          - action: get_group_info
      - expect:
          send_count: 1
          sends:
            - action: send_group_msg
              group_id: 42
              text_contains: "hi"

``send_event`` accepts either a raw OneBot event (it has ``post_type``) or one
of the shorthands ``group_message`` / ``private_message`` / ``member_join`` /
``member_leave`` / ``poke``.  ``expect_api`` asserts that the adapter issued an
API call to the mock (the adapter→OneBot direction); the mock never initiates
OneBot API calls itself.  A failed expectation prints an expected-vs-actual
diff and exits non-zero.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import aiohttp
from ruamel.yaml import YAML

DEFAULT_SELF_ID = 10000
_MESSAGE_ID_START = 1000


def text_segment(text: str) -> dict[str, Any]:
    return {"type": "text", "data": {"text": str(text)}}


def at_segment(qq: Any) -> dict[str, Any]:
    return {"type": "at", "data": {"qq": str(qq)}}


def image_segment(file: str) -> dict[str, Any]:
    return {"type": "image", "data": {"file": str(file)}}


def reply_segment(message_id: Any) -> dict[str, Any]:
    return {"type": "reply", "data": {"id": str(message_id)}}


@dataclass
class RecordedSend:
    """One ``send_group_msg`` / ``send_private_msg`` request from the adapter."""

    action: str
    chat_id: str
    group_id: int | None
    user_id: int | None
    message: list[dict[str, Any]]
    message_id: str
    timestamp: float

    @property
    def text(self) -> str:
        """Concatenated text segments (empty for media-only sends)."""
        return "".join(
            str((seg.get("data") or {}).get("text", ""))
            for seg in self.message
            if seg.get("type") == "text"
        )

    @property
    def segment_types(self) -> list[str]:
        return [str(seg.get("type", "")) for seg in self.message]

    @property
    def at_targets(self) -> list[str]:
        return [
            str((seg.get("data") or {}).get("qq", ""))
            for seg in self.message
            if seg.get("type") == "at"
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "chat_id": self.chat_id,
            "group_id": self.group_id,
            "user_id": self.user_id,
            "message": self.message,
            "message_id": self.message_id,
            "timestamp": self.timestamp,
        }


@dataclass
class ApiCall:
    """One API request received from the adapter."""

    action: str
    params: dict[str, Any]
    timestamp: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "params": self.params,
            "timestamp": self.timestamp,
        }


class ScenarioError(AssertionError):
    """A scenario step failed (bad step shape or expectation mismatch)."""


class MockOneBot:
    """A minimal OneBot v11 client that speaks to the adapter's reverse WS."""

    def __init__(
        self,
        adapter_url: str,
        token: str = "",
        *,
        self_id: int | str = DEFAULT_SELF_ID,
        group_names: dict[int, str] | None = None,
        nicknames: dict[int, str] | None = None,
        request_timeout: float = 5.0,
    ) -> None:
        self.adapter_url = adapter_url
        self.token = token
        self.self_id = int(self_id) if str(self_id).isdigit() else str(self_id)
        self.group_names = dict(group_names or {})
        self.nicknames = dict(nicknames or {})
        self.request_timeout = request_timeout

        self.sends: list[RecordedSend] = []
        self.api_calls: list[ApiCall] = []
        self.events_sent: list[dict[str, Any]] = []
        self.raw_frames: list[dict[str, Any]] = []
        self.inbound_messages: dict[str, dict[str, Any]] = {}

        self._session: aiohttp.ClientSession | None = None
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._connected = asyncio.Event()
        self._closed = False
        self._send_lock = asyncio.Lock()
        self._message_id = _MESSAGE_ID_START
        self._real_seqs: dict[str, int] = {}

    # ── Connection lifecycle ────────────────────────────────────────────

    @property
    def connected(self) -> bool:
        return self._ws is not None and not self._ws.closed

    async def connect(self, timeout: float = 10.0) -> MockOneBot:
        self._session = aiohttp.ClientSession()
        url = self.adapter_url
        if self.token:
            separator = "&" if "?" in url else "?"
            url = f"{url}{separator}access_token={quote(self.token)}"
        try:
            self._ws = await asyncio.wait_for(self._session.ws_connect(url), timeout=timeout)
        except Exception:
            await self._session.close()
            self._session = None
            raise
        self._reader_task = asyncio.create_task(self._read_loop(), name="mock-onebot-reader")
        self._connected.set()
        await asyncio.sleep(0)  # let the reader task start
        return self

    async def close(self) -> None:
        self._closed = True
        reader = self._reader_task
        self._reader_task = None
        if reader is not None:
            reader.cancel()
            await asyncio.gather(reader, return_exceptions=True)
        ws = self._ws
        self._ws = None
        if ws is not None and not ws.closed:
            await ws.close()
        session = self._session
        self._session = None
        if session is not None and not session.closed:
            await session.close()
        self._connected.clear()

    async def _read_loop(self) -> None:
        ws = self._ws
        assert ws is not None
        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    await self._on_frame(msg.data)
                elif msg.type in (
                    aiohttp.WSMsgType.ERROR,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSED,
                ):
                    break
        finally:
            self._connected.clear()

    async def _on_frame(self, raw: str) -> None:
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return
        if not isinstance(data, dict):
            return
        # The adapter→OneBot direction is an API request: action + echo.
        if "action" in data and "post_type" not in data:
            await self._handle_api_request(data)
            return
        self.raw_frames.append(data)

    async def _handle_api_request(self, request: dict[str, Any]) -> None:
        action = str(request.get("action", ""))
        params = request.get("params") or {}
        if not isinstance(params, dict):
            params = {}
        self.api_calls.append(ApiCall(action=action, params=dict(params), timestamp=time.time()))
        try:
            payload: dict[str, Any] = {
                "status": "ok",
                "retcode": 0,
                "data": self._api_result(action, params),
                "echo": request.get("echo"),
            }
        except Exception as exc:  # pragma: no cover - defensive
            payload = {
                "status": "failed",
                "retcode": 1,
                "data": None,
                "msg": str(exc),
                "echo": request.get("echo"),
            }
        await self._send_json(payload)

    def _api_result(self, action: str, params: dict[str, Any]) -> Any:
        if action == "get_login_info":
            return {"user_id": self.self_id, "nickname": "MockBot"}
        if action == "get_group_info":
            group_id = _as_int(params.get("group_id"))
            return {
                "group_id": group_id,
                "group_name": self.group_names.get(group_id, f"MockGroup{group_id}"),
                "member_count": 3,
            }
        if action == "get_group_member_info":
            group_id = _as_int(params.get("group_id"))
            user_id = _as_int(params.get("user_id"))
            return {
                "group_id": group_id,
                "user_id": user_id,
                "card": "",
                "nickname": self.nicknames.get(user_id, f"User{user_id}"),
            }
        if action == "get_stranger_info":
            user_id = _as_int(params.get("user_id"))
            return {"user_id": user_id, "nickname": self.nicknames.get(user_id, f"User{user_id}")}
        if action == "get_friend_list":
            return [
                {"user_id": user_id, "nickname": nickname}
                for user_id, nickname in self.nicknames.items()
            ]
        if action == "get_msg":
            message_id = str(params.get("message_id", ""))
            known = self.inbound_messages.get(message_id)
            if known is not None:
                return known
            return {
                "message_id": _maybe_int(message_id),
                "sender": {"user_id": 0, "nickname": "unknown"},
                "message": [text_segment(f"message {message_id}")],
                "time": int(time.time()),
            }
        if action == "send_group_msg":
            return {"message_id": self._record_send("send_group_msg", params)}
        if action == "send_private_msg":
            return {"message_id": self._record_send("send_private_msg", params)}
        if action in {"send_msg", "send_group_forward_msg", "send_private_forward_msg"}:
            group_id = params.get("group_id")
            action_name = "send_group_msg" if group_id not in (None, "") else "send_private_msg"
            return {"message_id": self._record_send(action_name, params)}
        if action == "get_group_msg_history":
            return self._group_history(params)
        if action == "get_forward_msg":
            return {"messages": []}
        if action == "set_msg_emoji_like":
            return {"message_id": _maybe_int(params.get("message_id"))}
        return {}

    def _record_send(self, action: str, params: dict[str, Any]) -> str:
        group_raw = params.get("group_id")
        user_raw = params.get("user_id")
        group_id = _as_int(group_raw) if group_raw not in (None, "") else None
        user_id = _as_int(user_raw) if user_raw not in (None, "") else None
        chat_id = f"group:{group_id}" if group_id is not None else str(user_id)
        message = params.get("message") or []
        if not isinstance(message, list):
            message = []
        message = json.loads(json.dumps(message, ensure_ascii=False))
        message_id = str(self._next_message_id())
        self.sends.append(
            RecordedSend(
                action=action,
                chat_id=chat_id,
                group_id=group_id,
                user_id=user_id,
                message=message,
                message_id=message_id,
                timestamp=time.time(),
            )
        )
        return message_id

    def _group_history(self, params: dict[str, Any]) -> dict[str, Any]:
        group_id = _as_int(params.get("group_id"))
        messages = [
            {
                "message_id": _maybe_int(send.message_id),
                "sender": {"user_id": self.self_id, "nickname": "MockBot"},
                "message": send.message,
                "time": int(send.timestamp),
            }
            for send in self.sends
            if send.action == "send_group_msg" and send.group_id == group_id
        ]
        return {"messages": messages}

    async def _send_json(self, payload: dict[str, Any]) -> None:
        ws = self._ws
        if ws is None or ws.closed:
            raise ConnectionError("MockOneBot is not connected")
        async with self._send_lock:
            await ws.send_json(payload)

    # ── Event construction / injection ──────────────────────────────────

    def _next_message_id(self) -> int:
        self._message_id += 1
        return self._message_id

    def _next_real_seq(self, group_id: int) -> int:
        key = str(group_id)
        self._real_seqs[key] = self._real_seqs.get(key, 0) + 1
        return self._real_seqs[key]

    def build_group_message(
        self,
        group_id: int,
        user_id: int,
        *,
        text: str = "",
        at: bool | int | str | None = False,
        at_qq: int | str | None = None,
        image: str | None = None,
        reply_to: int | str | None = None,
        segments: list[dict[str, Any]] | None = None,
        real_seq: int | str | None = None,
        message_id: int | str | None = None,
        sender: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        group_id = int(group_id)
        user_id = int(user_id)
        if segments is None:
            segments = []
            if reply_to not in (None, ""):
                segments.append(reply_segment(reply_to))
            if at is True:
                segments.append(at_segment(self.self_id))
            elif at not in (None, False):
                segments.append(at_segment(at))
            if at_qq not in (None, "", False):
                segments.append(at_segment(at_qq))
            if text:
                segments.append(text_segment(text))
            if image:
                segments.append(image_segment(image))
        event: dict[str, Any] = {
            "post_type": "message",
            "message_type": "group",
            "sub_type": "normal",
            "group_id": group_id,
            "user_id": user_id,
            "message_id": _maybe_int(message_id) if message_id is not None else self._next_message_id(),
            "real_seq": real_seq if real_seq is not None else self._next_real_seq(group_id),
            "time": int(time.time()),
            "sender": sender or self._default_sender(user_id),
            "message": segments,
        }
        self.inbound_messages[str(event["message_id"])] = event
        return event

    def build_private_message(
        self,
        user_id: int,
        *,
        text: str = "",
        segments: list[dict[str, Any]] | None = None,
        message_id: int | str | None = None,
        sender: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        user_id = int(user_id)
        if segments is None:
            segments = [text_segment(text)] if text else []
        event: dict[str, Any] = {
            "post_type": "message",
            "message_type": "private",
            "sub_type": "friend",
            "user_id": user_id,
            "message_id": _maybe_int(message_id) if message_id is not None else self._next_message_id(),
            "time": int(time.time()),
            "sender": sender or self._default_sender(user_id),
            "message": segments,
        }
        self.inbound_messages[str(event["message_id"])] = event
        return event

    def build_member_join(
        self, group_id: int, user_id: int, *, operator_id: int = 0,
    ) -> dict[str, Any]:
        return {
            "post_type": "notice",
            "notice_type": "group_increase",
            "sub_type": "approve",
            "group_id": int(group_id),
            "user_id": int(user_id),
            "operator_id": str(operator_id),
            "time": int(time.time()),
        }

    def build_member_leave(
        self, group_id: int, user_id: int, *, kick: bool = False, operator_id: int = 0,
    ) -> dict[str, Any]:
        return {
            "post_type": "notice",
            "notice_type": "group_decrease",
            "sub_type": "kick" if kick else "leave",
            "group_id": int(group_id),
            "user_id": int(user_id),
            "operator_id": str(operator_id),
            "time": int(time.time()),
        }

    def build_poke(
        self,
        group_id: int | None,
        user_id: int,
        *,
        target_id: int | str | None = None,
        private: bool = False,
    ) -> dict[str, Any]:
        event: dict[str, Any] = {
            "post_type": "notice",
            "notice_type": "notify",
            "sub_type": "poke",
            "user_id": int(user_id),
            "target_id": str(target_id if target_id is not None else self.self_id),
            "time": int(time.time()),
        }
        if not private and group_id is not None:
            event["group_id"] = int(group_id)
        return event

    def _default_sender(self, user_id: int) -> dict[str, Any]:
        return {
            "user_id": int(user_id),
            "nickname": self.nicknames.get(int(user_id), f"User{user_id}"),
            "card": "",
        }

    async def send_event(self, event: dict[str, Any]) -> None:
        self.events_sent.append(event)
        if event.get("post_type") == "message":
            self.inbound_messages[str(event.get("message_id", ""))] = event
        await self._send_json(event)

    async def send_group_message(
        self,
        group_id: int,
        user_id: int,
        *,
        text: str = "",
        at: bool | int | str | None = False,
        at_qq: int | str | None = None,
        image: str | None = None,
        reply_to: int | str | None = None,
        segments: list[dict[str, Any]] | None = None,
        real_seq: int | str | None = None,
        message_id: int | str | None = None,
        sender: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        event = self.build_group_message(
            group_id,
            user_id,
            text=text,
            at=at,
            at_qq=at_qq,
            image=image,
            reply_to=reply_to,
            segments=segments,
            real_seq=real_seq,
            message_id=message_id,
            sender=sender,
        )
        await self.send_event(event)
        return event

    async def send_private_message(self, user_id: int, *, text: str = "", **kwargs: Any) -> dict[str, Any]:
        event = self.build_private_message(user_id, text=text, **kwargs)
        await self.send_event(event)
        return event

    async def send_member_join(self, group_id: int, user_id: int, *, operator_id: int = 0) -> None:
        await self.send_event(self.build_member_join(group_id, user_id, operator_id=operator_id))

    async def send_member_leave(self, group_id: int, user_id: int, *, kick: bool = False) -> None:
        await self.send_event(self.build_member_leave(group_id, user_id, kick=kick))

    async def send_poke_notice(
        self,
        group_id: int | None,
        user_id: int,
        *,
        target_id: int | str | None = None,
        private: bool = False,
    ) -> None:
        await self.send_event(self.build_poke(group_id, user_id, target_id=target_id, private=private))

    # ── Waiting / assertions for tests ──────────────────────────────────

    async def wait_until(self, predicate: Any, *, timeout: float = 3.0, interval: float = 0.02) -> None:
        deadline = time.monotonic() + timeout
        while not predicate():
            if time.monotonic() >= deadline:
                raise AssertionError(f"timed out after {timeout:.1f}s waiting for condition")
            await asyncio.sleep(interval)

    async def wait_for_sends(self, count: int = 1, *, timeout: float = 3.0) -> RecordedSend:
        await self.wait_until(lambda: len(self.sends) >= count, timeout=timeout)
        return self.sends[count - 1]

    async def wait_for_api(self, action: str, count: int = 1, *, timeout: float = 3.0) -> ApiCall:
        await self.wait_until(
            lambda: sum(1 for call in self.api_calls if call.action == action) >= count,
            timeout=timeout,
        )
        return [call for call in self.api_calls if call.action == action][count - 1]

    def has_api_call(self, action: str, **params: Any) -> bool:
        for call in self.api_calls:
            if call.action != action:
                continue
            if all(call.params.get(key) == value for key, value in params.items()):
                return True
        return False


# ── Scenario engine ─────────────────────────────────────────────────────


def load_scenario(path: str | Path) -> dict[str, Any]:
    yaml = YAML(typ="safe")
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.load(handle)
    if not isinstance(data, dict):
        raise ScenarioError(f"scenario {path} must be a mapping")
    return data


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _maybe_int(value: Any) -> int | str:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return value


def _expand_event(client: MockOneBot, spec: Any) -> dict[str, Any]:
    if not isinstance(spec, dict):
        raise ScenarioError("send_event must be a mapping")
    if "post_type" in spec:
        return dict(spec)
    kind = spec.get("type")
    if kind == "group_message":
        kwargs = {
            key: spec[key]
            for key in (
                "text", "at", "at_qq", "image", "reply_to", "segments",
                "real_seq", "message_id", "sender",
            )
            if key in spec
        }
        return client.build_group_message(
            int(spec["group_id"]), int(spec["user_id"]), **kwargs,
        )
    if kind == "private_message":
        kwargs = {key: spec[key] for key in ("text", "segments", "message_id", "sender") if key in spec}
        return client.build_private_message(int(spec["user_id"]), **kwargs)
    if kind == "member_join":
        return client.build_member_join(
            int(spec["group_id"]), int(spec["user_id"]), operator_id=int(spec.get("operator_id", 0)),
        )
    if kind == "member_leave":
        return client.build_member_leave(
            int(spec["group_id"]), int(spec["user_id"]), kick=bool(spec.get("kick", False)),
        )
    if kind == "poke":
        return client.build_poke(
            int(spec["group_id"]) if spec.get("group_id") is not None else None,
            int(spec["user_id"]),
            target_id=spec.get("target_id"),
            private=bool(spec.get("private", False)),
        )
    raise ScenarioError(f"unknown send_event type {kind!r}")


def _send_summary(sends: list[RecordedSend]) -> str:
    if not sends:
        return "  (none)"
    return "\n".join(
        f"  [{index}] {send.action} chat={send.chat_id} "
        f"segments={send.segment_types} text={send.text!r}"
        for index, send in enumerate(sends, 1)
    )


def _matches_send(send: RecordedSend, matcher: dict[str, Any]) -> bool:
    if "action" in matcher and send.action != matcher["action"]:
        return False
    if "group_id" in matcher and send.group_id != int(matcher["group_id"]):
        return False
    if "user_id" in matcher and send.user_id != int(matcher["user_id"]):
        return False
    if "chat_id" in matcher and send.chat_id != str(matcher["chat_id"]):
        return False
    if "text_contains" in matcher:
        needles = matcher["text_contains"]
        if isinstance(needles, str):
            needles = [needles]
        if not all(str(needle) in send.text for needle in needles):
            return False
    if "text_equals" in matcher and send.text != str(matcher["text_equals"]):
        return False
    if "segment_types" in matcher and send.segment_types != list(matcher["segment_types"]):
        return False
    if "segment_types_contain" in matcher:
        if not all(item in send.segment_types for item in matcher["segment_types_contain"]):
            return False
    if "at" in matcher and str(matcher["at"]) not in send.at_targets:
        return False
    return True


def _expect_sends(client: MockOneBot, spec: Any, step_no: int) -> None:
    if not isinstance(spec, dict):
        raise ScenarioError(f"step {step_no}: expect must be a mapping")
    sends = list(client.sends)
    problems: list[str] = []
    if spec.get("no_sends") and sends:
        problems.append(f"expected no sends, got {len(sends)}")
    if "send_count" in spec and len(sends) != int(spec["send_count"]):
        problems.append(f"expected send_count={spec['send_count']}, got {len(sends)}")
    used: set[int] = set()
    for matcher in spec.get("sends") or []:
        if not isinstance(matcher, dict):
            raise ScenarioError(f"step {step_no}: each sends matcher must be a mapping")
        match = next(
            (index for index, send in enumerate(sends) if index not in used and _matches_send(send, matcher)),
            None,
        )
        if match is None:
            problems.append(f"no send matched {json.dumps(matcher, ensure_ascii=False)}")
        else:
            used.add(match)
    if problems:
        raise ScenarioError(
            f"step {step_no} expect failed: {'; '.join(problems)}\n"
            f"expected:\n{json.dumps(spec, ensure_ascii=False, indent=2)}\n"
            f"actual sends:\n{_send_summary(sends)}"
        )


def _matches_api(call: ApiCall, matcher: dict[str, Any]) -> bool:
    if call.action != matcher.get("action"):
        return False
    for key, expected in (matcher.get("params") or {}).items():
        if call.params.get(key) != expected:
            return False
    return True


def _expect_api(client: MockOneBot, spec: Any, step_no: int) -> None:
    matchers = spec if isinstance(spec, list) else [spec]
    for matcher in matchers:
        if not isinstance(matcher, dict) or not matcher.get("action"):
            raise ScenarioError(f"step {step_no}: expect_api entries need an action")
    calls = list(client.api_calls)
    used: set[int] = set()
    for matcher in matchers:
        match = next(
            (index for index, call in enumerate(calls) if index not in used and _matches_api(call, matcher)),
            None,
        )
        if match is None:
            actual = ", ".join(
                f"{call.action}({json.dumps(call.params, ensure_ascii=False)})" for call in calls
            ) or "(none)"
            raise ScenarioError(
                f"step {step_no} expect_api failed: no API call matched "
                f"{json.dumps(matcher, ensure_ascii=False)}\n"
                f"expected:\n{json.dumps(matchers, ensure_ascii=False, indent=2)}\n"
                f"actual API calls: {actual}"
            )
        used.add(match)


async def _run_step(client: MockOneBot, step: Any, step_no: int) -> None:
    if not isinstance(step, dict) or len(step) != 1:
        raise ScenarioError(f"step {step_no} must be a single-key mapping, got {step!r}")
    kind, payload = next(iter(step.items()))
    if kind == "sleep":
        await asyncio.sleep(float(payload))
        return
    if kind == "send_event":
        await client.send_event(_expand_event(client, payload))
        return
    if kind == "expect":
        _expect_sends(client, payload, step_no)
        return
    if kind == "expect_api":
        _expect_api(client, payload, step_no)
        return
    raise ScenarioError(f"step {step_no}: unknown step {kind!r}")


async def run_scenario(client: MockOneBot, scenario: dict[str, Any]) -> None:
    steps = scenario.get("steps") or []
    if not isinstance(steps, list):
        raise ScenarioError("scenario 'steps' must be a list")
    for index, step in enumerate(steps, 1):
        await _run_step(client, step, index)


# ── CLI ────────────────────────────────────────────────────────────────


def _int_keyed(mapping: Any) -> dict[int, str]:
    if not isinstance(mapping, dict):
        return {}
    result: dict[int, str] = {}
    for key, value in mapping.items():
        try:
            result[int(key)] = str(value)
        except (TypeError, ValueError):
            continue
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MockOneBot — OneBot v11 client for adapter integration tests",
    )
    parser.add_argument("--adapter-url", default="", help="ws://host:port/onebot (or set adapter_url in the YAML)")
    parser.add_argument("--token", default=None, help="OneBot access_token (overrides the scenario)")
    parser.add_argument("--scenario", required=True, help="scenario YAML file")
    parser.add_argument("--report", default="", help="write a JSON report to this path")
    parser.add_argument("--self-id", default=str(DEFAULT_SELF_ID), help="bot QQ id (default: 10000)")
    parser.add_argument("--timeout", type=float, default=10.0, help="connect timeout in seconds")
    return parser


async def _run_cli(args: argparse.Namespace) -> int:
    scenario_path = Path(args.scenario)
    scenario = load_scenario(scenario_path)
    url = args.adapter_url or str(scenario.get("adapter_url") or "")
    if not url:
        raise ScenarioError("--adapter-url is required (or set adapter_url in the scenario)")
    token = args.token if args.token is not None else str(scenario.get("token") or "")
    client = MockOneBot(
        url,
        token,
        self_id=args.self_id,
        group_names=_int_keyed(scenario.get("group_names")),
        nicknames=_int_keyed(scenario.get("nicknames")),
    )
    started_at = time.time()
    report: dict[str, Any] = {
        "scenario": scenario.get("name") or scenario_path.stem,
        "adapter_url": url,
        "ok": False,
        "failure": None,
        "sends": [],
        "api_calls": [],
        "events_sent": [],
        "started_at": started_at,
    }
    try:
        await client.connect(timeout=args.timeout)
        await run_scenario(client, scenario)
        report["ok"] = True
    except Exception as exc:
        report["failure"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["sends"] = [send.to_dict() for send in client.sends]
        report["api_calls"] = [call.to_dict() for call in client.api_calls]
        report["events_sent"] = client.events_sent
        report["finished_at"] = time.time()
        report["duration_seconds"] = report["finished_at"] - started_at
        if args.report:
            Path(args.report).write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
            )
        await client.close()
    print(
        f"OK: {report['scenario']} — {len(client.sends)} send(s), "
        f"{len(client.api_calls)} API call(s), {len(client.events_sent)} event(s) sent"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return asyncio.run(_run_cli(args))
    except ScenarioError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    except (ConnectionError, TimeoutError, OSError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
