"""Fake Hermes plugin endpoint client for adapter integration tests.

The real plugin lives inside the Hermes gateway process and speaks the
adapter's ``/hermes`` WebSocket protocol.  This stand-in connects, receives
``ready`` / ``event`` frames, acknowledges events, and lets a test script the
plugin side:

* send a normal reply (``send_text`` / ``send_action``);
* stay silent (NO_REPLY — the default: connecting and acking only);
* report Hermes mode / commands / plugin info like the real plugin does.

It is a test helper, not production code.
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any
from urllib.parse import quote

import aiohttp

from onebot_adapter.relay.protocol import (
    api_call_message,
    envelope,
    hermes_mode_report_message,
    pong_message,
    send_message,
)

DEFAULT_COMMANDS: list[dict[str, Any]] = [
    {"name": "reset", "description": "reset the session", "aliases": []},
    {"name": "new", "description": "start a new session", "aliases": []},
    {"name": "stop", "description": "stop the running turn", "aliases": []},
    {"name": "clean", "description": "clear the adapter queue", "aliases": []},
]


class FakeHermesPlugin:
    """Minimal ``/hermes`` plugin-side client."""

    def __init__(
        self,
        hermes_url: str,
        token: str = "",
        *,
        auto_ack: bool = True,
        group_sessions_per_user: bool = True,
        commands: list[dict[str, Any]] | None = None,
        plugin_version: str = "0.0.0-test",
        request_timeout: float = 5.0,
    ) -> None:
        self.hermes_url = hermes_url
        self.token = token
        self.auto_ack = auto_ack
        self.group_sessions_per_user = group_sessions_per_user
        self.commands = list(DEFAULT_COMMANDS if commands is None else commands)
        self.plugin_version = plugin_version
        self.request_timeout = request_timeout

        self.ready: dict[str, Any] | None = None
        self.events: list[dict[str, Any]] = []
        self.frames: list[dict[str, Any]] = []

        self._session: aiohttp.ClientSession | None = None
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._ready_event = asyncio.Event()
        self._pending: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self._send_lock = asyncio.Lock()

    # ── Connection lifecycle ────────────────────────────────────────────

    @property
    def connected(self) -> bool:
        return self._ws is not None and not self._ws.closed

    async def connect(self, timeout: float = 10.0) -> FakeHermesPlugin:
        self._session = aiohttp.ClientSession()
        url = self.hermes_url
        if self.token:
            separator = "&" if "?" in url else "?"
            url = f"{url}{separator}token={quote(self.token)}"
        try:
            self._ws = await asyncio.wait_for(self._session.ws_connect(url), timeout=timeout)
        except Exception:
            await self._session.close()
            self._session = None
            raise
        self._reader_task = asyncio.create_task(self._read_loop(), name="fake-hermes-plugin-reader")
        try:
            await asyncio.wait_for(self._ready_event.wait(), timeout=timeout)
        except Exception:
            await self.close()
            raise
        await self.send_frame(hermes_mode_report_message(self.group_sessions_per_user))
        await self.send_frame(self._commands_frame())
        await self.send_frame(envelope("plugin_info", plugin_version=self.plugin_version))
        return self

    async def close(self) -> None:
        reader = self._reader_task
        self._reader_task = None
        if reader is not None:
            reader.cancel()
            await asyncio.gather(reader, return_exceptions=True)
        for future in self._pending.values():
            if not future.done():
                future.cancel()
        self._pending.clear()
        ws = self._ws
        self._ws = None
        if ws is not None and not ws.closed:
            await ws.close()
        session = self._session
        self._session = None
        if session is not None and not session.closed:
            await session.close()

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
            for future in self._pending.values():
                if not future.done():
                    future.set_exception(ConnectionError("fake plugin WebSocket closed"))
            self._pending.clear()

    async def _on_frame(self, raw: str) -> None:
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return
        if not isinstance(data, dict):
            return
        self.frames.append(data)
        kind = data.get("type")
        if kind == "ready":
            self.ready = data
            self._ready_event.set()
        elif kind == "event":
            self.events.append(data)
            if self.auto_ack:
                await self.send_frame(self._ack_frame(data))
        elif kind == "result":
            future = self._pending.pop(str(data.get("req_id", "")), None)
            if future is not None and not future.done():
                future.set_result(data)
        elif kind == "ping":
            await self.send_frame(pong_message())
        elif kind == "commands_refresh":
            await self.send_frame(self._commands_frame())
        elif kind == "mode_refresh":
            await self.send_frame(hermes_mode_report_message(self.group_sessions_per_user))

    def _commands_frame(self) -> dict[str, Any]:
        return envelope("commands_snapshot", commands=self.commands)

    @staticmethod
    def _ack_frame(event: dict[str, Any]) -> dict[str, Any]:
        delivery_ids = event.get("delivery_ids") or []
        if not delivery_ids and event.get("delivery_id"):
            delivery_ids = [event["delivery_id"]]
        return envelope(
            "event_ack",
            delivery_id=event.get("delivery_id", ""),
            delivery_ids=list(delivery_ids),
        )

    # ── Sending frames ──────────────────────────────────────────────────

    async def send_frame(self, frame: dict[str, Any]) -> None:
        ws = self._ws
        if ws is None or ws.closed:
            raise ConnectionError("FakeHermesPlugin is not connected")
        async with self._send_lock:
            await ws.send_json(frame)

    async def send_action(
        self,
        action: str,
        chat_id: str,
        payload: dict[str, Any] | None = None,
        *,
        wait_result: bool = True,
        timeout: float | None = None,
    ) -> dict[str, Any] | None:
        req_id = uuid.uuid4().hex
        frame = send_message(action, req_id, chat_id, **(payload or {}))  # type: ignore[arg-type]
        future: asyncio.Future[dict[str, Any]] | None = None
        if wait_result:
            future = asyncio.get_running_loop().create_future()
            self._pending[req_id] = future
        await self.send_frame(frame)
        if future is None:
            return None
        return await asyncio.wait_for(future, timeout=timeout or self.request_timeout)

    async def send_text(
        self,
        chat_id: str,
        content: str,
        *,
        reply_to: int | str | None = None,
        wait_result: bool = True,
        timeout: float | None = None,
    ) -> dict[str, Any] | None:
        payload: dict[str, Any] = {"content": content}
        if reply_to is not None:
            payload["reply_to"] = reply_to
        return await self.send_action(
            "send_text", chat_id, payload, wait_result=wait_result, timeout=timeout,
        )

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        *,
        caption: str = "",
        wait_result: bool = True,
    ) -> dict[str, Any] | None:
        payload: dict[str, Any] = {"image_url": image_url}
        if caption:
            payload["caption"] = caption
        return await self.send_action("send_image", chat_id, payload, wait_result=wait_result)

    async def call_api(
        self,
        action: str,
        params: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> Any:
        req_id = uuid.uuid4().hex
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending[req_id] = future
        await self.send_frame(api_call_message(action, req_id, params or {}))
        result = await asyncio.wait_for(future, timeout=timeout or self.request_timeout)
        if not result.get("success"):
            raise RuntimeError(f"api_call {action} failed: {result.get('error')}")
        return result.get("data")

    async def ping(self, *, timeout: float = 2.0) -> dict[str, Any]:
        frame = envelope("ping")
        await self.send_frame(frame)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for candidate in reversed(self.frames):
                if candidate.get("type") == "pong":
                    return candidate
            await asyncio.sleep(0.02)
        raise AssertionError("timed out waiting for pong")

    # ── Event waiting / assertions ──────────────────────────────────────

    @property
    def event_texts(self) -> list[str]:
        return [str(event.get("text", "")) for event in self.events]

    async def wait_event(self, count: int = 1, *, timeout: float = 3.0) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while len(self.events) < count:
            if time.monotonic() >= deadline:
                raise AssertionError(
                    f"timed out waiting for event #{count}; got {len(self.events)}: {self.event_texts}"
                )
            await asyncio.sleep(0.02)
        return self.events[count - 1]
