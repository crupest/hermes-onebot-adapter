"""Tests for ``tools/mock_onebot.py`` (library + CLI).

The fake adapter below plays the adapter's role: it pushes OneBot API requests
to MockOneBot and consumes the events MockOneBot injects.  No adapter service
is needed here — these tests pin down the mock's own protocol behaviour, the
scenario engine, and the CLI exit codes / report format.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import aiohttp
import aiohttp.web
import pytest
from aiohttp.test_utils import TestServer
from mock_onebot import (
    MockOneBot,
    ScenarioError,
    at_segment,
    image_segment,
    load_scenario,
    reply_segment,
    run_scenario,
    text_segment,
)

pytestmark = pytest.mark.integration

_ROOT = Path(__file__).resolve().parents[2]
_MOCK_SCRIPT = _ROOT / "tools" / "mock_onebot.py"
_SELFCHECK_SCENARIO = _ROOT / "scenarios" / "mock_selfcheck.yaml"
_SCENARIOS_DIR = _ROOT / "scenarios"


class FakeAdapterServer:
    """A tiny ``/onebot`` server that requests API calls and records events."""

    def __init__(self) -> None:
        self.events: list[dict] = []
        self.api_responses: list[dict] = []
        self.server: TestServer | None = None
        self.url = ""
        self.reply_text = "自动回复"
        self._send_counter = 0

    async def _handler(self, request: aiohttp.web.Request) -> aiohttp.web.WebSocketResponse:
        if request.query.get("access_token") != "tok":
            return aiohttp.web.Response(status=403)
        ws = aiohttp.web.WebSocketResponse()
        await ws.prepare(request)
        await ws.send_json({"action": "get_login_info", "params": {}, "echo": "login-1"})
        async for msg in ws:
            if msg.type != aiohttp.WSMsgType.TEXT:
                continue
            data = json.loads(msg.data)
            if data.get("post_type") == "message":
                self.events.append(data)
                self._send_counter += 1
                await ws.send_json({
                    "action": "send_group_msg",
                    "params": {
                        "group_id": data.get("group_id", 0),
                        "message": [{"type": "text", "data": {"text": self.reply_text}}],
                    },
                    "echo": f"send-{self._send_counter}",
                })
            elif "echo" in data:
                self.api_responses.append(data)
        return ws

    async def __aenter__(self) -> FakeAdapterServer:
        app = aiohttp.web.Application()
        app.router.add_get("/onebot", self._handler)
        self.server = TestServer(app, port=0)
        await self.server.start_server()
        self.url = f"ws://127.0.0.1:{self.server.port}/onebot"
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        assert self.server is not None
        await self.server.close()


def test_checked_in_scenarios_all_load():
    scenario_paths = sorted(_SCENARIOS_DIR.glob("*.yaml"))
    assert scenario_paths, "expected at least one scenario YAML"
    for path in scenario_paths:
        scenario = load_scenario(path)
        assert isinstance(scenario.get("steps"), list), path.name
        assert scenario["steps"], path.name


def test_segment_helpers_build_expected_frames():
    assert text_segment("hi") == {"type": "text", "data": {"text": "hi"}}
    assert at_segment(10001) == {"type": "at", "data": {"qq": "10001"}}
    assert image_segment("http://x/1.png") == {"type": "image", "data": {"file": "http://x/1.png"}}
    assert reply_segment(7) == {"type": "reply", "data": {"id": "7"}}


def test_group_message_event_contains_requested_segments():
    mock = MockOneBot("ws://127.0.0.1:1/onebot", self_id=10000, nicknames={100: "小明"})
    event = mock.build_group_message(
        42, 100, text="hello", at=True, image="http://x/1.png", reply_to=7,
    )
    types = [segment["type"] for segment in event["message"]]
    assert types == ["reply", "at", "text", "image"]
    assert event["post_type"] == "message"
    assert event["message_type"] == "group"
    assert event["group_id"] == 42
    assert event["sender"]["nickname"] == "小明"
    assert mock.inbound_messages[str(event["message_id"])] == event


async def test_mock_answers_api_with_echo_and_records_sends():
    async with FakeAdapterServer() as fake:
        mock = MockOneBot(fake.url, "tok", self_id=10000, group_names={42: "测试群"})
        await mock.connect()
        try:
            await mock.wait_for_api("get_login_info")
            await mock.wait_until(lambda: bool(fake.api_responses), timeout=2.0)
            response = next(item for item in fake.api_responses if item["echo"] == "login-1")
            assert response["status"] == "ok"
            assert response["retcode"] == 0
            assert response["data"]["user_id"] == 10000

            await mock.send_group_message(42, 100, text="你好", at=True)
            send = await mock.wait_for_sends(1)
            assert send.action == "send_group_msg"
            assert send.group_id == 42
            assert send.text == "自动回复"
            assert fake.events[0]["message"][0] == {"type": "at", "data": {"qq": "10000"}}
        finally:
            await mock.close()


async def test_scenario_runner_asserts_api_and_sends():
    async with FakeAdapterServer() as fake:
        scenario = {
            "name": "selfcheck",
            "steps": [
                {
                    "send_event": {
                        "type": "group_message",
                        "group_id": 42,
                        "user_id": 100,
                        "text": "hi",
                        "at": True,
                    },
                },
                {"sleep": 0.3},
                {"expect_api": [{"action": "get_login_info"}]},
                {
                    "expect": {
                        "send_count": 1,
                        "sends": [{"action": "send_group_msg", "group_id": 42, "text_contains": "自动回复"}],
                    },
                },
            ],
        }
        mock = MockOneBot(fake.url, "tok", self_id=10000)
        await mock.connect()
        try:
            await run_scenario(mock, scenario)
        finally:
            await mock.close()


async def test_scenario_failure_prints_expected_vs_actual_diff():
    async with FakeAdapterServer() as fake:
        scenario = {
            "name": "failing",
            "steps": [
                {
                    "send_event": {
                        "type": "group_message",
                        "group_id": 42,
                        "user_id": 100,
                        "text": "hi",
                        "at": True,
                    },
                },
                {"sleep": 0.3},
                {"expect": {"sends": [{"action": "send_group_msg", "text_contains": "不存在的回复"}]}},
            ],
        }
        mock = MockOneBot(fake.url, "tok", self_id=10000)
        await mock.connect()
        try:
            with pytest.raises(ScenarioError) as excinfo:
                await run_scenario(mock, scenario)
        finally:
            await mock.close()

    message = str(excinfo.value)
    assert "expect failed" in message
    assert "expected:" in message
    assert "actual sends:" in message
    assert "自动回复" in message


async def _run_cli(args: list[str]) -> tuple[int, bytes, bytes]:
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        str(_MOCK_SCRIPT),
        *args,
        cwd=str(_ROOT),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=30)
    return process.returncode, stdout, stderr


async def test_cli_runs_scenario_and_writes_report(tmp_path):
    assert _SELFCHECK_SCENARIO.is_file()
    async with FakeAdapterServer() as fake:
        report_path = tmp_path / "report.json"
        code, stdout, stderr = await _run_cli([
            "--adapter-url", fake.url,
            "--token", "tok",
            "--scenario", str(_SELFCHECK_SCENARIO),
            "--report", str(report_path),
        ])
        assert code == 0, f"stdout={stdout!r} stderr={stderr!r}"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        assert report["ok"] is True
        assert report["failure"] is None
        assert report["sends"][0]["action"] == "send_group_msg"
        assert report["sends"][0]["message"][0]["type"] == "text"
        assert any(call["action"] == "get_login_info" for call in report["api_calls"])


async def test_cli_failure_exits_non_zero_and_reports(tmp_path):
    scenario_path = tmp_path / "broken.yaml"
    scenario_path.write_text(
        "name: broken\n"
        "steps:\n"
        "  - send_event: {type: group_message, group_id: 42, user_id: 100, text: hi, at: true}\n"
        "  - sleep: 0.3\n"
        "  - expect:\n"
        "      sends:\n"
        "        - action: send_group_msg\n"
        "          text_contains: 不存在的回复\n",
        encoding="utf-8",
    )
    async with FakeAdapterServer() as fake:
        report_path = tmp_path / "failure-report.json"
        code, stdout, stderr = await _run_cli([
            "--adapter-url", fake.url,
            "--token", "tok",
            "--scenario", str(scenario_path),
            "--report", str(report_path),
        ])
        combined = (stdout + stderr).decode("utf-8", errors="replace")

    assert code != 0
    assert "FAIL" in combined
    assert "actual sends:" in combined
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["ok"] is False
    assert "expect failed" in report["failure"]
