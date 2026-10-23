"""Harness for adapter-level integration tests.

Each harness starts a real in-process :class:`AdapterService` (no subprocess,
no WebUI) on two random loopback ports:

* ``/onebot`` — :class:`MockOneBot` plays NapCat: it injects events and
  answers the adapter's OneBot API calls on the same bidirectional socket;
* ``/hermes`` — :class:`FakeHermesPlugin` plays the Hermes plugin: it receives
  normalized events, acks them, and can send replies or stay silent.

No external network, no QQ, no NapCat.  Ports are always ``0`` (kernel-assigned)
so parallel/local runs never collide.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiohttp
import pytest
from aiohttp.test_utils import TestServer

_ROOT = Path(__file__).resolve().parents[2]
_TOOLS = _ROOT / "tools"
for _path in (str(_ROOT), str(_TOOLS)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from fake_hermes_plugin import FakeHermesPlugin  # noqa: E402
from mock_onebot import MockOneBot  # noqa: E402

from onebot_adapter.app import AdapterService  # noqa: E402
from onebot_adapter.config import AdapterConfig, ConfigStore  # noqa: E402

SELF_ID = "10000"
GROUP_ID = 424242
USER_ID = 10001
OTHER_USER_ID = 10002
ONEBOT_TOKEN = "integration-onebot-token"
HERMES_TOKEN = "integration-hermes-token"


@dataclass
class AdapterHarness:
    """A running AdapterService plus helpers for both WebSocket peers."""

    service: AdapterService
    onebot_url: str
    hermes_url: str
    servers: list[TestServer] = field(default_factory=list)
    clients: list[Any] = field(default_factory=list)
    stopped: bool = False

    async def new_mock(self, **kwargs: Any) -> MockOneBot:
        mock = MockOneBot(
            self.onebot_url,
            ONEBOT_TOKEN,
            self_id=SELF_ID,
            group_names={GROUP_ID: "集成测试群"},
            nicknames={USER_ID: "小明", OTHER_USER_ID: "小红"},
            **kwargs,
        )
        await mock.connect()
        self.clients.append(mock)
        return mock

    async def new_plugin(self, **kwargs: Any) -> FakeHermesPlugin:
        plugin = FakeHermesPlugin(self.hermes_url, HERMES_TOKEN, **kwargs)
        await plugin.connect()
        self.clients.append(plugin)
        return plugin

    async def stop(self) -> None:
        if self.stopped:
            return
        self.stopped = True
        for client in self.clients:
            try:
                await client.close()
            except Exception:  # pragma: no cover - teardown must not mask failures
                pass
        # Closing the Hermes app runs AdapterService._on_hermes_cleanup, which
        # stops the relay + OneBot reverse server and closes the HTTP session.
        for server in reversed(self.servers):
            try:
                await server.close()
            except Exception:  # pragma: no cover - teardown must not mask failures
                pass


@pytest.fixture
async def adapter_factory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Build independent AdapterService instances with random ports."""
    monkeypatch.setattr(
        "onebot_adapter.hermes_config.materialize_channel_prompts",
        lambda *args, **kwargs: None,
    )
    created: list[AdapterHarness] = []
    counter = {"value": 0}

    async def make(**overrides: Any) -> AdapterHarness:
        counter["value"] += 1
        monkeypatch.setenv(
            "ONEBOT_ADAPTER_CONFIG", str(tmp_path / f"adapter-config-{counter['value']}.json"),
        )
        config_values: dict[str, Any] = {
            "self_id": SELF_ID,
            "onebot_ws_token": ONEBOT_TOKEN,
            "hermes_ws_token": HERMES_TOKEN,
            "onebot_mode": "reverse",
            "group_require_mention": True,
            "dm_policy": "allow",
            "log_file_enabled": False,
            "usage_stats_enabled": False,
            "reaction_emoji_enabled": False,
            "bot_blacklist_enabled": False,
            "event_queue_enabled": False,
        }
        config_values.update(overrides)
        store = ConfigStore(AdapterConfig(**config_values))
        service = AdapterService(store)
        service._session = aiohttp.ClientSession()
        service._init_components()

        onebot_server = TestServer(service.build_onebot_app(), port=0)
        hermes_server = TestServer(service.build_hermes_app(), port=0)
        try:
            await onebot_server.start_server()
            await hermes_server.start_server()
        except Exception:
            await onebot_server.close()
            await hermes_server.close()
            await service._session.close()
            raise

        harness = AdapterHarness(
            service=service,
            onebot_url=f"ws://127.0.0.1:{onebot_server.port}/onebot",
            hermes_url=f"ws://127.0.0.1:{hermes_server.port}/hermes",
            servers=[onebot_server, hermes_server],
        )
        created.append(harness)
        return harness

    yield make

    for harness in reversed(created):
        await harness.stop()


@pytest.fixture
async def adapter(adapter_factory) -> AdapterHarness:
    """A default AdapterService using the production window defaults."""
    return await adapter_factory()
