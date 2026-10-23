"""Registration allowlist hides every OneBot tool outside the allowlist."""
from __future__ import annotations

from unittest.mock import MagicMock

from onebot_adapter.hermes_plugin import onebot_tools


def test_allowlist_hides_non_matching_tools(monkeypatch):
    monkeypatch.setattr(onebot_tools, "_load_tool_policies", lambda: {})
    monkeypatch.setattr(
        onebot_tools,
        "_load_tool_allowlist",
        lambda: (
            {"onebot_send_message", "onebot_mark_msg_as_read"},
            ("onebot_get_", "onebot_fetch_", "onebot_nc_get_"),
        ),
    )
    ctx = MagicMock()
    onebot_tools.register_tools(ctx)
    names = {call.kwargs["name"] for call in ctx.register_tool.call_args_list}
    assert "onebot_send_message" in names
    assert "onebot_mark_msg_as_read" in names  # default-hidden, explicitly allowed
    assert "onebot_get_group_info" in names
    assert "onebot_get_recent_contact" in names  # default-hidden, matched by prefix
    assert "onebot_fetch_custom_face" in names
    assert "onebot_nc_get_user_status" in names
    assert "onebot_kick_group_member" not in names
    assert "onebot_mute_group_member" not in names
    assert "onebot_recall_message" not in names
    assert "onebot_upload_image_to_qun_album" not in names
