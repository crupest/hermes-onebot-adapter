"""The api_call allowlist is the hard boundary against terminal bypass."""
from __future__ import annotations

import pytest

from onebot_adapter.relay.hermes_ws import api_action_allowed, api_call_target_allowed


@pytest.mark.parametrize("action", [
    "get_login_info",
    "get_group_list",
    "get_group_member_info",
    "get_group_msg_history",
    "fetch_custom_face",
    "nc_get_user_status",
    "send_msg",
    "send_group_msg",
    "send_private_msg",
    "send_forward_msg",
    "forward_group_single_msg",
    "upload_group_file",
    "upload_private_file",
    "set_msg_emoji_like",
    "mark_msg_as_read",
    "send_like",
    "send_poke",
    "adapter_get_bot_blacklist",
    "set_qq_profile",
    "set_qq_avatar",
    "set_group_card",
])
def test_allowed_actions(action):
    assert api_action_allowed(action) is True


@pytest.mark.parametrize("action", [
    "",
    "set_group_kick",
    "set_group_ban",
    "set_group_whole_ban",
    "set_group_admin",
    "set_group_name",
    "set_group_leave",
    "set_group_add_request",
    "set_friend_add_request",
    "delete_friend",
    "set_group_special_title",
    "set_online_status",
    "delete_msg",
    "delete_group_file",
    "send_group_notice",
    "adapter_edit_bot_blacklist",
])
def test_blocked_actions(action):
    assert api_action_allowed(action) is False


def test_group_card_target_must_be_self():
    assert api_call_target_allowed("set_group_card", {"user_id": 10001}, "10001") is True
    assert api_call_target_allowed("set_group_card", {"user_id": "10001"}, 10001) is True
    assert api_call_target_allowed("set_group_card", {"user_id": 20002}, "10001") is False
    assert api_call_target_allowed("set_group_card", {}, "10001") is False
    assert api_call_target_allowed("set_group_card", {"user_id": 10001}, "") is False
    assert api_call_target_allowed("set_group_card", ["not", "a", "dict"], "10001") is False


def test_profile_actions_need_no_target_guard():
    assert api_call_target_allowed("set_qq_profile", {"nickname": "x"}, "10001") is True
    assert api_call_target_allowed("set_qq_avatar", {"file": "x.png"}, "10001") is True
