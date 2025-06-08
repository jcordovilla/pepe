import pytest
from utils.helpers import build_jump_url

pytestmark = [pytest.mark.unit, pytest.mark.utils]

def test_build_jump_url_valid():
    url = build_jump_url(1, 2, 3)
    assert url == "https://discord.com/channels/1/2/3"

@pytest.mark.parametrize("args", [
    {"guild_id": -1, "channel_id": 2, "message_id": 3},
    {"guild_id": 1.5, "channel_id": 2, "message_id": 3},
])
def test_build_jump_url_invalid(args):
    with pytest.raises(ValueError):
        build_jump_url(**args)
