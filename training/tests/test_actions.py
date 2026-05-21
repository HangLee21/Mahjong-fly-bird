import pytest

from mahjong_ai.env.actions import (
    ACTION_PASS,
    ACTION_SPACE_SIZE,
    ACTION_WIN,
    MahjongAction,
    build_action_mask,
    decode_action,
    encode_action,
    fallback_action,
)


def test_action_roundtrip():
    for tile in (0, 18, 33):
        action = MahjongAction("discard", tile=tile)
        assert decode_action(encode_action(action)) == action
    assert decode_action(encode_action(MahjongAction("win"))).type == "win"


def test_action_mask_and_fallback():
    mask = build_action_mask([0, ACTION_PASS, ACTION_WIN])
    assert mask.shape == (ACTION_SPACE_SIZE,)
    assert mask[ACTION_WIN]
    assert fallback_action([0, ACTION_WIN]) == ACTION_WIN
    assert fallback_action([0, ACTION_PASS]) == ACTION_PASS


def test_invalid_action():
    with pytest.raises(ValueError):
        decode_action(999)
    with pytest.raises(ValueError):
        build_action_mask([ACTION_SPACE_SIZE])

