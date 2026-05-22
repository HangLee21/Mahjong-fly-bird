from mahjong_ai.eval.play_game import action_text, render_text, tile_text, tiles_text


def test_tile_and_action_text_are_readable():
    assert tile_text(0) == "1万"
    assert tile_text(18) == "1条"
    assert tiles_text([0, 9, 27]) == "1万 1筒 东"
    assert action_text(0) == "打1万"
    assert action_text(101) == "胡"


def test_render_text_contains_decision_fields():
    text = render_text(
        {
            "model": "m.zip",
            "seed": 1,
            "opponent": "heuristic",
            "deterministic": True,
            "reward_config_loaded": True,
            "records": [
                {
                    "step": 1,
                    "action_text": "打9万",
                    "fallback_used": False,
                    "reward": 0.1,
                    "before": {
                        "hand_text": "1万 2万 9万",
                        "melds": [],
                        "goal": "平胡/顺子",
                        "shanten": 1,
                        "kong_pool": "东 南",
                        "wall_count": 50,
                        "goal_scores": {"平胡/顺子": 1.0},
                        "last_discard_player": None,
                        "last_discard": "-",
                        "legal_text": ["打9万"],
                    },
                    "after": {"hand_text": "1万 2万"},
                }
            ],
            "winner": None,
            "winners": [],
            "draw": True,
            "win_type": None,
            "payer": None,
            "win_points": 0,
            "win_names": [],
            "scores": [0, 0, 0, 0],
        }
    )
    assert "目标: 平胡/顺子" in text
    assert "选择: 打9万" in text
