from mahjong_ai.eval.play_game import action_text, render_text, tile_text, tiles_text


def test_tile_and_action_text_are_readable():
    assert tile_text(0)
    assert tile_text(18)
    assert tiles_text([0, 9, 27])
    assert action_text(0)
    assert action_text(101)


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
                    "action_text": action_text(8),
                    "fallback_used": False,
                    "reward": 0.1,
                    "draw_into_decision": "起手/首个决策",
                    "transition": {"removed_text": tile_text(8), "added_text": "-"},
                    "before": {
                        "hand_text": tiles_text([0, 1, 8]),
                        "melds": [],
                        "all_melds": [[], [], [], []],
                        "discards": ["-", "-", "-", "-"],
                        "goal": "平胡/顺子",
                        "shanten": 1,
                        "kong_pool": tiles_text([27, 28]),
                        "wall_count": 50,
                        "goal_scores": {"平胡/顺子": 1.0},
                        "last_discard_player": None,
                        "last_discard": "-",
                        "legal_text": [action_text(8)],
                        "scores": [0, 0, 0, 0],
                        "dealer": 0,
                        "current_player": 0,
                        "phase": "discard",
                        "xiaoji_disabled": False,
                        "last_kong_player": None,
                        "last_kong_tile": "-",
                        "last_draw_from_kong": False,
                        "pending": None,
                        "public_events_tail": [],
                    },
                    "after": {"hand_text": tiles_text([0, 1])},
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
    assert "目标:" in text
    assert "选择:" in text
    assert "本次摸入" in text
    assert "杠牌:" in text
