from mahjong_ai.rules.shanten import best_shanten, effective_tile_count


def test_complete_hand_shanten():
    tiles = [0, 1, 2, 3, 4, 5, 9, 10, 11, 27, 27, 27, 33, 33]
    assert best_shanten(tiles, wildcard_enabled=True) == -1


def test_wildcard_improves_shanten():
    no_wild = [0, 1, 2, 3, 4, 5, 9, 10, 11, 27, 27, 31, 32]
    with_wild = no_wild + [18]
    assert best_shanten(with_wild, wildcard_enabled=True) <= best_shanten(no_wild, wildcard_enabled=True)


def test_effective_tile_count_positive_for_incomplete_hand():
    tiles = [0, 1, 3, 4, 5, 9, 10, 11, 27, 27, 31, 32, 33]
    assert effective_tile_count(tiles, wildcard_enabled=True) > 0

