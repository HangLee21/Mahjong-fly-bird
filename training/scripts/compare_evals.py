import json

files = [
    "eval_v6_20m_refined.json",
    "eval_v8_10m_1000g.json",
    "eval_v8_20m_refined.json",
    "eval_v8b_15m_refined.json",
    "eval_v9fast_30m_refined.json",
    "eval_v10_20m_refined.json",
]
print(f"{'model':<34}{'games':<6}{'avg_score':<10}{'win':<7}{'deal_in':<8}{'handval':<8}{'big':<6}{'kongC':<7}{'missWin'}")
for f in files:
    try:
        d = json.load(open(r"training/artifacts/" + f, encoding="utf-8"))
        sq = d.get("score_quality", {})
        ar = d.get("action_rates", {})
        dq = d.get("decision_quality", {})
        print(
            f"{f:<34}{d['num_games']:<6}{d['avg_score']:+.4f}    "
            f"{d['win_rate']:.3f}   {d['deal_in_rate']:.3f}   "
            f"{sq.get('avg_hand_value_when_win', 0):.2f}    {sq.get('big_hand_rate', 0):.2f}   "
            f"{ar.get('kong_concealed', 0):.4f}  {dq.get('missed_win_rate', 'n/a')}"
        )
    except Exception as e:
        print(f, "ERR", e)
