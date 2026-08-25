import json, glob, os

files = [
    "artifacts/reports/eval_heuristic_baseline.json",
    "artifacts/reports/eval_v3_lite_bc_finetune_999996_heuristic.json",
    "artifacts/reports/eval_v3_lite_bc_finetune_3999984_heuristic.json",
    "artifacts/reports/eval_v3_lite_bc_finetune_5999976_heuristic.json",
    "artifacts/reports/eval_v3_full_action_scorer_9999960_heuristic.json",
    "artifacts/reports/eval_v2_honor_fix_2400w_heuristic.json",
    "artifacts/reports/eval_v2_gpu_stable_1900w_heuristic.json",
    "artifacts/reports/eval_v2_1500w_heuristic.json",
    "artifacts/reports/eval_v25_bc_finetune_200w_heuristic.json",
    "artifacts/reports/eval_v26_scratch_800w_heuristic.json",
    "artifacts/reports/eval_v27_taatsu_800w_heuristic.json",
    "artifacts/eval_mahjong_attention_v1.json",
]
print(f"{'report':55s} win  score  deal_in kong   steps games")
for f in files:
    if not os.path.exists(f):
        print(f"{os.path.basename(f)}: MISSING")
        continue
    r = json.load(open(f, encoding="utf-8"))
    print(
        f"{os.path.basename(f):55s} {r.get('win_rate', 0):.3f} {r.get('avg_score', 0):6.2f} "
        f"{r.get('deal_in_rate', 0):.3f} {r.get('action_rates', {}).get('kong', 0):.3f} "
        f"{r.get('avg_steps', 0):5.1f} {r.get('num_games', 0)}"
    )
print("---best_report.json files---")
for bf in sorted(glob.glob("artifacts/checkpoints/*/best_report.json")):
    r = json.load(open(bf, encoding="utf-8"))
    print(
        f"{bf:78s} win={r.get('win_rate', 0):.3f} score={r.get('avg_score', 0):6.2f} "
        f"deal_in={r.get('deal_in_rate', 0):.3f}"
    )
