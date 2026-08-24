"""Pull human game steps from the production backend and maintain a cumulative BC dataset.

Pipeline:
  1. SSH to the backend, run dump_human_steps.sql against PostgreSQL (in docker).
  2. Download the raw JSONL locally (kept for re-export with future exporters).
  3. Run export_human_traces.py to rebuild training observations (static + table).
  4. Optionally merge new traces into a cumulative dataset, dedup by (game_id, step).

Credentials:
  - Read from training/.secrets/backend.env if present (SSH_BACKEND_USER /
    SSH_BACKEND_PASSWORD), else from env vars, else prompt.

Usage:
  python scripts/pull_human_data.py                              # one-shot pull
  python scripts/pull_human_data.py --merge-into artifacts/human_traces.jsonl
  python scripts/pull_human_data.py --stats                      # only print DB counts
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ssh_backend import HOST, get_client, run  # noqa: E402

SECRETS_FILE = Path(__file__).resolve().parent.parent / ".secrets" / "backend.env"
CONTAINER = "mahjong-fly-bird-postgres-1"


def load_secrets() -> dict[str, str]:
    secrets: dict[str, str] = {}
    if SECRETS_FILE.exists():
        for line in SECRETS_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                secrets[key.strip()] = value.strip()
    return secrets


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--user", default=os.environ.get("SSH_BACKEND_USER", "ubuntu"))
    parser.add_argument("--password", default=os.environ.get("SSH_BACKEND_PASSWORD", ""))
    parser.add_argument("--out-dir", default="artifacts")
    parser.add_argument("--merge-into", default="", help="Cumulative traces file to merge new traces into (dedup by game_id+step).")
    parser.add_argument("--stats", action="store_true", help="Only print DB counts and exit.")
    args = parser.parse_args()

    secrets = load_secrets()
    if not args.user or args.user == "ubuntu":
        args.user = secrets.get("SSH_BACKEND_USER", args.user)
    if not args.password:
        args.password = secrets.get("SSH_BACKEND_PASSWORD", args.password)

    if not args.password:
        import getpass

        args.password = getpass.getpass(f"Password for {args.user}@{args.host}: ")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    steps_path = out_dir / "human_steps.jsonl"
    traces_path = out_dir / "human_traces_latest.jsonl"

    print(f"[1/3] connecting to {args.user}@{args.host} ...")
    client = get_client(args.user, args.password)

    if args.stats:
        code, out, err = run(
            client,
            (
                f"sudo docker exec {CONTAINER} sh -c "
                "'psql -U \"$POSTGRES_USER\" -d \"$POSTGRES_DB\" -t -A -c "
                "\"SELECT count(*), count(DISTINCT \\\"gameId\\\") FROM \\\"GameStep\\\" WHERE \\\"actionSource\\\"='HUMAN'\"'"
            ),
        )
        print(out or err)
        client.close()
        return code

    print("[2/3] dumping human steps from PostgreSQL ...")
    dump_sql = Path(__file__).resolve().parent / "dump_human_steps.sql"
    sftp = client.open_sftp()
    sftp.put(str(dump_sql), "/tmp/dump_human_steps.sql")
    sftp.close()
    code, out, err = run(
        client,
        (
            f"sudo docker cp /tmp/dump_human_steps.sql {CONTAINER}:/tmp/dump_human_steps.sql && "
            f"sudo docker exec {CONTAINER} sh -c "
            "'psql -U \"$POSTGRES_USER\" -d \"$POSTGRES_DB\" -t -A -f /tmp/dump_human_steps.sql' "
            "> /tmp/human_steps.jsonl && wc -l < /tmp/human_steps.jsonl"
        ),
    )
    if code != 0:
        print(err or out, file=sys.stderr)
        client.close()
        return code
    n_steps = out.strip().splitlines()[-1]
    print(f"    {n_steps} human steps in DB")
    sftp = client.open_sftp()
    sftp.get("/tmp/human_steps.jsonl", str(steps_path))
    sftp.close()
    client.close()
    print(f"    saved -> {steps_path}")

    print("[3/3] exporting training traces ...")
    exporter = Path(__file__).resolve().parent / "export_human_traces.py"
    proc = subprocess.run(
        [sys.executable, str(exporter), "--input", str(steps_path), "--output", str(traces_path)],
        capture_output=True,
        text=True,
    )
    print(proc.stderr.strip() or proc.stdout.strip())
    if proc.returncode != 0:
        return proc.returncode

    new_traces = [json.loads(l) for l in traces_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    if args.merge_into:
        merge_path = Path(args.merge_into)
        existing: list[dict] = []
        if merge_path.exists():
            existing = [json.loads(l) for l in merge_path.read_text(encoding="utf-8").splitlines() if l.strip()]

        seen = {(r["meta"]["game_id"], r["meta"]["step"]) for r in existing}
        added = [r for r in new_traces if (r["meta"]["game_id"], r["meta"]["step"]) not in seen]
        merged = existing + added
        merged.sort(key=lambda r: (r["meta"]["game_id"], r["meta"]["step"]))
        merge_path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in merged) + ("\n" if merged else ""),
            encoding="utf-8",
        )
        games = len({r["meta"]["game_id"] for r in merged})
        print(f"    merged -> {merge_path}: {len(existing)} kept, {len(added)} new, {len(merged)} total ({games} games)")
    else:
        print(f"    saved -> {traces_path} ({len(new_traces)} traces)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
