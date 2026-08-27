#!/usr/bin/env python3
"""Run an arbitrary SQL file against the production backend PostgreSQL.

Usage:
  python scripts/run_db_sql.py scripts/check_rooms.sql
  python scripts/run_db_sql.py scripts/check_db.sql

Reads SSH credentials from training/.secrets/backend.env and runs the SQL via
the postgres docker container (same mechanism as pull_human_data.py).
"""
from __future__ import annotations

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
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    sql_path = Path(sys.argv[1]).resolve()
    if not sql_path.exists():
        print(f"SQL file not found: {sql_path}", file=sys.stderr)
        return 1

    secrets = load_secrets()
    user = secrets.get("SSH_BACKEND_USER", "ubuntu")
    password = secrets.get("SSH_BACKEND_PASSWORD", "")

    client = get_client(user, password)
    remote = f"/tmp/{sql_path.name}"
    sftp = client.open_sftp()
    sftp.put(str(sql_path), remote)
    sftp.close()

    code, out, err = run(
        client,
        (
            f"sudo docker cp {remote} {CONTAINER}:/tmp/{sql_path.name} && "
            f"sudo docker exec {CONTAINER} sh -c "
            f"'psql -U \"$POSTGRES_USER\" -d \"$POSTGRES_DB\" -f /tmp/{sql_path.name}'"
        ),
    )
    print(out or err)
    client.close()
    return code


if __name__ == "__main__":
    sys.exit(main())
