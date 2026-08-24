"""SSH helper for the mahjong backend (49.232.32.4).

Usage:
  python ssh_backend.py whoami
  python ssh_backend.py "ls -la /root" [--user ubuntu] [--password ...]

Reads password from SSH_BACKEND_PASSWORD env var or --password.
"""
import argparse
import getpass
import os
import sys

import paramiko

HOST = "49.232.32.4"
DEFAULT_USER = "root"
DEFAULT_PORT = 22


def get_client(user: str, password: str | None, key_path: str | None = None) -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    kwargs: dict = {"hostname": HOST, "port": DEFAULT_PORT, "username": user, "timeout": 15}
    if password:
        kwargs["password"] = password
        kwargs["allow_agent"] = False
        kwargs["look_for_keys"] = False
    if key_path:
        kwargs["key_filename"] = key_path
    client.connect(**kwargs)
    return client


def run(client: paramiko.SSHClient, command: str, timeout: int = 60) -> tuple[int, str, str]:
    stdin, stdout, stderr = client.exec_command(command, timeout=timeout, get_pty=True)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    code = stdout.channel.recv_exit_status()
    return code, out, err


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a command on the mahjong backend via SSH.")
    parser.add_argument("command", nargs="*", help="Remote command to run (after optional upload).")
    parser.add_argument("--user", default=os.environ.get("SSH_BACKEND_USER", DEFAULT_USER))
    parser.add_argument("--password", default=os.environ.get("SSH_BACKEND_PASSWORD", ""))
    parser.add_argument("--key", default=os.environ.get("SSH_BACKEND_KEY", ""))
    parser.add_argument("--upload", default="", help="Local file to upload (SFTP).")
    parser.add_argument("--remote-path", default="/tmp/uploaded_file", help="Remote destination for --upload.")
    parser.add_argument("--download", default="", help="Remote file to download (SFTP).")
    parser.add_argument("--local-path", default="", help="Local destination for --download.")
    args = parser.parse_args()

    password = args.password or getpass.getpass(f"Password for {args.user}@{HOST}: ")
    client = get_client(args.user, password, args.key or None)
    if args.upload:
        sftp = client.open_sftp()
        sftp.put(args.upload, args.remote_path)
        sftp.close()
        print(f"uploaded {args.upload} -> {args.remote_path}")
    if args.download:
        sftp = client.open_sftp()
        local = args.local_path or Path(args.download).name
        sftp.get(args.download, local)
        sftp.close()
        print(f"downloaded {args.download} -> {local}")
    if args.command:
        code, out, err = run(client, " ".join(args.command))
        if out:
            print(out)
        if err:
            print(err, file=sys.stderr)
        client.close()
        return code
    client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
