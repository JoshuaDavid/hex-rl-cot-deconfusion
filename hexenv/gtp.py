"""Minimal GTP client for benzene's mohex binary (used as solver oracle)."""

from __future__ import annotations
import subprocess


class GTPEngine:
    def __init__(self, binary: str, args: list[str] | None = None):
        self.proc = subprocess.Popen(
            [binary] + (args or []),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )

    def send(self, cmd: str) -> str:
        assert self.proc.stdin and self.proc.stdout
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()
        lines = []
        while True:
            line = self.proc.stdout.readline()
            if line == "":
                raise RuntimeError(f"engine died on command: {cmd}")
            line = line.rstrip("\n")
            if line == "" and lines:
                break
            if line:
                lines.append(line)
        resp = "\n".join(lines)
        if resp.startswith("? "):
            raise RuntimeError(f"GTP error for {cmd!r}: {resp[2:]}")
        return resp[2:] if resp.startswith("= ") else resp.lstrip("= ")

    def close(self):
        try:
            self.proc.stdin.write("quit\n")
            self.proc.stdin.flush()
        except Exception:
            pass
        self.proc.terminate()
