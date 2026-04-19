from __future__ import annotations

import sys

try:
    from termcolor import colored
except ImportError:
    def colored(text, color=None, attrs=None, **kwargs):
        return str(text)


def info(msg: str) -> None:
    print(colored(f"[INFO] {msg}", "green"))


def warn(msg: str) -> None:
    print(colored(f"[WARNING] {msg}", "yellow"))


def error(msg: str) -> None:
    print(colored(f"[ERROR] {msg}", "red"), file=sys.stderr)


def success(msg: str) -> None:
    print(colored(f"[SUCCESS] {msg}", "green", attrs=["bold"]))


def phase(msg: str) -> None:
    print(colored(f"[BLASPHEMOUS] {msg}", "cyan", attrs=["bold"]))


def trial_log(msg: str) -> None:
    print(colored(f"  {msg}", "blue"))


def metric(label: str, value: str) -> None:
    print(f"  {label}: {colored(value, 'green')}")
