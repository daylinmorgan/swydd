#!/usr/bin/env python3

__import__("sys").path.append("src")

import shutil
import sys

from swydd import asset, cli, get, option, proc, sub, task


@task
def bootstrap():
    """setup swydd dev environment"""
    if not shutil.which("uv"):
        sys.exit("uv necessary for swydd development")
    (proc("uv sync").then("uv run pre-commit install").run())


@task
def tests():
    """run pytest"""
    sub("uv run pytest")


@task
@option("skip-mypy", "skip mypy")
def check(skip_mypy: bool = False):
    """run pre-commit (and mypy)"""
    sub("uv run pre-commit run --all")
    if not skip_mypy:
        sub("uv run mypy src/")


@task
def docs():
    """build docs"""
    tags = get("git tag --list")
    versions = [line for line in tags.splitlines() if line.startswith("v")]
    for tag in versions:
        (
            asset(f"docs/{tag}/swydd.py").write(
                get(f"git show {tag}:src/swydd/__init__.py")
            )
        )
    asset("docs/swydd.py").write(asset("src/swydd/__init__.py"))


cli("check")
