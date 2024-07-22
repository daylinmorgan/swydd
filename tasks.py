#!/usr/bin/env python3

__import__("sys").path.append("src")

import shutil
import sys

from swydd import cli, get, option, path, sub, task


@task
def bootstrap():
    """setup swydd dev environment"""
    if not shutil.which("uv"):
        sys.exit("uv necessary for swydd development")
    sub < "uv sync"
    sub < "uv run pre-commit install"


@task
def tests():
    """run pytest"""
    sub < "uv run pytest"


@task
@option("skip-mypy", "skip mypy")
def check(skip_mypy: bool = False):
    """run pre-commit (and mypy)"""
    sub < "uv run pre-commit run --all"
    if not skip_mypy:
        sub < "uv run mypy src/"


def copy_source():
    tags = get < "git tag --list"
    versions = [line for line in tags.splitlines() if line.startswith("v")]
    for tag in versions:
        (path / f"docs/{tag}/swydd.py") < (
            get < f"git show {tag}:src/swydd/__init__.py"
        )
    (path / "docs/swydd.py") < (path / "src/swydd/__init__.py")


@task
def docs():
    """build docs"""
    copy_source()


cli()
