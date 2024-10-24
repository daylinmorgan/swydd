#!/usr/bin/env python3

__import__("sys").path.append("src")

import shutil
import sys

import swydd as s


@s.task
def bootstrap():
    """setup swydd dev environment"""
    if not shutil.which("uv"):
        sys.exit("uv necessary for swydd development")
    (s.Proc("uv sync").then("uv run pre-commit install").run())


@s.task
def tests():
    """run pytest"""
    s.sub("uv run pytest")


@s.task
@s.option("skip-mypy", "skip mypy")
def check(skip_mypy: bool = False):
    """run pre-commit (and mypy)"""
    s.sub("uv run pre-commit run --all")
    if not skip_mypy:
        s.sub("uv run mypy src/")


@s.task
def serve_docs():
    """serve the docs with sphinx-autobuild"""
    s.sub("uv run sphinx-autobuild docs/source docs/build --watch src", rest=True)


@s.task
def build_docs():
    """build docs"""

    s.sub("uv run sphinx-build docs/source docs/build")

    tags = s.get("git tag --list")
    versions = [line for line in tags.splitlines() if line.startswith("v")]
    for tag in versions:
        version_swydd_src = s.get(f"git show {tag}:src/swydd/__init__.py")
        s.Asset(f"docs/build/{tag}/swydd.py").write(version_swydd_src)
    s.Asset("docs/build/swydd.py").copy(s.Asset("src/swydd/__init__.py"))


s.cli("check --skip-mypy")
