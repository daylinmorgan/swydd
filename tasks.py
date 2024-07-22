#!/usr/bin/env python3

__import__("sys").path.append("src")  # noqa

import shutil
import sys
from pathlib import Path

import swydd as s


@s.task
def bootstrap():
    """setup swydd dev environment"""
    if not shutil.which("pdm"):
        sys.exit("pdm necessary for swydd development")
    s.sh("pdm install")
    s.sh("pdm run pre-commit install")


@s.task
@s.option("no-mypy", "skip mypy")
def check(no_mypy: bool = False):
    """run pre-commit (and mypy)"""
    s.sh("pre-commit run --all")
    if not no_mypy:
        s.sh("mypy src/")


def write_docs_src(tag):
    src_text = s.Exec(f"git show {tag}:src/swydd/__init__.py", output=True).get().stdout
    (verdir := (Path(__file__).parent / "docs" / tag)).mkdir(exist_ok=True)
    (verdir / "swydd.py").write_text(src_text)


def copy_source():
    p = s.Exec("git tag --list", output=True).get()
    versions = [line for line in p.stdout.splitlines() if line.startswith("v")]
    for ver in versions:
        write_docs_src(ver)
    shutil.copyfile(
        Path(__file__).parent / "src" / "swydd" / "__init__.py",
        Path(__file__).parent / "docs" / "swydd.py",
    )


@s.task
def docs():
    """build docs"""
    copy_source()


s.cli()
