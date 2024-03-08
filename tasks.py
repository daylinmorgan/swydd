#!/usr/bin/env python3

__import__("sys").path.append("src")  # noqa

import shutil
import sys

import swydd as s


@s.task
def bootstrap():
    """setup swydd dev environment"""
    if not shutil.which("pdm"):
        sys.exit("pdm necessary for swydd development")
    s.sh("pdm install")
    s.sh("pdm run pre-commit install")


@s.task
@s.option("types", "also run mypy")
def check(types: bool = False):
    """run pre-commit (and mypy)"""
    s.sh("pre-commit run --all")
    if types:
        s.sh("mypy src/")


s.cli()
