#!/usr/bin/env python3

__import__("sys").path.append("src")  # noqa

import swydd as s


@s.task
@s.option("types", "also run mypy")
def check(types: bool = False):
    """run pre-commit (and mypy)"""
    s.sh("pre-commit run --all")
    if types:
        s.sh("mypy src/")


s.cli()
