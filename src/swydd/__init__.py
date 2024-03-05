import argparse
import inspect
import shlex
import subprocess
import sys
from argparse import (
    Action,
    ArgumentParser,
    RawDescriptionHelpFormatter,
    _SubParsersAction,
)
from inspect import Parameter
from typing import Any, Callable, Dict, List, Optional, Tuple

__version__ = "0.1.0"


class Context:
    def __init__(self) -> None:
        self.show_targets = True
        self.dag = False
        self.dry = False
        self.tasks: Dict[str, Any] = {}
        self.targets: Dict[str, Any] = {}
        self.data: Any = None
        self.flags: Dict[str, Any] = {}
        self._flag_defs: List[Tuple[Tuple[str, ...], Any]] = []
        self.verbose = False

    def add_task(
        self, func: Callable[..., Any], help: Optional[Dict[str, str]] = None
    ) -> None:
        name = func.__name__
        if name == "inner":
            return
        if name in self.tasks:
            raise ValueError(f"{name} task is repeated.")
        else:
            self.tasks[name] = dict(
                func=func, signature=inspect.signature(func), help=help
            )

    def add_flag(self, *args: str, **kwargs: Any) -> None:
        name = max(args, key=len).split("-")[-1]
        self.flags[name] = None
        self._flag_defs.append((args, kwargs))


ctx = Context()


class Exec:
    def __init__(self, cmd: str, shell: bool = False) -> None:
        self.shell = shell
        self.cmd = cmd

    def execute(self) -> int:
        if ctx.verbose:
            sys.stdout.write(f"exec: {self.cmd}\n")
        if self.shell:
            return subprocess.run(self.cmd, shell=True).returncode
        else:
            return subprocess.run(shlex.split(self.cmd)).returncode


def sh(cmd: str, shell: bool = False) -> int:
    return Exec(cmd, shell=shell).execute()


class SubcommandHelpFormatter(RawDescriptionHelpFormatter):
    """custom help formatter to remove bracketed list of subparsers"""

    def _format_action(self, action: Action) -> str:
        # TODO: actually modify the real "format_action for better control"
        parts = super(RawDescriptionHelpFormatter, self)._format_action(action)
        if action.nargs == argparse.PARSER:
            lines = parts.split("\n")[1:]
            tasks, targets = [], []
            for line in lines:
                if len(line) > 0 and line.strip().split()[0] in ctx.targets:
                    targets.append(line)
                else:
                    tasks.append(line)
            parts = "\n".join(tasks)
            if len(targets) > 0 and ctx.show_targets:
                parts += "\n".join(("\ntargets:", *targets))

        return parts


ctx = Context()


def task(func: Callable[..., Any]) -> Callable[..., None]:
    ctx.add_task(func)

    def wrap(*args: Any, **kwargs: Any) -> None:
        return func(*args, **kwargs)

    return wrap


def targets(
    *args: str,
) -> Callable[[Callable[..., Any]], Callable[..., Callable[..., None]]]:
    def wrapper(func: Callable[..., Any]) -> Callable[..., Callable[..., None]]:
        for arg in args:
            ctx.targets[arg] = func

        def inner(*args: Any, **kwargs: Any) -> Callable[..., None]:
            return func(*args, **kwargs)

        return inner

    return wrapper


def help(
    **help_kwargs: str,
) -> Callable[[Callable[..., Any]], Callable[..., Callable[..., None]]]:
    def wrapper(func: Callable[..., Any]) -> Callable[..., Callable[..., None]]:
        ctx.add_task(func, help=help_kwargs)

        def inner(*args: Any, **kwargs: Any) -> Callable[..., None]:
            return func(*args, **kwargs)

        return inner

    return wrapper


def manage(version: bool = False) -> None:
    """manage self"""
    print("self management ey")
    if version:
        print("current version", __version__)


def generate_subparser(
    shared: ArgumentParser,
    subparsers: _SubParsersAction,
    name: str,
    info: Dict[str, Any],
) -> ArgumentParser:
    func = info["func"]
    signature = info["signature"]
    help = info.get("help")
    doc = func.__doc__.splitlines()[0] if func.__doc__ else ""
    subparser = subparsers.add_parser(
        name, help=doc, description=func.__doc__, parents=[shared]
    )
    for name, param in signature.parameters.items():
        args = (f"--{name}",)
        kwargs = {"help": help.get(name, "")} if help else {}

        if param.annotation == bool:
            kwargs.update({"default": False, "action": "store_true"})
        elif param.annotation != Parameter.empty:
            kwargs.update({"type": param.annotation})
        kwargs.update(
            {"required": True}
            if param.default == Parameter.empty
            else {"default": param.default}
        )

        subparser.add_argument(*args, **kwargs)
    subparser.set_defaults(func=func)
    return subparser


def add_targets(
    parent: ArgumentParser, subparsers: _SubParsersAction, ctx: Context
) -> None:
    for target, target_func in ctx.targets.items():
        subp = generate_subparser(
            parent,
            subparsers,
            target,
            dict(func=target_func, signature=inspect.signature(target_func)),
        )
        subp.add_argument("--dag", help="show target dag", action="store_true")


def cli() -> None:
    parser = ArgumentParser(formatter_class=SubcommandHelpFormatter)
    shared = ArgumentParser(add_help=False)

    for flag_args, flag_kwargs in ctx._flag_defs:
        shared.add_argument(*flag_args, **flag_kwargs)

    shared.add_argument("--verbose", help="use verbose output", action="store_true")
    shared.add_argument(
        "-n", "--dry-run", help="don't execute tasks", action="store_true"
    )

    subparsers = parser.add_subparsers(
        title="tasks",
        required=True,
    )

    if len(sys.argv) > 1 and sys.argv[1] == "self":
        generate_subparser(
            shared,
            subparsers,
            "self",
            dict(func=manage, signature=inspect.signature(manage)),
        )

    add_targets(shared, subparsers, ctx)

    for name, info in ctx.tasks.items():
        generate_subparser(shared, subparsers, name, info)

    args = vars(parser.parse_args())
    ctx.verbose = args.pop("verbose", False)
    ctx.dry = args.pop("dry_run", False)
    ctx.dag = args.pop("dag", False)
    for name in ctx.flags:
        ctx.flags[name] = args.pop(name)

    if f := args.pop("func", None):
        if ctx.dry:
            print("dry run >>>")
            print("  args:", args)
            print(
                "\n".join(
                    f"  {line}"
                    for line in inspect.getsource(f).splitlines()
                    if not line.startswith("@")
                )
            )
        else:
            f(**args)


if __name__ == "__main__":
    sys.stderr.write("this module should not be invoked directly\n")
    sys.exit(1)
