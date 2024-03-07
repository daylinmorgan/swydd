import argparse
import inspect
import os
import shlex
import subprocess
import sys
from argparse import (
    Action,
    ArgumentParser,
    RawDescriptionHelpFormatter,
    _SubParsersAction,
)
from functools import wraps
from inspect import Parameter
from typing import Any, Callable, Dict, List, Optional, Tuple

__version__ = "0.1.0"


class SubcommandHelpFormatter(RawDescriptionHelpFormatter):
    """custom help formatter to remove bracketed list of subparsers"""

    def _format_action(self, action: Action) -> str:
        # TODO: actually modify the real "format_action for better control"
        print(action)
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


def _id_from_func(f: Callable[..., Any]):
    return str(id(wrapped) if (wrapped := getattr(f, "__wrapped__", None)) else id(f))


class Task:
    def __init__(self, func=Callable[..., Any]) -> None:
        self.show = False
        self.id = _id_from_func(func)
        self.name = func.__name__
        self.func = func
        self.targets = []

        self._process_signature()

    def _process_signature(self) -> None:
        self.signature = inspect.signature(self.func)
        self.params = {}
        for name, param in self.signature.parameters.items():
            self.params[name] = {"Parameter": param}

    def _update_option(self, name: str, help: str, **kwargs) -> None:
        self.params[name] = {
            **self.params.get(name, {}),
            "help": help,
            "kwargs": kwargs,
        }

    def _mark(self) -> None:
        self.show = True


class Context:
    def __init__(self) -> None:
        self._tasks: Dict[str, Any] = {}
        self.targets: Dict[str, Any] = {}
        self.data: Any = None
        self.flags: Dict[str, Any] = {}
        self._flag_defs: List[Tuple[Tuple[str, ...], Any]] = []
        self.show_targets = True

        # global flags
        self.dry = False
        self.dag = False
        self.verbose = False

    def _add_task(self, func: Callable[..., Any], show: bool = False) -> None:
        if (id_ := _id_from_func(func)) not in self._tasks:
            self._tasks[id_] = Task(func)
        if show:
            self._tasks[id_]._mark()

    def _update_option(self, func: Callable[..., Any], name: str, help: str, **kwargs):
        if (id_ := _id_from_func(func)) not in self._tasks:
            raise ValueError
        self._tasks[id_]._update_option(name, help, **kwargs)

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


# decorators


def task(func: Callable[..., Any]) -> Callable[..., None]:
    ctx._add_task(func, show=True)

    def wrap(*args: Any, **kwargs: Any) -> None:
        return func(*args, **kwargs)

    return wrap


# def inspect_wrapper(place, func):
#     if wrapped := getattr(func, "__wrapped__", None):
#         print(place, "wrapped->", id(wrapped))
#
#     print(
#         place,
#         id(func),
#     )
#


def targets(
    *args: str,
) -> Callable[[Callable[..., Any]], Callable[..., Callable[..., None]]]:
    def wrapper(func: Callable[..., Any]) -> Callable[..., Callable[..., None]]:
        ctx._add_task(func)
        for arg in args:
            ctx.targets[arg] = _id_from_func(func)

        @wraps(func)
        def inner(*args: Any, **kwargs: Any) -> Callable[..., None]:
            return func(*args, **kwargs)

        return inner

    return wrapper


def option(
    name: str,
    help: str,
    **help_kwargs: str,
) -> Callable[[Callable[..., Any]], Callable[..., Callable[..., None]]]:
    def wrapper(func: Callable[..., Any]) -> Callable[..., Callable[..., None]]:
        ctx._add_task(func)
        ctx._update_option(func, name, help, **help_kwargs)

        @wraps(func)
        def inner(*args: Any, **kwargs: Any) -> Callable[..., None]:
            return func(*args, **kwargs)

        return inner

    return wrapper


def manage(version: bool = False) -> None:
    """manage self"""
    print("self management stuff")
    if version:
        print("current version", __version__)


def generate_task_subparser(
    shared: ArgumentParser,
    subparsers: _SubParsersAction,
    task: Task,
    target: Optional[str] = None,
) -> Optional[ArgumentParser]:
    if not task.show and not target:
        return

    prog = os.path.basename(sys.argv[0])
    name = task.name if not target else target
    doc = task.func.__doc__.splitlines()[0] if task.func.__doc__ else ""
    subparser = subparsers.add_parser(
        name,
        help=doc,
        description=task.func.__doc__,
        parents=[shared],
        usage=f"%(prog)s {name} [opts]",
        prog=prog,
    )
    for name, info in task.params.items():
        param = info.get("Parameter")  # must check signature for args?
        args = (f"--{name}",)
        kwargs = {"help": info.get("help", "")}

        if param.annotation == bool:
            kwargs.update({"default": False, "action": "store_true"})
        elif param.annotation != Parameter.empty:
            kwargs.update({"type": param.annotation})
        kwargs.update(
            {"required": True}
            if param.default == Parameter.empty
            else {"default": param.default}
        )

        kwargs.update(info.get("kwargs", {}))
        subparser.add_argument(*args, **kwargs)
    subparser.set_defaults(func=task.func)
    return subparser


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
    shared: ArgumentParser, subparsers: _SubParsersAction, ctx: Context
) -> None:
    for target, id_ in ctx.targets.items():
        subp = generate_task_subparser(shared, subparsers, ctx._tasks[id_], str(target))
        if subp:
            subp.add_argument("--dag", help="show target dag", action="store_true")


def cli() -> None:
    parser = ArgumentParser(
        formatter_class=SubcommandHelpFormatter, usage="%(prog)s <task/target> [opts]"
    )
    shared = ArgumentParser(add_help=False)

    for flag_args, flag_kwargs in ctx._flag_defs:
        shared.add_argument(*flag_args, **flag_kwargs)

    shared.add_argument(
        "-v", "--verbose", help="use verbose output", action="store_true"
    )
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

    for _, task in ctx._tasks.items():
        generate_task_subparser(shared, subparsers, task)

    args = vars(parser.parse_args())
    ctx.verbose = args.pop("verbose", False)
    ctx.dry = args.pop("dry_run", False)
    ctx.dag = args.pop("dag", False)
    for name in ctx.flags:
        ctx.flags[name] = args.pop(name)

    if f := args.pop("func", None):
        if ctx.dry:
            sys.stderr.write("dry run >>>\n" f"  args: {args}\n")
            sys.stderr.write(
                (
                    "\n".join(
                        f"  {line}"
                        for line in inspect.getsource(f).splitlines()
                        if not line.startswith("@")
                    )
                    + "\n"
                )
            )
        elif ctx.dag:
            sys.stderr.write(
                "currently --dag is a noop\n"
                "future versions will generate a dag for specified target\n"
            )
        else:
            f(**args)


if __name__ == "__main__":
    sys.stderr.write("this module should not be invoked directly\n")
    sys.exit(1)
