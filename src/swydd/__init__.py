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
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

__version__ = "0.1.0"


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


def _id_from_func(f: Callable[..., Any]):
    return str(id(wrapped) if (wrapped := getattr(f, "__wrapped__", None)) else id(f))


class Task:
    def __init__(
        self, func=Callable[..., Any], name: Optional[str] = None, show: bool = False
    ) -> None:
        self.show = show
        self.id = _id_from_func(func)
        self.name = name if name else func.__name__
        self.func = func
        self.targets = []
        self.needs = []
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


class Graph:
    def __init__(self) -> None:
        self.nodes = {}
        self.edges = {}

    def add_nodes(self, task, node1, node2):
        if node1 not in self.nodes:
            self.nodes[node1] = []
        if node2 not in self.nodes:
            self.nodes[node2] = []

        self.edges[node1] = task
        if node2:
            self.nodes[node1].append(node2)


class Context:
    def __init__(self) -> None:
        self._tasks: Dict[str, Any] = {}
        self.targets: Dict[str, Any] = {}
        self.data: Any = None
        self.flags: Dict[str, Any] = {}
        self._flag_defs: List[Tuple[Tuple[str, ...], Any]] = []
        self.show_targets = True
        self._graph = Graph()

        # global flags
        self.dry = False
        self.dag = False
        self.verbose = False
        self.force = False

    def _add_task(self, func: Callable[..., Any], show: bool = False) -> str:
        if (id_ := _id_from_func(func)) not in self._tasks:
            self._tasks[id_] = Task(func)
        if show:
            self._tasks[id_]._mark()
        return id_

    def _update_option(self, func: Callable[..., Any], name: str, help: str, **kwargs):
        if (id_ := _id_from_func(func)) not in self._tasks:
            raise ValueError
        self._tasks[id_]._update_option(name, help, **kwargs)

    def _add_target(self, func: Callable[..., Any], target: str) -> None:
        self._add_task(func)
        id_ = _id_from_func(func)
        self._tasks[id_].targets.append(target)

    def _add_need(self, func: Callable[..., Any], need: str) -> None:
        self._add_task(func)
        id_ = _id_from_func(func)
        self._tasks[id_].needs.append(need)

    def _generate_graph(self) -> None:
        for task in self._tasks.values():
            if not task.targets:
                continue

            for target in task.targets:
                if not task.needs:
                    self._graph.add_nodes(task, target, None)
                else:
                    for need in task.needs:
                        self._graph.add_nodes(task, target, need)

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
            sys.stdout.write(f"swydd exec | {self.cmd}\n")
        if self.shell:
            return subprocess.run(self.cmd, shell=True).returncode
        else:
            return subprocess.run(shlex.split(self.cmd)).returncode


def sh(cmd: str, shell: bool = False) -> int:
    return Exec(cmd, shell=shell).execute()


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
            ctx._add_target(func, arg)
            ctx.targets[arg] = _id_from_func(func)

        @wraps(func)
        def inner(*args: Any, **kwargs: Any) -> Callable[..., None]:
            return func(*args, **kwargs)

        return inner

    return wrapper


def needs(
    *args: str,
) -> Callable[[Callable[..., Any]], Callable[..., Callable[..., None]]]:
    def wrapper(func: Callable[..., Any]) -> Callable[..., Callable[..., None]]:
        for arg in args:
            ctx._add_need(func, arg)

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
        ctx._update_option(func, name.replace("-", "_"), help, **help_kwargs)

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


def noop(*args, **kwargs) -> Any:
    pass


def target_generator(
    target: str,
    needs: List[str] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Callable[..., None]]]:
    def wrapper(func: Callable[..., Any]) -> Callable[..., Callable[..., None]]:
        @wraps(func)
        def inner(*args: Any, **kwargs: Any) -> Callable[..., None]:
            if not (target_path := Path(target)).is_file():
                return func(*args, **kwargs)
            elif not needs:
                sys.stderr.write(f"{target} already exists\n")
            else:
                target_stats = target_path.stat()
                needs_stats = [Path(need).stat() for need in needs]
                if any((stat.st_mtime > target_stats.st_mtime for stat in needs_stats)):
                    return func(*args, **kwargs)
                else:
                    sys.stderr.write("doing nothing\n")

            return noop(*args, **kwargs)

        return inner

    return wrapper


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

        args = (f"--{name.replace('_','-')}",)
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

    f = (
        target_generator(target, ctx._graph.nodes[target])(task.func)
        if target
        else task.func
    )
    subparser.set_defaults(func=f)
    return subparser


def add_targets(
    shared: ArgumentParser, subparsers: _SubParsersAction, ctx: Context
) -> None:
    for target, id_ in ctx.targets.items():
        subp = generate_task_subparser(shared, subparsers, ctx._tasks[id_], str(target))

        if subp:
            subp.add_argument("--dag", help="show target dag", action="store_true")
            subp.add_argument("--force", help="force execution", action="store_true")


def cli() -> None:
    ctx._generate_graph()

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
        title="tasks", required=True, dest="pos-arg", metavar="<task/target>"
    )

    if len(sys.argv) > 1 and sys.argv[1] == "self":
        generate_task_subparser(
            shared, subparsers, Task(manage, name="self", show=True)
        )

    add_targets(shared, subparsers, ctx)

    for task in ctx._tasks.values():
        generate_task_subparser(shared, subparsers, task)

    args = vars(parser.parse_args())
    _ = args.pop("pos-arg", None)
    ctx.verbose = args.pop("verbose", False)
    ctx.dry = args.pop("dry_run", False)
    ctx.dag = args.pop("dag", False)
    ctx.force = args.pop("force", False)
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
