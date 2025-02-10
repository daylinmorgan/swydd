from __future__ import annotations

import argparse
import inspect
import os
import shlex
import shutil
import signal
import sys
from argparse import (
    ArgumentParser,
    RawDescriptionHelpFormatter,
)
from functools import wraps
from pathlib import Path
from subprocess import PIPE, Popen
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from argparse import Action, _SubParsersAction
    from subprocess import CompletedProcess
    from typing import Any, Callable, Dict, List, Optional, Tuple

__version__ = "0.2.0"


# TODO: make this ouput wider help when necessary
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
        self, func: Callable[..., Any], name: Optional[str] = None, show: bool = False
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

    def _update_option(self, name: str, help: str, short: str, **kwargs) -> None:
        self.params[name] = {
            **self.params.get(name, {}),
            "help": help,
            "kwargs": kwargs,
        }
        if short != "":
            self.params[name]["short"] = short

    def _mark(self) -> None:
        self.show = True

    def _task_repr(self) -> str:
        return (
            "\n".join(
                f"  {line}"
                for line in inspect.getsource(self.func).splitlines()
                if not line.startswith("@")
            )
            + "\n"
        )


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
        self._env: Dict[str, str] = {}
        self._tasks: Dict[str, Any] = {}
        self._ids: Dict[str, str] = {}
        self.targets: Dict[str, Any] = {}
        self.data: Any = None
        self.show_targets = True
        self._graph = Graph()
        self.rest = []  # remaining positional args

        # global flags
        self.dry = False
        self.dag = False
        self.verbose = False
        self.force = False

    def _add_task(self, func: Callable[..., Any], show: bool = False) -> str:
        if (id_ := _id_from_func(func)) not in self._tasks:
            self._tasks[id_] = Task(func)
            self._ids[func.__name__] = id_
        if show:
            self._tasks[id_]._mark()
        return id_

    def _update_option(
        self, func: Callable[..., Any], name: str, help: str, short: str, **kwargs
    ):
        id_ = self._add_task(func)
        self._tasks[id_]._update_option(name, help, short, **kwargs)

    def _add_target(self, func: Callable[..., Any], target: str) -> None:
        id_ = self._add_task(func)
        self._tasks[id_].targets.append(target)

    def _add_need(self, func: Callable[..., Any], need: str) -> None:
        id_ = self._add_task(func)
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

    def _get_task(self, name: str) -> Task:
        assert name in self._ids
        return self._tasks[self._ids[name]]


class Flags:
    def __init__(self) -> None:
        self._flags: Dict[str, Any] = {}
        self._flag_defs: List[Tuple[Tuple[str, ...], Any]] = []

    def __getattr__(self, name: str):
        if name in self._flags:
            return self._flags[name]
        else:
            # TODO: better error/sys.exit
            raise AttributeError

    def add(self, *args: str, **kwargs: Any) -> None:
        name = max(args, key=len).split("-")[-1]
        self._flags[name] = None
        self._flag_defs.append((args, kwargs))

    def _parse_args(self, args):
        for name in self._flags:
            self._flags[name] = args.pop(name)

    def _add_arguments(self, shared):
        for flag_args, flag_kwargs in self._flag_defs:
            shared.add_argument(*flag_args, **flag_kwargs)


ctx = Context()
flags = Flags()


class SubResult:
    def __init__(
        self,
        code: int,
        stdout: None | str,
        stderr: None | str,
        process: CompletedProcess | Popen,
    ) -> None:
        self.code = code
        self.stdout = stdout
        self.stderr = stderr
        self._proces = process

    @classmethod
    def from_completed_process(cls, p: CompletedProcess) -> "SubResult":
        return cls(p.returncode, p.stdout, p.stderr, p)

    @classmethod
    def from_popen(
        cls,
        p: Popen,
        stdout: str = "",
        stderr: str = "",
    ) -> "SubResult":
        return cls(p.returncode, stdout, stderr, p)


class Proc:
    """a class"""

    def __init__(self, cmd: str, **kwargs: Any) -> None:
        self._cmd = cmd
        if cmd:
            self.cmd = shlex.split(cmd)
        self.cmd_kwargs = kwargs

    def pipe(self, proc: str | Proc) -> ProcPipe:
        """ProcPipe the output into `proc`

        Parameters
        ----------
        proc
            Command to feed stdout into
        Returns
        -------
        ProcPipe
        """

        if isinstance(proc, str):
            return ProcPipe(self, proc)
        elif isinstance(proc, Proc):
            return ProcPipe(proc)

    def then(self, proc: str | Proc | ProcSeq) -> ProcSeq:
        """If successful execute next `proc`

        Parameters
        ----------
        proc
            Command to execute after
        Returns
        -------
        ProcSeq
        """

        if isinstance(proc, Proc):
            return ProcSeq(proc)
        # should swydd seq even be supported here?
        elif isinstance(proc, ProcSeq):
            return proc
        else:
            return ProcSeq(Proc(proc))

    def _build_kwargs(self, output: bool) -> Dict[str, Any]:
        sub_kwargs: Dict[str, Any] = dict(env={**os.environ, **ctx._env})

        if output:
            sub_kwargs["text"] = True  # assume text is the desired output
            sub_kwargs["stdout"] = PIPE

        sub_kwargs.update(**self.cmd_kwargs)
        return sub_kwargs

    def _show_command(self) -> None:
        if ctx.verbose:
            sys.stdout.write(f"swydd exec | {self._cmd}\n")

    def execute(self, output: bool = False) -> SubResult:
        self._show_command()

        p = Popen(self.cmd, **self._build_kwargs(output))

        try:
            out, err = p.communicate()
        except KeyboardInterrupt:
            sys.stderr.write("forwarding CTRL+C\n")
            sys.stderr.flush()
            p.send_signal(signal.SIGINT)
            p.wait()
            out, err = p.communicate()

        return SubResult.from_popen(p, out, err)

    def check(self) -> bool:
        return self.execute().code == 0


class ProcPipe:
    """
    prefer: :func:`Proc.pipe`
    """

    def __init__(self, *procs: str | Proc | ProcPipe) -> None:
        self._procs = []
        for proc in procs:
            if isinstance(proc, str):
                self._procs.append(Proc(proc))
            elif isinstance(proc, Proc):
                self._procs.append(proc)
            elif isinstance(proc, ProcPipe):
                self._procs.extend(proc._procs)

    def check(self) -> bool:
        return self.execute().code == 0

    def _show_command(self) -> None:
        if ctx.verbose:
            cmd_str = " | ".join([p._cmd for p in self._procs])
            sys.stdout.write(f"swydd exec | {cmd_str}\n")

    def execute(self, output: bool = False) -> SubResult:
        procs = []
        sub_kwargs: Dict[str, Any] = dict(env={**os.environ, **ctx._env})
        self._show_command()

        for i, cmd in enumerate(self._procs, 1):
            kwargs = {}
            if i > 1:
                kwargs.update(stdin=procs[-1].stdout)
            if i != len(self._procs):
                kwargs.update(stdout=PIPE)
            elif i == len(self._procs):
                if output:
                    kwargs.update({"text": True, "stdout": PIPE, "stderr": PIPE})

            procs.append(Popen(cmd.cmd, **{**sub_kwargs, **kwargs}))

        for p in procs[:-1]:
            if p.stdout:
                p.stdout.close()

        try:
            out, err = procs[-1].communicate()
        except KeyboardInterrupt:
            sys.stderr.write("forwarding CTRL+C\n")
            sys.stderr.flush()
            # ALL of them?
            procs[-1].send_signal(signal.SIGINT)
            procs[-1].wait()
            out, err = procs[-1].communicate()

        return SubResult.from_popen(procs[-1], out, err)

    def pipe(self, proc: "str | Proc | ProcPipe") -> "ProcPipe":
        return ProcPipe(self, proc)


class ProcSeq:
    """
    prefer: :func:`Proc.then`
    """

    def __init__(self, *procs: "str | Proc | ProcSeq") -> None:
        self._procs = []
        for proc in procs:
            if isinstance(proc, ProcSeq):
                self._procs.extend(self._procs)
            if isinstance(proc, Proc):
                self._procs.append(proc)
            elif isinstance(proc, str):
                self._procs.append(Proc(proc))

    def _show_command(self) -> None:
        if ctx.verbose:
            cmd_str = " && ".join([p._cmd for p in self._procs])
            sys.stderr.write(f"sywdd exec | {cmd_str}\n")

    def then(self, proc: "str | Proc | ProcSeq") -> "ProcSeq":
        return ProcSeq(*self._procs, proc)

    def execute(self, output: bool = False) -> "SubResult":
        self._show_command()

        results = []
        for proc in self._procs:
            results.append(result := proc.execute(output=output))
            if result.code != 0:
                return result

        return results[-1]

    def run(self) -> int:
        self._show_command()

        rc = 0
        for proc in self._procs:
            if (rc := proc.execute().code) != 0:
                return rc
        return rc

    def check(self) -> bool:
        return self.run() == 0


def _get_caller_path() -> Path:
    # NOTE: jupyter will hate this code I'm sure
    for i, frame in enumerate(inspect.stack()):
        if (name := frame.filename) != __file__:
            return Path(name).parent
    raise ValueError("failed to find root directory of runner")


# Asset(ag) / f
# Asset / a <- also support!


class Asset:
    def __init__(self, p: str) -> None:
        self._root = _get_caller_path()
        self._path = self._root / p

    def read(self) -> str:
        if self._path:
            return self._path.read_text()
        else:
            raise ValueError("path is not set")

    def __truediv__(self, p: str) -> "Asset":
        self._path = self._path / p
        return self

    def _check(self) -> Path:
        if self._path is None:
            raise ValueError("todo")
        return self._path

    def _write_text(self, txt: str) -> "Asset":
        p = self._check()
        p.parent.mkdir(exist_ok=True)
        p.write_text(txt + "\n")
        return self

    def write(self, src: "str | Asset") -> "Asset":
        if isinstance(src, str):
            return self._write_text(src)
        elif isinstance(src, Asset):
            return self._write_text(src.read())

    def _append_text(self, txt: str) -> "Asset":
        p = self._check()
        p.parent.mkdir(exist_ok=True)
        with p.open("a") as f:
            f.write(txt)
            f.write("\n")
        return self

    def rename(self, dst: "str | Asset | Path") -> None:
        if isinstance(dst, str):
            dst_p = Asset(dst)._check()
        elif isinstance(dst, Path):
            dst_p = dst
        else:
            dst_p = dst._check()
        src_p = self._check()
        src_p.rename(dst_p)

    def copy(self, src: "str | Asset") -> None:
        if isinstance(src, str):
            self.write(src)
        elif isinstance(src, Asset):
            dst_p = self._check()
            src_p = src._check()
            shutil.copyfile(src_p, dst_p)

    def append(self, src: "str | Asset") -> "Asset":
        if isinstance(src, str):
            return self._append_text(src)
        elif isinstance(src, Asset):
            return self._append_text(src.read().strip())


def _inspect_wrapper(place, func):
    if wrapped := getattr(func, "__wrapped__", None):
        print(place, "wrapped->", id(wrapped))

    print(
        place,
        id(func),
    )


def task(
    arg: Callable[..., Any] | None = None,
):
    """decorator to convert a function into a swydd task"""

    def wrapper(
        func: Callable[..., Any] | None = None,
    ) -> Callable[..., Callable[..., None]]:
        # added this to prevent lsp errors
        # about missing argument on wrapped functions
        if func is None:
            raise ValueError

        ctx._add_task(func, show=not func.__name__.startswith("_"))

        @wraps(func)
        def inner(*args: Any, **kwargs: Any) -> Callable[..., None]:
            return func(*args, **kwargs)

        return inner

    if callable(arg):
        return wrapper(arg)
    else:
        return wrapper


def targets(
    *args: str,
) -> Callable[[Callable[..., Any]], Callable[..., Callable[..., None]]]:
    def wrapper(func: Callable[..., Any]) -> Callable[..., Callable[..., None]]:
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
    short: str = "",
    **help_kwargs: str,
) -> Callable[[Callable[..., Any]], Callable[..., Callable[..., None]]]:
    """add help and additional args for option

    Parameters
    ----------
    name
        variable name used in wrapped function, subsitute underscores with hypens
    help
        help description for variable
    **help_kwargs
        kwargs which will be passed onto :func:`argparse.ArgumentParser.add_argument`
    """

    def wrapper(func: Callable[..., Any]) -> Callable[..., Callable[..., None]]:
        ctx._update_option(func, name.replace("-", "_"), help, short, **help_kwargs)

        @wraps(func)
        def inner(*args: Any, **kwargs: Any) -> Callable[..., None]:
            return func(*args, **kwargs)

        return inner

    return wrapper


def noop(*args, **kwargs) -> Any:
    _ = args, kwargs


def _target_generator(
    target: str,
    needs: List[str] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Callable[..., None]]]:
    def wrapper(func: Callable[..., Any]) -> Callable[..., Callable[..., None]]:
        @wraps(func)
        def inner(*args: Any, **kwargs: Any) -> Callable[..., None]:
            if not (target_path := Path(target)).is_file():
                return func(*args, **kwargs)
            elif not needs:
                sys.stderr.write(f"{target_path} up to date, exiting\n")
            else:
                target_stats = target_path.stat()
                needs_stats = [Path(need).stat() for need in needs]
                if any((stat.st_mtime > target_stats.st_mtime for stat in needs_stats)):
                    return func(*args, **kwargs)
                else:
                    sys.stderr.write(f"{target_path} up to date, exiting\n")

            return noop(*args, **kwargs)

        return inner

    return wrapper


def _internal_cli():
    name = "+swydd"
    parser = ArgumentParser(
        usage=f"%(prog)s {name} [opts]",
        description="""internal cli""",
        prog=os.path.basename(sys.argv[0]),
    )
    parser.add_argument("-V", "--version", help="show version", action="store_true")
    parser.add_argument("--which", help="show path to swydd", action="store_true")

    if len(sys.argv) == 2:
        parser.print_help(sys.stderr)
        sys.exit(1)
    args = parser.parse_args(sys.argv[2:])

    if args.version:
        print("swydd, v" + __version__)
    if args.which:
        print("current swydd:", __file__)

    sys.exit(0)


# TODO: reduce the load bearing on this function
# seperate by subparser for tasks vs subparser for target
def _generate_task_subparser(
    shared: ArgumentParser,
    subparsers: _SubParsersAction,
    task: Task,
    target: Optional[str] = None,
    doc: str = "",
) -> Optional[ArgumentParser]:
    # TODO: don't return an option
    if not task.show and not target:
        return

    prog = os.path.basename(sys.argv[0])
    name = task.name if not target else target
    if doc == "" and task.func.__doc__:
        doc = task.func.__doc__.splitlines()[0]
    subparser = subparsers.add_parser(
        name.replace("_", "-"),
        help=doc,
        description=task.func.__doc__,
        parents=[shared],
        usage=f"%(prog)s {name} [opts]",
        prog=prog,
    )

    for name, info in task.params.items():
        param = info.get("Parameter")  # must check signature for args?
        annotation = param.annotation

        args = []
        if "short" in info:
            args.append("-" + info["short"])
        args.append(f"--{name.replace('_','-')}")
        kwargs = {"help": info.get("help", "")}

        str_lit_types = {"bool": bool, "int": int, "float": float, "Path": Path}
        annotation = str_lit_types.get(annotation, annotation)

        if annotation is bool:
            kwargs.update({"default": False, "action": "store_true"})
        elif (annotation != inspect._empty) and (type(annotations) is not str):
            kwargs.update({"type": annotation})

        kwargs.update(
            {"required": True}
            if param.default == inspect._empty
            else {"default": param.default}
        )

        kwargs.update(info.get("kwargs", {}))
        subparser.add_argument(*args, **kwargs)

    # TODO: properly build out a dag from tasks "needs"
    # for now add a simple check for existense
    # this check needs to exit early
    # if nothing can produce said asset

    def executor(*args, **kwargs):
        for need in task.needs:
            Asset(need)._check()

        f = (
            _target_generator(target, ctx._graph.nodes[target])(task.func)
            if target
            else task.func
        )
        return f(*args, **kwargs)

    subparser.set_defaults(func=executor)
    return subparser


def _target_status(target: str) -> str:
    if not (target_path := Path(target)).is_file():
        return "missing target"
    needs = ctx._graph.nodes[target]
    target_stat = target_path.stat()
    needs_stats = []
    for need in needs:
        if not (p := Path(need)).is_file():
            return "missing inputs!"
        needs_stats.append(p.stat())

    if any((stat.st_mtime > target_stat.st_mtime for stat in needs_stats)):
        return "out of date"

    return " "


def _add_targets(
    shared: ArgumentParser, subparsers: _SubParsersAction, ctx: Context
) -> None:
    for target, id_ in ctx.targets.items():
        subp = _generate_task_subparser(
            shared, subparsers, ctx._tasks[id_], str(target), doc=_target_status(target)
        )
        if subp:
            subp.add_argument("--dag", help="show target dag", action="store_true")
            subp.add_argument(
                "-f", "--force", help="force execution", action="store_true"
            )


def cli(default: str | None = None) -> None:
    """activate swydd cli

    Parameters
    ----------
    default
        The default args passed line passed on to sywdd.
    """
    ctx._generate_graph()
    if len(sys.argv) > 1 and sys.argv[1] == "+swydd":
        _internal_cli()

    parser = ArgumentParser(
        formatter_class=SubcommandHelpFormatter, usage="%(prog)s <task/target> [opts]"
    )
    shared = ArgumentParser(add_help=False)

    flags._add_arguments(shared)

    shared.add_argument(
        "-v", "--verbose", help="use verbose output", action="store_true"
    )
    shared.add_argument(
        "-n", "--dry-run", help="don't execute tasks", action="store_true"
    )

    subparsers = parser.add_subparsers(
        title="tasks", required=True, dest="pos-arg", metavar="<task/target>"
    )

    _add_targets(shared, subparsers, ctx)

    for task in ctx._tasks.values():
        _generate_task_subparser(shared, subparsers, task)

    if len(sys.argv) == 1:
        if default:
            sys.argv.extend(shlex.split(default))
        else:
            parser.print_help(sys.stderr)
            sys.exit(1)

    if "--" in sys.argv:
        i = sys.argv.index("--")
        args = vars(parser.parse_args(sys.argv[1:i]))
        ctx.rest = sys.argv[i + 1 :]
    else:
        args = vars(parser.parse_args())

    pos_arg = args.pop("pos-arg", None)
    ctx.verbose = args.pop("verbose", False)
    ctx.dry = args.pop("dry_run", False)
    ctx.dag = args.pop("dag", False)
    ctx.force = args.pop("force", False)
    flags._parse_args(args)

    if f := args.pop("func", None):
        if ctx.dry:
            sys.stderr.write("dry run >>>\n" f"  args: {args}\n")
            if ctx._env:
                sys.stderr.write(f"  env: {ctx._env}\n")
            sys.stderr.write(ctx._get_task(pos_arg)._task_repr())
        elif ctx.dag:
            sys.stderr.write(
                "currently --dag is a noop\n"
                "future versions will generate a dag for specified target\n"
            )
        else:
            f(**args)


def get(proc: str | Proc | ProcPipe | ProcSeq, stdout=True, stderr=False) -> str:
    """execute subprocess and capture outputs

    see also: :func:`geterr`

    Parameters
    ----------
    proc
        Command to execute.
    stdout
        If true, capture stdout
    stderr
        If true, capture stderr

    Returns
    -------
    str
        Captured output
    """
    if isinstance(proc, str):
        result = Proc(proc).execute(output=True)
    elif isinstance(proc, ProcPipe):
        result = proc.execute(output=True)
    elif isinstance(proc, Proc):
        result = proc.execute()
    elif isinstance(proc, ProcSeq):
        result = proc.execute(output=True)
    else:
        raise NotImplementedError(f"not implemented for type: {type(exec)}")

    output = ""
    if stdout and result.stdout:
        output += result.stdout.strip()
    if stderr and result.stderr:
        output += result.stderr.strip()
    return output


def geterr(
    proc: str | Proc | ProcPipe | ProcSeq,
) -> str:
    """execute subprocess and return stderr


    Parameters
    ----------
    proc : str, :class:`.Proc`, :class:`.ProcSeq`, :class:`.ProcPipe`
        Command to execute.

    Returns
    -------
    str
        Output captured from stderr.
    """
    return get(proc, stdout=True, stderr=False)


# TODO: think more about how "rest" should work...
def sub(proc: str | Proc | ProcPipe | ProcSeq, rest: bool = False) -> bool:
    if isinstance(proc, str):
        cmd = proc if not rest else (proc + " " + " ".join(ctx.rest))
        return Proc(cmd).check()
    elif (
        isinstance(proc, Proc)
        or isinstance(proc, ProcSeq)
        or isinstance(proc, ProcPipe)
    ):
        if rest:
            raise ValueError("rest is only supported when passing str")
        return proc.check()
    else:
        raise ValueError(f"unspported type: {type(exec)}")


def setenv(key: str, value: str) -> None:
    """Set environment variable shared by all Procs"""
    ctx._env.update({key: value})


__all__ = [
    "sub",
    "get",
    "ctx",
    "geterr",
    "setenv",
    "cli",
    "task",
    "targets",
    "needs",
    "option",
    "flags",
    "Proc",
    "ProcPipe",
    "ProcSeq",
    "Asset",
]

if __name__ == "__main__":
    sys.stderr.write("this module should not be invoked directly\n")
    sys.exit()
