import argparse
import inspect
import os
import shlex
import shutil
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
from subprocess import PIPE, CompletedProcess, Popen
from typing import Any, Callable, Dict, List, Optional, Tuple

__version__ = "0.1.0"


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
        self._env: Dict[str, str] = {}
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


def define_env(key: str, value: str) -> None:
    ctx._env.update({key: value})


class SwyddSubResult:
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
    def from_completed_process(cls, p: CompletedProcess) -> "SwyddSubResult":
        return cls(p.returncode, p.stdout, p.stderr, p)

    @classmethod
    def from_popen(
        cls,
        p: Popen,
        stdout: str = "",
        stderr: str = "",
    ) -> "SwyddSubResult":
        return cls(p.returncode, stdout, stderr, p)


class SwyddProc:
    """
    usage:
        sub < proc("echo $WORD world", env={'WORD':'hello'})
        sub < (proc | "single proc")

        sub < (proc | "echo hello" | "wc -c")
        sub(proc("hello world").pipe("wc -c"))

        sub < (proc & "cat -a" & "echo unreachable")
        sub(proc("cat -a").then("echo unreachable")
    """

    def __init__(
        self, cmd: str | None = None, output: bool = False, **kwargs: Any
    ) -> None:
        self._cmd = cmd
        if cmd:
            self.cmd = shlex.split(cmd)
        self.output = output
        self.cmd_kwargs = kwargs

    def pipe(self, proc: "str | SwyddProc") -> "SwyddPipe | SwyddProc":
        if isinstance(proc, str):
            if self._cmd is None:
                return SwyddPipe(SwyddProc(proc))
            else:
                return SwyddPipe(self, proc)
        elif isinstance(proc, SwyddProc):
            return SwyddPipe(proc)

    def __or__(self, proc: "str | SwyddProc") -> "SwyddPipe | SwyddProc":
        return self.pipe(proc)

    def then(self, proc: "str | SwyddProc | SwyddSeq") -> "SwyddSeq":
        if self._cmd:
            return SwyddSeq(self, proc)

        if isinstance(proc, SwyddProc):
            return SwyddSeq(proc)
        # should swydd seq even be supported here?
        elif isinstance(proc, SwyddSeq):
            return proc
        else:
            return SwyddSeq(SwyddProc(proc))

    def __and__(self, proc: "str | SwyddProc | SwyddSeq") -> "SwyddSeq":
        return self.then(proc)

    def _build_kwargs(self) -> Dict[str, Any]:
        sub_kwargs: Dict[str, Any] = dict(env={**os.environ, **ctx._env})

        if self.output:
            sub_kwargs["text"] = True  # assume text is the desired output
            sub_kwargs["capture_output"] = True

        sub_kwargs.update(**self.cmd_kwargs)
        return sub_kwargs

    def execute(self, output: bool = False) -> SwyddSubResult:
        if ctx.verbose:
            sys.stdout.write(f"swydd exec | {self._cmd}\n")

        self.output = self.output or output
        return SwyddSubResult.from_completed_process(
            subprocess.run(self.cmd, **self._build_kwargs())
        )

    def check(self) -> bool:
        return self.execute().code == 0


class SwyddPipe:
    def __init__(self, *procs: "str | SwyddProc | SwyddPipe") -> None:
        self._procs = []
        for proc in procs:
            if isinstance(proc, str):
                self._procs.append(SwyddProc(proc))
            elif isinstance(proc, SwyddProc):
                self._procs.append(proc)
            elif isinstance(proc, SwyddPipe):
                self._procs.extend(proc._procs)

    @classmethod
    def __call__(cls, *args, **kwargs) -> "SwyddPipe":
        return cls(*args, **kwargs)

    def check(self) -> bool:
        return self.execute().code == 0

    def execute(self, output: bool = False) -> SwyddSubResult:
        procs = []
        sub_kwargs: Dict[str, Any] = dict(env={**os.environ, **ctx._env})
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

        out, err = procs[-1].communicate()
        return SwyddSubResult.from_popen(procs[-1], out, err)

    def pipe(self, proc: "str | SwyddProc | SwyddPipe") -> "SwyddPipe":
        return SwyddPipe(self, proc)

    def __or__(self, proc: "str | SwyddProc | SwyddPipe") -> "SwyddPipe":
        return self.pipe(proc)


class SwyddSeq:
    def __init__(self, *procs: "str | SwyddProc | SwyddSeq") -> None:
        self._procs = []
        for proc in procs:
            if isinstance(proc, SwyddSeq):
                self._procs.extend(self._procs)
            if isinstance(proc, SwyddProc):
                self._procs.append(proc)
            elif isinstance(proc, str):
                self._procs.append(SwyddProc(proc))

    @classmethod
    def __call__(cls, *args, **kwargs) -> "SwyddSeq":
        return cls(*args, **kwargs)

    def then(self, proc: "str | SwyddProc | SwyddSeq") -> "SwyddSeq":
        return SwyddSeq(*self._procs, proc)

    def __and__(self, proc: "str | SwyddProc | SwyddSeq") -> "SwyddSeq":
        return self.then(proc)

    def execute(self, output: bool = False) -> "SwyddSubResult":
        results = []
        for proc in self._procs:
            results.append(result := proc.execute(output=output))
            if result.code != 0:
                return result

        return results[-1]

    def run(self) -> int:
        rc = 0
        for proc in self._procs:
            if (rc := proc.execute().code) != 0:
                return rc
        return rc

    def check(self) -> bool:
        return self.run() == 0


class SwyddGet:
    def __call__(
        self, proc: str | SwyddProc | SwyddPipe | SwyddSeq, stdout=True, stderr=False
    ) -> str:
        if isinstance(proc, str):
            result = SwyddProc(proc, output=True).execute()
        elif isinstance(proc, SwyddPipe):
            result = proc.execute(output=True)
        elif isinstance(proc, SwyddProc):
            result = proc.execute()
        elif isinstance(proc, SwyddSeq):
            result = proc.execute(output=True)
        else:
            raise NotImplementedError(f"not implemented for type: {type(exec)}")

        output = ""
        if stdout and result.stdout:
            output += result.stdout.strip()
        if stderr and result.stderr:
            output += result.stderr.strip()
        return output

    def __lt__(self, proc: str | SwyddProc | SwyddPipe | SwyddSeq) -> str:
        return self.__call__(proc)

    def __lshift__(self, proc: str | SwyddProc | SwyddPipe | SwyddSeq) -> str:
        return self.__call__(proc, stdout=False, stderr=True)


def _get_caller_path() -> Path:
    # NOTE: jupyter will hate this code I'm sure
    for i, frame in enumerate(inspect.stack()):
        if (name := frame.filename) != __file__:
            return Path(name).parent
    raise ValueError("failed to find root directory of runner")


class SwyddSub:
    def __call__(self, proc: str | SwyddPipe | SwyddProc | SwyddSeq) -> bool:
        if isinstance(proc, str):
            return SwyddProc(proc).check()
        elif (
            isinstance(proc, SwyddProc)
            or isinstance(proc, SwyddSeq)
            or isinstance(proc, SwyddPipe)
        ):
            return proc.check()
        else:
            raise ValueError(f"unspported type: {type(exec)}")

    def __lt__(self, proc: str | SwyddPipe | SwyddProc | SwyddSeq) -> bool:
        return self.__call__(proc)


class SwyddPath:
    _root = None
    _path = None

    def __init__(self, p: Path | None = None) -> None:
        if p:
            self._path = p

    @classmethod
    def from_str(cls, p: str) -> "SwyddPath":
        return cls() / p

    def __truediv__(self, p: str | Path) -> "SwyddPath":
        if not (root := self._root):
            root = _get_caller_path()

        if not self._path:
            if isinstance(p, str):
                return SwyddPath(root / p)
            elif isinstance(p, Path):
                return SwyddPath(p)
        else:
            og = self._path.relative_to(root)

        return SwyddPath(og / p)

    def _check(self) -> Path:
        if self._path is None:
            raise ValueError("todo")
        return self._path

    def write(self, txt: str) -> None:
        p = self._check()
        p.parent.mkdir(exist_ok=True)
        p.write_text(txt + "\n")

    def append(self, txt: str) -> None:
        p = self._check()
        p.parent.mkdir(exist_ok=True)
        with p.open("a") as f:
            f.write(txt)
            f.write("\n")

    def __mod__(self, dst: "str | SwyddPath | Path") -> None:
        if isinstance(dst, str):
            dst_p = SwyddPath.from_str(
                dst
            )._check()  # <- TODO: ensure this uses self._root?
        elif isinstance(dst, Path):
            dst_p = dst
        else:
            dst_p = dst._check()
        src_p = self._check()
        src_p.rename(dst_p)

    def __lt__(self, src: "str | SwyddPath") -> None:
        if isinstance(src, str):
            self.write(src)
        elif isinstance(src, SwyddPath):
            dst_p = self._check()
            src_p = src._check()
            shutil.copyfile(src_p, dst_p)

    def __lshift__(self, txt: str) -> None:
        self.append(txt)


def task(func: Callable[..., Any]) -> Callable[..., None]:
    ctx._add_task(func, show=True)

    def wrap(*args: Any, **kwargs: Any) -> None:
        return func(*args, **kwargs)

    return wrap


def _inspect_wrapper(place, func):
    if wrapped := getattr(func, "__wrapped__", None):
        print(place, "wrapped->", id(wrapped))

    print(
        place,
        id(func),
    )


def task2(
    hidden: bool = False,
) -> Callable[[Callable[..., Any]], Callable[..., Callable[..., None]]]:
    def wrapper(func: Callable[..., Any]) -> Callable[..., Callable[..., None]]:
        ctx._add_task(func, show=True)

        _inspect_wrapper("task", func)

        @wraps(func)
        def inner(*args: Any, **kwargs: Any) -> Callable[..., None]:
            return func(*args, **kwargs)

        return inner

    return wrapper


def targets(
    *args: str,
) -> Callable[[Callable[..., Any]], Callable[..., Callable[..., None]]]:
    def wrapper(func: Callable[..., Any]) -> Callable[..., Callable[..., None]]:
        _inspect_wrapper("targets", func)
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
    _ = args, kwargs


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


def _generate_task_subparser(
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
        name.replace("_", "-"),
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

        if param.annotation is bool:
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


def _add_targets(
    shared: ArgumentParser, subparsers: _SubParsersAction, ctx: Context
) -> None:
    for target, id_ in ctx.targets.items():
        subp = _generate_task_subparser(
            shared, subparsers, ctx._tasks[id_], str(target)
        )

        if subp:
            subp.add_argument("--dag", help="show target dag", action="store_true")
            subp.add_argument("--force", help="force execution", action="store_true")


def _task_repr(func: Callable) -> str:
    return (
        "\n".join(
            f"  {line}"
            for line in inspect.getsource(func).splitlines()
            if not line.startswith("@")
        )
        + "\n"
    )


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

    if len(sys.argv) > 1 and sys.argv[1] == "+swydd":
        _generate_task_subparser(
            shared, subparsers, Task(manage, name="+swydd", show=True)
        )

    _add_targets(shared, subparsers, ctx)

    for task in ctx._tasks.values():
        _generate_task_subparser(shared, subparsers, task)

    # TODO: add support for default arg?
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

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
            sys.stderr.write(_task_repr(f))
        elif ctx.dag:
            sys.stderr.write(
                "currently --dag is a noop\n"
                "future versions will generate a dag for specified target\n"
            )
        else:
            f(**args)


(
    proc,
    pipe,
    seq,
    sub,
    get,
    path,
) = (
    SwyddProc(),
    SwyddPipe(),
    SwyddSeq(),
    SwyddSub(),
    SwyddGet(),
    SwyddPath(),
)

if __name__ == "__main__":
    sys.stderr.write("this module should not be invoked directly\n")
    sys.exit(1)
