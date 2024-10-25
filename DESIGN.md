# Swydd Design

Goals:

- Generic task runner
- Portable (single python module)
- library not an exe

basic design:

`tasks.py`:
```python
from swydd import taks, option, cli


@task
@option("program", "name of program to compile")
def build(program: str = "hello"):
    """build a program"""
    sub < f"gcc -o {program} {program}.c"


cli()
```

```sh
./tasks.py build
./tasks.py build --program goodbye
```

## Ideas

### Simple shell pipelines

```python
@task
def pipe_commands():
    """run pre-commit (and mypy)"""
    p = Pipe(Exec("find . -name file"), Exec("grep 'pattern'")).get()
    print(p.stdout)
```

Made even simpler with operator overloading:


```python
@task
def run_commands():
    stdout = get < (pipe | "find . -name file" | "grep 'pattern'")
    print(stdout)
```

Upon reflection I think the operator overloading is wildly confusing.
I think it will make more sense to try to develop chainable objects:

Make `sub(cmd, **kwargs)` and alias for `proc(cmd).run(**kwargs)`

```python
@task
def run_commands():
    stdout = proc("find . -name file").pipe("grep 'pattern'").get()
    print(stdout)
```

## Handling a DAG

```python
@task
@needs("src/a-file.txt")
@targets("build/output.txt")
def build():
    """build step with a dependency"""
    lines = asset("src/a-filter.txt").read().splitlines()
    # or lines = ctx.needs[0].read().splitlines()
    ctx.targets.output.write("\n".join([f"{i}: {line}" for i, line in enumerate(lines)]))
```



## Internal CLI

```sh
./tasks.py +swydd --version # <- current favorite (use a none valid function name to possibly prevent overlap)
```
