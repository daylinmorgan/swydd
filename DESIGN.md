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
@option("program","name of program to compile")
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
def pipe_commands()
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


## Internal CLI

```sh
./tasks.py _ swydd-subcmd
# vs
./tasks.py _swydd-subcmd
# or
./tasks.py +swydd-subcmd # <- current favorite (use a none valid function name to possibly prevent overlap)
```
