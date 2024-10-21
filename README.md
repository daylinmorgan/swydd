[![EffVer][effver-shield]][effver-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![Ruff][ruff-shield]][ruff-url]
[![pre-commit][pre-commit-shield]][pre-commit-url]

<div align="center">
<h1>sywdd</h1>
<p>sywdd will yield desired deliverables </p>
</div>


## Automagic Snippet

```python
# https://github.com/daylinmorgan/swydd?tab=readme-ov-file#automagic-snippet
_src = (_i := __import__)("pathlib").Path(__file__).parent / "swydd/__init__.py"
if not (_i("importlib.util").util.find_spec("sywdd") or _src.is_file()):
    _r = _i("urllib.request").request.urlopen("https://swydd.dayl.in/swydd.py")
    _src.parent.mkdir(exist_ok=True)
    _src.write_text(_r.read().decode("utf-8"))
```

## Alternatives

- [task.mk](https://gh.dayl.in/task.mk)
- [make](https://www.gnu.org/software/make/)
- [just](https://just.systems)
- [task](https://taskfile.dev)
- [nox](https://nox.thea.codes/en/stable/)
- [pypyr](https://pypyr.io)
- [pydoit](https://pydoit.org)
- [knit](https://github.com/zyedidia/knit)

<!-- badges -->
[pre-commit-shield]: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white
[pre-commit-url]: https://pre-commit.com
[ruff-shield]: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
[ruff-url]: https://github.com/astral-sh/ruff
[pypi-shield]: https://img.shields.io/pypi/v/swydd
[pypi-url]: https://pypi.org/project/sywdd
[issues-shield]: https://img.shields.io/github/issues/daylinmorgan/swydd.svg
[issues-url]: https://github.com/daylinmorgan/swydd/issues
[license-shield]: https://img.shields.io/github/license/daylinmorgan/swydd.svg
[license-url]: https://github.com/daylinmorgan/swydd/blob/main/LICENSE
[effver-shield]: https://img.shields.io/badge/version_scheme-EffVer-0097a7
[effver-url]: https://jacobtomlinson.dev/effver
