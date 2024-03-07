[![EffVer][effver-shield]][effver-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![Ruff][ruff-shield]][ruff-url]
[![pre-commit][pre-commit-shield]][pre-commit-url]
<!-- [![Stargazers][stars-shield]][stars-url] -->
<!-- [![PYPI][pypi-shield]][pypi-url] -->

<div align="center">
<h1>sywdd</h1>
<p>sywdd will yield desired deliverables </p>
</div>


## Automagic Snippet

```python
# https://github.com/daylinmorgan/swydd?tab=readme-ov-file#automagic-snippet
# fmt: off
if not (src := __import__("pathlib").Path(__file__).parent / "swydd/__init__.py").is_file(): # noqa
    try: __import__("swydd") # noqa
    except ImportError:
        import sys; from urllib.request import urlopen; from urllib.error import URLError # noqa
        try: r = urlopen("https://raw.githubusercontent.com/daylinmorgan/swydd/main/src/swydd/__init__.py") # noqa
        except URLError as e: sys.exit(f"{e}\n") # noqa
        src.parent.mkdir(exists_ok=True); src.write_text(r.read().decode("utf-8")); # noqa
# fmt: on
```

## Alternatives

- make
- just
- task
- nox
- pypyr
- pydoit
https://github.com/zyedidia/knit

<!-- badges -->
[pre-commit-shield]: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white
[pre-commit-url]: https://pre-commit.com
[ruff-shield]: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
[ruff-url]: https://github.com/astral-sh/ruff
[pypi-shield]: https://img.shields.io/pypi/v/swydd
[pypi-url]: https://pypi.org/project/sywdd
[issues-shield]: https://img.shields.io/github/issues/daylinmorgan/swydd.svg
[issues-url]: https://github.com/daylinmorgan/swydd/issues
[license-shield]: https://img.shields.io/github/license/daylinmorgan/sywdd.svg
[license-url]: https://github.com/daylinmorgan/swydd/blob/main/LICENSE
[effver-shield]: https://img.shields.io/badge/version_scheme-EffVer-0097a7
[effver-url]: https://jacobtomlinson.dev/effver
