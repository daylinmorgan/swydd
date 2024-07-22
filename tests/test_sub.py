from swydd import get, pipe, seq, sub


def test_pipes():
    assert sub(
        pipe("cat ../src/swydd/__init__.py").pipe("grep '__version__'").pipe("wc -l")
    )


def test_seqs():
    # -a is not an arg to cat so the should return false
    assert not sub(seq("cat -a").then("echo hello"))


def test_capture():
    result = get("ls src not-src")
    assert result == "src:\nswydd"
    result = get("ls src not-src", stdout=False, stderr=True)
    assert result == "ls: cannot access 'not-src': No such file or directory"
    assert "2" == get(
        pipe("cat src/swydd/__init__.py").pipe("grep '__version__'").pipe("wc -l")
    )
    assert "hello part deux" == get(seq("echo 'hello'").then("echo hello part deux"))
