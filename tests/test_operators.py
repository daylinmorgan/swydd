from swydd import SwyddPath, asset, get, geterr, pipe, proc, seq, sub


def test_sub():
    assert sub("echo 'hello'")


def test_pipes():
    assert sub(
        pipe("cat ../src/swydd/__init__.py").pipe("grep '__version__'").pipe("wc -l")
    )
    assert sub(
        proc("cat ../src/swydd/__init__.py").pipe("grep '__version__'").pipe("wc -l")
    )


def test_seqs():
    # -a is not an arg to cat so the subprocess should return false
    assert not sub(seq("cat -a").then("echo hello"))
    assert not sub(proc("cat -a").then("echo hello"))


def test_capture():
    result = get("ls src not-src")
    assert result == "src:\nswydd"
    result = geterr("ls src not-src")
    assert result == "ls: cannot access 'not-src': No such file or directory"

    assert "hello part deux" == get(proc("echo 'hello'").then("echo hello part deux"))
    commands = proc("cp").then("echo hello part deux")
    assert "" == get(commands)
    assert "cp: missing file operand\nTry 'cp --help' for more information." == (
        geterr(commands)
    )


def check_result_file(file: SwyddPath, text: str) -> bool:
    if p := file._path:
        return p.read_text() == text
    return False


def test_write_to_path():
    result_f = asset("products/result.txt")
    result_txt = "text to file"
    result_f.write(result_txt)
    assert check_result_file(result_f, result_txt + "\n")


def test_copy_and_rename():
    src_f = asset("fixtures/input.txt")
    dst_f = asset("products/input.txt")
    dst_f.copy(src_f)
    assert check_result_file(dst_f, "data to copy to another file\n")

    dst_f.rename("products/input2.txt")
    assert check_result_file(
        asset("products/input2.txt"), "data to copy to another file\n"
    )
