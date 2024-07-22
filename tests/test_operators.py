from swydd import SwyddPath, get, path, pipe, proc, seq, sub


def test_operators():
    assert sub < "echo 'hello'"


def test_pipes():
    assert sub < (
        pipe | "cat ../src/swydd/__init__.py" | "grep '__version__'" | "wc -l"
    )
    assert sub < (
        proc | "cat ../src/swydd/__init__.py" | "grep '__version__'" | "wc -l"
    )


def test_seqs():
    # -a is not an arg to cat so the subprocess should return false
    assert not sub < (seq & "cat -a" & "echo hello")
    assert not sub < (proc & "cat -a" & "echo hello")


def test_capture():
    result = get < "ls src not-src"
    assert result == "src:\nswydd"
    result = get << "ls src not-src"
    assert result == "ls: cannot access 'not-src': No such file or directory"
    assert "hello part deux" == (get < seq & "echo 'hello'" & "echo hello part deux")
    assert "" == (get < (seq & "cp" & "echo hello part deux"))
    assert "cp: missing file operand\nTry 'cp --help' for more information." == (
        get << (seq & "cp" & "echo hello part deux")
    )


def check_result_file(file: SwyddPath, text: str) -> bool:
    if p := file._path:
        return p.read_text() == text
    return False


def test_write_to_path():
    result_f = path / "products/result.txt"
    result_txt = "text to file"
    result_f < result_txt
    assert check_result_file(result_f, result_txt + "\n")


def test_copy_and_rename():
    src_f = path / "fixtures/input.txt"
    dst_f = path / "products/input.txt"
    dst_f < src_f
    assert check_result_file(dst_f, "data to copy to another file\n")

    dst_f % "products/input2.txt"
    assert check_result_file(
        path / "products/input2.txt", "data to copy to another file\n"
    )
