"""Tests for ``src.parser.result_parser.ResultParser._clean_user_name``.

Specification: docs/00-Architecture/src_parser_result_parser__ResultParser__clean_user_name.yaml

``_clean_user_name`` cleans an OCR-derived user name string: it
removes the trailing parenthesized annotation, collapses a repeated
identical line into one line, then joins the lines and removes all
whitespace:

- parenthesis removal: apply the regex ``[（\\(].*$`` with ``re.sub``
  (no re.M); what is actually removed is the interval from the first
  （ or ( on the final line (the line reaching the end of the string)
  to the end of the string (or just before the final newline);
  parentheses on non-final lines remain
- split the result by newlines, strip each line and drop empty lines
  to make a line list
- if the line list is non-empty and all elements are identical
  (``len(set(lines)) == 1``), collapse to the single first line
- join the line list without a separator
- remove all whitespace (``re.sub(r"\\s+", "", name)``; ``\\s``
  includes Unicode whitespace in Python 3 str regex)
- finally call ``.strip()`` and return that string (a no-op in
  practice)

No mocked / stand-in dependencies: the method runs real on a real
``ResultParser`` instance.

``errors`` section of the spec (documented here only, NOT tested, per
the test-generation rules):

- condition: "name が str 以外（例: None または int）"
  behavior: "re.sub が TypeError (expected string or bytes-like object) を送出する。コード内では捕捉されず呼び出し側に伝播する。"
"""

from src.parser.result_parser import ResultParser


def _clean(name):
    """Call the real method on a fresh parser instance."""
    return ResultParser()._clean_user_name(name)


def test_edge_01():
    """
    input: ''
    expected: 空文字列を返す。
    """
    assert _clean("") == ""


def test_edge_02():
    """
    input: '山田（たか）'
    expected: "山田" を返す（単一行のため （ から行末までが除去される）。
    """
    assert _clean("山田（たか）") == "山田"


def test_edge_03():
    """
    input: 'abc(def)'
    expected: "abc" を返す（半角括弧 ( も除去対象である）。
    """
    assert _clean("abc(def)") == "abc"


def test_edge_04():
    """
    input: '山田\\n山田'
    expected: "山田" を返す（同一行の繰返しは1行に集約される。\\n は改行文字を意味する）。
    """
    assert _clean("山田\n山田") == "山田"


def test_edge_05():
    """
    input: '山田\\n花子'
    expected: "山田花子" を返す（異なる行は区切り文字なしで連結される。\\n は改行文字を意味する）。
    """
    assert _clean("山田\n花子") == "山田花子"


def test_edge_06():
    """
    input: 'a\\nb\\na'
    expected: "aba" を返す（集約は全行が同一の場合のみ適用され、混在する重複は除去されない。\\n は改行文字を意味する）。
    """
    assert _clean("a\nb\na") == "aba"


def test_edge_07():
    """
    input: '  山田  太郎  '
    expected: "山田太郎" を返す（空白はすべて除去される）。
    """
    assert _clean("  山田  太郎  ") == "山田太郎"


def test_edge_08():
    """
    input: 'abc（x\\ndef'
    expected: "abc（xdef" を返す（括弧が非最終行にあるため除去は発生せず （x は残る。行連結後は空白のみ除去される。\\n は改行文字を意味する）。
    """
    assert _clean("abc（x\ndef") == "abc（xdef"


def test_edge_09():
    """
    input: 'abc（x\\ndef（y'
    expected: "abc（xdef" を返す（除去されるのは最終行の （y のみで、1行目の （x は残る。\\n は改行文字を意味する）。
    """
    assert _clean("abc（x\ndef（y") == "abc（xdef"
