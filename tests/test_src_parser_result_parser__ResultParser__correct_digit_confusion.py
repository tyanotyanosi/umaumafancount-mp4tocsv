r"""Tests for ``src.parser.result_parser.ResultParser._correct_digit_confusion``.

Specification: docs/00-Architecture/src_parser_result_parser__ResultParser__correct_digit_confusion.yaml

``_correct_digit_confusion`` replaces OCR-confused characters in the
string with digits using the class attribute ``DIGIT_CONFUSION``
substitution table and returns the substitution result; if characters
other than digits / comma / period / whitespace remain after
substitution, it rejects and returns None:

- maps each character of ``s`` in order with
  ``DIGIT_CONFUSION.get(ch, ch)`` (characters in the table are
  substituted, others are kept) and joins them into the string
  ``corrected``
- runs ``re.search(r"[^0-9,.\s]", corrected)``; if at least one
  character not in [0-9, comma, period, whitespace] is present,
  returns None
- if no such character is present, returns ``corrected``
- the substitution table DIGIT_CONFUSION is O→0, Q→0, D→0, l→1,
  I→1, |→1, Z→2, z→2, S→5, G→5, B→8, A→4, E→3; characters other than
  these are not substituted

No mocked / stand-in dependencies: the method runs real on a real
``ResultParser`` instance.

``errors`` section of the spec (documented here only, NOT tested, per
the test-generation rules):

- condition: "s が str 以外（例: None または int）"
  behavior: "for ch in s のジェネレータ式で TypeError（例: \"NoneType is not iterable\"）が発生する。コード内では捕捉されず呼び出し側に伝播する。"
"""

from src.parser.result_parser import ResultParser


def _correct(s):
    """Call the real method on a fresh parser instance."""
    return ResultParser()._correct_digit_confusion(s)


def test_edge_01():
    """
    input: ''
    expected: 空文字列を返す（空文字列は拒否されず None にはならない）。
    """
    assert _correct("") == ""


def test_edge_02():
    """
    input: '1,234'
    expected: "1,234" を返す（混同文字が無くそのまま通過。カンマは残留を許容）。
    """
    assert _correct("1,234") == "1,234"


def test_edge_03():
    """
    input: 'O'
    expected: "0" を返す。
    """
    assert _correct("O") == "0"


def test_edge_04():
    """
    input: 'Q'
    expected: "0" を返す。
    """
    assert _correct("Q") == "0"


def test_edge_05():
    """
    input: 'D'
    expected: "0" を返す。
    """
    assert _correct("D") == "0"


def test_edge_06():
    """
    input: 'lI|'
    expected: "111" を返す。
    """
    assert _correct("lI|") == "111"


def test_edge_07():
    """
    input: 'Zz'
    expected: "22" を返す。
    """
    assert _correct("Zz") == "22"


def test_edge_08():
    """
    input: 'S G'
    expected: "5 5" を返す（空白は置換対象でなく、残留を許容）。
    """
    assert _correct("S G") == "5 5"


def test_edge_09():
    """
    input: 'A'
    expected: "4" を返す。
    """
    assert _correct("A") == "4"


def test_edge_10():
    """
    input: 'E'
    expected: "3" を返す。
    """
    assert _correct("E") == "3"


def test_edge_11():
    """
    input: 'B'
    expected: "8" を返す。
    """
    assert _correct("B") == "8"


def test_edge_12():
    """
    input: 'X'
    expected: None を返す（X は置換表になく、許容集合にも含まれないため残留して拒否される）。
    """
    assert _correct("X") is None


def test_edge_13():
    """
    input: 'a'
    expected: None を返す（小書き a は置換表になく、そのまま残留して拒否される）。
    """
    assert _correct("a") is None


def test_edge_14():
    """
    input: '1-2'
    expected: None を返す（ハイフン - は許容集合に含まれず拒否される）。
    """
    assert _correct("1-2") is None


def test_edge_15():
    """
    input: '12.34'
    expected: "12.34" を返す（ピリオドは残留を許容）。
    """
    assert _correct("12.34") == "12.34"
