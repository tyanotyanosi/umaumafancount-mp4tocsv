"""Tests for ``src.parser.result_parser.ResultParser._parse_number``.

Specification: docs/00-Architecture/src_parser_result_parser__ResultParser__parse_number.yaml

``_parse_number`` normalizes a numeric text to a string of half-width
digits and returns an int only when the value is at least 3 digits
and between 0 and 10000000000 inclusive, otherwise None:

- if ``num_str`` is falsy (empty string), return None
- call ``self._clean_number(num_str)`` and obtain the
  digits-only string ``cleaned``
- if ``cleaned`` is empty, or ``cleaned.isdigit()`` is falsy, return
  None
- if ``len(cleaned) < 3``, return None
- compute ``int(cleaned)`` (a raised ValueError falls through to the
  final ``return None``)
- return ``result`` if ``0 <= result <= 10000000000``, otherwise None

No mocked / stand-in dependencies: the method runs real on a real
``ResultParser`` instance (``_clean_number`` also runs real).

``errors`` section of the spec (documented here only, NOT tested, per
the test-generation rules):

- condition: "int(cleaned) が ValueError を送出する（CPython 3.11+ では int 文字列の桁数上限（デフォルト 4300 桁）を超える場合）"
  behavior: "例外は try/except で捕捉され、None が返る。"
"""

from src.parser.result_parser import ResultParser


def _parse(num_str):
    """Call the real method on a fresh parser instance."""
    return ResultParser()._parse_number(num_str)


def test_edge_01():
    """
    input: num_str=''
    expected: None が返る。
    """
    assert _parse("") is None


def test_edge_02():
    """
    input: num_str='12'
    expected: cleaned='12' が 3 桁未満のため None が返る。
    """
    assert _parse("12") is None


def test_edge_03():
    """
    input: num_str='0123'
    expected: 123 が返る。
    """
    assert _parse("0123") == 123


def test_edge_04():
    """
    input: num_str='1,234,567'
    expected: 1234567 が返る。
    """
    assert _parse("1,234,567") == 1234567


def test_edge_05():
    """
    input: num_str='12.3'
    expected: 123 が返る（小数点は _clean_number で単純除去される）。
    """
    assert _parse("12.3") == 123


def test_edge_06():
    """
    input: num_str='10000000000'
    expected: 10000000000 が返る（上限は等しい値を含む）。
    """
    assert _parse("10000000000") == 10000000000


def test_edge_07():
    """
    input: num_str='10000000001'
    expected: 範囲超過のため None が返る。
    """
    assert _parse("10000000001") is None


def test_edge_08():
    """
    input: num_str='12a'
    expected: cleaned='12' が 3 桁未満のため None が返る。
    """
    assert _parse("12a") is None
