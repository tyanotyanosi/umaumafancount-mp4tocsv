"""Tests for ``src.parser.result_parser.ResultParser._parse_fan_text``.

Specification: docs/00-Architecture/src_parser_result_parser__ResultParser__parse_fan_text.yaml

``_parse_fan_text`` converts a fan-count text (raw text including
suffixes and OCR-confused characters) to an int, returning None when
it cannot be converted:

- if ``s`` is falsy (empty string), return None
- remove '人', '名' and 'J' (half-width uppercase only) from ``s``
- call ``self._correct_digit_confusion(s)``; if the return value is
  None, return None
- otherwise return the return value of
  ``self._parse_number(corrected)`` (int or None) as-is

No mocked / stand-in dependencies: the method runs real on a real
``ResultParser`` instance (``_correct_digit_confusion`` and
``_parse_number`` also run real).

``errors`` section of the spec: empty (no error cases documented).
"""

from src.parser.result_parser import ResultParser


def _parse(s):
    """Call the real method on a fresh parser instance."""
    return ResultParser()._parse_fan_text(s)


def test_edge_01():
    """
    input: s=''
    expected: None が返る。
    """
    assert _parse("") is None


def test_edge_02():
    """
    input: s='1,234人'
    expected: 1234 が返る。
    """
    assert _parse("1,234人") == 1234


def test_edge_03():
    """
    input: s='1,000名'
    expected: 1000 が返る。
    """
    assert _parse("1,000名") == 1000


def test_edge_04():
    """
    input: s='12J34'
    expected: 1234 が返る（'J' が削除される）。
    """
    assert _parse("12J34") == 1234


def test_edge_05():
    """
    input: s='O1,234'
    expected: 1234 が返る（'O'→'0' 補正後の '01,234' を _parse_number が int 化する）。
    """
    assert _parse("O1,234") == 1234


def test_edge_06():
    """
    input: s='99'
    expected: None が返る（_parse_number の 3 桁未満チェック）。
    """
    assert _parse("99") is None


def test_edge_07():
    """
    input: s='abc'
    expected: None が返る（'a' が置換表に無く _correct_digit_confusion が None を返す）。
    """
    assert _parse("abc") is None
