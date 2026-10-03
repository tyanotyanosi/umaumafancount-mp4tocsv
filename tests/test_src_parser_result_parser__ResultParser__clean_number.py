"""Tests for ``src.parser.result_parser.ResultParser._clean_number``.

Specification: docs/00-Architecture/src_parser_result_parser__ResultParser__clean_number.yaml

``_clean_number`` removes suffixes (名・人) and symbols from a numeric
text and returns a string of half-width digits only (or an empty
string):

- remove '名' and '人' from ``num_str``
- apply ``re.sub(r'[^0-9.,]', '', ...)`` to remove every character
  other than half-width digits, ',' and '.'
- remove '.' and ',' from the remaining string
- return the result (half-width digits only, or an empty string)

No mocked / stand-in dependencies: the method runs real on a real
``ResultParser`` instance.

``errors`` section of the spec: empty (no error cases documented).
"""

from src.parser.result_parser import ResultParser


def _clean(num_str):
    """Call the real method on a fresh parser instance."""
    return ResultParser()._clean_number(num_str)


def test_edge_01():
    """
    input: num_str='1,234名'
    expected: '1234' が返る。
    """
    assert _clean("1,234名") == "1234"


def test_edge_02():
    """
    input: num_str='12.34人'
    expected: '1234' が返る。
    """
    assert _clean("12.34人") == "1234"


def test_edge_03():
    """
    input: num_str='1,2,3'
    expected: '123' が返る。
    """
    assert _clean("1,2,3") == "123"


def test_edge_04():
    """
    input: num_str='abc名'
    expected: '' が返る。
    """
    assert _clean("abc名") == ""


def test_edge_05():
    """
    input: num_str=''
    expected: '' が返る。
    """
    assert _clean("") == ""


def test_edge_06():
    """
    input: num_str='１２３'（全角数字）
    expected: '' が返る（全角数字は [0-9] に含まれず除去される）。
    """
    assert _clean("１２３") == ""
