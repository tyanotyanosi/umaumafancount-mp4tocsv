"""Tests for ``src.parser.name_mapper.levenshtein_distance``.

Specification: docs/00-Architecture/src_parser_name_mapper__levenshtein_distance.yaml

``levenshtein_distance`` returns the Levenshtein (edit) distance
between two strings:

- if ``a == b``, return 0 (early return)
- if ``a`` is falsy, return ``len(b)``
- if ``b`` is falsy, return ``len(a)``
- otherwise compute the minimum number of insertions, deletions and
  substitutions (each operation costs 1) to transform ``a`` into
  ``b`` via 1-D dynamic programming:
  ``prev = list(range(len(b) + 1))``; for each character ``ca`` of
  ``a`` (indexed from 1) build ``cur = [i]``; for each character
  ``cb`` of ``b`` (indexed from 1) with ``cost = 0 if ca == cb else
  1``, append ``min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)``;
  update ``prev = cur`` per row and return ``prev[-1]``

The result is always an int, 0 or more and at most
``max(len(a), len(b))``. There is no type checking in the code.

Mocked / stand-in dependencies (per the test-generation rules):
none — the function is pure and is called directly with the spec
edge-case inputs.

``errors`` section of the spec (documented here only, NOT tested, per
the test-generation rules):

- condition: "a か b がコードが使う演算（==, truthiness, len, enumerate, インデックス参照）に耐えない型（例: int）"
  behavior: "コードに型チェックがないため、その型の演算子サポートに応じた組み込み例外（例: enumerate(int) による TypeError）が送出される。"
"""

from src.parser.name_mapper import levenshtein_distance


def test_edge_01():
    """
    input: "a='abc', b='abc'"
    expected: "0 を返す。"
    """
    assert levenshtein_distance("abc", "abc") == 0


def test_edge_02():
    """
    input: "a='', b='abc'"
    expected: "3 を返す（len(b)）。"
    """
    assert levenshtein_distance("", "abc") == 3


def test_edge_03():
    """
    input: "a='abc', b=''"
    expected: "3 を返す（len(a)）。"
    """
    assert levenshtein_distance("abc", "") == 3


def test_edge_04():
    """
    input: "a='kitten', b='sitting'"
    expected: "3 を返す。"
    """
    assert levenshtein_distance("kitten", "sitting") == 3


def test_edge_05():
    """
    input: "a='abc', b='abd'"
    expected: "1 を返す。"
    """
    assert levenshtein_distance("abc", "abd") == 1
