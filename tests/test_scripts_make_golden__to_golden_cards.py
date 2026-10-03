"""Tests for ``scripts.make_golden.to_golden_cards``.

Specification: docs/00-Architecture/scripts_make_golden__to_golden_cards.yaml

``to_golden_cards`` converts detected card objects into a list of
JSON-serializable 4-element lists ``[role, badge_box, name_box,
fan_box or None]`` in the original order:

- prepare an empty list ``out``
- for each ``c`` in cards: if ``c.fan_box`` is None then
  ``fb = None``, otherwise build the 4-element integer list
  ``fb`` from the first 4 elements of ``c.fan_box`` via ``int()``
- append ``[c.role, badge_box integer list, name_box integer list,
  fb]`` in that order
- return ``out`` after processing all elements

No side effects.

Mocked / stand-in dependencies (per the test-generation rules):
the card objects are ``types.SimpleNamespace`` stand-ins carrying
the attributes the code references (``role``, ``badge_box``,
``name_box``, ``fan_box``); the real CardDetector card class is out
of read scope per the spec's ``missing`` section.

``errors`` section of the spec (documented here only, NOT tested, per
the test-generation rules):

- condition: c.fan_box（None 以外）、c.badge_box、c.name_box のいずれかが4要素未満
  behavior: IndexError が発生する。
- condition: ボックスの要素が int() で変換できない文字列
  behavior: ValueError が発生する。
- condition: ボックスの要素が None など int() が拒否する値
  behavior: TypeError が発生する。
- condition: c.badge_box または c.name_box が None
  behavior: TypeError が発生する（None へのインデックス参照）。
- condition: c に role / badge_box / name_box / fan_box のいずれかの属性がない
  behavior: AttributeError が発生する。
- condition: cards がイテラブルでない
  behavior: TypeError が発生する。
"""

import pytest
from types import SimpleNamespace

from scripts.make_golden import to_golden_cards


def _card(role, badge_box, name_box, fan_box):
    """Build a SimpleNamespace stand-in card object."""
    return SimpleNamespace(role=role, badge_box=badge_box,
                           name_box=name_box, fan_box=fan_box)


def test_edge_01():
    """
    input: 'cards = []（空リスト）'
    expected: '[] を返す。'
    """
    assert to_golden_cards([]) == []


def test_edge_02():
    """
    input: 'c.fan_box が None'
    expected: '対応する結果行の第4要素が None になる。'
    """
    c = _card("r", (1, 2, 3, 4), (5, 6, 7, 8), None)
    result = to_golden_cards([c])
    assert result == [["r", [1, 2, 3, 4], [5, 6, 7, 8], None]]


def test_edge_03():
    """
    input: 'c.fan_box = (0, 0, 0, 0)'
    expected: '第4要素が [0, 0, 0, 0] になる（除外されるのは None のみで、全ゼロのボックスもそのままボックスとして保持される）。'
    """
    c = _card("r", (1, 2, 3, 4), (5, 6, 7, 8), (0, 0, 0, 0))
    result = to_golden_cards([c])
    assert result[0][3] == [0, 0, 0, 0]


def test_edge_04():
    """
    input: 'c.fan_box = (10.7, 20.2, 30.9, 40.1)'
    expected: '第4要素が [10, 20, 30, 40] になる（int() により小数点以下が切り捨てられる）。'
    """
    c = _card("r", (1, 2, 3, 4), (5, 6, 7, 8), (10.7, 20.2, 30.9, 40.1))
    result = to_golden_cards([c])
    assert result[0][3] == [10, 20, 30, 40]


def test_edge_05():
    """
    input: 'c.badge_box = (1.5, 2.5, 3.5, 4.5)'
    expected: '第2要素が [1, 2, 3, 4] になる（int() により小数点以下が切り捨てられる）。'
    """
    c = _card("r", (1.5, 2.5, 3.5, 4.5), (5, 6, 7, 8), (9, 10, 11, 12))
    result = to_golden_cards([c])
    assert result[0][1] == [1, 2, 3, 4]


def test_edge_06():
    """
    input: 'c.role が文字列（例: "a"）'
    expected: '第1要素が文字列 "a" のまま追加される（role は変換されない）。'
    """
    c = _card("a", (1, 2, 3, 4), (5, 6, 7, 8), None)
    result = to_golden_cards([c])
    assert result[0][0] == "a"
    assert isinstance(result[0][0], str)


def test_edge_07():
    """
    input: 'c.fan_box が4要素未満（例: (1, 2)）'
    expected: 'c.fan_box[2] の参照で IndexError が発生する。'
    """
    c = _card("r", (1, 2, 3, 4), (5, 6, 7, 8), (1, 2))
    with pytest.raises(IndexError):
        to_golden_cards([c])


def test_edge_08():
    """
    input: 'ボックスの要素が文字列 "abc"'
    expected: 'int("abc") で ValueError が発生する。'
    """
    c = _card("r", ("abc", 1, 2, 3), (5, 6, 7, 8), None)
    with pytest.raises(ValueError):
        to_golden_cards([c])


def test_edge_09():
    """
    input: 'ボックスの要素が None（例: c.badge_box = (None, 1, 2, 3)）'
    expected: 'int(None) で TypeError が発生する。'
    """
    c = _card("r", (None, 1, 2, 3), (5, 6, 7, 8), None)
    with pytest.raises(TypeError):
        to_golden_cards([c])


def test_edge_10():
    """
    input: 'c.badge_box が None'
    expected: 'c.badge_box[0] のインデックス操作で TypeError が発生する（None は subscript できない）。'
    """
    c = _card("r", None, (5, 6, 7, 8), None)
    with pytest.raises(TypeError):
        to_golden_cards([c])


def test_edge_11():
    """
    input: 'c に fan_box 属性がない'
    expected: 'c.fan_box の参照で AttributeError が発生する。'
    """
    c = SimpleNamespace(role="r", badge_box=(1, 2, 3, 4),
                        name_box=(5, 6, 7, 8))
    with pytest.raises(AttributeError):
        to_golden_cards([c])


def test_edge_12():
    """
    input: 'cards がイテラブルでないオブジェクト（例: 42）'
    expected: 'for ループで TypeError が発生する（42 は iterable ではない）。'
    """
    with pytest.raises(TypeError):
        to_golden_cards(42)
