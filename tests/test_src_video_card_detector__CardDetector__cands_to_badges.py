"""Tests for ``src.video.card_detector.CardDetector._cands_to_badges``.

Specification: docs/00-Architecture/src_video_card_detector__CardDetector__cands_to_badges.yaml

``CardDetector._cands_to_badges`` converts the badge candidates produced by
``_collect_scale_cands`` (``cands``, a list of 5-element sequences
``[x, y, s, sm, sl]``; ``sm``/``sl`` are cv2.matchTemplate-derived numeric
scores and ``s`` a numeric scale) into a list of badge coordinate tuples
``(x, y, w, h, role)``:

- Iterates over ``cands`` in input order, unpacking each element as
  ``(x, y, s, sm, sl)``
- Role is ``"member"`` when ``sm >= sl`` (ties count as ``"member"``),
  otherwise ``"leader"``
- ``w = int(124 * s)`` and ``h = int(37 * s)`` (truncation by ``int()``)
- Returns a list of ``(x, y, w, h, role)`` tuples with the same length and
  order as the input; ``sm``/``sl`` are discarded

Per the spec the function is pure (no I/O, no external calls, no
instance or global state change), so the tests invoke the unbound method
with ``self`` bound to ``None`` and no mocks.

``errors`` section of the spec (documented here only, NOT tested, per the
test-generation rules):

- condition: 'cands の要素が長さ5でないシーケンス（例 3要素リスト）'
  behavior: 'アンパック箇所（for x, y, s, sm, sl in cands）で ValueError が送出される'
- condition: 'sm と sl が比較不可な型（例 str と int の混合）'
  behavior: 'sm >= sl の比較で TypeError が送出される'
"""

from src.video.card_detector import CardDetector


def test_edge_01():
    """
    input: cands = []
    expected: [] を返す
    """
    result = CardDetector._cands_to_badges(None, [])
    assert result == []
    assert isinstance(result, list)


def test_edge_02():
    """
    input: cands = [[10, 20, 1.0, 0.8, 0.5]]
    expected: [(10, 20, 124, 37, 'member')] を返す
    """
    result = CardDetector._cands_to_badges(None, [[10, 20, 1.0, 0.8, 0.5]])
    assert result == [(10, 20, 124, 37, "member")]


def test_edge_03():
    """
    input: cands = [[10, 20, 1.0, 0.5, 0.8]]
    expected: [(10, 20, 124, 37, 'leader')] を返す
    """
    result = CardDetector._cands_to_badges(None, [[10, 20, 1.0, 0.5, 0.8]])
    assert result == [(10, 20, 124, 37, "leader")]


def test_edge_04():
    """
    input: cands = [[10, 20, 1.0, 0.6, 0.6]]（sm == sl）
    expected: [(10, 20, 124, 37, 'member')] を返す（sm >= sl は同値を含み "member"）
    """
    result = CardDetector._cands_to_badges(None, [[10, 20, 1.0, 0.6, 0.6]])
    assert result == [(10, 20, 124, 37, "member")]


def test_edge_05():
    """
    input: cands = [[5, 5, 0.5, 0.9, 0.1]]
    expected: [(5, 5, 62, 18, 'member')] を返す（w=int(124*0.5)=62、h=int(37*0.5)=18）
    """
    result = CardDetector._cands_to_badges(None, [[5, 5, 0.5, 0.9, 0.1]])
    assert result == [(5, 5, 62, 18, "member")]


def test_edge_06():
    """
    input: cands = [[0, 0, 1.0, 0.9, 0.1], [100, 100, 1.0, 0.1, 0.9]]
    expected: [(0, 0, 124, 37, 'member'), (100, 100, 124, 37, 'leader')] を返す（入力の順を保持）
    """
    result = CardDetector._cands_to_badges(
        None, [[0, 0, 1.0, 0.9, 0.1], [100, 100, 1.0, 0.1, 0.9]]
    )
    assert result == [(0, 0, 124, 37, "member"), (100, 100, 124, 37, "leader")]
