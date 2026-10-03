"""Tests for ``src.video.card_detector.CardDetector._pick_winning_scale``.

Specification: docs/00-Architecture/src_video_card_detector__CardDetector__pick_winning_scale.yaml

``CardDetector._pick_winning_scale(self, cands, s0)`` (per the spec's
``signature``, ``purpose`` and ``behavior`` fields) picks the winning scale
from the candidate list:

- Step 1: initialize an empty dict ``scores``
- Step 2: for each candidate ``c`` (in list order) let ``s = c[2]``,
  ``sm = c[3]``, ``sl = c[4]`` and add ``max(sm, sl)`` to ``scores[s]``
  (starting from ``0.0`` when the scale is not yet registered)
- Step 3: pick the scale ``s`` maximizing the tuple
  ``(score(s), -abs(s - s0))`` under tuple comparison — a larger score
  wins; on a score tie the scale with the smaller ``abs(s - s0)`` (closer
  to ``s0``) wins; on a fully tied value the key inserted first into
  ``scores`` (i.e. the first scale seen in ``cands``) is kept
- Step 4: return that winning scale

The method is pure logic with no external dependencies (no files, GUI,
network, time, randomness, or CV2/OpenCV use), so the tests below invoke
it directly with no mocks. Per the spec the body uses none of the instance
state (``side_effects: []``), so it is called in unbound form as
``CardDetector._pick_winning_scale(None, cands, s0)``.

``errors`` section of the spec (documented here only, NOT tested, per the
test-generation rules):

- condition: 'cands が空 list である'
  behavior: 'ValueError（max() arg is an empty sequence）。空 dict に対する max(scores, key=...) で発生'
- condition: 'c[2] がハッシュ可能でない型（list / dict 等）である'
  behavior: 'TypeError。scores[s] = ... の dict キー代入で発生'
- condition: 'c[3] または c[4] が数値でない（None 等）'
  behavior: 'TypeError。max(sm, sl) の比較で発生'
- condition: 's0 が数値でない（str 等）'
  behavior: 'TypeError。キー関数内の s - s0 引き算で発生'
- condition: 'cands の要素 c の長さが 5 未満である'
  behavior: 'IndexError。c[3] または c[4] のアクセスで発生'
"""

from src.video.card_detector import CardDetector


def test_edge_01():
    """
    input: cands = [], s0 = 1.0
    expected: ValueError（max() arg is an empty sequence）。scores が空 dict となるため max() が例外を送出
    """
    raised = None
    try:
        CardDetector._pick_winning_scale(None, [], 1.0)
    except ValueError as exc:
        raised = exc
    assert raised is not None


def test_edge_02():
    """
    input: cands = [[10, 20, 1.0, 0.9, 0.1]], s0 = 1.0
    expected: 1.0 を返す。scores はキー 1.0 のみ（スコア 0.9）を持ち、単一キーが argmax となる
    """
    result = CardDetector._pick_winning_scale(None, [[10, 20, 1.0, 0.9, 0.1]], 1.0)
    assert result == 1.0


def test_edge_03():
    """
    input: cands = [[0, 0, 0.8, 0.3, 0.0], [0, 0, 0.8, 0.4, 0.0]], s0 = 1.0
    expected: 0.8 を返す。同じスケール 0.8 の候補 2 件が足し算され score(0.8) = 0.3 + 0.4 = 0.7 となる
    """
    result = CardDetector._pick_winning_scale(None, [[0, 0, 0.8, 0.3, 0.0], [0, 0, 0.8, 0.4, 0.0]], 1.0)
    assert result == 0.8


def test_edge_04():
    """
    input: cands = [[0, 0, 0.8, 0.9, 0.0], [0, 0, 1.1, 0.9, 0.0]], s0 = 1.0
    expected: 1.1 を返す。score(0.8) = score(1.1) = 0.9 で同点のため s0 = 1.0 に近い方が優先され、abs(1.1 - 1.0) = 0.1 < abs(0.8 - 1.0) = 0.2 により 1.1 が選定される
    """
    result = CardDetector._pick_winning_scale(None, [[0, 0, 0.8, 0.9, 0.0], [0, 0, 1.1, 0.9, 0.0]], 1.0)
    assert result == 1.1


def test_edge_05():
    """
    input: cands = [[0, 0, 0.9, 0.9, 0.0], [0, 0, 1.1, 0.9, 0.0]], s0 = 1.0
    expected: 0.9 を返す。スコア同点（0.9）かつ abs(0.9 - 1.0) == abs(1.1 - 1.0) == 0.1 で完全同値のため、max() が scores dict における初出キー（cands 中の初出スケール）である 0.9 を採用する
    """
    result = CardDetector._pick_winning_scale(None, [[0, 0, 0.9, 0.9, 0.0], [0, 0, 1.1, 0.9, 0.0]], 1.0)
    assert result == 0.9


def test_edge_06():
    """
    input: cands = [[0, 0, 0.9, 0.3, 0.2], [0, 0, 1.1, 0.9, 0.1]], s0 = 1.0
    expected: 1.1 を返す。score(1.1) = 0.9 > score(0.9) = 0.3 であり、スコア差がある限り s0 からの近接性は影響しない
    """
    result = CardDetector._pick_winning_scale(None, [[0, 0, 0.9, 0.3, 0.2], [0, 0, 1.1, 0.9, 0.1]], 1.0)
    assert result == 1.1
