"""Tests for ``src.video.card_detector.CardDetector._nms_badges``.

Specification: docs/00-Architecture/src_video_card_detector__CardDetector__nms_badges.yaml

``CardDetector._nms_badges`` is an instance method
(``def _nms_badges(self, cands: list) -> list:``) that applies NMS
(suppression of duplicate candidates whose IoU exceeds 0.5) to a badge
candidate list ``cands`` and returns the kept candidates as a list in
high-to-low score order (per the spec's ``purpose`` field):

- Step 1: sort ``cands`` by score(c) = max(c[3], c[4]) in descending order
  (``sorted`` is a stable sort, so same-score candidates keep their
  original order)
- Step 2: initialize ``kept`` to an empty list
- Step 3: scan the sorted candidates in order; for each candidate c, compute
  ``_iou(box(c), box(k))`` against every candidate k in kept, where box(c)
  is ``self._as_box(c, c[2])``
- Step 4: if ``kept`` is empty, or the IoU is 0.5 or less for every k,
  append c to ``kept``; if some k has an IoU greater than 0.5, c is
  suppressed (not appended)
- Step 5: return ``kept`` as is

Per the spec's ``side_effects`` field the method only calls ``self._as_box``
and ``self._iou`` (no I/O, no global-state change), so it is pure logic and
the tests below invoke it directly with no mocks. ``_nms_badges`` is an
instance method whose behavior does not depend on constructor state
(``_as_box``/``_iou`` use only their arguments), so each test binds an
uninitialised instance created via ``object.__new__(CardDetector)`` to
invoke the real method.

``errors`` section of the spec (documented here only, NOT tested, per the
test-generation rules):

- condition: '候補要素の長さが 5 未満（例: [10, 20, 1.0, 0.9]）'
  behavior: 'sorted の key 関数 score 内の c[4] アクセス時に IndexError が送出される'
- condition: 'c[3] と c[4] が比較できない（例: [0, 0, 1.0, None, 0.5]）'
  behavior: 'max() 呼び出し中に TypeError が送出される'
- condition: 'c[2] が数値以外で _as_box が w/h を生成できない（例: [0, 0, None, 0.9, 0.1]）'
  behavior: '_as_box 内の 124 * s の計算時に TypeError が送出される'
"""

from src.video.card_detector import CardDetector


def _detector():
    """Return an uninitialised ``CardDetector`` instance.

    ``_nms_badges`` only uses ``self._as_box`` and ``self._iou`` (spec
    ``side_effects``), neither of which depends on ``__init__`` state, so
    skipping the constructor still lets the real method be invoked directly.
    """
    return object.__new__(CardDetector)


def test_edge_01():
    """
    input: cands = []
    expected: [] を返す
    """
    det = _detector()
    cands = []
    result = det._nms_badges(cands)
    assert result == []
    # postcondition: sorted() produces a new list, not the argument itself
    assert result is not cands


def test_edge_02():
    """
    input: cands = [[10, 20, 1.0, 0.9, 0.1]]（候補 1 個）
    expected: [[10, 20, 1.0, 0.9, 0.1]] を返す（kept が空のため抑制されない）
    """
    det = _detector()
    cands = [[10, 20, 1.0, 0.9, 0.1]]
    result = det._nms_badges(cands)
    assert result == [[10, 20, 1.0, 0.9, 0.1]]
    # postcondition: the element of the original cands itself is stored
    assert result[0] is cands[0]


def test_edge_03():
    """
    input: cands = [[10, 20, 1.0, 0.9, 0.1], [10, 20, 1.0, 0.8, 0.2]]（同一位置・同一スケール、IoU = 1.0）
    expected: [[10, 20, 1.0, 0.9, 0.1]] のみを返す（2 つ目の候補は IoU = 1.0 > 0.5 で抑制される）
    """
    det = _detector()
    cands = [[10, 20, 1.0, 0.9, 0.1], [10, 20, 1.0, 0.8, 0.2]]
    result = det._nms_badges(cands)
    assert result == [[10, 20, 1.0, 0.9, 0.1]]
    assert result[0] is cands[0]


def test_edge_04():
    """
    input: cands = [[10, 20, 1.0, 0.5, 0.1], [10, 20, 1.0, 0.9, 0.1]]（元リストが低スコア先頭）
    expected: [[10, 20, 1.0, 0.9, 0.1]] のみを返す（スコア 0.9 の候補が先頭に来て処理され、スコア 0.5 の候補が抑制される）
    """
    det = _detector()
    cands = [[10, 20, 1.0, 0.5, 0.1], [10, 20, 1.0, 0.9, 0.1]]
    result = det._nms_badges(cands)
    assert result == [[10, 20, 1.0, 0.9, 0.1]]
    assert result[0] is cands[1]
    # postcondition: the argument list itself is not mutated (still low-score first)
    assert cands == [[10, 20, 1.0, 0.5, 0.1], [10, 20, 1.0, 0.9, 0.1]]


def test_edge_05():
    """
    input: cands = [[0, 0, 1.0, 0.9, 0.1], [62, 0, 1.0, 0.8, 0.1]]（2 つの box の IoU がちょうど 0.5）
    expected: [[0, 0, 1.0, 0.9, 0.1], [62, 0, 1.0, 0.8, 0.1]] の順で両候補を返す（IoU = 0.5 は 0.5 以下なので抑制されない）
    """
    det = _detector()
    cands = [[0, 0, 1.0, 0.9, 0.1], [62, 0, 1.0, 0.8, 0.1]]
    result = det._nms_badges(cands)
    assert result == [[0, 0, 1.0, 0.9, 0.1], [62, 0, 1.0, 0.8, 0.1]]
    assert result[0] is cands[0]
    assert result[1] is cands[1]


def test_edge_06():
    """
    input: cands = [[0, 0, 1.0, 0.9, 0.1], [500, 0, 1.0, 0.8, 0.1]]（box が重ならない、IoU = 0）
    expected: [[0, 0, 1.0, 0.9, 0.1], [500, 0, 1.0, 0.8, 0.1]] の順で両候補を返す
    """
    det = _detector()
    cands = [[0, 0, 1.0, 0.9, 0.1], [500, 0, 1.0, 0.8, 0.1]]
    result = det._nms_badges(cands)
    assert result == [[0, 0, 1.0, 0.9, 0.1], [500, 0, 1.0, 0.8, 0.1]]
    assert result[0] is cands[0]
    assert result[1] is cands[1]


def test_edge_07():
    """
    input: cands = [[0, 0, 1.0, 0.9, 0.1], [0, 0, 0.5, 0.8, 0.1]]（同一位置・スケール違い）
    expected: [[0, 0, 1.0, 0.9, 0.1], [0, 0, 0.5, 0.8, 0.1]] の順で両候補を返す（box2 = (0, 0, 62, 18) は box1 = (0, 0, 124, 37) に完全包含され IoU = 1116/4588 ≒ 0.243 で 0.5 以下のため抑制されない）
    """
    det = _detector()
    cands = [[0, 0, 1.0, 0.9, 0.1], [0, 0, 0.5, 0.8, 0.1]]
    result = det._nms_badges(cands)
    assert result == [[0, 0, 1.0, 0.9, 0.1], [0, 0, 0.5, 0.8, 0.1]]
    assert result[0] is cands[0]
    assert result[1] is cands[1]
