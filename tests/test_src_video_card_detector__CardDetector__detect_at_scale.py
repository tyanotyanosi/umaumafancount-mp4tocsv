"""Tests for ``src.video.card_detector.CardDetector._detect_at_scale``.

Specification: docs/00-Architecture/src_video_card_detector__CardDetector__detect_at_scale.yaml

``CardDetector._detect_at_scale`` performs template matching at a single
scale ``s`` (for cache reuse):

1. Call ``self._collect_scale_cands(gray, [s])`` to collect badge candidates
   of the form ``[x, y, s, score_m, score_l]`` at scale ``s``.
2. If the candidates form an empty list, return the empty list as-is
   (without calling ``_nms_badges``).
3. If candidates exist, run non-maximum suppression via
   ``self._nms_badges(cands)``.
4. Convert the surviving candidates to ``(x, y, w, h, role)`` tuples with
   ``self._cands_to_badges(cands)`` and return them; every element has
   ``w = int(124 * s)`` and ``h = int(37 * s)``.

Per the spec, the external CV2 dependency (``cv2.matchTemplate``) lives
inside ``_collect_scale_cands`` (see the ``errors`` section), so the tests
mock the instance methods ``_collect_scale_cands``, ``_nms_badges``, and
``_cands_to_badges`` with ``unittest.mock`` and exercise the wiring of
``_detect_at_scale`` itself.

``errors`` section of the spec (documented here only, NOT tested, per the
test-generation rules):

- condition: 'スケール s 倍後のテンプレートの幅または高さが gray より大きい'
  behavior: 'cv2.matchTemplate の制約により cv2.error が送出される（_collect_scale_cands 内）'
- condition: 'gray が有効な画像配列でない（None、dtype 不適合等）'
  behavior: 'cv2.matchTemplate 由来の例外（cv2.error 等）が送出される'
"""

from types import SimpleNamespace
from unittest import mock

from src.video.card_detector import CardDetector


def test_edge_01():
    """
    input: gray がスケール s で self.badge_threshold 以上のマッチを一切持たない画像
    expected: _collect_scale_cands が [] を返し、本関数も [] を返す（_nms_badges 未呼び出し）
    """
    detector = CardDetector.__new__(CardDetector)
    gray = SimpleNamespace(shape=(240, 426))
    detector._collect_scale_cands = mock.Mock(return_value=[])
    detector._nms_badges = mock.Mock(return_value=[])
    result = detector._detect_at_scale(gray, 1.0)
    assert result == []
    # 仕様ステップ1: s は単一要素リスト [s] として渡される
    detector._collect_scale_cands.assert_called_once_with(gray, [1.0])
    # 候補が空なら _nms_badges は呼び出されない
    detector._nms_badges.assert_not_called()


def test_edge_02():
    """
    input: gray に1箇所のみピークが閾値超過（score_m=0.9, score_l=0.3）
    expected: [(x, y, int(124*s), int(37*s), 'member')]（x, y はピーク座標）を返す
    """
    detector = CardDetector.__new__(CardDetector)
    gray = SimpleNamespace(shape=(240, 426))
    s = 1.5
    cand = [10.0, 20.0, s, 0.9, 0.3]
    detector._collect_scale_cands = mock.Mock(return_value=[cand])
    detector._nms_badges = mock.Mock(return_value=[cand])
    detector._cands_to_badges = mock.Mock(
        return_value=[(10, 20, int(124 * s), int(37 * s), "member")]
    )
    result = detector._detect_at_scale(gray, s)
    # int(124 * 1.5) = 186、int(37 * 1.5) = 55。score_m >= score_l なら 'member'
    assert result == [(10, 20, 186, 55, "member")]
    detector._nms_badges.assert_called_once_with([cand])
    detector._cands_to_badges.assert_called_once_with([cand])


def test_edge_03():
    """
    input: s = 1.0 かつ1件以上マッチ
    expected: 返り値の各要素は w = 124, h = 37（int(124*1.0), int(37*1.0)）
    """
    detector = CardDetector.__new__(CardDetector)
    gray = SimpleNamespace(shape=(240, 426))
    cands = [
        [5.0, 6.0, 1.0, 0.8, 0.2],
        [100.0, 30.0, 1.0, 0.4, 0.7],
    ]
    detector._collect_scale_cands = mock.Mock(return_value=cands)
    detector._nms_badges = mock.Mock(return_value=cands)
    detector._cands_to_badges = mock.Mock(
        return_value=[
            (5, 6, 124, 37, "member"),
            (100, 30, 124, 37, "leader"),
        ]
    )
    result = detector._detect_at_scale(gray, 1.0)
    assert result == [
        (5, 6, 124, 37, "member"),
        (100, 30, 124, 37, "leader"),
    ]
    # s = 1.0 のため各要素は w = int(124 * 1.0) = 124、h = int(37 * 1.0) = 37
    assert [(w, h) for (_x, _y, w, h, _role) in result] == [
        (124, 37),
        (124, 37),
    ]
    detector._collect_scale_cands.assert_called_once_with(gray, [1.0])
