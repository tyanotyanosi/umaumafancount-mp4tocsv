"""Tests for ``src.video.card_detector.CardDetector._detect_badges``.

Specification: docs/00-Architecture/src_video_card_detector__CardDetector__detect_badges.yaml

``CardDetector._detect_badges`` detects badges at the fixed scale
``s = 1.0`` and returns a list of ``(x, y, w, h, role)`` tuples:

1. Call ``self._collect_scale_cands(gray, [1.0])`` to collect badge
   candidates ``[x, y, s, score_m, score_l]`` at the fixed scale 1.0.
2. Pass ALL obtained candidates (including an empty list) to
   ``self._nms_badges(cands)``.
3. Convert with ``self._cands_to_badges(cands)`` into a list of
   ``(x, y, w, h, role)`` and return it. Since ``s`` is fixed at 1.0, every
   element has ``w = 124`` and ``h = 37``.

Unlike ``_detect_at_scale``, this function passes the empty candidate list
to ``_nms_badges`` as well. Per the spec, the external CV2 dependency
(``cv2.matchTemplate``) lives inside ``_collect_scale_cands`` (see the
``errors`` section), so the tests mock the instance methods
``_collect_scale_cands``, ``_nms_badges``, and ``_cands_to_badges`` with
``unittest.mock`` and exercise the wiring of ``_detect_badges`` itself.

``errors`` section of the spec (documented here only, NOT tested, per the
test-generation rules):

- condition: 's=1.0 テンプレートの幅または高さが gray より大きい'
  behavior: 'cv2.matchTemplate の制約により cv2.error が送出される（_collect_scale_cands 内）'
- condition: 'gray が有効な画像配列でない（None、dtype 不適合等）'
  behavior: 'cv2.matchTemplate 由来の例外（cv2.error 等）が送出される'
"""

from types import SimpleNamespace
from unittest import mock

from src.video.card_detector import CardDetector


def test_edge_01():
    """
    input: gray が badge_threshold 以上のマッチを一切持たない画像
    expected: cands = [] が _nms_badges へ渡られ _cands_to_badges([]) となり最終的に [] を返す（_nms_badges が空リストを正常に処理する場合。実装は未確認）
    """
    detector = CardDetector.__new__(CardDetector)
    gray = SimpleNamespace(shape=(240, 426))
    detector._collect_scale_cands = mock.Mock(return_value=[])
    detector._nms_badges = mock.Mock(return_value=[])
    detector._cands_to_badges = mock.Mock(return_value=[])
    result = detector._detect_badges(gray)
    assert result == []
    # 仕様ステップ1: 固定スケール 1.0 は [1.0] として渡される
    detector._collect_scale_cands.assert_called_once_with(gray, [1.0])
    # _detect_at_scale と異なり、空の候補リストも _nms_badges に渡される
    detector._nms_badges.assert_called_once_with([])
    detector._cands_to_badges.assert_called_once_with([])


def test_edge_02():
    """
    input: gray に1箇所マッチ（ピークで score_m=0.9, score_l=0.3）
    expected: [(x, y, 124, 37, 'member')]（x, y はピーク座標。score_m >= score_l なら 'member'、そうでなければ 'leader'）を返す
    """
    detector = CardDetector.__new__(CardDetector)
    gray = SimpleNamespace(shape=(240, 426))
    cand = [12.0, 15.0, 1.0, 0.9, 0.3]
    detector._collect_scale_cands = mock.Mock(return_value=[cand])
    detector._nms_badges = mock.Mock(return_value=[cand])
    detector._cands_to_badges = mock.Mock(
        return_value=[(12, 15, 124, 37, "member")]
    )
    result = detector._detect_badges(gray)
    # s は 1.0 に固定: w = 124、h = 37。score_m (0.9) >= score_l (0.3) なので role = 'member'
    assert result == [(12, 15, 124, 37, "member")]
    detector._collect_scale_cands.assert_called_once_with(gray, [1.0])
    detector._nms_badges.assert_called_once_with([cand])
    detector._cands_to_badges.assert_called_once_with([cand])
