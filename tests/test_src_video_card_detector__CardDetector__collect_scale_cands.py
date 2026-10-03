"""Tests for ``src.video.card_detector.CardDetector._collect_scale_cands``.

Specification: docs/00-Architecture/src_video_card_detector__CardDetector__collect_scale_cands.yaml

``CardDetector._collect_scale_cands`` (per the spec's ``purpose`` field)
runs member/leader badge template matching for every scale ``s`` in the
scale candidate set ``S`` and collects a list of badge candidates of the
form ``[x, y, s, score_m, score_l]`` with the global coordinate offsets
``ox``/``oy`` applied.

Behavior (per the spec's ``behavior`` field):

- ``pyramid = self._build_scale_pyramid(S)`` is obtained first (templates
  pre-resized at each s; definition is out of read scope and mocked here).
- For each ``s`` in order:
  - ``m_t = pyramid[s]["member"]``, ``l_t = pyramid[s]["leader"]``
  - ``res_m = cv2.matchTemplate(gray, m_t, cv2.TM_CCOEFF_NORMED)``
  - ``res_l = cv2.matchTemplate(gray, l_t, cv2.TM_CCOEFF_NORMED)``
  - ``m_peaks = self._find_peaks(res_m, self.badge_threshold, s)`` and
    ``l_peaks = self._find_peaks(res_l, self.badge_threshold, s)``
    (peaks are ``(x, y, score)`` triples; the score is unused here)
  - every member peak ``(x, y)`` unconditionally appends
    ``[ox + x, oy + y, s, float(res_m[y, x]), self._sample(res_l, y, x)]``
  - every leader peak ``(x, y)`` appends
    ``[ox + x, oy + y, s, self._sample(res_m, y, x), float(res_l[y, x])]``
    only when ``self._close(cands, gx, gy, int(24 * s))`` is False
- ``cands`` is returned; ``gray`` is not modified.

External dependencies are mocked per the test-generation rules:
``cv2.matchTemplate`` (OpenCV) and the out-of-scope helper methods
(``_build_scale_pyramid``, ``_find_peaks``, ``_close``, ``_sample``) are
replaced with ``unittest.mock`` objects; the detector instance is created
via ``CardDetector.__new__`` (bypassing ``__init__``) with
``badge_threshold`` set, since the real method only reads those.

``errors`` section of the spec (documented here only, NOT tested, per the
test-generation rules):

- condition: 'S の s が pyramid のキーに存在しない、または pyramid[s] に member / leader キーがない'
  behavior: 'KeyError（未処理で呼び出し側に伝播）'
- condition: 'gray がテンプレートより小さい（テンプレートが画像より大きい）'
  behavior: '未確認。コードにガードはなく、その場合の cv2.matchTemplate の挙動はこのファイル内に記載がない（OpenCV の実装依存）'
- condition: '_find_peaks が結果マップ範囲外の (y, x) を返す'
  behavior: '未確認。res_m[y, x]・res_l[y, x] の直接インデックス参照にガードがなく、範囲外座標では IndexError が発生し得る（_find_peaks が範囲内座標を返すことが前提）'
"""

from unittest import mock

import numpy as np

from src.video.card_detector import CardDetector


def _make_detector():
    """Create a ``CardDetector`` instance without running ``__init__``.

    The real method only reads ``self.badge_threshold`` plus the mocked
    helper methods, so a bare instance with that attribute set is a
    faithful stand-in without any template loading.
    """
    detector = CardDetector.__new__(CardDetector)
    detector.badge_threshold = 0.5
    return detector


def test_edge_01():
    """
    input: S = []（gray は有効な 2 次元画像）
    expected: ループ体が実行されず、空リスト [] が返る（_build_scale_pyramid が空リストの S を例外なしで受け入れることを前提とする。その挙動は未確認）
    """
    detector = _make_detector()
    detector._build_scale_pyramid = mock.MagicMock(return_value={})
    detector._find_peaks = mock.MagicMock()
    detector._sample = mock.MagicMock()
    detector._close = mock.MagicMock()

    gray = np.zeros((100, 100), dtype=np.uint8)

    with mock.patch("cv2.matchTemplate") as match:
        result = detector._collect_scale_cands(gray, [])

    assert result == []
    match.assert_not_called()
    detector._build_scale_pyramid.assert_called_once_with([])
    detector._find_peaks.assert_not_called()
    assert (gray == 0).all()


def test_edge_02():
    """
    input: scale s で member ピークが (x, y) にあり、かつ (y, x) が res_l の範囲外である場合（member/leader のテンプレート幅差による結果マップサイズ差）
    expected: 候補 [ox + x, oy + y, s, float(res_m[y, x]), 0.0] が追加される（score_l は _sample の境界チェックにより 0.0 になる）
    """
    detector = _make_detector()
    detector._build_scale_pyramid = mock.MagicMock(
        return_value={1.0: {"member": "m_t", "leader": "l_t"}}
    )
    # member peak (30, 75); no leader peaks
    detector._find_peaks = mock.MagicMock(side_effect=[[(30, 75, 0.7)], []])
    # (y, x) = (75, 30) is out of res_l range -> boundary check yields 0.0
    detector._sample = mock.MagicMock(return_value=0.0)
    detector._close = mock.MagicMock(return_value=False)

    # member result map is 100x100 so (75, 30) is inside; leader result map
    # is 50x50 so the same (y, x) is outside it
    res_m = np.full((100, 100), 0.75, dtype=np.float32)
    res_l = np.full((50, 50), 0.5, dtype=np.float32)
    gray = np.zeros((200, 200), dtype=np.uint8)

    with mock.patch("cv2.matchTemplate", side_effect=[res_m, res_l]) as match:
        result = detector._collect_scale_cands(gray, [1.0])

    assert result == [[30, 75, 1.0, 0.75, 0.0]]
    assert match.call_count == 2
    detector._sample.assert_called_once_with(res_l, 75, 30)


def test_edge_03():
    """
    input: scale s で leader ピークが (x, y) にあり、かつ (y, x) が res_m の範囲外である場合
    expected: _close によるスキップが起きなければ、候補 [gx, gy, s, 0.0, float(res_l[y, x])] が追加される（score_m は _sample により 0.0 になる）
    """
    detector = _make_detector()
    detector._build_scale_pyramid = mock.MagicMock(
        return_value={1.0: {"member": "m_t", "leader": "l_t"}}
    )
    # no member peaks; leader peak (40, 20)
    detector._find_peaks = mock.MagicMock(side_effect=[[], [(40, 20, 0.9)]])
    # (y, x) = (20, 40) is out of res_m range -> score_m becomes 0.0
    detector._sample = mock.MagicMock(return_value=0.0)
    # not skipped by _close
    detector._close = mock.MagicMock(return_value=False)

    # leader result map is 100x100 so (20, 40) is inside; member result map
    # is 10x10 so the same (y, x) is outside it
    res_m = np.full((10, 10), 0.75, dtype=np.float32)
    res_l = np.full((100, 100), 0.75, dtype=np.float32)
    gray = np.zeros((200, 200), dtype=np.uint8)

    with mock.patch("cv2.matchTemplate", side_effect=[res_m, res_l]):
        result = detector._collect_scale_cands(gray, [1.0], ox=5, oy=10)

    # gx, gy = ox + x, oy + y (global coordinates)
    assert result == [[45, 30, 1.0, 0.0, 0.75]]
    detector._sample.assert_called_once_with(res_m, 20, 40)
    detector._close.assert_called_once_with(result, 45, 30, int(24 * 1.0))


def test_edge_04():
    """
    input: 既存候補と同一位置（距離 0）に leader ピークが存在する場合
    expected: _close がその位置を近いと判定（True を返す）すれば候補は cands に追加されない（_close の定義は読取範囲外のため、閾値 int(24 * s) での判定内容自体は未確認）
    """
    detector = _make_detector()
    detector._build_scale_pyramid = mock.MagicMock(
        return_value={1.0: {"member": "m_t", "leader": "l_t"}}
    )
    # member peak and leader peak at the same position (30, 75)
    detector._find_peaks = mock.MagicMock(
        side_effect=[[(30, 75, 0.7)], [(30, 75, 0.9)]]
    )
    detector._sample = mock.MagicMock(return_value=0.6)
    # _close judges the position as close -> leader candidate is skipped
    detector._close = mock.MagicMock(return_value=True)

    res_m = np.full((100, 100), 0.75, dtype=np.float32)
    res_l = np.full((100, 100), 0.5, dtype=np.float32)
    gray = np.zeros((200, 200), dtype=np.uint8)

    with mock.patch("cv2.matchTemplate", side_effect=[res_m, res_l]):
        result = detector._collect_scale_cands(gray, [1.0])

    # the leader candidate is skipped: only the member candidate remains
    assert result == [[30, 75, 1.0, 0.75, 0.6]]
    assert len(result) == 1
    detector._close.assert_called_once_with(result, 30, 75, 24)


def test_edge_05():
    """
    input: 同一位置が S の 2 つの異なる scale s1, s2 両方で member ピークとして検出される場合
    expected: member ピークには重複排除処理がないため、s1 と s2 の両方の候補が cands に追加される
    """
    detector = _make_detector()
    detector._build_scale_pyramid = mock.MagicMock(
        return_value={
            1.0: {"member": "m_t1", "leader": "l_t1"},
            2.0: {"member": "m_t2", "leader": "l_t2"},
        }
    )
    # the same member peak (30, 75) is detected at both scales
    # s1 = 1.0 and s2 = 2.0; no leader peaks at either scale
    detector._find_peaks = mock.MagicMock(
        side_effect=[
            [(30, 75, 0.7)], [],
            [(30, 75, 0.8)], [],
        ]
    )
    detector._sample = mock.MagicMock(side_effect=[0.1, 0.2])
    detector._close = mock.MagicMock(return_value=False)

    res_m1 = np.full((100, 100), 0.75, dtype=np.float32)
    res_l1 = np.full((50, 50), 0.5, dtype=np.float32)
    res_m2 = np.full((80, 80), 0.75, dtype=np.float32)
    res_l2 = np.full((40, 40), 0.5, dtype=np.float32)
    gray = np.zeros((200, 200), dtype=np.uint8)

    with mock.patch(
        "cv2.matchTemplate", side_effect=[res_m1, res_l1, res_m2, res_l2]
    ) as match:
        result = detector._collect_scale_cands(gray, [1.0, 2.0])

    # no dedup for member peaks: both candidates are added
    assert result == [[30, 75, 1.0, 0.75, 0.1], [30, 75, 2.0, 0.75, 0.2]]
    assert len(result) == 2
    assert match.call_count == 4
