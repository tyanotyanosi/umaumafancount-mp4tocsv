"""Tests for ``src.video.card_detector.CardDetector._detect_coarse``.

Specification: docs/00-Architecture/src_video_card_detector__CardDetector__detect_coarse.yaml

``CardDetector._detect_coarse`` downsamples the frame by ``self.coarse_scale``
to detect badges at low resolution, and returns approximate badge positions
in the original-resolution coordinates:

1. Read ``cs = self.coarse_scale`` and take ``h, w`` from ``gray.shape``.
2. Compute the downsampled size ``sh = max(1, int(h * cs))``,
   ``sw = max(1, int(w * cs))``.
3. Resize ``gray`` to ``(sw, sh)`` with ``cv2.resize`` (INTER_AREA
   interpolation) to produce ``small``.
4. Correct the scale candidates to the downsampled resolution:
   ``S_c = [s * cs for s in S]``, ``s0_c = s0 * cs``.
5. Call ``self._detect_badges_multiscale(small, S_c, s0_c)`` to get
   ``(badges, s_w)``.
6. Return ``[]`` if ``badges`` is empty (early return at spec line 293).
7. Otherwise append ``[x / cs, y / cs, s_w / cs, role]`` for each
   ``badges`` element ``(x, y, bw, bh, role)`` and return that list
   (``bw``, ``bh`` discarded; every element shares the same ``s_w / cs``).

The spec indicates external CV2 dependencies for this function (``cv2.resize``
directly, and ``cv2.matchTemplate`` inside the delegated multiscale path),
so the tests mock ``cv2.resize`` and the instance's
``_detect_badges_multiscale`` method with ``unittest.mock`` and then exercise
the pure coordinate/scale mapping logic directly.

``errors`` section of the spec (documented here only, NOT tested, per the
test-generation rules):

- condition: 'gray.shape が2次元でない（例: 3チャネルカラー）'
  behavior: 'h, w = gray.shape のアンパッキングで ValueError。try/except なしで呼び出し元へ伝播。'
- condition: '_collect_scale_cands 内で cv2.matchTemplate が呼ばれるとき、リサイズ後テンプレートが下サンプリング画像 small より大きい'
  behavior: '本関数側にガードはない。_collect_scale_cands が cv2.matchTemplate を直接呼び出す（186-187行目）ため、テンプレートが画像より大きいときの OpenCV の挙動（cv2.error 等を送出）が捕捉されず本関数を透過して伝播し得る。実際にこの入力が生じるかはテンプレートの実サイズ次第で未確認。'
"""

from types import SimpleNamespace
from unittest import mock

import pytest

from src.video.card_detector import CardDetector


def test_edge_01():
    """
    input: S = []（gray は任意の2次元配列、s0 = 1.0、coarse_scale = 0.5）
    expected: 空リスト [] を返す。S_c も空になるため _collect_scale_cands は候補を生成せず、_detect_badges_multiscale は ([], None) を返し、293行目で早期 return。例外は発生しない。
    """
    detector = CardDetector.__new__(CardDetector)
    detector.coarse_scale = 0.5
    gray = SimpleNamespace(shape=(100, 200))
    small = SimpleNamespace(shape=(50, 100))
    detector._detect_badges_multiscale = mock.Mock(return_value=([], None))
    with mock.patch("cv2.resize", return_value=small) as resize_mock:
        result = detector._detect_coarse(gray, [], 1.0)
    assert result == []
    # 仕様ステップ4: S_c = [s * 0.5 for s in []] = []、s0_c = 1.0 * 0.5 = 0.5
    detector._detect_badges_multiscale.assert_called_once_with(small, [], 0.5)
    # side_effects に従い cv2.resize は1回呼ばれる
    resize_mock.assert_called_once()


def test_edge_02():
    """
    input: gray 任意、S = [1.0]、s0 = 1.0、_detect_badges_multiscale を ([], None) を返すようモック
    expected: 空リスト [] を返す（293行目の if not badges 分岐）。
    """
    detector = CardDetector.__new__(CardDetector)
    detector.coarse_scale = 0.5
    gray = SimpleNamespace(shape=(100, 200))
    small = SimpleNamespace(shape=(50, 100))
    detector._detect_badges_multiscale = mock.Mock(return_value=([], None))
    with mock.patch("cv2.resize", return_value=small):
        result = detector._detect_coarse(gray, [1.0], 1.0)
    assert result == []
    # 仕様ステップ4: S_c = [1.0 * 0.5] = [0.5]、s0_c = 1.0 * 0.5 = 0.5
    detector._detect_badges_multiscale.assert_called_once_with(small, [0.5], 0.5)


def test_edge_03():
    """
    input: coarse_scale = 0.5、gray が 200x100（h x w）、_detect_badges_multiscale を ([(10, 20, 62, 19, "member")], 1.0) を返すようモック
    expected: [[20.0, 40.0, 2.0, "member"]] を返す。x / cs = 20.0、y / cs = 40.0、s_w / cs = 2.0、bw = 62 と bh = 19 は破棄される。
    """
    detector = CardDetector.__new__(CardDetector)
    detector.coarse_scale = 0.5
    gray = SimpleNamespace(shape=(200, 100))
    small = SimpleNamespace(shape=(50, 100))
    detector._detect_badges_multiscale = mock.Mock(
        return_value=([(10, 20, 62, 19, "member")], 1.0)
    )
    with mock.patch("cv2.resize", return_value=small) as resize_mock:
        result = detector._detect_coarse(gray, [1.0], 1.0)
    assert result == [[20.0, 40.0, 2.0, "member"]]
    # sh = max(1, int(200 * 0.5)) = 100、sw = max(1, int(100 * 0.5)) = 50
    assert resize_mock.call_args[0] == (gray, (50, 100))
    detector._detect_badges_multiscale.assert_called_once_with(small, [0.5], 0.5)


def test_edge_04():
    """
    input: gray が3チャネルカラー配列（shape が (h, w, 3)）
    expected: h, w = gray.shape のアンパッキングで ValueError（3要素を2変数へ）が発生し、捕捉されないまま呼び出し元へ伝播する。
    """
    detector = CardDetector.__new__(CardDetector)
    detector.coarse_scale = 0.5
    gray = SimpleNamespace(shape=(100, 200, 3))
    with pytest.raises(ValueError):
        detector._detect_coarse(gray, [1.0], 1.0)
