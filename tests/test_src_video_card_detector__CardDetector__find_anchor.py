"""Tests for src.video.card_detector.CardDetector._find_anchor (spec-based).

Spec: docs/00-Architecture/src_video_card_detector__CardDetector__find_anchor.yaml
function: src.video.card_detector.CardDetector._find_anchor
signature: 'def _find_anchor(self, gray, tpl_name: str, threshold: float, x0: int, y0: int, x1: int, y1: int, s: float = 1.0) -> Optional[tuple]'

------------------------------------------------------------------------
`errors` セクション（ルール8によりテスト化せず、文書化のみ）:
  - condition: tpl_name が self.templates にないキー（かつ pyramid 分岐でも選択されない）
    behavior: KeyError（self.templates[tpl_name] の参照）
  - condition: gray の shape の要素数が 2 ではない（1 次元配列は gray.shape[1] で IndexError、3 次元配列等は cv2.matchTemplate で cv2.error が推定）
    behavior: IndexError または cv2.error（非 ndarray 入力は AttributeError。推定）
  - condition: gray とテンプレートの dtype（depth）が一致しない（例：gray が float32、テンプレートが uint8）
    behavior: cv2.matchTemplate が cv2.error を送出（OpenCV ライブラリ制約。推定・本ファイルでは未検証）
------------------------------------------------------------------------
構築メモ: _find_anchor は self.templates と self._pyramid のみを読む
（仕様書 inputs の self: 「self._pyramid と self.templates を読む（変更しない）」）ため、
CardDetector.__new__(CardDetector) でインスタンスを作成し、それらの属性を
直接設定して構築する。これにより __init__ のテンプレートファイル読み込み
（_load 経由の cv2.imread / cv2.cvtColor）を避け、テストを決定論的にする。

外部依存: cv2（cv2.matchTemplate / cv2.minMaxLoc / cv2.TM_CCOEFF_NORMED）は
仕様書が言及している依存のため、各テストで unittest.mock の mock.patch により
モックする。マッチングスコアは cv2.minMaxLoc の戻り値から制御する。
"""

from unittest import mock

import numpy as np

from src.video.card_detector import CardDetector


def _gray_2d(height, width, value=128):
    """値一様な 2 次元グレースケール配列（uint8）を作る。"""
    return np.full((height, width), value, dtype=np.uint8)


def _pyramid_entry(tpl_gray):
    """_build_scale_pyramid の形に倣い、4 テンプレート名を持つ pyramid[s] エントリを作る。"""
    return {
        "member": tpl_gray,
        "leader": tpl_gray,
        "i_icon": tpl_gray,
        "label": tpl_gray,
    }


def _make_detector(i_icon_gray, pyramid=None):
    """_find_anchor が読む属性（self.templates, self._pyramid）だけを持つ CardDetector を作る。"""
    det = CardDetector.__new__(CardDetector)
    det.templates = {
        "member": {"gray": _gray_2d(8, 16), "w": 16, "h": 8},
        "leader": {"gray": _gray_2d(10, 20), "w": 20, "h": 10},
        "i_icon": {
            "gray": i_icon_gray,
            "w": int(i_icon_gray.shape[1]),
            "h": int(i_icon_gray.shape[0]),
        },
        "label": {"gray": _gray_2d(12, 24), "w": 24, "h": 12},
    }
    det._pyramid = {} if pyramid is None else pyramid
    return det


def test_edge_01():
    """edge_case #1（仕様書の原文引用）

    input: gray が 200x200 配列、x0 = 0, y0 = 0, x1 = 10, y1 = 10、tpl_name = "member"、threshold = 0.7、s = 1.0（テンプレートの幅または高さが 10 を超えるものとする）
    expected: None を返す（x1 - x0 < tw または y1 - y0 < th のため matchTemplate を計算する前に終了）
    """
    # helper 内の member テンプレートは 幅16 x 高さ8（テンプレートの幅が 10 を超える）
    det = _make_detector(i_icon_gray=_gray_2d(8, 8))
    gray = _gray_2d(200, 200)
    with mock.patch("src.video.card_detector.cv2") as cv2_mock:
        result = det._find_anchor(gray, "member", 0.7, 0, 0, 10, 10, 1.0)
    assert result is None
    # 「matchTemplate を計算する前に終了」を検証
    cv2_mock.matchTemplate.assert_not_called()


def test_edge_02():
    """edge_case #2（仕様書の原文引用）

    input: x0 = x1 = 100、y0 = 0, y1 = 200（幅ゼロの領域）、gray は 200x200 以上、他の引数は妥当
    expected: None を返す（クランプ後に x1 - x0 = 0 < tw になるためテンプレート寸法に依存せず常に成立）
    """
    # 小さいテンプレート（8x8）でも常に成立するはず（テンプレート寸法に依存しない）
    det = _make_detector(i_icon_gray=_gray_2d(8, 8))
    gray = _gray_2d(200, 200)
    with mock.patch("src.video.card_detector.cv2") as cv2_mock:
        result = det._find_anchor(gray, "i_icon", 0.7, 100, 0, 100, 200, 1.0)
    assert result is None
    cv2_mock.matchTemplate.assert_not_called()


def test_edge_03():
    """edge_case #3（仕様書の原文引用）

    input: 領域内の最良マッチングスコアが 0.65、threshold = 0.7
    expected: None を返す
    """
    det = _make_detector(i_icon_gray=_gray_2d(8, 8))
    gray = _gray_2d(200, 200)
    with mock.patch("src.video.card_detector.cv2") as cv2_mock:
        # cv2.minMaxLoc は (minVal, maxVal, minLoc, maxLoc) を返す
        # （本 .venv の cv2 5.0.0 の実際の戻り値順を検証して合わせた）
        cv2_mock.minMaxLoc.return_value = (0.0, 0.65, (0, 0), (10, 20))
        result = det._find_anchor(gray, "i_icon", 0.7, 0, 0, 200, 200, 1.0)
    assert result is None
    # マッチングは計算された上、mx < threshold で None になることを検証
    cv2_mock.matchTemplate.assert_called_once()


def test_edge_04():
    """edge_case #4（仕様書の原文引用）

    input: 領域内の最良マッチングスコアが threshold と厳密に一致（mx == 0.7、threshold = 0.7）
    expected: 4 要素タプルを返す（ミス判定は mx < threshold のみであり、等値はヒット）
    """
    # i_icon テンプレートは 8x8（tw = 8, th = 8）
    det = _make_detector(i_icon_gray=_gray_2d(8, 8))
    gray = _gray_2d(200, 200)
    with mock.patch("src.video.card_detector.cv2") as cv2_mock:
        cv2_mock.minMaxLoc.return_value = (0.0, 0.7, (0, 0), (5, 7))
        result = det._find_anchor(gray, "i_icon", 0.7, 0, 0, 200, 200, 1.0)
    assert isinstance(result, tuple) and len(result) == 4
    # ヒット位置の左上を全体フレーム座標に変換: (x0 + px, y0 + py, tw, th)
    assert result == (5, 7, 8, 8)


def test_edge_05():
    """edge_case #5（仕様書の原文引用）

    input: x0 = -50, y0 = -50, x1 = gray の幅 + 100, y1 = gray の高さ + 100（領域が画像外に溢出）
    expected: 例外を投げない。x0, y0 は 0 に、x1, y1 は gray の端にクランプされて全体領域でのマッチングが行われる（スコア次第で None またはタプル）
    """
    det = _make_detector(i_icon_gray=_gray_2d(8, 8))
    gray = _gray_2d(200, 200)  # 幅200 / 高さ200
    with mock.patch("src.video.card_detector.cv2") as cv2_mock:
        # スコア 0.8 >= threshold 0.7 なのでヒット（タプル側を検証する）
        cv2_mock.minMaxLoc.return_value = (0.0, 0.8, (0, 0), (10, 15))
        # x1 = gray の幅 + 100 = 300, y1 = gray の高さ + 100 = 300
        result = det._find_anchor(gray, "i_icon", 0.7, -50, -50, 300, 300, 1.0)
    assert result == (10, 15, 8, 8)
    # クランプ後の検索領域は全体（200x200）であることを検証
    sub = cv2_mock.matchTemplate.call_args.args[0]
    assert sub.shape == (200, 200)


def test_edge_06():
    """edge_case #6（仕様書の原文引用）

    input: s = 0.5、tpl_name = "i_icon"、self._pyramid にキー 0.5 が存在し self._pyramid[0.5] に "i_icon" が存在
    expected: スケール済みテンプレートでマッチングし、返るタプルの w, h はスケール済みテンプレートの幅・高さ
    """
    base = _gray_2d(16, 16)    # 元サイズ i_icon（16x16）
    scaled = _gray_2d(8, 8)    # pyramid[0.5]["i_icon"]（スケール済み 8x8）
    det = _make_detector(i_icon_gray=base, pyramid={0.5: _pyramid_entry(scaled)})
    gray = _gray_2d(200, 200)
    with mock.patch("src.video.card_detector.cv2") as cv2_mock:
        cv2_mock.minMaxLoc.return_value = (0.0, 0.9, (0, 0), (3, 4))
        result = det._find_anchor(gray, "i_icon", 0.7, 0, 0, 200, 200, 0.5)
    # タプルの w, h はスケール済みテンプレートの幅・高さ
    assert result == (3, 4, 8, 8)
    # スケール済みテンプレートでマッチングしたことを検証
    tpl_arg = cv2_mock.matchTemplate.call_args.args[1]
    assert tpl_arg.shape == (8, 8)


def test_edge_07():
    """edge_case #7（仕様書の原文引用）

    input: s = 0.5、tpl_name = "i_icon"、self._pyramid にキー 0.5 が存在しない（または 0.5 下になき "i_icon"）
    expected: 例外を投げず、self.templates["i_icon"]["gray"]（元サイズ）に黙ってフォールバックしてマッチングする
    """
    base = _gray_2d(16, 16)  # 元サイズ i_icon（16x16）
    det = _make_detector(i_icon_gray=base)  # _pyramid = {}（キー 0.5 不在）
    gray = _gray_2d(200, 200)
    with mock.patch("src.video.card_detector.cv2") as cv2_mock:
        cv2_mock.minMaxLoc.return_value = (0.0, 0.9, (0, 0), (3, 4))
        result = det._find_anchor(gray, "i_icon", 0.7, 0, 0, 200, 200, 0.5)
    # 例外を投げず、元サイズテンプレート（w, h = 16, 16）でヒット
    assert result == (3, 4, 16, 16)
    # self.templates["i_icon"]["gray"]（元サイズ）にフォールバックしたことを検証
    tpl_arg = cv2_mock.matchTemplate.call_args.args[1]
    assert tpl_arg.shape == (16, 16)


def test_edge_08():
    """edge_case #8（仕様書の原文引用）

    input: s = 1.0、かつ self._pyramid にキー 1.0 が存在する場合
    expected: self.templates[tpl_name]["gray"] を使う（pyramid は s != 1.0 のときのみ参照される）
    """
    base = _gray_2d(16, 16)  # 元サイズ i_icon（16x16）
    scaled = _gray_2d(8, 8)  # pyramid[1.0]["i_icon"]（s = 1.0 なら参照されないはず）
    det = _make_detector(i_icon_gray=base, pyramid={1.0: _pyramid_entry(scaled)})
    gray = _gray_2d(200, 200)
    with mock.patch("src.video.card_detector.cv2") as cv2_mock:
        cv2_mock.minMaxLoc.return_value = (0.0, 0.9, (0, 0), (3, 4))
        result = det._find_anchor(gray, "i_icon", 0.7, 0, 0, 200, 200, 1.0)
    # self.templates[tpl_name]["gray"]（元サイズ 16x16）を使う
    assert result == (3, 4, 16, 16)
    tpl_arg = cv2_mock.matchTemplate.call_args.args[1]
    assert tpl_arg.shape == (16, 16)


def test_edge_09():
    """edge_case #9（仕様書の原文引用）

    input: tpl_name = "foo"（self.templates にないキー）、他の引数は妥当
    expected: KeyError を送出
    """
    det = _make_detector(i_icon_gray=_gray_2d(8, 8))
    gray = _gray_2d(200, 200)
    with mock.patch("src.video.card_detector.cv2") as cv2_mock:
        captured = None
        try:
            det._find_anchor(gray, "foo", 0.7, 0, 0, 200, 200, 1.0)
        except KeyError as exc:
            captured = exc
    assert isinstance(captured, KeyError)
    # self.templates["foo"] の参照で失敗するためマッチングは計算されない
    cv2_mock.matchTemplate.assert_not_called()
