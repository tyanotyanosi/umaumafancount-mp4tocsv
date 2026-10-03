"""scripts.measure_tail.dr の仕様書駆動 pytest テスト。

spec: docs/00-Architecture/scripts_measure_tail__dr.yaml
function: scripts.measure_tail.dr

検証する行動:
  d = cv2.absdiff(a, b) を計算し、np.count_nonzero(d) / d.size（非ゼロ要素数 ÷ 要素総数）を
  float（np.float64）として返す。shape・dtype・null 性は検証せず、無効入力時の挙動は
  cv2.absdiff と numpy の挙動に委ねられる。

モックした依存: なし（実 cv2 / numpy を使用）。

errors セクション（記録のみ・テストしない）:
  - a と b の shape が異なる、または cv2.absdiff が処理できない入力 → cv2.absdiff が例外を送出し伝播する。
  - a と b が空配列で d.size が 0 → 0/0 の評価結果は numpy バージョンに依存（未確認）。

spec 乖離（下方報告に集計）:
  - edge_04: 仕様 expected は「nan（RuntimeWarning 付き）か例外（未確認）」とするが、実 cv2 5.0.0 では
    1-D 空配列 (0,) uint8 の cv2.absdiff が 4x1 float64 ゼロ配列を返すため d.size=4・count_nonzero=0 で
    0.0 が返る（nan ではない）。観測挙動を assert。
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

import scripts.measure_tail as m


def test_edge_01():
    """input: a と b が同一 shape で全要素等しい（例: a is b、shape (4,4) uint8 で全要素 128）
    expected: d の全要素が 0 であり、戻り値は 0.0（float）
    """
    a = np.full((4, 4), 128, np.uint8)
    assert m.dr(a, a) == 0.0


def test_edge_02():
    """input: a と b が同一 shape (2,2) uint8 で a が全 0、b が全 255
    expected: d の全要素が 255（非ゼロ）であり、戻り値は 1.0（float）
    """
    assert m.dr(np.zeros((2, 2), np.uint8), np.full((2, 2), 255, np.uint8)) == 1.0


def test_edge_03():
    """input: a が shape (2,2)、b が shape (3,3) と形状が異なる
    expected: 関数は検証せず cv2.absdiff に渡す。OpenCV 仕様上 cv2.error が送出される（未確認: 本環境で未検証）
    """
    with pytest.raises(cv2.error):
        m.dr(np.zeros((2, 2), np.uint8), np.zeros((3, 3), np.uint8))


def test_edge_04():
    """input: a と b が同一 shape の空配列（例: ともに np.zeros((0,), dtype=np.uint8)）
    expected: cv2.absdiff が空配列を受理した場合、d.size が 0 となり返り式は 0 / 0 を評価する。結果が float の nan（RuntimeWarning 付き）になるか例外になるかは numpy の挙動に従う（未確認: 本環境で未検証）。cv2.absdiff が空配列を受理するか自体も未確認
    """
    # 備考（spec 乖離）: 実 cv2 5.0.0 では 1-D 空配列 (0,) の absdiff が 4x1 float64 ゼロ配列を返すため
    # d.size=4・count_nonzero=0 で 0.0 が返る（nan ではなく 0.0）。観測挙動を assert。
    assert m.dr(np.zeros((0,), np.uint8), np.zeros((0,), np.uint8)) == 0.0
