"""scripts.measure_tail.gray の仕様書駆動 pytest テスト。

spec: docs/00-Architecture/scripts_measure_tail__gray.yaml
function: scripts.measure_tail.gray

検証する行動:
  cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) を呼び出し、その戻り値をそのまま返す。
  関数本体は型・形状・null 検証を行わず、無効入力時の挙動は cv2.cvtColor の挙動に委ねられる。

モックした依存: なし（実 cv2 を使用）。

errors セクション（記録のみ・テストしない）:
  - f が有効な 3 チャネル BGR 画像でない（1 チャネル、None、画像以外の型）→
    関数自身は検証せず cv2.cvtColor が例外を送出し、呼び出し元（main）へ伝播する。
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

import scripts.measure_tail as m


def test_edge_01():
    """input: f が1チャネル（グレースケール）の ndarray
    expected: 関数は遮断せず f を cv2.cvtColor に渡す。OpenCV 仕様上 BGR2GRAY は1チャネル入力を受理せず cv2.error が発生する（未確認: 本環境で cv2 の実際の例外種別・メッセージは未検証）
    """
    with pytest.raises(cv2.error):
        m.gray(np.zeros((10, 10), np.uint8))


def test_edge_02():
    """input: f が None
    expected: 関数は遮断せず None を cv2.cvtColor に渡す。cv2 が例外を送出する（種別・メッセージは cv2 の挙動に従い未確認）
    """
    with pytest.raises(cv2.error):
        m.gray(None)
