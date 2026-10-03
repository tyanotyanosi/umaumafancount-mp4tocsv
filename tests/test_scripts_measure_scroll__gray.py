"""Tests for ``scripts.measure_scroll.gray``.

Specification: docs/00-Architecture/scripts_measure_scroll__gray.yaml

``gray(f)`` converts a BGR image array to a grayscale image using
``cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)`` and returns the result as-is (per the
spec's ``purpose`` and ``behavior``). It performs no validation of ``f``.

Mocked / stand-in dependencies: none — ``gray`` only calls the installed
``cv2`` library, which is used as a real object (no mocking).

``errors`` section of the spec (documented here only, NOT tested, per the
test-generation rules):

- condition: 'f が BGR2GRAY に対して無効なチャネル数・dtype・サイズの配列'
  behavior: 'cv2.error が発生し、関数内で捕まらず伝播する。'
"""

import cv2
import numpy as np
import pytest

from scripts.measure_scroll import gray


def test_edge_01():
    """
    input: shape (h, w, 3) の 3チャネル BGR uint8 配列
    expected: shape (h, w) の2次元配列が返る。
    """
    f = np.zeros((5, 7, 3), dtype=np.uint8)
    result = gray(f)
    assert result.shape == (5, 7)


def test_edge_02():
    """
    input: 4チャネル BGRA 配列
    expected: cv2.error が発生する（BGR2GRAY には4チャネル入力の変換が定義されていない）。
    """
    f = np.zeros((5, 7, 4), dtype=np.uint8)
    # spec 乖離: cv2 5.0.0 では 4 チャネル入力の BGR2GRAY 変換は cv2.error を
    # 送出せず、shape (h, w) の2次元配列を返す（alpha チャネルを無視して
    # 変換に成功する）。観察挙動を assert する。
    result = gray(f)
    assert result.shape == (5, 7)


def test_edge_03():
    """
    input: shape (h, w) の1チャネル配列
    expected: cv2.error が発生する（コードと入力のチャネル数が不一致）。
    """
    f = np.zeros((5, 7), dtype=np.uint8)
    with pytest.raises(cv2.error):
        gray(f)


def test_edge_04():
    """
    input: 画像でないオブジェクト（例: Python int のリスト）
    expected: cv2.error が発生する。
    """
    with pytest.raises(cv2.error):
        gray([1, 2, 3])
