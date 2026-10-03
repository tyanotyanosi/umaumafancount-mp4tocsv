"""Tests for ``scripts.measure_shift.crop_gray``.

Specification: docs/00-Architecture/scripts_measure_shift__crop_gray.yaml

``crop_gray(f)`` crops the fixed list region (rows Y0:Y1=470:960, columns
X0:X1=580:1500) from a BGR video frame and converts it to grayscale using
``cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)``, returning the result (per the
spec's ``purpose`` and ``behavior``). It uses the module constants
``X0=580, X1=1500, Y0=470, Y1=960`` and performs no validation of ``f``.

Mocked / stand-in dependencies: none — ``crop_gray`` only calls the installed
``cv2`` library, which is used as a real object (no mocking).

``errors`` section of the spec (documented here only, NOT tested, per the
test-generation rules):

- condition: "スライス結果が空（f が470行未満または580列未満など）、または f が3チャネルBGR画像でない"
  behavior: "cv2.cvtColor が cv2.error（OpenCV例外。プロジェクトvenvのcv2 5.0.0で計測）を送出し、呼び出し元に伝播する。この関数には例外処理がない"
"""

import cv2
import numpy as np
import pytest

from scripts.measure_shift import crop_gray


def test_edge_01():
    """
    input: shape (960, 1500, 3) の f（フルクロップを得られる最小サイズ）
    expected: shape (490, 920) の2次元グレースケール配列を返す
    """
    f = np.zeros((960, 1500, 3), dtype=np.uint8)
    assert crop_gray(f).shape == (490, 920)


def test_edge_02():
    """
    input: shape (500, 1600, 3) の f
    expected: スライス f[470:960, 580:1500] は 30行 x 920列 となり、shape (30, 920) のグレースケール配列を返す
    """
    f = np.zeros((500, 1600, 3), dtype=np.uint8)
    assert crop_gray(f).shape == (30, 920)


def test_edge_03():
    """
    input: shape (100, 100, 3) の f（スライス領域が空）
    expected: スライス結果が空配列 (shape (0, 0, 3)) となり、cv2.cvtColor が cv2.error を送出する（プロジェクトvenvのcv2 5.0.0で計測）。例外処理がなく呼び出し元に伝播する
    """
    f = np.zeros((100, 100, 3), dtype=np.uint8)
    with pytest.raises(cv2.error):
        crop_gray(f)


def test_edge_04():
    """
    input: f = None
    expected: スライス式で TypeError（None は subscript 不可）が送出され、呼び出し元に伝播する
    """
    with pytest.raises(TypeError):
        crop_gray(None)
