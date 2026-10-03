"""Tests for ``scripts.extract_keyframes.gray``.

Specification: docs/00-Architecture/scripts_extract_keyframes__gray.yaml

``gray`` converts a BGR color image ``f`` to a grayscale image with
``cv2.cvtColor`` and returns it:

- call ``cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)`` and return its result
  as-is

No state changes: only the cv2.cvtColor call; no file, network, or
global-state operations.

Mocked / stand-in dependencies (per the test-generation rules):
none — ``cv2.cvtColor`` is the real library (cv2 5.0.0) and the
inputs are plain numpy uint8 arrays / plain objects as in the spec
edge cases.

``errors`` section of the spec (documented here only, NOT tested, per
the test-generation rules):

- condition: fがcv2.cvtColorに対して有効な画像入力でない（shape・dtype・チャンネル数が不正、非Matオブジェクト等）
  behavior: cv2.cvtColorがOpenCVエラーを送出する（捕捉されない。例外種別・メッセージは未確認）
"""

import cv2
import numpy as np
import pytest

from scripts.extract_keyframes import gray


def test_edge_01():
    """
    input: f = np.zeros((4, 4, 3), dtype=np.uint8)（全ゼロのBGR画像）
    expected: shape (4, 4) の全ゼロ単一チャネル画像が返る（出力の内容・dtypeはcv2.cvtColorのBGR→グレースケール変換契約に従う・未確認）
    """
    f = np.zeros((4, 4, 3), dtype=np.uint8)
    g = gray(f)
    assert g.shape == (4, 4)
    assert np.all(g == 0)


def test_edge_02():
    """
    input: f = shape (H, W, 3) の任意uint8 ndarray
    expected: cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)の戻り値（単一チャネル・shape (H, W)・dtype uint8 と推定されるが出力仕様はcv2の契約であり未確認）が返る
    """
    f = np.random.default_rng(0).integers(
        0, 256, size=(6, 8, 3), dtype=np.uint8)
    g = gray(f)
    expected = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    assert g.shape == (6, 8)
    assert g.dtype == np.uint8
    assert np.array_equal(g, expected)


def test_edge_03():
    """
    input: f = 画像で無いオブジェクト（例、Python list、0次元ndarray）
    expected: cv2.cvtColorがOpenCVエラーを送出する（本関数にはtry/exceptが無いため例外がそのまま伝播する。例外種別は未確認）
    """
    with pytest.raises(cv2.error):
        gray([0, 0, 0])
    with pytest.raises(cv2.error):
        gray(np.array(5))
