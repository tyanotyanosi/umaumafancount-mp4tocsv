"""Tests for ``scripts.diagnose_diff.gray``.

Specification: docs/00-Architecture/scripts_diagnose_diff__gray.yaml

``gray`` converts a BGR image to a grayscale image and returns the
result (a thin wrapper around ``cv2.cvtColor``):

- call ``cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)`` and return its result
  as-is

The function is pure: no file I/O, no network I/O, no state changes.

Mocked / stand-in dependencies (per the test-generation rules):
none — ``cv2.cvtColor`` is the real library (cv2 5.0.0) and the
inputs are plain numpy uint8 arrays as in the spec edge cases.

``errors`` section of the spec (documented here only, NOT tested, per
the test-generation rules):

- condition: f のチャンネル数がCOLOR_BGR2GRAYのコードと不一致（例: 1チャンネル、4チャンネル）
  behavior: cv2.cvtColorがエラーを送出する（このファイルからは確認不可、missing参照）
- condition: f がndarrayでない（例: None）
  behavior: cv2.cvtColorがエラーを送出する（このファイルからは確認不可、missing参照）
"""

import cv2
import numpy as np
import pytest

from scripts.diagnose_diff import gray


def test_edge_01():
    """
    input: 'f が shape (h, w, 3) の3チャンネル画像'
    expected: 'shape (h, w) の1チャンネル画像が返る'
    """
    f = np.full((5, 7, 3), 200, dtype=np.uint8)
    g = gray(f)
    assert g.shape == (5, 7)


def test_edge_02():
    """
    input: 'f が shape (h, w) の1チャンネル画像'
    expected: 'cv2.cvtColorがエラーを送出する（COLOR_BGR2GRAYは3チャンネル入力を要求。具体的な例外種別はこのファイルからは確認不可、missing参照）'
    """
    f = np.full((4, 4), 128, dtype=np.uint8)
    with pytest.raises(cv2.error):
        gray(f)
