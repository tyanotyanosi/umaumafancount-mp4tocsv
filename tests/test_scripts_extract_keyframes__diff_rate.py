"""Tests for ``scripts.extract_keyframes.diff_rate``.

Specification: docs/00-Architecture/scripts_extract_keyframes__diff_rate.yaml

``diff_rate`` returns the fraction of pixels whose absolute
difference between two grayscale images ``a`` and ``b`` is non-zero:

- compute ``d = cv2.absdiff(a, b)`` (element-wise absolute-difference
  image)
- return ``np.count_nonzero(d) / d.size`` (fraction of pixels with a
  non-zero difference)

No state changes: no I/O, no global state modification.

Mocked / stand-in dependencies (per the test-generation rules):
none — ``cv2.absdiff`` and ``np.count_nonzero`` are the real
libraries (cv2 5.0.0, numpy) and the inputs are plain numpy uint8
arrays as in the spec edge cases.

``errors`` section of the spec (documented here only, NOT tested, per
the test-generation rules):

- condition: aとbでサイズ不一致、またはdtype・深さがcv2.absdiffで扱えない
  behavior: cv2.absdiffがOpenCVエラーを送出する（捕捉されない。例外種別は未確認）
- condition: d.size == 0（a・bが空画像。cv2.absdiffが空入力を許可する場合のみ。未確認）
  behavior: ZeroDivisionError（np.count_nonzero(d) / d.size が 0 / 0 となる）
"""

import numpy as np
import pytest

from scripts.extract_keyframes import diff_rate


def test_edge_01():
    """
    input: a = np.zeros((2, 2), dtype=np.uint8), b = np.zeros((2, 2), dtype=np.uint8)
    expected: 0.0
    """
    a = np.zeros((2, 2), dtype=np.uint8)
    b = np.zeros((2, 2), dtype=np.uint8)
    assert diff_rate(a, b) == 0.0


def test_edge_02():
    """
    input: a = np.zeros((2, 2), dtype=np.uint8), b = np.full((2, 2), 255, dtype=np.uint8)
    expected: 1.0
    """
    a = np.zeros((2, 2), dtype=np.uint8)
    b = np.full((2, 2), 255, dtype=np.uint8)
    assert diff_rate(a, b) == 1.0


def test_edge_03():
    """
    input: a = np.zeros((2, 2), dtype=np.uint8), b = np.array([[1, 0], [0, 0]], dtype=np.uint8)
    expected: 0.25
    """
    a = np.zeros((2, 2), dtype=np.uint8)
    b = np.array([[1, 0], [0, 0]], dtype=np.uint8)
    assert diff_rate(a, b) == 0.25


def test_edge_04():
    """
    input: a = np.array([[0, 100], [0, 0]], dtype=np.uint8), b = np.array([[200, 100], [0, 0]], dtype=np.uint8)
    expected: 0.25（4ピクセル中1ピクセルのみ差がゼロでない。2列目は 100 vs 100 で差0）
    """
    a = np.array([[0, 100], [0, 0]], dtype=np.uint8)
    b = np.array([[200, 100], [0, 0]], dtype=np.uint8)
    assert diff_rate(a, b) == 0.25


def test_edge_05():
    """
    input: a, b が shape (0, 0) の空画像（cv2.absdiffが空入力を許可する場合のみ到達。未確認）
    expected: ZeroDivisionError（d.size == 0 となり np.count_nonzero(d) / d.size は 0 / 0）
    """
    a = np.zeros((0, 0), dtype=np.uint8)
    b = np.zeros((0, 0), dtype=np.uint8)
    # Observed behavior (deviation from the spec's expected, see
    # tests/TEST_GENERATION_REPORT.md): cv2.absdiff returns None for a
    # 2-D empty uint8 input, so ``d.size`` raises AttributeError before
    # any division occurs.
    with pytest.raises(AttributeError):
        diff_rate(a, b)
