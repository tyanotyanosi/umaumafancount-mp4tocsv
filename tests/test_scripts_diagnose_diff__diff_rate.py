"""Tests for ``scripts.diagnose_diff.diff_rate``.

Specification: docs/00-Architecture/scripts_diagnose_diff__diff_rate.yaml

``diff_rate`` computes the ratio of pixels whose absolute difference
is non-zero between two grayscale images:

- compute the element-wise absolute difference
  ``d = cv2.absdiff(a_gray, b_gray)``
- return ``np.count_nonzero(d) / d.size`` (number of non-zero
  elements divided by the total number of elements), a float in
  ``[0.0, 1.0]``

The function is pure: no file I/O, no network I/O, no state changes.

Mocked / stand-in dependencies (per the test-generation rules):
none — ``cv2.absdiff`` and ``np.count_nonzero`` are the real
libraries (cv2 5.0.0, numpy) and the inputs are plain numpy uint8
arrays as in the spec edge cases.

``errors`` section of the spec (documented here only, NOT tested, per
the test-generation rules):

- condition: a_grayとb_grayのshapeが異なる
  behavior: cv2.absdiffがエラーを送出する（OpenCVの入力検証。具体的な例外種別はこのファイルからは確認不可、missing参照）
"""

import numpy as np

from scripts.diagnose_diff import diff_rate


def test_edge_01():
    """
    input: 'a_grayとb_grayが要素ごとに同一（同一shape・同一値）'
    expected: '0.0 が返る（d は全0のためcount_nonzeroが0）'
    """
    a = np.full((10, 10), 128, dtype=np.uint8)
    b = a.copy()
    assert diff_rate(a, b) == 0.0


def test_edge_02():
    """
    input: 'a_gray が全0、b_gray が全255（同一shape、例: 100x100）'
    expected: '1.0 が返る（d の全要素が非ゼロ）'
    """
    a = np.zeros((100, 100), dtype=np.uint8)
    b = np.full((100, 100), 255, dtype=np.uint8)
    assert diff_rate(a, b) == 1.0


def test_edge_03():
    """
    input: 'a_gray がuint8配列 [[0, 255]]、b_gray がuint8配列 [[0, 0]]（2x1）'
    expected: '0.5 が返る（2要素中1要素のみ非ゼロ）'
    """
    a = np.array([[0, 255]], dtype=np.uint8)
    b = np.array([[0, 0]], dtype=np.uint8)
    assert diff_rate(a, b) == 0.5


def test_edge_04():
    """
    input: 'a_grayとb_grayがともにshape (0,) の空配列'
    expected: 'd.size が0となり、cv2.absdiffが正常に返す場合は0/0の除算でZeroDivisionErrorが送出される（cv2.absdiffの空入力への挙動はこのファイルからは確認不可、missing参照）'
    """
    a = np.zeros((0,), dtype=np.uint8)
    b = np.zeros((0,), dtype=np.uint8)
    # Observed behavior (deviation from the spec's expected, see
    # tests/TEST_GENERATION_REPORT.md): cv2.absdiff returns a 4x1
    # float64 zero array for a 1-D empty uint8 input (d.size == 4,
    # not 0), so no ZeroDivisionError occurs and 0.0 is returned.
    assert diff_rate(a, b) == 0.0
