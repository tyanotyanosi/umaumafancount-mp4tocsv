"""Tests for ``cli.main._extract_crop``.

Specification: docs/00-Architecture/cli_main___extract_crop.yaml

``_extract_crop`` cuts out the crop corresponding to ``box`` from
``frame``; when the height h is less than ``min_height``, it is
upscaled 2x with INTER_CUBIC and returned:

- ``x, y, w, h = box`` (unpack)
- ``crop = frame[y:y+h, x:x+w]``
- if ``crop.size == 0``, return crop as-is
- if ``h < min_height``, return
  ``cv2.resize(crop, (w*2, h*2), interpolation=cv2.INTER_CUBIC)``;
  otherwise return crop

No mocked / stand-in dependencies: ``frame`` is a real numpy array
(100x100x3 BGR image) and ``cv2.resize`` runs real.

``errors`` section of the spec (documented here only, NOT tested, per the
test-generation rules):

- condition: "frameが2次元スライスを支援しない、またはインデックス要素型が不適"
  behavior: "スライス評価時のTypeError / ValueError / KeyError等が本関数で捕捉されず送出される"
- condition: "cv2.resizeが失敗する（例: cropがsize>0の1次元配列・非画像dtype等）"
  behavior: "cv2.errorが本関数で捕捉されず送出される"
"""

import numpy as np
import pytest

from cli.main import _extract_crop


def _frame():
    """Create a real 100x100x3 BGR numpy frame."""
    return np.zeros((100, 100, 3), dtype=np.uint8)


def test_edge_01():
    """
    input: box=(0, 0, 10, 10), frameが100x100の配列
    expected: h=10が60未満のため、INTER_CUBICで (w*2, h*2)=(20, 20) の配列を返す。
    """
    ret = _extract_crop(_frame(), (0, 0, 10, 10))
    assert ret.shape == (20, 20, 3)


def test_edge_02():
    """
    input: box=(0, 0, 100, 60), frameが100x100の配列
    expected: h=60は60未満ではないため、リサイズしない100x60のcropを返す。
    """
    frame = _frame()
    ret = _extract_crop(frame, (0, 0, 100, 60))
    assert ret.shape == (60, 100, 3)


def test_edge_03():
    """
    input: box=(50, 50, 100, 30), frameが100x100の配列
    expected: スライスが範囲内にクリップされて実際は50x30だが、h=30<60のため(w*2, h*2)=(200, 60)へリサイズされる（boxのw, h基準であり、クリップ後サイズ基準ではない）。
    """
    ret = _extract_crop(_frame(), (50, 50, 100, 30))
    # 実際のcropは 30x50（x方向がクリップされる）だが、
    # リサイズ目標は box の w, h 基準: (200, 60)
    assert ret.shape == (60, 200, 3)


def test_edge_04():
    """
    input: box=(200, 0, 10, 10), frameが100x100の配列（xが範囲外）
    expected: crop.size == 0 となり、空の配列がリサイズされずに返る。
    """
    ret = _extract_crop(_frame(), (200, 0, 10, 10))
    assert ret.size == 0
    assert ret.shape == (10, 0, 3)


def test_edge_05():
    """
    input: boxが3要素のシーケンス (0, 0, 10)
    expected: x, y, w, h = box のアンパックでValueErrorが送出される。
    """
    with pytest.raises(ValueError):
        _extract_crop(_frame(), (0, 0, 10))


def test_edge_06():
    """
    input: min_height=0, box=(0, 0, 10, 10)
    expected: h=10は0未満ではないため、10x10のcropがリサイズされずに返る。
    """
    ret = _extract_crop(_frame(), (0, 0, 10, 10), min_height=0)
    assert ret.shape == (10, 10, 3)
