"""Tests for ``src.video.card_detector.CardDetector._find_peaks``.

Specification: docs/00-Architecture/src_video_card_detector__CardDetector__find_peaks.yaml

``CardDetector._find_peaks`` (per the spec's ``purpose`` and ``behavior``
fields) extracts peaks from a template-matching result map using NMS
(non-maximum suppression) and returns them as a list of ``(x, y, score)``
tuples:

- Copy ``res`` (the caller's original array is never mutated).
- Obtain ``h, w`` from ``res.shape`` (the unpacking raises ``ValueError``
  for arrays whose shape has a length other than 2) and compute the
  suppression radius ``r = int(40 * s)``.
- Loop: obtain the global maximum ``mx`` and its location ``(x, y)`` on the
  copy via ``cv2.minMaxLoc``.
- If ``mx < threshold``, stop (an empty list is returned when even the
  first pass finds nothing).
- Append ``(x, y, float(mx))`` to ``peaks``.
- Rewrite the square region of radius ``r`` centered on ``(x, y)`` (rows
  ``y-r..y+r``, columns ``x-r..x+r``, clipped at the array bounds) to -1
  (suppressing the neighborhood of the detected peak).
- Repeat until the global maximum drops below ``threshold``, then return
  ``peaks``. Postconditions: a list of ``(x, y, score)`` triples with int
  coordinates and float scores, every score >= threshold, in
  non-increasing score order, and the input array unchanged.

The function depends on the OpenCV library function ``cv2.minMaxLoc`` (the
spec describes ``res`` as numpy arrays, e.g. the result of
``cv2.matchTemplate``), so the tests build real numpy result maps and call
the method directly. The method reads no instance attribute at all (per
the spec's ``inputs`` field for ``self``), so it is invoked unbound as
``CardDetector._find_peaks(None, res, threshold, s)``. No mock is used in
``test_edge_01`` through ``test_edge_04`` and in ``test_edge_06``. The sole
exception is ``test_edge_05``: the spec's expected behavior for ``s < 0``
is a non-terminating infinite loop, which cannot be observed directly;
there, ``cv2.minMaxLoc`` is temporarily replaced with a bounded wrapper
around the real implementation that counts scans and raises the
``_PeakScanLimitExceeded`` sentinel once the scan count exceeds the bound,
and the test asserts that the sentinel propagates out of ``_find_peaks``
(proving the loop never terminates).

``errors`` section of the spec (documented here only, NOT tested, per the
test-generation rules):

- condition: 'res の shape の要素数が 2 ではない（1 次元・3 次元配列等）'
  behavior: 'ValueError（h, w = res.shape の展開）'
- condition: 'res が要素数 0 の空 2 次元配列'
  behavior: 'cv2.minMaxLoc が cv2.error を送出（OpenCV アサーション。推定・本ファイルでは未検証）'
- condition: 's < 0 かつ全領域最大値が threshold 以上'
  behavior: '返さない（無限ループ。抑制領域が空のため最大値が更新されない）'
- condition: '全領域最大値が -1 以下かつ threshold 以上、かつ抑制が -1 を書く（例：res が全て -1 且つ threshold <= -1）'
  behavior: '返さない（無限ループ。-1 への書換で値が変化しないため）。コードの論理からの推定'
"""

from unittest import mock

import cv2
import numpy as np
import pytest

from src.video.card_detector import CardDetector


class _PeakScanLimitExceeded(RuntimeError):
    """Sentinel raised by the bounded stand-in for cv2.minMaxLoc in test_edge_05."""


_MAX_SCANS_BEFORE_SENTINEL = 8


def test_edge_01():
    """
    input: res が 10x10 の 0.5 一様配列、threshold = 0.6、s = 1.0
    expected: [] を返す（最大値 0.5 が 0.6 未満で最初の一巡で終了）
    """
    original = np.full((10, 10), 0.5)
    result = CardDetector._find_peaks(None, original, 0.6, 1.0)
    assert result == []
    # 後条件: 入力配列は変更されない（内部でコピーを使用）
    assert np.array_equal(original, np.full((10, 10), 0.5))


def test_edge_02():
    """
    input: res が 1x1 配列 [[0.9]]、threshold = 0.6、s = 1.0
    expected: [(0, 0, 0.9)] を返す
    """
    original = np.array([[0.9]])
    result = CardDetector._find_peaks(None, original, 0.6, 1.0)
    assert result == [(0, 0, 0.9)]
    # 後条件: 入力配列は変更されない（内部でコピーを使用）
    assert np.array_equal(original, np.array([[0.9]]))


def test_edge_03():
    """
    input: res が 100x100 の 0 配列で res[10, 10] = 0.9、res[50, 50] = 0.8 のみ、threshold = 0.5、s = 1.0（r = 40）
    expected: [(10, 10, 0.9)] を返す（第2ピーク (50, 50) は第1ピークの半径40抑制領域（行・列 0..50）に含まれ -1 に書換えられるため検出されない）
    """
    original = np.zeros((100, 100))
    original[10, 10] = 0.9
    original[50, 50] = 0.8
    snapshot = original.copy()
    result = CardDetector._find_peaks(None, original, 0.5, 1.0)
    assert result == [(10, 10, 0.9)]
    assert len(result) == 1
    # 後条件: 入力配列は変更されない（内部でコピーを使用）
    assert np.array_equal(original, snapshot)


def test_edge_04():
    """
    input: res が 1 次元配列 np.array([0.1, 0.9, 0.5])、threshold = 0.6
    expected: ValueError（h, w = res.shape の展開で not enough values to unpack）
    """
    res = np.array([0.1, 0.9, 0.5])
    with pytest.raises(ValueError) as excinfo:
        CardDetector._find_peaks(None, res, 0.6)
    # 仕様書 expected に明記される展開失敗メッセージを含むことを確認
    assert "not enough values to unpack" in str(excinfo.value)


def test_edge_05():
    """
    input: res が 1x1 配列 [[0.8]]、threshold = 0.5、s = -1.0（r = -40）
    expected: 返さない（無限ループ。抑制スライスが空のため最大値が更新されず、(0, 0, 0.8) を繰り返し追加し続ける）
    """
    res = np.array([[0.8]])
    real_min_max_loc = cv2.minMaxLoc
    scans = {"count": 0}

    def bounded_min_max_loc(image):
        scans["count"] += 1
        if scans["count"] > _MAX_SCANS_BEFORE_SENTINEL:
            # 抑制スライスが空（r = -40）のため最大値は更新されず、
            # (0, 0, 0.8) が繰り返し追加され続け、ループは終了しない。
            # 有限の走査回数を越えたらセンチネルを送出してループを打ち切る。
            raise _PeakScanLimitExceeded(
                "cv2.minMaxLoc was called "
                f"{scans['count']} times; the NMS loop never terminated"
            )
        return real_min_max_loc(image)

    with mock.patch.object(cv2, "minMaxLoc", new=bounded_min_max_loc):
        with pytest.raises(_PeakScanLimitExceeded):
            CardDetector._find_peaks(None, res, 0.5, -1.0)

    # 上限より多くの走査が試みられたことを確認（= (0, 0, 0.8) の
    # 繰り返し追加が発生し続けていた証拠）
    assert scans["count"] == _MAX_SCANS_BEFORE_SENTINEL + 1


def test_edge_06():
    """
    input: res が shape (0, 5) の空の 2 次元配列、threshold = 0.6
    expected: cv2.minMaxLoc が例外を送出（OpenCV の空配列アサーション、推定）
    """
    # 仕様書 expected は「推定」であることが明示されている（spec errors:
    # "cv2.minMaxLoc が cv2.error を送出（OpenCV アサーション。推定・本ファイルでは
    # 未検証）"、unconfirmed: 例外の正確な種別・メッセージは未検証）。
    # 実際にインストールされている OpenCV（cv2 5.0.0）では空配列の
    # minMaxLoc は例外を送出せず (0.0, 0.0, (-1, -1), (-1, -1)) を返すため、
    # 観察可能な挙動は仕様書 behavior「mx < threshold ならループを終了する
    # （最初の一巡ですら該当値がなければ空リストを返す）」に従い、
    # mx = 0.0 < 0.6 として最初の一巡で空リストを返す。
    res = np.empty((0, 5))
    result = CardDetector._find_peaks(None, res, 0.6)
    assert isinstance(result, list)
    assert result == []
