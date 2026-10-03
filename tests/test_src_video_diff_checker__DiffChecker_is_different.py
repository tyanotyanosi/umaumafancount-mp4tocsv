"""Tests for ``src.video.diff_checker.DiffChecker.is_different``.

Specification: docs/00-Architecture/src_video_diff_checker__DiffChecker_is_different.yaml

``is_different(self, current_frame)`` determines whether the pixel-difference
ratio between the passed current frame and the retained previous frame
(``self.last_frame``) is at or above ``self.threshold``, and returns it as a
bool. The behaviors verified here:

- if ``self.last_frame`` is None (previous frame not yet saved), True is
  returned immediately
- both frames are converted from BGR to grayscale with ``cv2.cvtColor``; if
  the converted shapes differ, True is returned
- otherwise the absolute difference is computed with ``cv2.absdiff``, the
  non-zero pixel count is obtained with ``np.count_nonzero``, and
  ``diff_rate = non-zero pixel count / (height * width)`` is compared with
  ``self.threshold`` (``diff_rate >= self.threshold``)
- the instance state (``threshold`` / ``last_frame``) is not modified by this
  method

Mocked / stand-in dependencies: none. ``cv2`` and ``numpy`` are installed
libraries and are used as real objects; the frames are real BGR ``numpy``
ndarrays, and ``self.last_frame`` is set by direct attribute assignment
(matching the spec's precondition that it is a stored frame). No unrelated
dependency needed mocking.

``errors`` section of the spec (documented here only, NOT tested, per the
test-generation rules):

- condition: 'current_frame または self.last_frame が cv2.COLOR_BGR2GRAY で変換できない入力（None・チャンネル数の不一致・非対応 dtype など）'
  behavior: 'cv2.cvtColor が cv2.error を送出し、呼び出し側に伝播する'
- condition: '両画像の shape が等しく高さ * 幅が 0 の場合（例 0x100 の空画像 pair）'
  behavior: '0 での除算となり ZeroDivisionError が送出される'
"""

import cv2
import numpy as np
import pytest

from src.video.diff_checker import DiffChecker


def _bgr_frame(height, width, value):
    """Create a BGR ``numpy`` frame filled with ``value``."""
    return np.full((height, width, 3), value, dtype=np.uint8)


def test_edge_01():
    """
    input: 直後に生成したインスタンス（self.last_frame が None）に対して任意の BGR 画像を渡す
    expected: True が返る
    """
    checker = DiffChecker()
    assert checker.last_frame is None
    assert checker.is_different(_bgr_frame(100, 100, 128)) is True


def test_edge_02():
    """
    input: self.last_frame が 100x100 の BGR 画像、current_frame が内容同一の 200x200 の BGR 画像
    expected: shape 不一致のため True が返る
    """
    checker = DiffChecker()
    checker.last_frame = _bgr_frame(100, 100, 128)
    current = _bgr_frame(200, 200, 128)
    assert checker.is_different(current) is True


def test_edge_03():
    """
    input: self.last_frame と current_frame が内容完全に同一の 100x100 BGR 画像で、threshold = 0.1
    expected: diff_rate = 0.0 となり False が返る
    """
    checker = DiffChecker(threshold=0.1)
    frame = _bgr_frame(100, 100, 128)
    checker.last_frame = frame.copy()
    result = checker.is_different(frame)
    # spec 乖離: 戻り値は Python の bool ではなく np.bool_（np.False_）。
    # 真偽値は spec の expected（False）と一致する。
    assert result is np.False_


def test_edge_04():
    """
    input: self.last_frame が全黒の 100x100 BGR 画像、current_frame が全白の 100x100 BGR 画像、threshold = 0.1
    expected: diff_rate = 1.0 となり True が返る
    """
    checker = DiffChecker(threshold=0.1)
    checker.last_frame = _bgr_frame(100, 100, 0)
    result = checker.is_different(_bgr_frame(100, 100, 255))
    # spec 乖離: 戻り値は Python の bool ではなく np.bool_（np.True_）。
    # 真偽値は spec の expected（True）と一致する。
    assert result is np.True_


def test_edge_05():
    """
    input: 内容同一の画像 pair で threshold = 0.0
    expected: diff_rate = 0.0 が 0.0 >= 0.0 を満たすため True が返る
    """
    checker = DiffChecker(threshold=0.0)
    frame = _bgr_frame(100, 100, 128)
    checker.last_frame = frame.copy()
    result = checker.is_different(frame)
    # spec 乖離: 戻り値は Python の bool ではなく np.bool_（np.True_）。
    # 真偽値は spec の expected（True）と一致する。
    assert result is np.True_


def test_edge_06():
    """
    input: current_frame が 1 チャンネル（2 次元）のグレースケール画像
    expected: cv2.cvtColor の BGR2GRAY 変換に失敗し cv2.error が送出される
    """
    checker = DiffChecker()
    checker.last_frame = _bgr_frame(100, 100, 128)
    gray = np.full((100, 100), 128, dtype=np.uint8)
    with pytest.raises(cv2.error):
        checker.is_different(gray)


def test_edge_07():
    """
    input: current_frame が None
    expected: cv2.cvtColor が cv2.error を送出する
    """
    checker = DiffChecker()
    checker.last_frame = _bgr_frame(100, 100, 128)
    with pytest.raises(cv2.error):
        checker.is_different(None)
