"""Tests for ``src.video.diff_checker.DiffChecker.update``.

Specification: docs/00-Architecture/src_video_diff_checker__DiffChecker_update.yaml

``update(self, frame)`` stores a copy of the passed frame into
``self.last_frame`` as the previous frame for the next ``is_different``
determination. The behaviors verified here:

- ``frame.copy()`` is called and the returned copy is assigned to
  ``self.last_frame``
- no type / content validation, conversion, or I/O is performed on ``frame``;
  the method returns None
- the content of ``frame`` itself is not modified by this method
- when ``frame`` is a ``numpy`` ndarray, ``.copy()`` returns a separate
  object with data independent from ``frame``

Mocked / stand-in dependencies: none. ``numpy`` is an installed library and
is used as a real object; the frames are real ndarrays. No unrelated
dependency needed mocking.

``errors`` section of the spec (documented here only, NOT tested, per the
test-generation rules):

- condition: 'frame が .copy() メソッドを持たないオブジェクト（None・list・int など）'
  behavior: 'frame.copy() の評価時に AttributeError が送出される'
"""

import numpy as np
import pytest

from src.video.diff_checker import DiffChecker


def test_edge_01():
    """
    input: frame に 100x100 の BGR ndarray を渡し、update 後に frame 自体を in-place 変更（例 全要素を 0 に上書き）
    expected: self.last_frame は update 時点の内容を保持し、frame への後続 in-place 変更は反映されない（frame が ndarray の場合 copy() は frame と独立したデータの別オブジェクトを返す）
    """
    checker = DiffChecker()
    frame = np.full((100, 100, 3), 42, dtype=np.uint8)
    checker.update(frame)
    frame.fill(0)
    assert checker.last_frame is not frame
    assert np.array_equal(checker.last_frame, np.full((100, 100, 3), 42, dtype=np.uint8))


def test_edge_02():
    """
    input: frame に list（例 [1, 2, 3]）を渡す
    expected: AttributeError が送出される（list 型には copy 属性がない）
    """
    # spec 乖離: Python の list には .copy() メソッドが存在するため、
    # AttributeError は送出されず、update は list のコピーを
    # self.last_frame に設定する。ここでは観察された挙動を assert する。
    checker = DiffChecker()
    frame = [1, 2, 3]
    checker.update(frame)
    assert checker.last_frame == [1, 2, 3]
    assert checker.last_frame is not frame


def test_edge_03():
    """
    input: frame に None を渡す
    expected: AttributeError が送出される（NoneType には copy 属性がない）
    """
    checker = DiffChecker()
    with pytest.raises(AttributeError):
        checker.update(None)


def test_edge_04():
    """
    input: update を 2 回連続呼び出す
    expected: self.last_frame は 2 回目で渡した frame のコピーになる
    """
    checker = DiffChecker()
    frame1 = np.full((10, 10, 3), 1, dtype=np.uint8)
    frame2 = np.full((10, 10, 3), 2, dtype=np.uint8)
    checker.update(frame1)
    checker.update(frame2)
    assert checker.last_frame is not frame1
    assert checker.last_frame is not frame2
    assert np.array_equal(checker.last_frame, frame2)
