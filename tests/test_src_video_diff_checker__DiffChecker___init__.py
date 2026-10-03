"""Tests for ``src.video.diff_checker.DiffChecker.__init__``.

Specification: docs/00-Architecture/src_video_diff_checker__DiffChecker___init__.yaml

``__init__(self, threshold: float = 0.1)`` initializes the instance. The
behaviors verified here:

- the ``threshold`` argument (default 0.1) is assigned to ``self.threshold``
- ``self.last_frame`` is set to None
- no argument validation, I/O, or external call is performed; the method
  ends immediately, returns None, and raises no exception
- values outside the 0..1 range (negative or greater than 1) are accepted
  as-is, since the code performs no range validation

Mocked / stand-in dependencies: none. ``DiffChecker.__init__`` has no
external dependencies and is exercised directly. No unrelated dependency
needed mocking.

The spec's ``errors`` section is empty (no error conditions documented), so
there is nothing to document here.
"""

from src.video.diff_checker import DiffChecker


def test_edge_01():
    """
    input: 引数なしで DiffChecker() をインスタンス生成
    expected: self.threshold が 0.1 かつ self.last_frame が None になる
    """
    checker = DiffChecker()
    assert checker.threshold == 0.1
    assert checker.last_frame is None


def test_edge_02():
    """
    input: DiffChecker(0.5) をインスタンス生成
    expected: self.threshold が 0.5 かつ self.last_frame が None になる
    """
    checker = DiffChecker(0.5)
    assert checker.threshold == 0.5
    assert checker.last_frame is None


def test_edge_03():
    """
    input: DiffChecker(-1.0) や DiffChecker(2.0) のような 0..1 範囲外の値を渡して生成
    expected: 例外は発生せず、指定値がそのまま self.threshold に入る
    """
    negative = DiffChecker(-1.0)
    overflow = DiffChecker(2.0)
    assert negative.threshold == -1.0
    assert overflow.threshold == 2.0
    assert negative.last_frame is None
    assert overflow.last_frame is None
