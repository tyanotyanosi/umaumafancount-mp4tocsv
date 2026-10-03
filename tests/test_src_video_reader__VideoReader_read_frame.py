"""Tests for ``src.video.reader.VideoReader.read_frame``.

Specification: docs/00-Architecture/src_video_reader__VideoReader_read_frame.yaml

``read_frame`` reads one frame from the current position and returns
the frame object (returns None when it cannot be read):

- calls ``self.cap.read()`` and obtains the 2-tuple ``(ret, frame)``
- if ``ret`` is falsy, return None
- if ``ret`` is truthy, return ``frame``

Mocked / stand-in dependencies (per the test-generation rules):
``self.cap`` is a mock on a bare ``VideoReader`` instance
(``object.__new__``) whose ``read`` returns the specified
``(ret, frame)`` tuple, creating the states described in the spec
edge cases; the frame object is a real numpy array.

``errors`` section of the spec (documented here only, NOT tested, per
the test-generation rules):

- condition: "self.cap.read() が例外を送出する"
  behavior: "例外をそのまま呼び出し側へ伝播する（この関数には catch 処理がない）"
"""

from unittest import mock

import numpy as np

from src.video.reader import VideoReader


def _bare_reader(read_result):
    """Create a bare ``VideoReader`` instance with a mock ``cap``
    whose ``read`` returns ``read_result``."""
    reader = object.__new__(VideoReader)
    reader.cap = mock.Mock()
    reader.cap.read.return_value = read_result
    return reader


def test_edge_01():
    """
    input: self.cap.read() が (True, <フレーム>) を返す状態
    expected: そのフレーム（<フレーム>）を返す
    """
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    reader = _bare_reader((True, frame))
    ret = reader.read_frame()
    assert ret is frame
    reader.cap.read.assert_called_once()


def test_edge_02():
    """
    input: self.cap.read() が (False, None) を返す状態
    expected: None を返す
    """
    reader = _bare_reader((False, None))
    ret = reader.read_frame()
    assert ret is None
    reader.cap.read.assert_called_once()
