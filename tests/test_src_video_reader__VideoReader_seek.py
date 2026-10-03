"""Tests for ``src.video.reader.VideoReader.seek``.

Specification: docs/00-Architecture/src_video_reader__VideoReader_seek.yaml

``seek`` seeks the read position to the specified frame number and
returns the result of the seek operation:

- calls ``self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)`` and
  returns its return value as-is

Mocked / stand-in dependencies (per the test-generation rules):
``self.cap`` is a mock on a bare ``VideoReader`` instance
(``object.__new__``) whose ``set`` returns the specified value,
creating the states described in the spec edge cases.

``errors`` section of the spec (documented here only, NOT tested, per
the test-generation rules):

- condition: "self.cap.set が例外を送出する"
  behavior: "例外をそのまま呼び出し側へ伝播する（この関数には catch 処理がない）"
"""

import cv2
from unittest import mock

from src.video.reader import VideoReader


def _bare_reader(set_result):
    """Create a bare ``VideoReader`` instance with a mock ``cap``
    whose ``set`` returns ``set_result``."""
    reader = object.__new__(VideoReader)
    reader.cap = mock.Mock()
    reader.cap.set.return_value = set_result
    return reader


def test_edge_01():
    """
    input: self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx) が True を返す状態
    expected: True を返す
    """
    reader = _bare_reader(True)
    ret = reader.seek(5)
    assert ret is True
    reader.cap.set.assert_called_once_with(cv2.CAP_PROP_POS_FRAMES, 5)


def test_edge_02():
    """
    input: self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx) が False を返す状態
    expected: False を返す
    """
    reader = _bare_reader(False)
    ret = reader.seek(5)
    assert ret is False
    reader.cap.set.assert_called_once_with(cv2.CAP_PROP_POS_FRAMES, 5)
