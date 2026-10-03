"""Tests for ``src.video.reader.VideoReader.get_frame_count``.

Specification: docs/00-Architecture/src_video_reader__VideoReader_get_frame_count.yaml

``get_frame_count`` gets and returns the total frame count of the open
video as an int:

- gets the property value of ``CAP_PROP_FRAME_COUNT`` via
  ``self.cap.get``
- converts the obtained value with ``int()`` and returns it

Mocked / stand-in dependencies (per the test-generation rules):
``self.cap`` is a mock on a bare ``VideoReader`` instance
(``object.__new__``) whose ``get`` returns the specified value,
creating the states described in the spec edge cases.

``errors`` section of the spec (documented here only, NOT tested, per
the test-generation rules):

- condition: "self.cap.get(cv2.CAP_PROP_FRAME_COUNT) が int に変換できない値（例: None）を返す"
  behavior: "TypeError（int() 呼び出しで送出）"
"""

import cv2
from unittest import mock

from src.video.reader import VideoReader


def _bare_reader(value):
    """Create a bare ``VideoReader`` instance with a mock ``cap``
    whose ``get`` returns ``value``."""
    reader = object.__new__(VideoReader)
    reader.cap = mock.Mock()
    reader.cap.get.return_value = value
    return reader


def test_edge_01():
    """
    input: self.cap.get(cv2.CAP_PROP_FRAME_COUNT) が 300.7 を返す状態
    expected: 300 を返す（int() による小数切り捨て）
    """
    reader = _bare_reader(300.7)
    assert reader.get_frame_count() == 300
    reader.cap.get.assert_called_once_with(cv2.CAP_PROP_FRAME_COUNT)


def test_edge_02():
    """
    input: self.cap.get(cv2.CAP_PROP_FRAME_COUNT) が 0.0 を返す状態
    expected: 0 を返す
    """
    reader = _bare_reader(0.0)
    assert reader.get_frame_count() == 0
    reader.cap.get.assert_called_once_with(cv2.CAP_PROP_FRAME_COUNT)
