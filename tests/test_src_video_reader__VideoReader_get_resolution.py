"""Tests for ``src.video.reader.VideoReader.get_resolution``.

Specification: docs/00-Architecture/src_video_reader__VideoReader_get_resolution.yaml

``get_resolution`` gets and returns the resolution of the open video
as a 2-tuple of ints ``(width, height)``:

- gets ``CAP_PROP_FRAME_WIDTH`` via ``self.cap.get``, converts it
  with ``int()`` and makes it width
- gets ``CAP_PROP_FRAME_HEIGHT`` via ``self.cap.get``, converts it
  with ``int()`` and makes it height
- returns the tuple ``(width, height)``

Mocked / stand-in dependencies (per the test-generation rules):
``self.cap`` is a mock on a bare ``VideoReader`` instance
(``object.__new__``) whose ``get`` returns the specified width /
height values (via ``side_effect``), creating the states described in
the spec edge cases.

``errors`` section of the spec (documented here only, NOT tested, per
the test-generation rules):

- condition: "self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) または self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) が int に変換できない値（例: None）を返す"
  behavior: "TypeError（int() 呼び出しで送出）"
"""

import cv2
from unittest import mock

from src.video.reader import VideoReader


def _bare_reader(width, height):
    """Create a bare ``VideoReader`` instance with a mock ``cap``
    whose ``get`` returns ``width`` then ``height``."""
    reader = object.__new__(VideoReader)
    reader.cap = mock.Mock()
    reader.cap.get.side_effect = [width, height]
    return reader


def test_edge_01():
    """
    input: self.cap.get が幅 1920.0、高さ 1080.0 を返す状態
    expected: (1920, 1080) を返す
    """
    reader = _bare_reader(1920.0, 1080.0)
    ret = reader.get_resolution()
    assert ret == (1920, 1080)
    assert reader.cap.get.call_args_list[0] == mock.call(
        cv2.CAP_PROP_FRAME_WIDTH)
    assert reader.cap.get.call_args_list[1] == mock.call(
        cv2.CAP_PROP_FRAME_HEIGHT)


def test_edge_02():
    """
    input: self.cap.get が幅 640.0、高さ 0.0 を返す状態
    expected: (640, 0) を返す
    """
    reader = _bare_reader(640.0, 0.0)
    ret = reader.get_resolution()
    assert ret == (640, 0)
    assert reader.cap.get.call_args_list[0] == mock.call(
        cv2.CAP_PROP_FRAME_WIDTH)
    assert reader.cap.get.call_args_list[1] == mock.call(
        cv2.CAP_PROP_FRAME_HEIGHT)
