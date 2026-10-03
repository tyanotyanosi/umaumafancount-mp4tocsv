"""Tests for ``src.video.reader.VideoReader.get_fps``.

Specification: docs/00-Architecture/src_video_reader__VideoReader_get_fps.yaml

``get_fps`` gets and returns the FPS of the open video:

- gets the property value of ``CAP_PROP_FPS`` via
  ``self.cap.get`` and returns the result as-is

Mocked / stand-in dependencies (per the test-generation rules):
``self.cap`` is a mock on a bare ``VideoReader`` instance
(``object.__new__``) whose ``get`` returns the specified FPS value,
creating the states described in the spec edge cases.

``errors`` section of the spec: empty (no error cases documented).
"""

import cv2
from unittest import mock

from src.video.reader import VideoReader


def _bare_reader(fps):
    """Create a bare ``VideoReader`` instance with a mock ``cap``
    whose ``get`` returns ``fps``."""
    reader = object.__new__(VideoReader)
    reader.cap = mock.Mock()
    reader.cap.get.return_value = fps
    return reader


def test_edge_01():
    """
    input: self.cap.get(cv2.CAP_PROP_FPS) が 30.0 を返す状態
    expected: 30.0 を返す
    """
    reader = _bare_reader(30.0)
    assert reader.get_fps() == 30.0
    reader.cap.get.assert_called_once_with(cv2.CAP_PROP_FPS)


def test_edge_02():
    """
    input: self.cap.get(cv2.CAP_PROP_FPS) が 0.0 を返す状態
    expected: 0.0 を返す
    """
    reader = _bare_reader(0.0)
    assert reader.get_fps() == 0.0
    reader.cap.get.assert_called_once_with(cv2.CAP_PROP_FPS)
