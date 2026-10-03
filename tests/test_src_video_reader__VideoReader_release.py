"""Tests for ``src.video.reader.VideoReader.release``.

Specification: docs/00-Architecture/src_video_reader__VideoReader_release.yaml

``release`` releases the video reading resources (it calls
``self.cap.release()`` only when the capture is open):

- checks ``self.cap.isOpened()``
- only if ``isOpened()`` is True, calls ``self.cap.release()``
- if ``isOpened()`` is False, does nothing and returns
- returns None

Mocked / stand-in dependencies (per the test-generation rules):
``self.cap`` is a mock on a bare ``VideoReader`` instance
(``object.__new__``) whose ``isOpened`` returns the specified value,
creating the states described in the spec edge cases.

``errors`` section of the spec: empty (no error cases documented).
"""

from unittest import mock

from src.video.reader import VideoReader


def _bare_reader(is_opened):
    """Create a bare ``VideoReader`` instance with a mock ``cap``
    whose ``isOpened`` returns ``is_opened``."""
    reader = object.__new__(VideoReader)
    reader.cap = mock.Mock()
    reader.cap.isOpened.return_value = is_opened
    return reader


def test_edge_01():
    """
    input: self.cap.isOpened() が True を返す状態
    expected: self.cap.release() が1回呼び出され、None を返す
    """
    reader = _bare_reader(True)
    ret = reader.release()
    assert ret is None
    reader.cap.release.assert_called_once()


def test_edge_02():
    """
    input: self.cap.isOpened() が False を返す状態（既に解放済み）
    expected: self.cap.release() は呼び出されず（no-op）、None を返す
    """
    reader = _bare_reader(False)
    ret = reader.release()
    assert ret is None
    reader.cap.release.assert_not_called()
