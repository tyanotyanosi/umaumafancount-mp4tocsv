"""Tests for ``src.video.reader.VideoReader.__exit__``.

Specification: docs/00-Architecture/src_video_reader__VideoReader___exit__.yaml

``__exit__`` releases the video reading resources when leaving a
``with`` block (it calls ``release`` regardless of whether an
exception occurred):

- does not use the exception information arguments
  (exc_type, exc_val, exc_tb) and calls ``self.release()``
- has no explicit return, so None is returned

No mocked / stand-in dependencies: a real ``VideoReader`` instance is
built from the real workspace video
``output/synthetic.mp4``; ``release`` / ``cv2.VideoCapture.release``
run real (the released state is observed via ``cap.isOpened()``).

``errors`` section of the spec: empty (no error cases documented).
"""

from pathlib import Path

import pytest

from src.video.reader import VideoReader

_VIDEO = Path(__file__).resolve().parent.parent / "output" / "synthetic.mp4"


def test_edge_01():
    """
    input: 例外なしで with 文ブロックを離脱した場合（exc_type, exc_val, exc_tb が全て None）
    expected: self.release() が呼び出され、None を返す
    """
    reader = VideoReader(str(_VIDEO))
    assert reader.cap.isOpened()
    ret = reader.__exit__(None, None, None)
    assert ret is None
    assert not reader.cap.isOpened()


def test_edge_02():
    """
    input: with 文ブロック内で例外が発生して離脱した場合（exc_type に例外型が入った状態）
    expected: self.release() が呼び出され、None を返す（例外は抑制されず伝播する）
    """
    reader = VideoReader(str(_VIDEO))
    with pytest.raises(ValueError):
        with reader:
            raise ValueError("boom")
    assert not reader.cap.isOpened()
