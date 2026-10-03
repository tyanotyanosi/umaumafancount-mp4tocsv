"""Tests for ``src.video.reader.VideoReader.__enter__``.

Specification: docs/00-Architecture/src_video_reader__VideoReader___enter__.yaml

``__enter__`` returns ``self`` for use as a context manager in a
``with`` statement:

- returns ``self``

No mocked / stand-in dependencies: a real ``VideoReader`` instance is
built from the real workspace video ``output/synthetic.mp4``.

``errors`` section of the spec: empty (no error cases documented).
"""

from pathlib import Path

from src.video.reader import VideoReader

_VIDEO = Path(__file__).resolve().parent.parent / "output" / "synthetic.mp4"


def test_edge_01():
    """
    input: __init__ が正常完了した VideoReader インスタンス
    expected: 渡された self と同一のインスタンスを返す
    """
    reader = VideoReader(str(_VIDEO))
    assert reader.__enter__() is reader
    reader.cap.release()
