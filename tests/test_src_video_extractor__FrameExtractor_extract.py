"""Tests for ``src.video.extractor.FrameExtractor.extract``.

Specification: docs/00-Architecture/src_video_extractor__FrameExtractor_extract.yaml

``extract`` collects all frames yielded by ``iter_frames()`` into a
list and returns it:

- generate ``self.iter_frames(max_frames=max_frames)`` and collect all
  elements with ``list()``, returning them as-is. No other processing.

``max_frames`` has the same meaning as in ``iter_frames`` (a limit on
the number of frames read; None means until the end of the video).

Mocked / stand-in dependencies (per the test-generation rules):
``self.video_reader`` is a stand-in class (``_FakeVideoReader``)
implementing ``get_fps()`` and ``read_frame()`` (returns None at the
end), counting the calls of both methods; the frames are plain ints.
The real ``VideoReader`` implementation is out of read scope per the
spec's ``missing`` section.

``errors`` section of the spec (documented here only, NOT tested, per
the test-generation rules):

- condition: get_fps() や read_frame() が例外を送出する
  behavior: list() 構築中に当該例外がそのまま送出される（iter_frames と同一の挙動）
"""

from src.video.extractor import FrameExtractor


class _FakeVideoReader:
    """Stand-in video_reader with a fixed frame list and call
    counters."""

    def __init__(self, fps, frames):
        self.fps = fps
        self.frames = frames
        self._pos = 0
        self.read_count = 0
        self.fps_calls = 0

    def get_fps(self):
        self.fps_calls += 1
        return self.fps

    def read_frame(self):
        if self._pos >= len(self.frames):
            return None
        self._pos += 1
        self.read_count += 1
        return self.frames[self._pos - 1]


def _extractor(fps, n_frames, interval_sec):
    """Build a FrameExtractor over a stand-in reader with
    ``n_frames`` frames (0..n_frames-1) and return
    (extractor, reader)."""
    reader = _FakeVideoReader(fps, list(range(n_frames)))
    extractor = FrameExtractor(reader, interval_sec)
    return extractor, reader


def test_edge_01():
    """
    input: fps=30、30 フレームの動画、interval_sec=1.0、max_frames=None
    expected: 1 要素の list [frame0] を返す。
    """
    extractor, reader = _extractor(30.0, 30, 1.0)
    result = extractor.extract()
    assert result == [0]
    assert reader.read_count == 30


def test_edge_02():
    """
    input: fps=30、30 フレームの動画、interval_sec=0.0、max_frames=None
    expected: 全 30 フレームからなる list を返す。
    """
    extractor, reader = _extractor(30.0, 30, 0.0)
    result = extractor.extract()
    assert result == list(range(30))
    assert reader.read_count == 30


def test_edge_03():
    """
    input: fps=30、30 フレームの動画、interval_sec=1.0、max_frames=10
    expected: read_frame() は 10 回のみ呼び出され、1 要素の list [frame0] を返す。
    """
    extractor, reader = _extractor(30.0, 30, 1.0)
    result = extractor.extract(max_frames=10)
    assert result == [0]
    assert reader.read_count == 10


def test_edge_04():
    """
    input: fps=30、任意の動画、interval_sec=1.0、max_frames=0
    expected: 空 list [] を返し、get_fps() は 1 回・read_frame() は 0 回呼び出される。
    """
    extractor, reader = _extractor(30.0, 30, 1.0)
    result = extractor.extract(max_frames=0)
    assert result == []
    assert reader.fps_calls == 1
    assert reader.read_count == 0


def test_edge_05():
    """
    input: read_frame() が最初から None を返す video_reader
    expected: 空 list [] を返す。
    """
    extractor, reader = _extractor(30.0, 0, 1.0)
    result = extractor.extract()
    assert result == []
    assert reader.fps_calls == 1
    assert reader.read_count == 0
