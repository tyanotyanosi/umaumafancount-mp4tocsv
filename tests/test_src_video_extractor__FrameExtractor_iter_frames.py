"""Tests for ``src.video.extractor.FrameExtractor.iter_frames``.

Specification: docs/00-Architecture/src_video_extractor__FrameExtractor_iter_frames.yaml

``iter_frames`` returns a generator that reads frames from
``video_reader`` and yields one frame at a time at an interval of
``int(fps * interval_sec)`` (minimum 1):

- the generator body does not run until the first ``next()``
- ``fps = self.video_reader.get_fps()`` and
  ``interval_frames = max(1, int(fps * self.interval_sec))``
  (``int()`` truncates)
- loop with ``count = 0``: at the top of the loop, if
  ``max_frames is not None and count >= max_frames``, break (the check
  happens before reading)
- ``frame = self.video_reader.read_frame()``; if None, break
- if ``count % interval_frames == 0``, yield the frame (the first
  frame at count=0 is always yielded)
- ``count += 1`` and continue

Iteration stops when ``read_frame()`` returns None or the number of
read frames reaches ``max_frames``.

Mocked / stand-in dependencies (per the test-generation rules):
``self.video_reader`` is a stand-in class (``_FakeVideoReader``)
implementing ``get_fps()`` and ``read_frame()`` (returns None at the
end), counting the calls of both methods; the frames are plain ints.
The real ``VideoReader`` implementation is out of read scope per the
spec's ``missing`` section.

``errors`` section of the spec (documented here only, NOT tested, per
the test-generation rules):

- condition: self.video_reader.get_fps() が例外を送出する
  behavior: ジェネレータの最初の next() 時点で当該例外が送出される（関数側でキャッチしない）
- condition: self.video_reader.read_frame() が例外を送出する
  behavior: その読み取りに対応する next() 時点で当該例外が送出され、ジェネレータは中絶される
"""

import pytest

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


def _extractor(fps, n_frames, interval_sec, max_frames=None):
    """Build a FrameExtractor over a stand-in reader with
    ``n_frames`` frames (0..n_frames-1) and return
    (extractor, reader, generator)."""
    reader = _FakeVideoReader(fps, list(range(n_frames)))
    extractor = FrameExtractor(reader, interval_sec)
    gen = extractor.iter_frames(max_frames=max_frames)
    return extractor, reader, gen


def test_edge_01():
    """
    input: fps=30、30 フレームの動画、interval_sec=1.0、max_frames=None
    expected: interval_frames=30 となり、frame 0 のみ（1 枚）yield される。
    """
    _, reader, gen = _extractor(30.0, 30, 1.0)
    assert list(gen) == [0]
    assert reader.read_count == 30


def test_edge_02():
    """
    input: fps=30、30 フレームの動画、interval_sec=0.0、max_frames=None
    expected: interval_frames = max(1, int(0.0)) = 1 となり、全 30 フレームが yield される。
    """
    _, reader, gen = _extractor(30.0, 30, 0.0)
    assert list(gen) == list(range(30))
    assert reader.read_count == 30


def test_edge_03():
    """
    input: fps=30、30 フレームの動画、interval_sec=0.5、max_frames=None
    expected: interval_frames=15 となり、frame 0 と frame 15 の 2 枚が yield される。
    """
    _, reader, gen = _extractor(30.0, 30, 0.5)
    assert list(gen) == [0, 15]
    assert reader.read_count == 30


def test_edge_04():
    """
    input: fps=30、30 フレームの動画、interval_sec=1.0、max_frames=10
    expected: read_frame() は 10 回のみ呼び出され、frame 0 のみ yield される。
    """
    _, reader, gen = _extractor(30.0, 30, 1.0, max_frames=10)
    assert list(gen) == [0]
    assert reader.read_count == 10


def test_edge_05():
    """
    input: fps=30、任意の動画、interval_sec=1.0、max_frames=0
    expected: get_fps() は 1 回呼び出されるが read_frame() は 0 回で、yield は 0 枚。
    """
    _, reader, gen = _extractor(30.0, 30, 1.0, max_frames=0)
    with pytest.raises(StopIteration):
        next(gen)
    assert reader.fps_calls == 1
    assert reader.read_count == 0


def test_edge_06():
    """
    input: fps=30、30 フレームの動画、interval_sec=1.0、max_frames=-1
    expected: 上限判定は常に偽となり、30 フレームを全読み取りして frame 0 のみ yield される。
    """
    _, reader, gen = _extractor(30.0, 30, 1.0, max_frames=-1)
    # Observed behavior (deviation from the spec's expected, see
    # tests/TEST_GENERATION_REPORT.md): the loop-top check
    # ``count(0) >= max_frames(-1)`` is True (0 >= -1), so the
    # generator breaks before any read_frame() call.
    assert list(gen) == []
    assert reader.read_count == 0


def test_edge_07():
    """
    input: fps=0、30 フレームの動画、interval_sec=1.0
    expected: interval_frames = max(1, 0) = 1 となり、全 30 フレームが yield される。
    """
    _, reader, gen = _extractor(0.0, 30, 1.0)
    assert list(gen) == list(range(30))
    assert reader.read_count == 30


def test_edge_08():
    """
    input: fps=29.97、interval_sec=2.0、60 フレーム以上の動画
    expected: interval_frames = int(59.94) = 59（切り捨て）の間隔で frame 0、frame 59 が yield される。
    """
    _, reader, gen = _extractor(29.97, 60, 2.0)
    assert list(gen) == [0, 59]
    assert reader.read_count == 60


def test_edge_09():
    """
    input: read_frame() が最初から None を返す video_reader
    expected: frame を 1 枚も yield せず、最初の next() で StopIteration となる。
    """
    _, reader, gen = _extractor(30.0, 0, 1.0)
    with pytest.raises(StopIteration):
        next(gen)
    assert reader.fps_calls == 1
    assert reader.read_count == 0
