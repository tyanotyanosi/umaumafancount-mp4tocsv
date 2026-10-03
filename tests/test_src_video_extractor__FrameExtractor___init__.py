"""Tests for ``src.video.extractor.FrameExtractor.__init__``.

Specification: docs/00-Architecture/src_video_extractor__FrameExtractor___init__.yaml

``__init__`` initializes a FrameExtractor instance holding the
VideoReader and the extraction interval ``interval_sec``:

- assign ``video_reader`` to ``self.video_reader`` and
  ``interval_sec`` to ``self.interval_sec``, then finish

There is no type checking (annotation only) and no I/O; any value
(including 0 or negative ``interval_sec``) is kept as-is.

Mocked / stand-in dependencies (per the test-generation rules):
``video_reader`` is a plain dummy object (``object()``); no methods
are required because ``__init__`` only stores the reference.

``errors`` section of the spec: empty (no error conditions documented).
"""

import pytest

from src.video.extractor import FrameExtractor


def test_edge_01():
    """
    input: FrameExtractor(reader)（reader はダミーオブジェクト）
    expected: 例外なく初期化され、self.interval_sec == 1.0 かつ self.video_reader は reader と同一オブジェクトである。
    """
    reader = object()
    inst = FrameExtractor(reader)
    assert inst.interval_sec == 1.0
    assert inst.video_reader is reader


def test_edge_02():
    """
    input: FrameExtractor(reader, 0.0)
    expected: 例外なく初期化され、self.interval_sec == 0.0 である。
    """
    reader = object()
    inst = FrameExtractor(reader, 0.0)
    assert inst.interval_sec == 0.0
    assert inst.video_reader is reader


def test_edge_03():
    """
    input: FrameExtractor(reader, -1.0)
    expected: 例外なく初期化され、self.interval_sec == -1.0 である。
    """
    reader = object()
    inst = FrameExtractor(reader, -1.0)
    assert inst.interval_sec == -1.0
    assert inst.video_reader is reader


def test_edge_04():
    """
    input: FrameExtractor(VideoReader 以外のオブジェクト, 1.0)
    expected: 初期化時に例外は起きない。型チェックが存在しないため、get_fps() や read_frame() の非存在は後続の iter_frames の最初の next() でのみ顕在化する。
    """
    reader = object()
    inst = FrameExtractor(reader, 1.0)
    assert inst.video_reader is reader
    # No exception at initialization; the missing get_fps() surfaces
    # only at the first next() of iter_frames.
    with pytest.raises(AttributeError):
        next(inst.iter_frames())
