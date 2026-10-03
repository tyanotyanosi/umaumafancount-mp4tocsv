"""Tests for ``src.video.reader.VideoReader.__init__``.

Specification: docs/00-Architecture/src_video_reader__VideoReader___init__.yaml

``__init__`` opens the video file at the given path and keeps the
``cv2.VideoCapture`` object as the instance attribute ``self.cap``:

- creates a ``cv2.VideoCapture`` object with ``video_path`` and
  assigns it to ``self.cap``
- if ``self.cap.isOpened()`` is False: calls ``self.cap.release()``
  and raises ``FileNotFoundError("動画を開けません: {video_path}")``
- if ``isOpened()`` is True, the constructor completes without
  exception

Mocked / stand-in dependencies (per the test-generation rules): for
the unopenable-path case, ``cv2.VideoCapture`` is patched with a mock
whose ``isOpened()`` returns False (so the ``release()`` call on
``self.cap`` can be observed); for the valid-path case a real
``VideoReader`` is built from the real workspace video
``output/synthetic.mp4``.

``errors`` section of the spec (documented here only, NOT tested, per
the test-generation rules):

- condition: "cv2.VideoCapture(video_path) の isOpened() が False"
  behavior: 'self.cap.release() を呼び、FileNotFoundError("動画を開けません: {video_path}") を送出する'
"""

import shutil
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from src.video.reader import VideoReader

_VIDEO = Path(__file__).resolve().parent.parent / "output" / "synthetic.mp4"


def _tmp_dir():
    """Create a temporary directory under the session workspace (the
    platform temp area is not writable for subdirectory creation under
    the file sandbox; the workspace is)."""
    workspace = Path(__file__).resolve().parent.parent
    return Path(tempfile.mkdtemp(dir=workspace))


def test_edge_01():
    """
    input: video_path が cv2.VideoCapture で開けないパス（例: 存在しないファイル）
    expected: メッセージ "動画を開けません: <video_path>" の FileNotFoundError が送出され、self.cap は release() が呼び出された状態
    """
    tmp = _tmp_dir()
    try:
        missing = tmp / "nope.mp4"
        with mock.patch("src.video.reader.cv2.VideoCapture") as mock_vc:
            cap = mock.Mock()
            cap.isOpened.return_value = False
            mock_vc.return_value = cap
            with pytest.raises(FileNotFoundError) as ei:
                VideoReader(str(missing))
            mock_vc.assert_called_once_with(str(missing))
            cap.release.assert_called_once()
            assert f"動画を開けません: {missing}" in str(ei.value)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_edge_02():
    """
    input: video_path が有効な動画ファイルのパス
    expected: 例外なしでコンストラクタが完了し、self.cap.isOpened() が True
    """
    reader = VideoReader(str(_VIDEO))
    assert reader.cap.isOpened()
    reader.cap.release()
