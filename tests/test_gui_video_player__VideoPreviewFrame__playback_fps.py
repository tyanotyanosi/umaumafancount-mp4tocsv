"""Tests for ``gui.video_player.VideoPreviewFrame._playback_fps``.

Specification: docs/00-Architecture/gui_video_player__VideoPreviewFrame__playback_fps.yaml

``VideoPreviewFrame._playback_fps`` returns the frame rate used for
playback: if ``self.fps`` is a truthy value and ``self.fps > 0``, it
returns ``self.fps`` as-is; otherwise it returns the fallback value
``30.0`` (the fallback for the case where fps cannot be read, per the
code comment on line 125). The method does not modify the state of
``self`` in any way.

``self.fps`` is not set by the visible ``__init__`` (lines 15-23) —
the setter is outside the visible range — so each test constructs a
bare ``VideoPreviewFrame`` instance via ``object.__new__`` (skipping
``__init__`` entirely, since the constructor body is an unrelated
dependency of ``_playback_fps``) and sets ``self.fps`` directly.
``_playback_fps`` only references ``self.fps``, so no Tk root window is
required.

``errors`` section of the spec (documented here only, NOT tested, per the
test-generation rules):

- condition: 'self.fps 属性が存在しない'
  behavior: 'AttributeError が送出され、捕捉されず呼び出し元に伝播する'
"""

import pytest

from gui.video_player import VideoPreviewFrame


def test_edge_01():
    """
    input: self.fps = 30.0
    expected: 30.0 を返す（self.fps そのまま）
    """
    frame = object.__new__(VideoPreviewFrame)
    frame.fps = 30.0
    result = frame._playback_fps()
    assert result == 30.0


def test_edge_02():
    """
    input: self.fps = 24
    expected: 24 を返す（self.fps そのまま。int の値であり float 注釈は強制されない）
    """
    frame = object.__new__(VideoPreviewFrame)
    frame.fps = 24
    result = frame._playback_fps()
    assert result == 24
    assert isinstance(result, int)


def test_edge_03():
    """
    input: self.fps = 29.97
    expected: 29.97 を返す
    """
    frame = object.__new__(VideoPreviewFrame)
    frame.fps = 29.97
    result = frame._playback_fps()
    assert result == 29.97


def test_edge_04():
    """
    input: self.fps = None
    expected: 30.0 を返す（and の左辺で偽値）
    """
    frame = object.__new__(VideoPreviewFrame)
    frame.fps = None
    result = frame._playback_fps()
    assert result == 30.0


def test_edge_05():
    """
    input: self.fps = 0
    expected: 30.0 を返す（偽値）
    """
    frame = object.__new__(VideoPreviewFrame)
    frame.fps = 0
    result = frame._playback_fps()
    assert result == 30.0


def test_edge_06():
    """
    input: self.fps = -5
    expected: 30.0 を返す（真値だが self.fps > 0 が偽）
    """
    frame = object.__new__(VideoPreviewFrame)
    frame.fps = -5
    result = frame._playback_fps()
    assert result == 30.0


def test_edge_07():
    """
    input: self.fps 属性が未作成
    expected: 123行目の self.fps 参照で AttributeError が発生する
    """
    frame = object.__new__(VideoPreviewFrame)
    with pytest.raises(AttributeError):
        frame._playback_fps()
