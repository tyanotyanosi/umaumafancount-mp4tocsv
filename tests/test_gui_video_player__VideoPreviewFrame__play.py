"""Tests for ``gui.video_player.VideoPreviewFrame._play``.

Specification: docs/00-Architecture/gui_video_player__VideoPreviewFrame__play.yaml

``_play`` starts playback: if the video is loaded and not already playing,
it sets the playing state, changes the button to "一時停止", and starts the
play tick:

- guard: if ``self.cap`` is falsy, or ``self.video_path`` is falsy, or
  ``self.is_playing`` is truthy, return immediately
- ``self.is_playing = True``
- ``self.btn_play``'s text is changed to "一時停止"
- ``self._play_tick()`` is called to start the play tick
- returns ``None``

Mocked / stand-in dependencies (per the test-generation rules): the
``VideoPreviewFrame`` instance is created without running ``__init__``
(``object.__new__``); ``self.cap`` is a ``MagicMock`` stand-in for an open
``cv2.VideoCapture`` (truthy) or ``None``; ``self.video_path`` and
``self.is_playing`` are set per test; ``self.btn_play`` is a minimal fake
widget recording ``configure`` calls; ``self._play_tick`` is a mock.

``errors`` section of the spec: none documented (empty).
"""

from unittest import mock

from gui.video_player import VideoPreviewFrame


class _FakeWidget:
    """Minimal stand-in for a Tk widget: records ``configure`` keyword calls
    and item assignments so tests can assert on ``text`` / ``state``."""

    def __init__(self):
        self.text = None
        self.state = None
        self.configure_calls = []

    def configure(self, **kwargs):
        self.configure_calls.append(kwargs)
        if "text" in kwargs:
            self.text = kwargs["text"]
        if "state" in kwargs:
            self.state = kwargs["state"]

    def __setitem__(self, key, value):
        if key == "text":
            self.text = value
        elif key == "state":
            self.state = value


def _make_self(cap, video_path, is_playing):
    """Build a ``VideoPreviewFrame`` instance (without running ``__init__``)
    with the dependencies ``_play`` touches mocked / faked."""
    self = object.__new__(VideoPreviewFrame)
    self.cap = cap
    self.video_path = video_path
    self.is_playing = is_playing
    self.btn_play = _FakeWidget()
    self._play_tick = mock.MagicMock(name="_play_tick")
    return self


def test_edge_01():
    """
    input: self.cap = None（動画未ロード）、self.video_path = None、self.is_playing = False
    expected: None を返す。is_playing は False のまま btn_play のテキストは不変で _play_tick は呼び出されない
    """
    self = _make_self(None, None, False)
    ret = self._play()
    assert ret is None
    assert self.is_playing is False
    assert self.btn_play.text is None
    self._play_tick.assert_not_called()


def test_edge_02():
    """
    input: self.cap が真値、self.video_path = None、self.is_playing = False
    expected: None を返す。状態変更は一切起きない（ガードで return）
    """
    self = _make_self(mock.MagicMock(name="cap"), None, False)
    ret = self._play()
    assert ret is None
    assert self.is_playing is False
    assert self.btn_play.text is None
    self._play_tick.assert_not_called()


def test_edge_03():
    """
    input: self.cap が真値、self.video_path = "C:\\videos\\test.mp4"、self.is_playing = True
    expected: None を返す。is_playing は True のままで btn_play のテキストはこの呼び出しでは変更されず _play_tick は呼び出されない
    """
    self = _make_self(mock.MagicMock(name="cap"), "C:\\videos\\test.mp4", True)
    ret = self._play()
    assert ret is None
    assert self.is_playing is True
    assert self.btn_play.text is None
    self._play_tick.assert_not_called()


def test_edge_04():
    """
    input: self.cap が真値、self.video_path = "C:\\videos\\test.mp4"、self.is_playing = False
    expected: is_playing が True になり btn_play のテキストが "一時停止" になり _play_tick が1回呼び出され、None を返す
    """
    self = _make_self(mock.MagicMock(name="cap"), "C:\\videos\\test.mp4", False)
    ret = self._play()
    assert ret is None
    assert self.is_playing is True
    assert self.btn_play.text == "一時停止"
    self._play_tick.assert_called_once()
