"""Tests for ``gui.video_player.VideoPreviewFrame._toggle_play``.

Specification: docs/00-Architecture/gui_video_player__VideoPreviewFrame__toggle_play.yaml

``_toggle_play`` toggles the playback state (the play button's command): if
playing, it delegates to ``self._stop()``; if stopped, it delegates to
``self._play()``. It returns ``None`` and does not modify
``self.is_playing`` or the UI directly (the changes happen in the delegate).

Mocked / stand-in dependencies (per the test-generation rules): the
``VideoPreviewFrame`` instance is created without running ``__init__``
(``object.__new__``); ``self.is_playing`` / ``self.cap`` /
``self.video_path`` are set per test; ``self._stop`` is a mock (its body is
outside the read scope); ``self._play`` runs real (its guard and effects are
documented in its spec); ``self.btn_play`` is a minimal fake widget
recording ``configure`` calls; ``self._play_tick`` is a mock.

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


def _make_self(is_playing, cap, video_path):
    """Build a ``VideoPreviewFrame`` instance (without running ``__init__``)
    with the dependencies ``_toggle_play`` touches mocked / faked."""
    self = object.__new__(VideoPreviewFrame)
    self.is_playing = is_playing
    self.cap = cap
    self.video_path = video_path
    self.btn_play = _FakeWidget()
    self._play_tick = mock.MagicMock(name="_play_tick")
    self._stop = mock.MagicMock(name="_stop")
    return self


def test_edge_01():
    """
    input: self.is_playing = True（その他の状態は任意）
    expected: self._stop() が恰好1回呼び出され、self._play() は呼び出されない。None を返す
    """
    self = _make_self(True, mock.MagicMock(name="cap"), "sample.mp4")
    ret = self._toggle_play()
    assert ret is None
    self._stop.assert_called_once()
    # _play が呼ばれない可観測効果: tick 未呼び出し・状態不変
    self._play_tick.assert_not_called()
    assert self.is_playing is True
    assert self.btn_play.text is None


def test_edge_02():
    """
    input: self.is_playing = False、self.cap が真値、self.video_path が真値
    expected: self._play() が恰好1回呼び出され、_play 内で is_playing が True になり btn_play のテキストが "一時停止" になり _play_tick が1回呼び出される
    """
    self = _make_self(False, mock.MagicMock(name="cap"), "sample.mp4")
    ret = self._toggle_play()
    assert ret is None
    self._stop.assert_not_called()
    assert self.is_playing is True
    assert self.btn_play.text == "一時停止"
    self._play_tick.assert_called_once()


def test_edge_03():
    """
    input: self.is_playing = False、self.cap = None
    expected: self._play() は呼び出されるがそのガード（134行目）で return するため、is_playing は False のまま btn_play のテキストも不変で _play_tick は呼び出されない
    """
    self = _make_self(False, None, None)
    ret = self._toggle_play()
    assert ret is None
    self._stop.assert_not_called()
    assert self.is_playing is False
    assert self.btn_play.text is None
    self._play_tick.assert_not_called()


def test_edge_04():
    """
    input: self.is_playing = False、self.cap が真値、self.video_path = None
    expected: self._play() は呼び出されるがガードで return するため、状態変更は一切起きない
    """
    self = _make_self(False, mock.MagicMock(name="cap"), None)
    ret = self._toggle_play()
    assert ret is None
    self._stop.assert_not_called()
    assert self.is_playing is False
    assert self.btn_play.text is None
    self._play_tick.assert_not_called()
