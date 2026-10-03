"""Tests for ``gui.video_player.VideoPreviewFrame._play_tick``.

Specification: docs/00-Architecture/gui_video_player__VideoPreviewFrame__play_tick.yaml

``_play_tick`` executes one step of video playback: if ``self.is_playing`` is
False or ``self.cap`` is None it returns immediately; otherwise it calls
``self.cap.read()`` and, on failure (falsy ``ret``), calls ``self._stop()``
and returns. On success it sets
``self.current_frame_idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))``,
calls ``self._draw_frame(frame)``, sets ``self.progress`` to
``self.current_frame_idx / self.frame_count`` (``0`` when
``frame_count`` is not > 0), computes
``delay_ms = max(1, int(1000 / self._playback_fps()))`` and stores the job
id returned by ``self.after(delay_ms, self._play_tick)`` in
``self._play_job_id``. It returns ``None``.

Mocked / stand-in dependencies (per the test-generation rules): the
``VideoPreviewFrame`` instance is created without running ``__init__``
(``object.__new__``); ``self.cap`` is a ``MagicMock`` stand-in for an open
``cv2.VideoCapture`` (``read`` / ``get`` / ``set`` controlled per test);
``self._draw_frame`` / ``self._show_frame`` are mocks; ``self._playback_fps``
is a mock returning the value per test; ``self.progress`` is a minimal fake
recording ``set`` calls; ``self.btn_play`` is a minimal fake widget
recording ``configure`` calls; ``self.after`` is a plain function returning
sequential job ids; ``self.after_cancel`` is a mock; ``self._stop`` is the
real method wrapped with ``mock.Mock(wraps=...)`` so its call is recorded
while its documented effects (per edge case 3) are executed for real.

``errors`` section of the spec (documented here only, NOT tested, per the
test-generation rules):

- condition: 'self._playback_fps() が 0 を返す'
  behavior: 'ZeroDivisionError が送出される。次の tick はスケジュールされない（after() 呼び出しに到達しない）'
- condition: 'self.frame_count 属性が存在しない（tick がスケジュールされる前に設定されない状態）'
  behavior: 'progress 計算時に AttributeError が送出される。次の tick はスケジュールされない'
- condition: 'self.cap.read() が例外を送出する（例: 基盤の capture 状態が無効）'
  behavior: '例外がそのまま送出される。次の tick はスケジュールされない'
"""

import itertools
from unittest import mock

import cv2
import pytest

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


class _FakeProgress:
    """Minimal stand-in for a CTkProgressBar: records ``set`` calls."""

    def __init__(self):
        self.value = None
        self.set_calls = []

    def set(self, value):
        self.set_calls.append(value)
        self.value = value


def _make_self(playback_fps=30.0, frame_count=100):
    """Build a ``VideoPreviewFrame`` instance (without running ``__init__``)
    with the dependencies ``_play_tick`` touches mocked / faked."""
    self = object.__new__(VideoPreviewFrame)
    self.is_playing = True
    self.cap = mock.MagicMock(name="cap")
    self.current_frame_idx = 0
    self.frame_count = frame_count
    self._play_job_id = None
    self._draw_frame = mock.MagicMock(name="_draw_frame")
    self._show_frame = mock.MagicMock(name="_show_frame")
    self._playback_fps = mock.MagicMock(name="_playback_fps",
                                        return_value=playback_fps)
    self.progress = _FakeProgress()
    self.btn_play = _FakeWidget()
    self.after_cancel = mock.MagicMock(name="after_cancel")
    counter = itertools.count(1)
    self.after = mock.MagicMock(
        name="after", side_effect=lambda ms, func, *args: next(counter))
    bound_stop = self._stop
    self._stop = mock.Mock(wraps=bound_stop)
    return self


def test_edge_01():
    """
    input: self.is_playing=False、self.cap=<オープン済みの VideoCapture>
    expected: 即 return。cap.read() は呼ばれず、新たな after() スケジュールも発生せず、self._play_job_id は変更されず、None を返す
    """
    self = _make_self()
    self.is_playing = False
    before = self._play_job_id
    ret = self._play_tick()
    assert ret is None
    self.cap.read.assert_not_called()
    self.after.assert_not_called()
    assert self._play_job_id is before


def test_edge_02():
    """
    input: self.is_playing=True、self.cap=None
    expected: 即 return。cap 関連操作も発生せず、self._play_job_id は変更されず、None を返す
    """
    self = _make_self()
    self.cap = None
    before = self._play_job_id
    ret = self._play_tick()
    assert ret is None
    self.after.assert_not_called()
    assert self._play_job_id is before


def test_edge_03():
    """
    input: self.is_playing=True、cap.read() が (False, None) を返す、cap は truthy な VideoCapture
    expected: _stop() が1回呼び出され、この tick は次の tick をスケジュールしない。実行後は is_playing=False、_play_job_id=None、btn_play の text は「再生」、cap がフレーム0に設定され、current_frame_idx=0、_show_frame(0) が1回呼び出されている
    """
    self = _make_self()
    self.cap.read.return_value = (False, None)
    ret = self._play_tick()
    assert ret is None
    self._stop.assert_called_once()
    self.after.assert_not_called()
    assert self.is_playing is False
    assert self._play_job_id is None
    assert self.btn_play.text == "再生"
    self.cap.set.assert_called_once_with(cv2.CAP_PROP_POS_FRAMES, 0)
    assert self.current_frame_idx == 0
    self._show_frame.assert_called_once_with(0)


def test_edge_04():
    """
    input: read 成功、self.frame_count=0
    expected: progress に 0 が設定される（渡される値は 0）。フレームは描画され、次の tick も通常通りスケジュールされる
    """
    self = _make_self(frame_count=0)
    self.cap.read.return_value = (True, "frame")
    self.cap.get.return_value = 5.0
    ret = self._play_tick()
    assert ret is None
    self._draw_frame.assert_called_once_with("frame")
    assert self.progress.set_calls == [0]
    self.after.assert_called_once()
    assert self._play_job_id == 1


def test_edge_05():
    """
    input: read 成功、self.frame_count=100、cap.get(cv2.CAP_PROP_POS_FRAMES) が 30.0 を返す
    expected: self.current_frame_idx は 30 になり、progress に 0.3（30/100 の浮動小数点の結果）が設定される
    """
    self = _make_self(frame_count=100)
    self.cap.read.return_value = (True, "frame")
    self.cap.get.return_value = 30.0
    ret = self._play_tick()
    assert ret is None
    assert self.current_frame_idx == 30
    assert self.progress.set_calls == [0.3]
    self.cap.get.assert_called_once_with(cv2.CAP_PROP_POS_FRAMES)


def test_edge_06():
    """
    input: read 成功、self._playback_fps() が 30 を返す
    expected: delay_ms は 33（max(1, int(1000/30))）になり、self.after(33, self._play_tick) が1回呼び出され、そのジョブIDが self._play_job_id に入る
    """
    self = _make_self(playback_fps=30)
    self.cap.read.return_value = (True, "frame")
    self.cap.get.return_value = 30.0
    ret = self._play_tick()
    assert ret is None
    self.after.assert_called_once()
    args = self.after.call_args.args
    assert args[0] == 33
    assert args[1].__func__ is VideoPreviewFrame._play_tick
    assert args[1].__self__ is self
    assert self._play_job_id == 1


def test_edge_07():
    """
    input: read 成功、self._playback_fps() が -10 を返す
    expected: delay_ms は 1（max(1, int(1000/-10)) = max(1, -100)）になり、self.after(1, self._play_tick) が1回呼び出される
    """
    self = _make_self(playback_fps=-10)
    self.cap.read.return_value = (True, "frame")
    self.cap.get.return_value = 30.0
    ret = self._play_tick()
    assert ret is None
    self.after.assert_called_once()
    args = self.after.call_args.args
    assert args[0] == 1
    assert args[1].__func__ is VideoPreviewFrame._play_tick
    assert args[1].__self__ is self
    assert self._play_job_id == 1


def test_edge_08():
    """
    input: read 成功、self._playback_fps() が 0 を返す
    expected: delay_ms 計算時点で ZeroDivisionError が送出される。その時点で _draw_frame と progress 設定はすでに実行済みで、次の tick はスケジュールされず、self._play_job_id は前回のスケジュール時の値のまま
    """
    self = _make_self(playback_fps=0)
    self._play_job_id = 99  # value from a previous schedule
    self.cap.read.return_value = (True, "frame")
    self.cap.get.return_value = 30.0
    with pytest.raises(ZeroDivisionError):
        self._play_tick()
    self._draw_frame.assert_called_once_with("frame")
    assert self.progress.set_calls == [0.3]
    self.after.assert_not_called()
    assert self._play_job_id == 99
