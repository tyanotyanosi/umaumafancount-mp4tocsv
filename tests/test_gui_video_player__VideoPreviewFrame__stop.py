"""Tests for ``gui.video_player.VideoPreviewFrame._stop``.

Specification: docs/00-Architecture/gui_video_player__VideoPreviewFrame__stop.yaml

``_stop`` stops playback: it cancels the play tick and, if a video is
loaded, seeks the capture back to the first frame and displays it:

- Step 1: ``self.is_playing = False``
- Step 2: ``self._cancel_play_tick()`` cancels any pending play tick
  (via which ``self._play_job_id`` becomes ``None``)
- Step 3: ``self.btn_play``'s text is reset to "再生"
- Step 4: if ``self.cap`` is truthy: ``self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)``
  seeks the capture to frame 0; otherwise steps 4-6 are skipped
- Step 5: in the truthy branch: ``self.current_frame_idx = 0``
- Step 6: in the truthy branch: ``self._show_frame(0)`` displays frame 0

The function returns ``None``.

Mocked / stand-in dependencies (per the test-generation rules): the
``VideoPreviewFrame`` instance is created without running ``__init__``
(``object.__new__``); ``self.cap`` is a ``MagicMock`` stand-in for an open
``cv2.VideoCapture`` (truthy) or ``None``; ``self.is_playing`` /
``self.current_frame_idx`` / ``self._play_job_id`` are set per test;
``self.btn_play`` is a minimal fake widget recording ``configure`` calls;
``self._show_frame`` is a mock; ``self.after_cancel`` is a mock (the real
``_cancel_play_tick`` runs and uses it); ``self._cancel_play_tick`` runs
real per its spec (cancels the job and clears ``_play_job_id`` only when
the job id is not None).

``errors`` section of the spec (documented here only, NOT tested, per the
test-generation rules):

- condition: "self.cap.set（seek の成否を示す戻り値）が無視されており（168行目）、seek 失敗時の分岐が存在しない。"
  behavior: "seek 失敗時も例外にならず、current_frame_idx=0 と _show_frame(0) がそのまま実行される"
- condition: "関数本体に明示的な raise 文がない。"
  behavior: "cap.set や _show_frame（実装は読込範囲外）から例外が起きれば、捕捉されず呼び出し側へ伝播する"
"""

from unittest import mock

import cv2

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


def _make_self(cap, is_playing, current_frame_idx, play_job_id):
    """Build a ``VideoPreviewFrame`` instance (without running ``__init__``)
    with the dependencies ``_stop`` touches mocked / faked."""
    self = object.__new__(VideoPreviewFrame)
    self.cap = cap
    self.is_playing = is_playing
    self.current_frame_idx = current_frame_idx
    self._play_job_id = play_job_id
    self.btn_play = _FakeWidget()
    self._show_frame = mock.MagicMock(name="_show_frame")
    self.after_cancel = mock.MagicMock(name="after_cancel")
    return self


def test_edge_01():
    """
    input: self.cap=None, self._play_job_id=None, self.is_playing=True, self.current_frame_idx=5（動画未ロード、または close 呼び出し後）
    expected: is_playing==False、_play_job_id は None、btn_play テキスト==再生、current_frame_idx は 5 のまま、cap.set と _show_frame は呼び出されず、戻り値 None
    """
    self = _make_self(None, True, 5, None)
    ret = self._stop()
    assert ret is None
    assert self.is_playing is False
    assert self._play_job_id is None
    assert self.btn_play.text == "再生"
    assert self.current_frame_idx == 5
    self._show_frame.assert_not_called()
    self.after_cancel.assert_not_called()


def test_edge_02():
    """
    input: self.cap=truthy な cv2.VideoCapture、self._play_job_id=有効なジョブID、self.is_playing=True、self.current_frame_idx=10（動画ロード済み・tick進行中）
    expected: is_playing==False、_play_job_id は None、btn_play テキスト==再生、cap.set が (cv2.CAP_PROP_POS_FRAMES, 0) を引数に1回呼び出され、current_frame_idx==0、_show_frame が引数 0 で1回呼び出され、戻り値 None
    """
    cap = mock.MagicMock(name="cap")
    self = _make_self(cap, True, 10, 12345)
    ret = self._stop()
    assert ret is None
    assert self.is_playing is False
    assert self._play_job_id is None
    assert self.btn_play.text == "再生"
    cap.set.assert_called_once_with(cv2.CAP_PROP_POS_FRAMES, 0)
    assert self.current_frame_idx == 0
    self._show_frame.assert_called_once_with(0)
    self.after_cancel.assert_called_once_with(12345)


def test_edge_03():
    """
    input: self.cap=truthy な cv2.VideoCapture、self._play_job_id=None、self.is_playing=False（既に停止済みでの再呼び出し）
    expected: idempotent: is_playing は False のまま、after_cancel は呼ばれず、btn_play テキスト==再生、cap.set(CAP_PROP_POS_FRAMES, 0)、current_frame_idx==0、_show_frame(0) 呼び出し、戻り値 None
    """
    cap = mock.MagicMock(name="cap")
    self = _make_self(cap, False, 7, None)
    ret = self._stop()
    assert ret is None
    assert self.is_playing is False
    self.after_cancel.assert_not_called()
    assert self._play_job_id is None
    assert self.btn_play.text == "再生"
    cap.set.assert_called_once_with(cv2.CAP_PROP_POS_FRAMES, 0)
    assert self.current_frame_idx == 0
    self._show_frame.assert_called_once_with(0)
