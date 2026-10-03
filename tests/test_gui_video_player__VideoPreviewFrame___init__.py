"""Tests for ``gui.video_player.VideoPreviewFrame.__init__``.

Specification: docs/00-Architecture/gui_video_player__VideoPreviewFrame___init__.yaml

``__init__`` creates the video preview frame (a ``ctk.CTkFrame``
subclass) and initializes the internal state (video path, capture
handle, playback state, etc.):

- Step 1: ``super().__init__(master, **kwargs)`` creates the frame widget
- Step 2: ``video_path=None, cap=None, is_playing=False,
  current_frame_idx=0, on_process=None`` are set
- Step 3: ``_play_job_id=None`` is set (the play tick's ``after()`` job id;
  no tick exists initially)
- Step 4: ``_setup_ui()`` is called to build the UI

``fps`` / ``frame_count`` / ``native_width`` / ``native_height`` /
``disp_width`` / ``disp_height`` are NOT set by this function (they are set
by ``load_video`` / ``_compute_display_size``).

Mocked / stand-in dependencies (per the test-generation rules): edge case
1 runs the real ``VideoPreviewFrame()`` construction (a real
``ctk.CTkFrame`` widget; a retry helper guards against unstable Tcl
environments and the widget is destroyed at the end of the test).
``VideoPreviewFrame._setup_ui`` is temporarily replaced by a counting
wrapper that records the call and then runs the real ``_setup_ui`` (so the
call count is asserted while the real UI is built). Edge case 2 calls
``VideoPreviewFrame.__init__`` directly on a bare
(``object.__new__``) instance with an invalid ``master`` so the partially
constructed object can be inspected.

``errors`` section of the spec (documented here only, NOT tested, per the
test-generation rules):

- condition: "super().__init__(master, **kwargs) で例外が発生する（不正な master、CTkFrame が受け付けない kwargs など）"
  behavior: "try/except なし。例外は呼び出し元へそのまま伝播し、6属性は設定されず _setup_ui も未呼び出しのまま"
- condition: "self._setup_ui() で例外が発生する"
  behavior: "try/except なし。例外は呼び出し元へそのまま伝播する。6属性はすでに初期値設定済み"
"""

import contextlib
import time
from unittest import mock

import customtkinter as ctk

from gui.video_player import VideoPreviewFrame

SIX_ATTRS = ("video_path", "cap", "is_playing", "current_frame_idx",
             "on_process", "_play_job_id")


def _create_frame():
    """Construct a real ``VideoPreviewFrame`` (default master), retrying up
    to 5 times at 0.25 s intervals against unstable Tcl environments;
    re-raise the original error if every attempt fails."""
    last_error = None
    for attempt in range(5):
        if attempt:
            time.sleep(0.25)
        try:
            return VideoPreviewFrame()
        except Exception as exc:
            last_error = exc
    raise last_error


@contextlib.contextmanager
def _count_setup_ui(calls):
    """Temporarily replace ``VideoPreviewFrame._setup_ui`` with a counting
    wrapper that records each call and then runs the real ``_setup_ui``."""
    orig = VideoPreviewFrame._setup_ui

    def counting_setup_ui(self):
        calls.append(1)
        return orig(self)

    with mock.patch.object(VideoPreviewFrame, "_setup_ui", counting_setup_ui):
        yield


def test_edge_01():
    """
    input: master=None, kwargs なし（デフォルト呼び出し）
    expected: デフォルトの親でフレームが生成され、6属性が上記初期値に設定され、_setup_ui() が1回呼び出され、戻り値は None
    """
    calls = []
    self = None
    with _count_setup_ui(calls):
        self = _create_frame()
        try:
            assert self is not None
            assert isinstance(self, ctk.CTkFrame)
            assert self.video_path is None
            assert self.cap is None
            assert self.is_playing is False
            assert self.current_frame_idx == 0
            assert self.on_process is None
            assert self._play_job_id is None
            # _setup_ui() が1回呼び出され、UI ウィジェットが作成された
            assert calls == [1]
            assert self.video_label is not None
            assert self.progress is not None
            assert self.btn_play is not None
            assert self.btn_process is not None
        finally:
            if self is not None:
                try:
                    self.destroy()
                except Exception:
                    pass


def test_edge_02():
    """
    input: super().__init__(master, **kwargs) が例外を送出する引数（不正な master や不正な kwargs）
    expected: __init__ は異常終了し、例外は呼び出し元へそのまま伝播する。6属性は設定されず _setup_ui も未呼び出し
    """
    obj = object.__new__(VideoPreviewFrame)
    try:
        VideoPreviewFrame.__init__(obj, master="invalid_master_string")
        raised = False
    except Exception:
        raised = True
    assert raised
    for attr in SIX_ATTRS:
        assert not hasattr(obj, attr)
