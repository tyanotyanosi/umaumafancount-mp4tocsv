"""Tests for ``gui.main_window.MainWindow._create_main_content``.

Specification: docs/00-Architecture/gui_main_window__MainWindow__create_main_content.yaml

``MainWindow._create_main_content`` creates the main content area (main
frame, video preview frame, result table) and connects the preview
frame's process callback to ``_on_video_process`` (per the spec's
``purpose`` field):

- Create ``ctk.CTkFrame(self)`` as ``self.main_frame`` and grid it at
  row=1 column=0 sticky="nsew" padx=10 pady=10
- Configure ``main_frame`` grid columns 0 and 1 each with weight=1
  (two equal columns)
- Create ``VideoPreviewFrame(self.main_frame)`` as ``self.video_preview``
  and grid it at row=0 column=0 sticky="nsew" padx=5 pady=5
- Assign ``self._on_video_process`` to ``self.video_preview.on_process``
- Create ``ResultTableView(self.main_frame)`` as ``self.result_table`` and
  grid it at row=0 column=1 sticky="nsew" padx=5 pady=5

The method only depends on the main window widget itself, so the tests
below build a real ``MainWindow`` (a ``ctk.CTk`` root window, i.e., the
minimal Tk root) and call ``_create_main_content`` directly, asserting
the created widgets and their state. No mocks are used. Window creation
is retried up to 5 times (0.25 s apart) in case the environment's Tcl is
unstable (TclError 'couldn't read file'); if all attempts fail, the last
TclError is raised.

``errors`` section of the spec (documented here only, NOT tested, per the
test-generation rules):

- condition: 'VideoPreviewFrame / ResultTableView のコンストラクタが例外を送出（実装依存）'
  behavior: '例外が呼び出し元（_setup_ui 経由の呼び出しチェーン）へそのまま送出される'
"""

import time
import tkinter

import customtkinter as ctk
from gui.main_window import MainWindow, ResultTableView, VideoPreviewFrame


def _create_main_window(fresh):
    """Create a MainWindow root window.

    fresh=True: the widget part is initialized via ``ctk.CTk.__init__``
    only, so the main_frame / video_preview / result_table attributes are
    unset (the spec's edge_01 input).
    fresh=False: ``__init__`` runs fully (``_setup_ui`` already executed),
    so the three attributes already exist (the spec's edge_02 input).

    Tk window creation is retried up to 5 times (0.25 s apart) in case the
    environment's Tcl is unstable (TclError 'couldn't read file'); if all
    attempts fail, the last TclError is raised.
    """
    last_error = None
    for _ in range(5):
        try:
            if fresh:
                win = object.__new__(MainWindow)
                ctk.CTk.__init__(win)
            else:
                win = MainWindow()
            return win
        except tkinter.TclError as exc:
            last_error = exc
            time.sleep(0.25)
    raise last_error


def test_edge_01():
    """
    input: 新規の MainWindow（main_frame / video_preview / result_table 属性が未設定）
    expected: 例外なし。3属性が設定され、video_preview.on_process が self._on_video_process となり、video_preview と result_table は main_frame を親とする
    """
    win = _create_main_window(fresh=True)
    try:
        assert not hasattr(win, "main_frame")
        assert not hasattr(win, "video_preview")
        assert not hasattr(win, "result_table")
        win._create_main_content()
        assert isinstance(win.main_frame, ctk.CTkFrame)
        mf_info = win.main_frame.grid_info()
        assert mf_info["row"] == 1
        assert mf_info["column"] == 0
        assert mf_info["sticky"] == "nesw"
        assert mf_info["padx"] == 10
        assert mf_info["pady"] == 10
        assert win.main_frame.grid_columnconfigure(0)["weight"] == 1
        assert win.main_frame.grid_columnconfigure(1)["weight"] == 1
        assert isinstance(win.video_preview, VideoPreviewFrame)
        assert win.video_preview.master is win.main_frame
        vp_info = win.video_preview.grid_info()
        assert vp_info["row"] == 0
        assert vp_info["column"] == 0
        assert vp_info["sticky"] == "nesw"
        assert vp_info["padx"] == 5
        assert vp_info["pady"] == 5
        assert win.video_preview.on_process == win._on_video_process
        assert isinstance(win.result_table, ResultTableView)
        assert win.result_table.master is win.main_frame
        rt_info = win.result_table.grid_info()
        assert rt_info["row"] == 0
        assert rt_info["column"] == 1
        assert rt_info["sticky"] == "nesw"
        assert rt_info["padx"] == 5
        assert rt_info["pady"] == 5
    finally:
        win.destroy()


def test_edge_02():
    """
    input: main_frame / video_preview / result_table 属性が既に存在する self（メソッドの二重呼び出し）
    expected: 例外なし。3属性が新規に作成したウィジェットへ上書きされる
    """
    win = _create_main_window(fresh=False)
    try:
        old_main_frame = win.main_frame
        old_video_preview = win.video_preview
        old_result_table = win.result_table
        win._create_main_content()
        assert win.main_frame is not old_main_frame
        assert win.video_preview is not old_video_preview
        assert win.result_table is not old_result_table
        assert isinstance(win.main_frame, ctk.CTkFrame)
        assert isinstance(win.video_preview, VideoPreviewFrame)
        assert win.video_preview.master is win.main_frame
        assert isinstance(win.result_table, ResultTableView)
        assert win.result_table.master is win.main_frame
        assert win.video_preview.on_process == win._on_video_process
    finally:
        win.destroy()
