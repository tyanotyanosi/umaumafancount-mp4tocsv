"""Tests for ``gui.main_window.MainWindow._create_status_bar``.

Specification: docs/00-Architecture/gui_main_window__MainWindow__create_status_bar.yaml

``MainWindow._create_status_bar`` creates the status bar at the bottom of
the window (a CTkFrame of height 30) and a status label whose initial
text is "準備完了" (per the spec's ``purpose`` field):

- Create ``ctk.CTkFrame(self, height=30)`` as ``self.status_bar`` and
  grid it at row=2 column=0 sticky="ew" padx=10 pady=(0, 10)
- Create ``ctk.CTkLabel(self.status_bar, text="準備完了", anchor="w")`` as
  ``self.status_label`` and pack it with side="left" padx=10 pady=5

The method only depends on the main window widget itself, so the tests
below build a real ``MainWindow`` (a ``ctk.CTk`` root window, i.e., the
minimal Tk root) and call ``_create_status_bar`` directly, asserting the
created widgets and their state. No mocks are used. Window creation is
retried up to 5 times (0.25 s apart) in case the environment's Tcl is
unstable (TclError 'couldn't read file'); if all attempts fail, the last
TclError is raised.

``errors`` section of the spec (documented here only, NOT tested, per the
test-generation rules):

- (empty — the spec's ``errors`` section is ``[]``; no error conditions
  are documented)
"""

import time
import tkinter

import customtkinter as ctk
from gui.main_window import MainWindow


def _create_main_window(fresh):
    """Create a MainWindow root window.

    fresh=True: the widget part is initialized via ``ctk.CTk.__init__``
    only, so the status_bar / status_label attributes are unset (the
    spec's edge_01 input).
    fresh=False: ``__init__`` runs fully (``_setup_ui`` already executed),
    so the two attributes already exist (the spec's edge_02 input).

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
    input: 新規の MainWindow（status_bar / status_label 属性が未設定）
    expected: 例外なし。2属性が設定され、ラベルテキストは「準備完了」、親子関係は self -> status_bar -> status_label となる
    """
    win = _create_main_window(fresh=True)
    try:
        assert not hasattr(win, "status_bar")
        assert not hasattr(win, "status_label")
        win._create_status_bar()
        assert isinstance(win.status_bar, ctk.CTkFrame)
        assert win.status_bar.cget("height") == 30
        sb_info = win.status_bar.grid_info()
        assert sb_info["row"] == 2
        assert sb_info["column"] == 0
        assert sb_info["sticky"] == "ew"
        assert sb_info["padx"] == 10
        assert sb_info["pady"] == (0, 10)
        assert win.status_bar.master is win
        assert isinstance(win.status_label, ctk.CTkLabel)
        assert win.status_label.cget("text") == "準備完了"
        assert win.status_label.cget("anchor") == "w"
        assert win.status_label.master is win.status_bar
        pack_info = win.status_label.pack_info()
        assert pack_info["side"] == "left"
        assert pack_info["padx"] == 10
        assert pack_info["pady"] == 5
    finally:
        win.destroy()


def test_edge_02():
    """
    input: status_bar / status_label 属性が既に存在する self（メソッドの二重呼び出し）
    expected: 例外なし。2属性が新規に作成したウィジェットへ上書きされる
    """
    win = _create_main_window(fresh=False)
    try:
        old_status_bar = win.status_bar
        old_status_label = win.status_label
        win._create_status_bar()
        assert win.status_bar is not old_status_bar
        assert win.status_label is not old_status_label
        assert isinstance(win.status_bar, ctk.CTkFrame)
        assert win.status_bar.cget("height") == 30
        assert isinstance(win.status_label, ctk.CTkLabel)
        assert win.status_label.cget("text") == "準備完了"
        assert win.status_label.master is win.status_bar
    finally:
        win.destroy()
