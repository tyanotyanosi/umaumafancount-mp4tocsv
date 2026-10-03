"""Tests for ``gui.main_window.MainWindow._on_close``.

Specification: docs/00-Architecture/gui_main_window__MainWindow__on_close.yaml

``MainWindow._on_close`` is the WM_DELETE_WINDOW close handler: it releases
the video preview capture by calling ``self.video_preview.close()`` once,
then destroys the main window via ``self.destroy()``, and returns ``None``.

Mocks: ``self.video_preview`` is an unrelated dependency (its concrete
effects are outside this file, per the spec ``missing``/``unconfirmed``
sections) replaced with a ``unittest.mock.Mock`` object. The window is a
``MainWindow`` instance whose widget base is a minimal ``tkinter.Tk()`` root
window initialized through the CTk base (the ``super().__init__()`` required
by the spec preconditions). ``MainWindow.__init__`` itself is not executed,
so ``_setup_ui`` and all unrelated widgets are not created.

``errors`` section of the spec (documented here only, NOT tested, per the
test-generation rules):

- condition: 'self.video_preview 属性が存在しない（_create_main_content が実行されていない等）'
  behavior: 'AttributeError が発生し外へ伝播する。self.destroy() は実行されない'
- condition: 'video_preview.close() が例外を投げる'
  behavior: '例外が外へ伝播する。self.destroy() は実行されない'
"""

import time
import tkinter as tk
from unittest.mock import Mock

import customtkinter as ctk

from gui.main_window import MainWindow


def _create_main_window():
    """Build a ``MainWindow`` on a minimal ``tkinter.Tk()`` root window.

    Only the widget base is initialized (``ctk.CTk.__init__`` is the
    ``super().__init__()`` required by the spec preconditions);
    ``MainWindow.__init__`` is not executed, so ``_setup_ui`` and all
    unrelated widgets are skipped. ``self.video_preview`` is mocked with
    ``unittest.mock.Mock``.

    Retries the Tk() creation up to 5 times with 0.25 s intervals to guard
    against unstable Tcl environments (TclError "couldn't read file").
    Raises the original TclError if all attempts fail.
    """
    last_error = None
    for _ in range(5):
        win = object.__new__(MainWindow)
        try:
            ctk.CTk.__init__(win)
            break
        except tk.TclError as exc:
            last_error = exc
            time.sleep(0.25)
    else:
        raise last_error
    win.video_preview = Mock(name="video_preview")
    return win


def _is_alive(win):
    """Return True if the Tk application of ``win`` still exists.

    After ``destroy()`` the Tcl application is gone and ``winfo_exists()``
    raises ``TclError``, which is interpreted as "not alive".
    """
    try:
        return win.winfo_exists() == 1
    except tk.TclError:
        return False


def test_edge_01():
    """
    input: ユーザーがウィンドウのクローズボタンを押す（WM_DELETE_WINDOW 経由）
    expected: video_preview.close() が1回、その後 self.destroy() が1回呼び出され、メインウィンドウは破棄される
    """
    win = _create_main_window()
    events = []
    try:
        win.video_preview.close.side_effect = lambda: events.append("close")
        original_destroy = win.destroy

        def _spied_destroy():
            events.append("destroy")
            original_destroy()

        win.destroy = _spied_destroy

        result = win._on_close()
        assert result is None
        assert win.video_preview.close.call_count == 1
        assert events == ["close", "destroy"]

        # The main window is destroyed: the Tcl application no longer exists.
        assert not _is_alive(win)
    finally:
        if _is_alive(win):
            win.destroy()
