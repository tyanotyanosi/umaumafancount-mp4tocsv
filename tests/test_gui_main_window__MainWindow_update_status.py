"""Tests for ``gui.main_window.MainWindow.update_status``.

Specification: docs/00-Architecture/gui_main_window__MainWindow_update_status.yaml

``MainWindow.update_status`` updates the status bar label text to the given
message: it calls ``self.status_label.configure(text=message)`` and returns
``None`` (no explicit return statement). No validation of ``message`` is
performed — it is passed through to ``configure(text=...)`` as-is.

Mocks: ``self.status_label`` is an unrelated dependency replaced with a
minimal ``tkinter.Label`` widget (a widget that can call
``configure(text=...)``), per the spec input constraints. The window is a
``MainWindow`` instance whose widget base is a minimal ``tkinter.Tk()`` root
window initialized through the CTk base (the ``super().__init__()`` required
by the spec preconditions). ``MainWindow.__init__`` itself is not executed,
so ``_setup_ui`` and all unrelated widgets are not created.

``errors`` section of the spec (documented here only, NOT tested, per the
test-generation rules):

- condition: 'self.status_label が存在しない（__init__/_setup_ui 未完了のインスタンスなど）'
  behavior: 'AttributeError がそのまま呼び出し元へ伝播する'
- condition: 'message が ウィジェットの configure(text=...) が受け付けない値（例: None、アノテーションはstrだが実行時チェックなし）'
  behavior: 'コード自体は何の検証もせず configure にそのまま渡す。最終的な挙動は tkinter/customtkinter ウィジェットの実装に依存し、本ファイルのコードからは検証できない'
"""

import time
import tkinter as tk

import customtkinter as ctk

from gui.main_window import MainWindow


def _create_main_window():
    """Build a ``MainWindow`` on a minimal ``tkinter.Tk()`` root window.

    Only the widget base is initialized (``ctk.CTk.__init__`` is the
    ``super().__init__()`` required by the spec preconditions);
    ``MainWindow.__init__`` is not executed, so ``_setup_ui`` and all
    unrelated widgets are skipped. ``self.status_label`` is a minimal
    ``tkinter.Label`` — a widget that can call ``configure(text=...)`` —
    per the spec input constraints.

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
    win.status_label = tk.Label(win, text="")
    return win


def test_edge_01():
    """
    input: message = ''（空文字列）
    expected: status_label のテキストが '' に設定され、状態バーは空（テキストなし）になる
    """
    win = _create_main_window()
    try:
        result = win.update_status("")
        assert result is None
        assert win.status_label.cget("text") == ""
    finally:
        win.destroy()


def test_edge_02():
    """
    input: message = '処理完了'
    expected: status_label のテキストが「処理完了」になる
    """
    win = _create_main_window()
    try:
        result = win.update_status("処理完了")
        assert result is None
        assert win.status_label.cget("text") == "処理完了"
    finally:
        win.destroy()
