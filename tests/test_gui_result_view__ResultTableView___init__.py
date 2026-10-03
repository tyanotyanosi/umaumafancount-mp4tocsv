"""Tests for ``gui.result_view.ResultTableView.__init__``.

Specification: docs/00-Architecture/gui_result_view__ResultTableView___init__.yaml

``__init__`` initializes a ResultTableView instance that inherits
from ctk.CTkFrame: it initializes the internal state (data, entries)
and then calls ``_setup_ui`` to build the UI:

1. call ``super().__init__(master, **kwargs)`` to delegate
   initialization to ctk.CTkFrame
2. set ``self.data`` to an empty dict
3. set ``self.entries`` to an empty list
4. call ``self._setup_ui()`` to build the header part and the scroll
   area

Mocked / stand-in dependencies (per the test-generation rules):
``ResultTableView`` is a real widget; ``test_edge_01`` builds it with
no arguments (``master=None`` creates its own root window) and
``test_edge_02`` builds it on a real ``ctk.CTkFrame`` parent created
on a minimal ``tkinter.Tk()`` root window (created with a retry
helper against unstable Tcl environments and destroyed at the end of
each test).

``errors`` section of the spec (documented here only, NOT tested, per
the test-generation rules):

- condition: "master か kwargs が ctk.CTkFrame.__init__ で例外を招く"
  behavior: "CTkFrame.__init__ が送出する例外が伝播する（例外種別は CTkFrame 実装に依存し本ファイルでは定義されない）"
"""

import contextlib
import time

import customtkinter as ctk
import tkinter as tk

from gui.result_view import ResultTableView


def _create_root():
    """Create a minimal ``tkinter.Tk()`` root window, retrying up to 5
    times at 0.25 s intervals against unstable Tcl environments;
    re-raise the original ``TclError`` if every attempt fails."""
    last_error = None
    for attempt in range(5):
        if attempt:
            time.sleep(0.25)
        try:
            return tk.Tk()
        except tk.TclError as exc:
            last_error = exc
    raise last_error


def _header_labels(view):
    """Return the ``text`` of the CTkLabel children of the header
    frame (the widget gridded at (0,0)) in creation order."""
    header = view.grid_slaves(0, 0)[0]
    return [c.cget("text")
            for c in header.winfo_children()
            if isinstance(c, ctk.CTkLabel)]


def test_edge_01():
    """
    input: ResultTableView()（引数なし）
    expected: master=None の CTkFrame が生成され、self.data == {}、self.entries == [] となり、ヘッダ部とスクロール領域が構築される
    """
    view = ResultTableView()
    try:
        assert isinstance(view, ctk.CTkFrame)
        # master=None was passed to CTkFrame.__init__; the widget
        # created its own root window, which is now its master.
        assert isinstance(view.master, tk.Tk)
        assert view.data == {}
        assert view.entries == []
        assert isinstance(view.scroll_frame, ctk.CTkScrollableFrame)
        assert _header_labels(view) == ["ユーザ名", "ファン数"]
        # The widget gridded at (1,0) is the outer CTkFrame wrapper of
        # the scrollable frame (its canvas's master).
        assert view.scroll_frame.master.master is view.grid_slaves(1, 0)[0]
    finally:
        view.destroy()


def test_edge_02():
    """
    input: ResultTableView(master=frame, fg_color='white')
    expected: master と fg_color は CTkFrame.__init__ にそのまま渡され、ウィジェットは CTkFrame のルールに従い生成される（挙動は CTkFrame 実装で決まる）
    """
    root = _create_root()
    try:
        frame = ctk.CTkFrame(root)
        view = ResultTableView(master=frame, fg_color="white")
        try:
            assert view in frame.winfo_children()
            assert view.cget("fg_color") == "white"
            assert view.data == {}
            assert view.entries == []
            assert isinstance(view.scroll_frame, ctk.CTkScrollableFrame)
            assert _header_labels(view) == ["ユーザ名", "ファン数"]
        finally:
            view.destroy()
        frame.destroy()
    finally:
        root.destroy()
