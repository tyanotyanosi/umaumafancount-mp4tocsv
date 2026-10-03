"""Tests for ``gui.result_view.ResultTableView._setup_ui``.

Specification: docs/00-Architecture/gui_result_view__ResultTableView__setup_ui.yaml

``_setup_ui`` builds the static UI of ResultTableView (the two-column
header labels「ユーザ名」「ファン数」and the scrollable content
frame ``self.scroll_frame``):

1. set grid row 0 and column 0 of ``self`` to weight=1
2. create a ``CTkFrame`` header_frame in ``self`` and grid it at
   (0,0) with sticky=ew, padx=5, pady=(5,0); set its column 0 to
   weight=1 and column 1 to weight=2
3. create a ``CTkLabel`` (text=ユーザ名, anchor=w) in header_frame
   and grid it at (0,0) with padx=5, pady=5, sticky=ew
4. create a ``CTkLabel`` (text=ファン数, anchor=e) in header_frame
   and grid it at (0,1) with padx=5, pady=5, sticky=ew
5. create a ``CTkScrollableFrame`` (label_text='') in ``self``,
   assign it to ``self.scroll_frame``, grid it at (1,0) with
   sticky=nsew, padx=5, pady=5, and set its column 0 to weight=1 and
   column 1 to weight=2

Mocked / stand-in dependencies (per the test-generation rules):
``ResultTableView`` is a real widget built on a minimal
``tkinter.Tk()`` root window (created with a retry helper against
unstable Tcl environments and destroyed at the end of each test).
``__init__`` already calls ``_setup_ui`` once, so each test starts
from that state and calls ``_setup_ui`` explicitly as specified.

``errors`` section of the spec (documented here only, NOT tested, per
the test-generation rules):

- condition: "ctk.CTkFrame / CTkLabel / CTkScrollableFrame のコンストラクタが例外を送出する（例。self が有効な親ウィジェットでない場合）"
  behavior: "そのコンストラクタが送出する例外が伝播する（例外種別は customtkinter 実装に依存する）"
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


@contextlib.contextmanager
def _view():
    """Build a real ``ResultTableView`` on a minimal Tk root and
    destroy both at the end."""
    root = _create_root()
    try:
        view = ResultTableView(master=root)
        try:
            yield view
        finally:
            try:
                view.destroy()
            except tk.TclError:
                pass
    finally:
        try:
            root.destroy()
        except tk.TclError:
            pass


def _label_texts(frame):
    """Return the ``text`` of the CTkLabel children of ``frame`` in
    creation order."""
    return [c.cget("text")
            for c in frame.winfo_children()
            if isinstance(c, ctk.CTkLabel)]


def test_edge_01():
    """
    input: __init__ 完了済みインスタンスで1回呼び出し
    expected: ヘッダフレーム・ラベル2個・self.scroll_frame が追加され、各 grid の weight 設定が反映される
    """
    with _view() as view:
        before = len(view.winfo_children())
        old_outer = view.scroll_frame.master.master
        view._setup_ui()
        kids = view.winfo_children()
        assert len(kids) == before + 2
        header = kids[-2]
        new_outer = kids[-1]
        assert isinstance(header, ctk.CTkFrame)
        # The widget gridded at (1,0) is the outer CTkFrame wrapper of
        # the new CTkScrollableFrame (its canvas's master).
        assert isinstance(new_outer, ctk.CTkFrame)
        assert isinstance(view.scroll_frame, ctk.CTkScrollableFrame)
        assert view.scroll_frame.master.master is new_outer
        assert new_outer is not old_outer
        assert _label_texts(header) == ["ユーザ名", "ファン数"]
        assert view.grid_rowconfigure(0)["weight"] == 1
        assert view.grid_columnconfigure(0)["weight"] == 1
        assert header.grid_columnconfigure(0)["weight"] == 1
        assert header.grid_columnconfigure(1)["weight"] == 2
        # The weight settings are applied to the CTkScrollableFrame
        # itself (not to its outer wrapper).
        assert view.scroll_frame.grid_columnconfigure(0)["weight"] == 1
        assert view.scroll_frame.grid_columnconfigure(1)["weight"] == 2


def test_edge_02():
    """
    input: 同一インスタンスで2回呼び出し
    expected: 2セット目のヘッダフレーム・ラベル2個・新しい scroll_frame が追加される。1セット目は破棄されず同じ grid セルに重複配置される（表示は Tk の grid 挙動に依存）。self.scroll_frame は最後に作成されたインスタンスへ再代入される
    """
    with _view() as view:
        first_kids = list(view.winfo_children())
        first_header, first_scroll = first_kids
        view._setup_ui()
        view._setup_ui()
        kids = view.winfo_children()
        assert len(kids) == len(first_kids) + 4
        assert first_header.winfo_exists() == 1
        assert first_scroll.winfo_exists() == 1
        last_outer = kids[-1]
        # self.scroll_frame is reassigned to the last-created
        # CTkScrollableFrame (its outer wrapper is the last child).
        assert view.scroll_frame.master.master is last_outer
        assert last_outer is not first_scroll
        assert _label_texts(kids[-2]) == ["ユーザ名", "ファン数"]
