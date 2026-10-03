"""Tests for ``gui.result_view.ResultTableView.update_data``.

Specification: docs/00-Architecture/gui_result_view__ResultTableView_update_data.yaml

``update_data`` regenerates all result rows in the scroll area (it
destroys the old row widgets and, for each key of ``data``, creates a
row frame with a user-name label and a fan-count label):

1. call ``entry.destroy()`` in order on each element of the current
   ``self.entries`` and empty the list
2. set ``self.data`` to the passed ``data``
3. for each ``(user_name, fan_count)`` of ``data.items()`` (in dict
   insertion order):
   a. create a ``CTkFrame`` row_frame in ``self.scroll_frame`` and
      grid it at (row=len(self.entries), column=0, columnspan=2,
      sticky=ew, padx=2, pady=2); the list is empty at start, so row
      increases 0, 1, 2, ... in processing order
   b. create a ``CTkLabel`` (text=user_name, anchor=w) in row_frame
      and grid it at (0,0) with padx=5, pady=5, sticky=ew
   c. create a ``CTkLabel`` (text=str(fan_count), anchor=e) in
      row_frame and grid it at (0,1) with padx=5, pady=5, sticky=ew
   d. append row_frame to ``self.entries``

Mocked / stand-in dependencies (per the test-generation rules):
``ResultTableView`` is a real widget built on a minimal
``tkinter.Tk()`` root window (created with a retry helper against
unstable Tcl environments and destroyed at the end of each test);
``data`` is a plain dict as specified.

``errors`` section of the spec (documented here only, NOT tested, per
the test-generation rules):

- condition: "data が dict ではない（list や None 等）"
  behavior: "data.items() の呼び出しで AttributeError となる"
- condition: "self.entries のウィジェットへの destroy() が例外を送出する"
  behavior: "その例外が伝播し、以降の処理（self.data の設定・行の生成）は実行されない"
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


def _row_texts(row):
    """Return the ``text`` of the CTkLabel children of ``row`` in
    creation order."""
    return [c.cget("text")
            for c in row.winfo_children()
            if isinstance(c, ctk.CTkLabel)]


def test_edge_01():
    """
    input: data = {}（空 dict）
    expected: 既存の entries は破棄され、self.entries == [] かつ self.data == {} となり、新規ウィジェットは生成されない
    """
    with _view() as view:
        view.update_data({"old": 1})
        old_rows = list(view.entries)
        view.update_data({})
        assert view.entries == []
        assert view.data == {}
        for row in old_rows:
            assert row.winfo_exists() == 0
        assert view.scroll_frame.winfo_children() == []


def test_edge_02():
    """
    input: 'data = {"alice": 123, "bob": 0}'
    expected: self.data は data と同一オブジェクト、len(self.entries) == 2、行0 のラベルは text=alice と text=123、行1 のラベルは text=bob と text=0（dict の挿入順に並ぶ）
    """
    with _view() as view:
        data = {"alice": 123, "bob": 0}
        view.update_data(data)
        assert view.data is data
        assert len(view.entries) == 2
        assert _row_texts(view.entries[0]) == ["alice", "123"]
        assert _row_texts(view.entries[1]) == ["bob", "0"]


def test_edge_03():
    """
    input: 'data = {1: 2.5}（キー・値が非文字列）'
    expected: 名前ラベルの text はキーの 1（CTkLabel へそのまま渡される）、件数ラベルの text は str(2.5) の 2.5 になる
    """
    with _view() as view:
        data = {1: 2.5}
        view.update_data(data)
        assert len(view.entries) == 1
        texts = _row_texts(view.entries[0])
        assert texts[0] == 1
        assert texts[1] == "2.5"
