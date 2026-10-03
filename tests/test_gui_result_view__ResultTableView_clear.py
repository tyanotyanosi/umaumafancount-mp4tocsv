"""Tests for ``gui.result_view.ResultTableView.clear``.

Specification: docs/00-Architecture/gui_result_view__ResultTableView_clear.yaml

``clear`` removes all result rows in the scroll area and resets the
data state to empty:

1. call ``entry.destroy()`` in order on each element of the current
   ``self.entries``
2. clear ``self.entries`` (empty it)
3. set ``self.data`` to an empty dict

Mocked / stand-in dependencies (per the test-generation rules):
``ResultTableView`` is a real widget built on a minimal
``tkinter.Tk()`` root window (created with a retry helper against
unstable Tcl environments and destroyed at the end of each test).

``errors`` section of the spec (documented here only, NOT tested, per
the test-generation rules):

- condition: "self.entries のウィジェットへの destroy() が例外を送出する"
  behavior: "その例外が伝播し、以降の処理（self.entries.clear() と self.data の再代入）は実行されない"
"""

import contextlib
import time

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


def test_edge_01():
    """
    input: self.entries == [] の self（行が1件もない状態）
    expected: destroy() は0回呼び出され、self.entries == [] かつ self.data == {} となる
    """
    with _view() as view:
        assert view.entries == []
        view.clear()
        assert view.entries == []
        assert view.data == {}


def test_edge_02():
    """
    input: 'update_data({"a": 1, "b": 2}) 後の self（行が2件ある状態）'
    expected: 2件の row_frame が破棄され、self.entries == [] かつ self.data == {} となる
    """
    with _view() as view:
        view.update_data({"a": 1, "b": 2})
        rows = list(view.entries)
        assert len(rows) == 2
        view.clear()
        for row in rows:
            assert row.winfo_exists() == 0
        assert view.entries == []
        assert view.data == {}
