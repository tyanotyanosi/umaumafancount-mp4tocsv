"""Tests for ``gui.result_view.ResultTableView.get_selected``.

Specification: docs/00-Architecture/gui_result_view__ResultTableView_get_selected.yaml

``get_selected`` is a method intended to return the selected item, but
the current implementation always returns ``None``:

1. immediately return ``None``

The method does not reference ``self`` at all; ``self`` is unchanged
(self.data, self.entries, and the UI are all invariant) and there are
no side effects.

Mocked / stand-in dependencies (per the test-generation rules):
``ResultTableView`` is a real widget built on a minimal
``tkinter.Tk()`` root window (created with a retry helper against
unstable Tcl environments and destroyed at the end of each test).

``errors`` section of the spec: empty (no error conditions documented).
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
    input: 'update_data({"a": 1}) 後のインスタンスで呼び出し'
    expected: None が返る
    """
    with _view() as view:
        view.update_data({"a": 1})
        data_before = view.data
        entries_before = list(view.entries)
        assert view.get_selected() is None
        # self is unchanged (data, entries, UI all invariant).
        assert view.data is data_before
        assert view.entries == entries_before
        for row in entries_before:
            assert row.winfo_exists() == 1


def test_edge_02():
    """
    input: 'clear() 後のインスタンスで呼び出し'
    expected: None が返る
    """
    with _view() as view:
        view.update_data({"a": 1})
        view.clear()
        assert view.get_selected() is None
        assert view.entries == []
        assert view.data == {}
