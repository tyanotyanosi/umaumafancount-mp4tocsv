"""Tests for ``gui.main_window.MainWindow._on_video_process._still_exists``.

Specification: docs/00-Architecture/gui_main_window__MainWindow__on_video_process__still_exists.yaml

``_still_exists`` is the nested function ``def _still_exists():`` inside
``MainWindow._on_video_process`` (per the spec's ``function`` field). It
returns ``self.winfo_exists()`` to determine whether the main window
(``self``) still exists on tkinter, and returns ``False`` if
``tk.TclError`` is raised.

- it calls ``self.winfo_exists()`` inside a ``try`` block
- if no exception is raised it returns that return value as-is
- if ``tk.TclError`` occurs it returns ``False``

Because ``_still_exists`` is a local function of ``_on_video_process`` it
cannot be imported by name; the module-level ``_still_exists`` object below
is recovered from the code object named ``_still_exists`` inside
``MainWindow._on_video_process.__code__.co_consts`` and rebound as a plain
function via ``types.FunctionType`` with the module's globals, supplying
the closure variable (``self``) through a ``types.CellType`` cell (per the
``nms_badges_score`` precedent).

``self`` is a minimal tkinter root window (``tkinter.Tk()``) built per the
spec's preconditions and destroyed at test end; the external dependency
``self.winfo_exists`` is mocked with ``unittest.mock`` for each edge case.

``errors`` section of the spec (documented here only, NOT tested, per the
test-generation rules):

- condition: 'self.winfo_exists() が tk.TclError を送出'
  behavior: 'False を返す（例外は伝播しない）'
- condition: 'self.winfo_exists() が tk.TclError 以外の例外を送出'
  behavior: '例外は捕捉されず関数の外へ伝播する'
"""

import time
import types
import tkinter as tk
from unittest import mock

from gui.main_window import MainWindow


def _make_root(max_attempts=5, delay=0.25):
    """Create a minimal ``tkinter.Tk()`` root window.

    Retries up to ``max_attempts`` times at ``delay`` second intervals if
    Tcl is unstable (``TclError`` on ``Tk()`` creation); raises the
    original ``TclError`` if every attempt fails. Does not change test
    semantics.
    """
    last_error = None
    for attempt in range(max_attempts):
        try:
            return tk.Tk()
        except tk.TclError as exc:
            last_error = exc
            if attempt < max_attempts - 1:
                time.sleep(delay)
    raise last_error


# Recover the nested ``_still_exists`` function object of
# ``MainWindow._on_video_process`` (see the module docstring for the
# rationale).
_still_exists_code = None
for _const in MainWindow._on_video_process.__code__.co_consts:
    if (
        isinstance(_const, types.CodeType)
        and _const.co_name == "_still_exists"
        and _const.co_argcount == 0
    ):
        _still_exists_code = _const
        break
if _still_exists_code is None:
    raise AssertionError(
        "nested function '_still_exists' (per the spec's function field) "
        "was not found inside MainWindow._on_video_process"
    )


def _make_fn(self_obj):
    """Build a callable ``_still_exists`` from the recovered code object,
    binding the closure variable ``self`` to ``self_obj``."""
    cells = []
    for name in _still_exists_code.co_freevars:
        if name == "self":
            cells.append(types.CellType(self_obj))
        else:
            cells.append(types.CellType(mock.Mock()))
    return types.FunctionType(
        _still_exists_code,
        MainWindow._on_video_process.__globals__,
        _still_exists_code.co_name,
        None,
        tuple(cells),
    )


def test_edge_01():
    """
    input: self.winfo_exists() が例外なしで 1 を返す場合（ウィンドウ生存）
    expected: 1（真値）を返す
    """
    root = _make_root()
    try:
        root.winfo_exists = mock.Mock(return_value=1)
        result = _make_fn(root)()
        assert result == 1
    finally:
        root.destroy()


def test_edge_02():
    """
    input: self.winfo_exists() が例外なしで 0 を返す場合
    expected: 0 を返す（False へ変換されない偽値）
    注: 仕様書 expected 原文「0 を返す（False へ変換されない偽値）」。実測挙動を assert
    """
    root = _make_root()
    try:
        root.winfo_exists = mock.Mock(return_value=0)
        result = _make_fn(root)()
        assert result == 0
        assert result is not False
    finally:
        root.destroy()


def test_edge_03():
    """
    input: self.winfo_exists() が tk.TclError を送出する場合
    expected: False を返す
    """
    root = _make_root()
    try:
        root.winfo_exists = mock.Mock(
            side_effect=tk.TclError("application has been destroyed")
        )
        result = _make_fn(root)()
        assert result is False
    finally:
        root.destroy()
