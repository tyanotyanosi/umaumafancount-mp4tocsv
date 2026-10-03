"""Tests for ``gui.main_window.MainWindow._on_video_process._safe_after``.

Specification: docs/00-Architecture/gui_main_window__MainWindow__on_video_process__safe_after.yaml

``_safe_after`` is the nested function ``def _safe_after(func):`` inside
``MainWindow._on_video_process`` (per the spec's ``function`` field). It
schedules ``func`` through the main window via ``self.after(0, func)`` and,
if ``tk.TclError`` is raised (per the spec's documented string, after the
window has been closed), silently gives up scheduling.

- it calls ``self.after(0, func)`` inside a ``try`` block
- if ``tk.TclError`` is raised it does nothing and returns (``pass``)
- if no exception is raised, the ``after`` callback registration completes
  and the function returns
- in every case the function returns ``None`` (no explicit ``return``
  statement)

Because ``_safe_after`` is a local function of ``_on_video_process`` it
cannot be imported by name; the module-level ``_safe_after`` object below
is recovered from the code object named ``_safe_after`` inside
``MainWindow._on_video_process.__code__.co_consts`` and rebound as a plain
function via ``types.FunctionType`` with the module's globals, supplying
the closure variable (``self``) through a ``types.CellType`` cell (per the
``nms_badges_score`` precedent).

``self`` is a minimal tkinter root window (``tkinter.Tk()``) built per the
spec's preconditions and destroyed at test end; the external dependency
``self.after`` is mocked with ``unittest.mock`` for each edge case.

``errors`` section of the spec (documented here only, NOT tested, per the
test-generation rules):

- condition: 'self.after(0, func) が tk.TclError を送出（ウィンドウが既に破棄済み）'
  behavior: '例外は飲み込まれ（pass）、関数は None で正常終了する'
- condition: 'self.after(0, func) が tk.TclError 以外の例外を送出'
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


# Recover the nested ``_safe_after`` function object of
# ``MainWindow._on_video_process`` (see the module docstring for the
# rationale).
_safe_after_code = None
for _const in MainWindow._on_video_process.__code__.co_consts:
    if (
        isinstance(_const, types.CodeType)
        and _const.co_name == "_safe_after"
        and _const.co_argcount == 1
        and _const.co_varnames[0] == "func"
    ):
        _safe_after_code = _const
        break
if _safe_after_code is None:
    raise AssertionError(
        "nested function '_safe_after' (per the spec's function field) was "
        "not found inside MainWindow._on_video_process"
    )


def _make_fn(self_obj):
    """Build a callable ``_safe_after`` from the recovered code object,
    binding the closure variable ``self`` to ``self_obj``."""
    cells = []
    for name in _safe_after_code.co_freevars:
        if name == "self":
            cells.append(types.CellType(self_obj))
        else:
            cells.append(types.CellType(mock.Mock()))
    return types.FunctionType(
        _safe_after_code,
        MainWindow._on_video_process.__globals__,
        _safe_after_code.co_name,
        None,
        tuple(cells),
    )


def test_edge_01():
    """
    input: ウィンドウが生きており func が引数0個の何もしない関数の場合
    expected: 例外なし。関数は None を返し、func は after(0, func) によりスケジューリングされる（イベントループで後刻実行）
    """
    root = _make_root()
    try:
        after = mock.Mock(return_value=1)
        root.after = after
        func = lambda: None
        result = _make_fn(root)(func)
        assert result is None
        after.assert_called_once_with(0, func)
    finally:
        root.destroy()


def test_edge_02():
    """
    input: self.after が tk.TclError を送出する場合（ウィンドウが既に破棄済み）
    expected: 例外なし。関数は None を返し、func はスケジューリングされない
    """
    root = _make_root()
    try:
        after = mock.Mock(
            side_effect=tk.TclError("application has been destroyed")
        )
        root.after = after
        func = lambda: None
        result = _make_fn(root)(func)
        assert result is None
        after.assert_called_once_with(0, func)
    finally:
        root.destroy()


def test_edge_03():
    """
    input: self.after が tk.TclError 以外の例外を送出する場合
    expected: その例外は捕捉されず関数の外へ伝播する
    """
    root = _make_root()
    try:
        after = mock.Mock(side_effect=ValueError("not a TclError"))
        root.after = after
        func = lambda: None
        try:
            _make_fn(root)(func)
        except ValueError:
            raised = True
        else:
            raised = False
        assert raised
    finally:
        root.destroy()
