"""Tests for ``gui.main_window.MainWindow._on_video_process.process_thread._apply``.

Specification: docs/00-Architecture/gui_main_window__MainWindow__on_video_process_process_thread__apply.yaml

``_apply`` is the nested function ``def _apply():`` inside
``process_thread`` (itself nested in ``MainWindow._on_video_process``; per
the spec's ``function`` field). Only when the main window still exists it
applies the return value of ``process_video`` (the closure variable
``result``) to the UI via ``self.set_result_data``.

- it calls ``_still_exists()`` and evaluates the truthiness of its return
  value
- if truthy it calls ``self.set_result_data(result)``
- if falsy it does nothing and returns
- the function returns ``None`` (no explicit ``return`` statement)

Because ``_apply`` is a local function of ``process_thread`` it cannot be
imported by name; the module-level ``_apply`` object below is recovered by
first finding the code object named ``process_thread`` inside
``MainWindow._on_video_process.__code__.co_consts`` and then the code
object named ``_apply`` inside that ``co_consts``, and rebound as a plain
function via ``types.FunctionType`` with the module's globals, supplying
the closure variables (``self``, ``result``, ``_still_exists``) through
``types.CellType`` cells (per the ``nms_badges_score`` precedent).

``self`` is a minimal tkinter root window (``tkinter.Tk()``) built per the
spec's preconditions and destroyed at test end; the external dependency
``self.set_result_data`` is mocked with ``unittest.mock`` and
``_still_exists`` is stubbed as a plain function per the edge-case inputs,
both supplied as closure variables for each edge case.

``errors`` section of the spec (documented here only, NOT tested, per the
test-generation rules):

- condition: 'self.set_result_data(result) の呼び出しで例外が送出'
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


# Recover the nested ``_apply`` function object of
# ``MainWindow._on_video_process.process_thread`` (see the module docstring
# for the rationale).
_process_thread_code = None
for _const in MainWindow._on_video_process.__code__.co_consts:
    if isinstance(_const, types.CodeType) and _const.co_name == "process_thread":
        _process_thread_code = _const
        break
if _process_thread_code is None:
    raise AssertionError(
        "nested function 'process_thread' (per the spec's function field) "
        "was not found inside MainWindow._on_video_process"
    )

_apply_code = None
for _const in _process_thread_code.co_consts:
    if isinstance(_const, types.CodeType) and _const.co_name == "_apply":
        _apply_code = _const
        break
if _apply_code is None:
    raise AssertionError(
        "nested function '_apply' (per the spec's function field) was not "
        "found inside process_thread"
    )


def _make_fn(self_obj, result, still_exists):
    """Build a callable ``_apply`` from the recovered code object, binding
    the closure variables ``self``, ``result`` and ``_still_exists``."""
    cells = []
    for name in _apply_code.co_freevars:
        if name == "self":
            cells.append(types.CellType(self_obj))
        elif name == "result":
            cells.append(types.CellType(result))
        elif name == "_still_exists":
            cells.append(types.CellType(still_exists))
        else:
            cells.append(types.CellType(mock.Mock()))
    return types.FunctionType(
        _apply_code,
        MainWindow._on_video_process.__globals__,
        _apply_code.co_name,
        None,
        tuple(cells),
    )


def test_edge_01():
    """
    input: _still_exists() が真値を返し（ウィンドウ生存）、クロージャの result が R の場合
    expected: self.set_result_data が引数 R でちょうど1回呼ばれる。関数は None を返す
    """
    root = _make_root()
    try:
        r = "R"
        root.set_result_data = mock.Mock()
        result = _make_fn(root, r, lambda: True)()
        assert result is None
        root.set_result_data.assert_called_once_with(r)
    finally:
        root.destroy()


def test_edge_02():
    """
    input: _still_exists() が False を返す場合（ウィンドウ破棄済み）
    expected: self.set_result_data は呼ばれない。関数は None を返す
    """
    root = _make_root()
    try:
        r = "R"
        root.set_result_data = mock.Mock()
        result = _make_fn(root, r, lambda: False)()
        assert result is None
        root.set_result_data.assert_not_called()
    finally:
        root.destroy()


def test_edge_03():
    """
    input: ウィンドウ生存で self.set_result_data(result) が ValueError を送出する場合
    expected: ValueError は捕捉されず関数の外へ伝播する（try/except がない）
    """
    root = _make_root()
    try:
        r = "R"
        root.set_result_data = mock.Mock(side_effect=ValueError("bad result"))
        try:
            _make_fn(root, r, lambda: True)()
        except ValueError:
            raised = True
        else:
            raised = False
        assert raised
    finally:
        root.destroy()
