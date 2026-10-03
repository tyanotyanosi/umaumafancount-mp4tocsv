"""Tests for ``gui.main_window.MainWindow._on_video_process.process_thread._reenable``.

Specification: docs/00-Architecture/gui_main_window__MainWindow__on_video_process_process_thread__reenable.yaml

``_reenable`` is the nested function ``def _reenable():`` inside
``process_thread`` (itself nested in ``MainWindow._on_video_process``)
(per the spec's ``function`` field). Per the spec's ``purpose`` and
``behavior``, it re-enables the process button that the caller disabled
before starting the worker thread:

- it calls ``_still_exists()`` and evaluates the truthiness of its return value
- if truthy, it sets the state of ``self.video_preview.btn_process`` to
  ``'normal'`` (``configure(state='normal')``, exactly once)
- if falsy, it does nothing and returns
- it returns ``None`` (no explicit return statement)

Because ``_reenable`` is a local function of ``process_thread`` it cannot
be imported by name; the module-level ``_reenable`` object below is
recovered from the nested code object named ``_reenable`` inside
``process_thread.__code__.co_consts`` (``process_thread`` itself being
found inside ``MainWindow._on_video_process.__code__.co_consts``) and
rebound as a plain function via ``types.FunctionType`` with the module's
globals, supplying its closure variables (``_still_exists``, ``self`` in
``co_freevars`` order) as ``types.CellType`` cells (same technique as the
``test_src_video_card_detector__CardDetector__nms_badges_score``
precedent).

Per the spec's ``preconditions``, ``self`` is built so that
``self.video_preview`` and its ``btn_process`` attribute exist with a
``configure`` method; ``_still_exists`` and the button widget are mocked
with ``unittest.mock`` (the tkinter/CTk button implementation and the
generated code are out of the reading scope, per the spec's ``missing``).

``errors`` section of the spec (documented here only, NOT tested, per the
test-generation rules):

- condition: 'self.video_preview.btn_process.configure の呼び出しで例外が送出'
  behavior: '例外は捕捉されず関数の外へ伝播する'
"""

import types
import unittest.mock

import tkinter as tk

from gui.main_window import MainWindow


def _find_nested_code(owner_code, name):
    """Return the code object named ``name`` among ``owner_code``'s consts."""
    for const in owner_code.co_consts:
        if isinstance(const, types.CodeType) and const.co_name == name:
            return const
    raise AssertionError(
        f"nested code object {name!r} not found inside {owner_code.co_name!r}"
    )


_process_thread_code = _find_nested_code(
    MainWindow._on_video_process.__code__, "process_thread"
)
_reenable_code = _find_nested_code(_process_thread_code, "_reenable")


def _make_reenable(self_obj, still_exists):
    """Bind the recovered ``_reenable`` code with the given closure cells."""
    values = {"_still_exists": still_exists, "self": self_obj}
    cells = []
    for name in _reenable_code.co_freevars:
        if name in values:
            cells.append(types.CellType(values[name]))
        else:
            cells.append(types.CellType(unittest.mock.MagicMock(name=name)))
    return types.FunctionType(
        _reenable_code,
        MainWindow._on_video_process.__globals__,
        None,
        None,
        tuple(cells),
    )


def test_edge_01():
    """
    input: _still_exists() が真値を返す場合（ウィンドウ生存）
    expected: self.video_preview.btn_process.configure が state='normal' でちょうど1回呼ばれる。関数は None を返す
    """
    root = tk.Tk()
    root.withdraw()
    try:
        btn_process = unittest.mock.Mock(name="btn_process")
        self = types.SimpleNamespace(
            video_preview=types.SimpleNamespace(btn_process=btn_process)
        )
        still_exists = unittest.mock.Mock(name="_still_exists", return_value=True)
        reenable = _make_reenable(self, still_exists)
        result = reenable()
        assert result is None
        still_exists.assert_called_once()
        btn_process.configure.assert_called_once_with(state="normal")
    finally:
        root.destroy()


def test_edge_02():
    """
    input: _still_exists() が False を返す場合（ウィンドウ破棄済み）
    expected: self.video_preview.btn_process.configure は呼ばれない。関数は None を返す
    """
    root = tk.Tk()
    root.withdraw()
    try:
        btn_process = unittest.mock.Mock(name="btn_process")
        self = types.SimpleNamespace(
            video_preview=types.SimpleNamespace(btn_process=btn_process)
        )
        still_exists = unittest.mock.Mock(name="_still_exists", return_value=False)
        reenable = _make_reenable(self, still_exists)
        result = reenable()
        assert result is None
        still_exists.assert_called_once()
        btn_process.configure.assert_not_called()
    finally:
        root.destroy()


def test_edge_03():
    """
    input: ウィンドウ生存で btn_process.configure が例外を送出する場合
    expected: その例外は捕捉されず関数の外へ伝播する
    """
    root = tk.Tk()
    root.withdraw()
    try:
        btn_process = unittest.mock.Mock(name="btn_process")
        btn_process.configure.side_effect = RuntimeError("boom")
        self = types.SimpleNamespace(
            video_preview=types.SimpleNamespace(btn_process=btn_process)
        )
        still_exists = unittest.mock.Mock(name="_still_exists", return_value=True)
        reenable = _make_reenable(self, still_exists)
        try:
            reenable()
        except RuntimeError:
            raised = True
        else:
            raised = False
        assert raised
    finally:
        root.destroy()
