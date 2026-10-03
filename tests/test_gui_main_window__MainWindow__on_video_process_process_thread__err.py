"""Tests for ``gui.main_window.MainWindow._on_video_process.process_thread._err``.

Specification: docs/00-Architecture/gui_main_window__MainWindow__on_video_process_process_thread__err.yaml

``_err`` is the nested function ``def _err():`` inside the ``except Exception``
block of ``process_thread`` (per the spec's ``function`` field). It is the
error-display after callback: it calls the closure ``_still_exists()`` (which
reads ``self.winfo_exists()`` and returns ``False`` when ``tk.TclError`` is
raised), and only when the window still exists it calls
``self.status_label.configure(text=f"Error: {e}")`` once. It returns ``None``
(no return statement).

Mocked / stand-in dependencies (per the test-generation rules): the closure
``self`` is a ``MainWindow`` instance created without running ``__init__``
(``object.__new__``) backed by a minimal ``tkinter.Tk()`` root window (created
with a retry helper against unstable Tcl environments and destroyed at the end
of each test); ``self.status_label`` is a minimal fake widget recording
``configure`` calls; ``e`` is a plain ``Exception`` instance whose ``str`` is
controlled per test.

Because ``_err`` is a local function of ``process_thread`` (itself a local
function of ``_on_video_process``) it cannot be imported by name; the
module-level function object below is recovered from the nested code object
named ``_err`` inside ``process_thread``'s code object (searched recursively
from ``MainWindow._on_video_process.__code__.co_consts``) and rebound as a
plain function via ``types.FunctionType`` with the module's globals, supplying
the closure variables (``self``, ``e`` per the spec's preconditions) via
``types.CellType``. The nested ``_still_exists`` closure is bound the same way.

``errors`` section of the spec (documented here only, NOT tested, per the
test-generation rules):

- condition: 'self.status_label が存在しない、または configure が例外を送出'
  behavior: '例外は本関数で捕捉されず、Tk メインループの after コールバック処理へ伝播する（_safe_after はスケジュール時の tk.TclError のみ握り潰し、コールバック実行時の例外は対象外）'
- condition: 'self.winfo_exists() が tk.TclError 以外の例外を送出'
  behavior: '_still_exists は tk.TclError のみ捕捉するため、例外は _err の外へ伝播する'
"""

import contextlib
import time
import types

import tkinter as tk

from gui.main_window import MainWindow


def _find_nested_code(parent_code, name):
    """Recursively find a nested code object named ``name`` under
    ``parent_code``."""
    stack = [parent_code]
    while stack:
        code = stack.pop()
        for const in code.co_consts:
            if not isinstance(const, types.CodeType):
                continue
            if const.co_name == name:
                return const
            stack.append(const)
    return None


_on_video_process_code = MainWindow._on_video_process.__code__
_process_thread_code = _find_nested_code(_on_video_process_code, "process_thread")
if _process_thread_code is None:
    raise AssertionError(
        "nested function 'process_thread' (per the spec's function field) was "
        "not found inside MainWindow._on_video_process"
    )
_err_code = _find_nested_code(_process_thread_code, "_err")
if _err_code is None:
    raise AssertionError(
        "nested function '_err' (per the spec's function field) was not found "
        "inside process_thread"
    )


def _bind_function(code, values, globals_dict):
    """Rebind ``code`` as a plain function, supplying each free variable from
    ``values`` (or recursively binding a nested code object of that name).
    ``globals_dict`` is the module globals of the defining function (code
    objects carry no globals of their own)."""
    cells = []
    for name in code.co_freevars:
        if name in values:
            value = values[name]
        else:
            nested = _find_nested_code(_on_video_process_code, name)
            if nested is None:
                raise AssertionError(
                    f"cannot resolve closure variable {name!r} of the nested "
                    f"function under process_thread"
                )
            value = _bind_function(nested, values, globals_dict)
        cells.append(types.CellType(value))
    fn = types.FunctionType(code, globals_dict, closure=tuple(cells))
    return fn


def _make_err(self, e):
    """Bind the nested ``_err`` function with closure variables ``self`` and
    ``e``."""
    return _bind_function(
        _err_code,
        {"self": self, "e": e},
        MainWindow._on_video_process.__globals__,
    )


class _FakeWidget:
    """Minimal stand-in for a Tk widget: records ``configure`` keyword calls
    and item assignments so tests can assert on ``text`` / ``state``."""

    def __init__(self):
        self.text = None
        self.state = None
        self.configure_calls = []

    def configure(self, **kwargs):
        self.configure_calls.append(kwargs)
        if "text" in kwargs:
            self.text = kwargs["text"]
        if "state" in kwargs:
            self.state = kwargs["state"]

    def __setitem__(self, key, value):
        if key == "text":
            self.text = value
        elif key == "state":
            self.state = value


def _create_root():
    """Create a minimal ``tkinter.Tk()`` root window, retrying up to 5 times
    at 0.25 s intervals against unstable Tcl environments; re-raise the
    original ``TclError`` if every attempt fails."""
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
def _root_window():
    root = _create_root()
    try:
        yield root
    finally:
        try:
            root.destroy()
        except tk.TclError:
            pass


def _make_self(root):
    """Build a ``MainWindow`` instance (without running ``__init__``) backed by
    the given Tk root, with the dependencies ``_err`` touches faked."""
    self = object.__new__(MainWindow)
    self.status_label = _FakeWidget()
    self.winfo_exists = root.winfo_exists
    return self


def test_edge_01():
    """
    input: self.winfo_exists() が True かつ例外 e の str が x
    expected: self.status_label.configure が 1 回呼ばれ、ラベルの text が Error: x になる
    """
    with _root_window() as root:
        self = _make_self(root)
        err = _make_err(self, Exception("x"))
        ret = err()
    assert ret is None
    assert self.status_label.configure_calls == [{"text": "エラー: x"}]
    assert self.status_label.text == "エラー: x"


def test_edge_02():
    """
    input: self.winfo_exists() が False（ウィンドウ破棄済み）
    expected: self.status_label.configure は呼ばれず、関数は None を返しラベルは不変
    """
    with _root_window() as root:
        self = _make_self(root)
        err = _make_err(self, Exception("x"))
        root.destroy()  # window disposed before _err runs
        ret = err()
    assert ret is None
    assert self.status_label.configure_calls == []
    assert self.status_label.text is None


def test_edge_03():
    """
    input: self.winfo_exists() が tk.TclError を送出
    expected: _still_exists() が False を返し、self.status_label.configure は呼ばれず、関数は例外を送出せず None を返す
    """
    with _root_window() as root:
        self = _make_self(root)

        def _winfo_raises():
            raise tk.TclError("winfo failed")

        self.winfo_exists = _winfo_raises
        err = _make_err(self, Exception("x"))
        ret = err()
    assert ret is None
    assert self.status_label.configure_calls == []
    assert self.status_label.text is None


def test_edge_04():
    """
    input: 例外 e の str(e) が空文字列
    expected: ラベルの text が Error: （コロンの後に半角スペース 1 つ、f"Error: {e}" の書式結果）になる
    """
    with _root_window() as root:
        self = _make_self(root)
        err = _make_err(self, Exception(""))
        ret = err()
    assert ret is None
    assert self.status_label.configure_calls == [{"text": "エラー: "}]
    assert self.status_label.text == "エラー: "
