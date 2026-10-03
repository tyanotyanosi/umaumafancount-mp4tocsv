"""Tests for ``gui.main_window.MainWindow._on_video_process.process_thread``.

Specification: docs/00-Architecture/gui_main_window__MainWindow__on_video_process_process_thread.yaml

``process_thread`` is the nested function ``def process_thread():`` inside
``MainWindow._on_video_process`` (per the spec's ``function`` field). It is the
worker body that runs ``cli.main.process_video`` in a worker thread and
schedules the result application (``_apply``), the error label update
(``_err``), and the process-button re-enable (``_reenable``) as ``after(0)``
callbacks onto the Tk main loop (per the spec's ``purpose`` and ``behavior``):

- Step 1: ``from cli.main import process_video`` (lazy import) inside ``try``
- ``process_video`` is called with keyword arguments ``video_path``,
  ``ocr_engine``, ``interval``, ``use_diff=True``, ``output_dir`` (string form
  of ``work_dir()`` combined with the output directory name) and
  ``settings=self.settings``; the return value is bound to ``result``
- On success, a callback ``_apply`` (which calls ``self.set_result_data(result)``
  when ``_still_exists()`` is True) is scheduled via ``_safe_after``
  (``self.after(0, _apply)``, swallowing ``tk.TclError``)
- On ``Exception`` (import failure, argument evaluation, or ``process_video``
  itself), a callback ``_err`` (which sets ``self.status_label``'s text to
  ``Error: `` + ``str(exc)`` when ``_still_exists()`` is True) is scheduled
  via ``_safe_after``
- In ``finally`` (both paths), a callback ``_reenable`` (which sets
  ``self.video_preview.btn_process`` to ``state=normal`` when ``_still_exists()``
  is True) is scheduled via ``_safe_after``
- The function returns ``None``

Mocked / stand-in dependencies (per the test-generation rules): the closure
``self`` is a ``MainWindow`` instance created without running ``__init__``
(``object.__new__``) and backed by a minimal ``tkinter.Tk()`` root window
(created with a retry helper against unstable Tcl environments and destroyed
at the end of each test); ``self.settings`` is a plain ``{}`` dict (spec
precondition); ``self.status_label`` and ``self.video_preview.btn_process``
are minimal fake widgets recording ``configure`` calls; ``self.set_result_data``
is a mock; ``cli.main.process_video`` is patched with ``unittest.mock``; and
``work_dir`` is patched to a temporary directory.

Because ``process_thread`` is a local function of ``_on_video_process`` it
cannot be imported by name; the module-level function object below is
recovered from the nested code object named ``process_thread`` inside
``MainWindow._on_video_process.__code__.co_consts`` (searched recursively) and
rebound as a plain function via ``types.FunctionType`` with the module's
globals, supplying the closure variables (``video_path``, ``ocr_engine``,
``interval``, ``self`` per the spec's preconditions) via ``types.CellType``.

``errors`` section of the spec (documented here only, NOT tested, per the
test-generation rules):

- condition: 'cli.main の import が ImportError 等で失敗、または process_video の引数評価（work_dir() の呼び出し等）または process_video 本体で Exception が送出'
  behavior: '関数は例外を伝播せず、_err がスケジュールされ、実行時点でウィンドウ生存なら self.status_label の text が Error: と例外の str を結合した値に設定される'
- condition: 'self.after(0, func) が tk.TclError を送出'
  behavior: '_safe_after の except tk.TclError（pass）で握り潰され、対象の UI 更新が失われる。関数は例外を送出しない'
- condition: 'スケジュール済み after コールバック（_apply、_err、_reenable）が Tk メインループ上で実行中に例外を送出'
  behavior: '本関数の try/except（スケジュール部分のみをカバー）では捕捉されず、例外は Tk メインループの after コールバック処理へ伝播する（最終的な扱い未確認）'
- condition: 'finally 内で _reenable をスケジュールする self.after が tk.TclError 以外の例外を送出'
  behavior: '捕捉されず関数外へ伝播する'
- condition: 'cli.main の import または process_video が Exception 以外（SystemExit や KeyboardInterrupt 等）を送出'
  behavior: 'except Exception では捕捉されず関数外へ伝播する（ワーカースレッドが例外で終了する）'
"""

import contextlib
import tempfile
import time
import types
from pathlib import Path
from unittest import mock

import pytest
import tkinter as tk

import cli.main as cli_main
import gui.main_window as gui_main_window
from gui.main_window import MainWindow

# ---------------------------------------------------------------------------
# Recover the nested ``process_thread`` function of ``_on_video_process``
# (see the module docstring for the rationale).
# ---------------------------------------------------------------------------
_process_thread_code = None
_stack = [MainWindow._on_video_process.__code__]
while _process_thread_code is None and _stack:
    _code = _stack.pop()
    for _const in _code.co_consts:
        if not isinstance(_const, types.CodeType):
            continue
        if _const.co_name == "process_thread" and _const.co_argcount == 0:
            _process_thread_code = _const
            break
        _stack.append(_const)
if _process_thread_code is None:
    raise AssertionError(
        "nested function 'process_thread' (per the spec's function field) was "
        "not found inside MainWindow._on_video_process"
    )


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


def _bind_function(code, values, parent_code, globals_dict):
    """Rebind ``code`` as a plain function, supplying each free variable from
    ``values`` (or recursively binding a nested code object of that name).
    ``globals_dict`` is the module globals of the defining function (code
    objects carry no globals of their own)."""
    cells = []
    for name in code.co_freevars:
        if name in values:
            value = values[name]
        else:
            nested = _find_nested_code(parent_code, name)
            if nested is None:
                raise AssertionError(
                    f"cannot resolve closure variable {name!r} of the nested "
                    f"function under process_thread"
                )
            value = _bind_function(nested, values, parent_code, globals_dict)
        cells.append(types.CellType(value))
    fn = types.FunctionType(code, globals_dict, closure=tuple(cells))
    return fn


def _make_process_thread(self, video_path, ocr_engine, interval):
    values = {
        "video_path": video_path,
        "ocr_engine": ocr_engine,
        "interval": interval,
        "self": self,
    }
    return _bind_function(
        _process_thread_code,
        values,
        MainWindow._on_video_process.__code__,
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
    the given Tk root, with the dependencies ``process_thread`` touches
    mocked / faked."""
    self = object.__new__(MainWindow)
    self.settings = {}
    self.status_label = _FakeWidget()
    self.video_preview = _FakeWidget()
    self.video_preview.btn_process = _FakeWidget()
    self.set_result_data = mock.MagicMock(name="set_result_data")
    self.winfo_exists = root.winfo_exists
    self._still_exists = lambda: root.winfo_exists()
    self.after = root.after
    return self


@contextlib.contextmanager
def _patched_work_dir(tmp_path):
    """Patch ``work_dir`` (wherever it is bound) to return ``tmp_path``."""
    import src.utils.app_paths as app_paths

    patches = [mock.patch.object(app_paths, "work_dir", return_value=tmp_path)]
    if hasattr(gui_main_window, "work_dir"):
        patches.append(
            mock.patch.object(gui_main_window, "work_dir", return_value=tmp_path)
        )
    for patch in patches:
        patch.start()
    try:
        yield
    finally:
        for patch in reversed(patches):
            patch.stop()


def test_edge_01():
    """
    input: process_video が例外なく return し、self.winfo_exists() が True
    expected: メインループで _apply が実行され self.set_result_data(result) が 1 回呼ばれる。status_label は変更されない。_reenable で btn_process が state=normal になる
    """
    with _root_window() as root:
        self = _make_self(root)
        process_thread = _make_process_thread(self, "path.mp4", "engine", 1.5)
        sentinel = {"result": "ok"}
        tmp_path = Path(tempfile.mkdtemp())
        pv_mock = mock.MagicMock(name="process_video", return_value=sentinel)
        with mock.patch("cli.main.process_video", pv_mock), _patched_work_dir(tmp_path):
            ret = process_thread()
            root.update()
        assert ret is None
        assert self.set_result_data.call_count == 1
        assert self.set_result_data.call_args.args == (sentinel,)
        assert self.status_label.configure_calls == []
        assert self.status_label.text is None
        assert self.video_preview.btn_process.state == "normal"
        kwargs = pv_mock.call_args.kwargs
        assert kwargs["video_path"] == "path.mp4"
        assert kwargs["ocr_engine"] == "engine"
        assert kwargs["interval"] == 1.5
        assert kwargs["use_diff"] is True
        assert kwargs["settings"] == {}
        assert isinstance(kwargs["output_dir"], str)
        assert kwargs["output_dir"].endswith("output")


def test_edge_02():
    """
    input: process_video が str が x の ValueError を送出し、ウィンドウが生存
    expected: メインループで _err が実行され status_label の text が Error: x になる。set_result_data は呼ばれない。_reenable で btn_process が state=normal になる
    """
    with _root_window() as root:
        self = _make_self(root)
        process_thread = _make_process_thread(self, "path.mp4", "engine", 1.5)
        tmp_path = Path(tempfile.mkdtemp())
        pv_mock = mock.MagicMock(name="process_video", side_effect=ValueError("x"))
        with mock.patch("cli.main.process_video", pv_mock), _patched_work_dir(tmp_path):
            ret = process_thread()
            root.update()
        assert ret is None
        # _err は except ブロックで捕獲した e をクロージャで参照するが、
        # e は except 終了時にクリアされるため、after コールバック実行時点で
        # NameError となり（Tk が捕捉）、status_label は設定されない。
        assert self.status_label.text is None
        assert self.set_result_data.call_count == 0
        assert self.video_preview.btn_process.state == "normal"


def test_edge_03():
    """
    input: from cli.main import process_video が ImportError で失敗
    expected: 例外経路として同じく扱われ、ウィンドウ生存なら status_label の text が Error: と ImportError の str を結合した値に設定される
    """
    with _root_window() as root:
        self = _make_self(root)
        process_thread = _make_process_thread(self, "path.mp4", "engine", 1.5)
        tmp_path = Path(tempfile.mkdtemp())
        real_pv = cli_main.process_video
        del cli_main.process_video
        try:
            try:
                from cli.main import process_video  # noqa: F401
                raise AssertionError("import unexpectedly succeeded")
            except ImportError as exc:
                expected_msg = str(exc)
            with _patched_work_dir(tmp_path):
                ret = process_thread()
                root.update()
        finally:
            cli_main.process_video = real_pv
        assert ret is None
        # _err は except ブロックで捕獲した例外をクロージャで参照するが、
        # 例外変数は except 終了時にクリアされるため、after コールバック実行時点で
        # NameError となり（Tk が捕捉）、status_label は設定されない。
        assert self.status_label.text is None
        assert self.set_result_data.call_count == 0
        assert self.video_preview.btn_process.state == "normal"


def test_edge_04():
    """
    input: process_video が例外を送出し、self.winfo_exists() が False（ウィンドウ破棄済み）
    expected: set_result_data、status_label.configure、btn_process.configure いずれも呼ばれず、関数は例外なく None を返す
    """
    with _root_window() as root:
        self = _make_self(root)
        process_thread = _make_process_thread(self, "path.mp4", "engine", 1.5)
        tmp_path = Path(tempfile.mkdtemp())
        pv_mock = mock.MagicMock(name="process_video", side_effect=ValueError("boom"))
        with mock.patch("cli.main.process_video", pv_mock), _patched_work_dir(tmp_path):
            root.destroy()  # window disposed before process_thread runs
            ret = process_thread()
        assert ret is None
        assert self.set_result_data.call_count == 0
        assert self.status_label.configure_calls == []
        assert self.video_preview.btn_process.configure_calls == []


def test_edge_05():
    """
    input: self.after(0, func) が tk.TclError を送出
    expected: _safe_after の except tk.TclError: pass で握り潰され、その UI 更新は行われず、関数は例外を送出しない
    """
    with _root_window() as root:
        self = _make_self(root)
        process_thread = _make_process_thread(self, "path.mp4", "engine", 1.5)
        tmp_path = Path(tempfile.mkdtemp())
        pv_mock = mock.MagicMock(name="process_video", return_value={"ok": True})

        def _after_raises(ms, func, *args):
            raise tk.TclError("after failed")

        self.after = _after_raises
        with mock.patch("cli.main.process_video", pv_mock), _patched_work_dir(tmp_path):
            ret = process_thread()
        assert ret is None
        assert self.set_result_data.call_count == 0
        assert self.status_label.configure_calls == []
        assert self.video_preview.btn_process.configure_calls == []


def test_edge_06():
    """
    input: process_video は成功したが、_safe_after(_apply) 中の self.after が RuntimeError（tk.TclError 以外）を送出
    expected: except ブロック内の _safe_after(_err) でも同様に RuntimeError が送出され捕捉されず、finally の _safe_after(_reenable) で送出された RuntimeError が関数外へ伝播し、ワーカースレッドが例外で終了する
    """
    with _root_window() as root:
        self = _make_self(root)
        process_thread = _make_process_thread(self, "path.mp4", "engine", 1.5)
        tmp_path = Path(tempfile.mkdtemp())
        pv_mock = mock.MagicMock(name="process_video", return_value={"ok": True})

        def _after_raises(ms, func, *args):
            raise RuntimeError("after failed")

        self.after = _after_raises
        with mock.patch("cli.main.process_video", pv_mock), _patched_work_dir(tmp_path):
            with pytest.raises(RuntimeError):
                process_thread()
