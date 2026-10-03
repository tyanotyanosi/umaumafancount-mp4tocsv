"""Tests for ``gui.main_window.MainWindow._create_toolbar``.

Specification: docs/00-Architecture/gui_main_window__MainWindow__create_toolbar.yaml

``MainWindow._create_toolbar`` creates the toolbar on the main window: a
``CTkFrame`` (height=50) placed at row 0, column 0 (sticky=ew, padx=10,
pady=10) whose columns 0..3 all get weight 1, and four ``CTkButton`` widgets
bound to the callbacks ``_open_settings`` / ``_process_video`` /
``_output_json`` / ``_output_csv``:

- '設定'    -> ``self._open_settings``   at row 0, column 0
- '動画処理' -> ``self._process_video``   at row 0, column 1
- 'JSON出力' -> ``self._output_json``     at row 0, column 2
- 'CSV出力'  -> ``self._output_csv``      at row 0, column 3

The toolbar frame is kept only as a local variable (NOT an instance
attribute), and the method returns ``None``.

Mocks: the four callback attributes (``self._open_settings`` /
``self._process_video`` / ``self._output_json`` / ``self._output_csv``) are
unrelated dependencies replaced with ``unittest.mock.Mock`` objects, per the
spec preconditions. The window is a ``MainWindow`` instance whose widget base
is a minimal ``tkinter.Tk()`` root window initialized through the CTk base
(the ``super().__init__()`` required by the spec preconditions; a plain
``tk.Tk`` master is not CTk-compatible). ``MainWindow.__init__`` itself is not
executed, so ``_setup_ui`` and all unrelated widgets are not created.

``errors`` section of the spec (documented here only, NOT tested, per the
test-generation rules):

- condition: 'self に _open_settings / _process_video / _output_json / _output_csv が存在しない'
  behavior: '対応するボタンの生成時、command 引数での属性アクセス（self._open_settings 等）により AttributeError が発生し外へ伝播する'
"""

import time
import tkinter as tk
from unittest.mock import Mock

import customtkinter as ctk

from gui.main_window import MainWindow


def _create_main_window():
    """Build a ``MainWindow`` on a minimal ``tkinter.Tk()`` root window.

    Only the widget base is initialized (``ctk.CTk.__init__`` is the
    ``super().__init__()`` required by the spec preconditions);
    ``MainWindow.__init__`` is not executed, so ``_setup_ui`` and all
    unrelated widgets are skipped. The four callback attributes are mocked
    with ``unittest.mock.Mock``.

    Retries the Tk() creation up to 5 times with 0.25 s intervals to guard
    against unstable Tcl environments (TclError "couldn't read file").
    Raises the original TclError if all attempts fail.
    """
    last_error = None
    for _ in range(5):
        win = object.__new__(MainWindow)
        try:
            ctk.CTk.__init__(win)
            break
        except tk.TclError as exc:
            last_error = exc
            time.sleep(0.25)
    else:
        raise last_error
    win._open_settings = Mock(name="open_settings")
    win._process_video = Mock(name="process_video")
    win._output_json = Mock(name="output_json")
    win._output_csv = Mock(name="output_csv")
    return win


def _toolbar_frame(win):
    """Locate the toolbar ``CTkFrame`` child of ``win``.

    Per the spec the toolbar frame is a local variable (NOT an instance
    attribute), so it is found via the parent window's children.
    """
    frames = [child for child in win.winfo_children() if isinstance(child, ctk.CTkFrame)]
    assert len(frames) == 1
    return frames[0]


def test_edge_01():
    """
    input: メインウィンドウ作成後、_setup_ui からの通常呼び出し
    expected: 例外なし。4 つのインスタンス属性が設定され、ツールバーと 4 ボタンが表示される
    """
    win = _create_main_window()
    try:
        result = win._create_toolbar()
        assert result is None

        # 4 instance attributes are set
        assert isinstance(win.btn_settings, ctk.CTkButton)
        assert isinstance(win.btn_process, ctk.CTkButton)
        assert isinstance(win.btn_output_json, ctk.CTkButton)
        assert isinstance(win.btn_output_csv, ctk.CTkButton)

        # Toolbar frame placed at row 0, column 0 (sticky=ew, padx=10, pady=10)
        frame = _toolbar_frame(win)
        info = frame.grid_info()
        assert info["row"] == 0
        assert info["column"] == 0
        assert info["sticky"] == "ew"
        assert info["padx"] == 10
        assert info["pady"] == 10

        # Toolbar columns 0..3 all have weight 1
        for col in range(4):
            assert frame.grid_columnconfigure(col)["weight"] == 1

        # Buttons: labels, command bindings, grid cells
        for button, label, command, col in (
            (win.btn_settings, "設定", win._open_settings, 0),
            (win.btn_process, "動画処理", win._process_video, 1),
            (win.btn_output_json, "JSON出力", win._output_json, 2),
            (win.btn_output_csv, "CSV出力", win._output_csv, 3),
        ):
            assert button.cget("text") == label
            assert button.cget("command") is command
            binfo = button.grid_info()
            assert binfo["row"] == 0
            assert binfo["column"] == col
            assert binfo["padx"] == 5
            assert binfo["pady"] == 10
            assert button.winfo_exists()
    finally:
        win.destroy()


def test_edge_02():
    """
    input: 2 回目の呼び出し
    expected: 例外なし。同一グリッドセル（row 0, column 0）に新しいフレームと 4 ボタンが追加され、インスタンス属性は新しいボタンに置き換わる
    """
    win = _create_main_window()
    try:
        win._create_toolbar()
        first_buttons = (
            win.btn_settings,
            win.btn_process,
            win.btn_output_json,
            win.btn_output_csv,
        )

        win._create_toolbar()

        new_buttons = (
            win.btn_settings,
            win.btn_process,
            win.btn_output_json,
            win.btn_output_csv,
        )

        # Instance attributes are replaced with the new buttons
        for old, new in zip(first_buttons, new_buttons):
            assert new is not old

        # A new frame is added to the same grid cell (row 0, column 0)
        frames = [child for child in win.winfo_children() if isinstance(child, ctk.CTkFrame)]
        assert len(frames) == 2
        for frame in frames:
            info = frame.grid_info()
            assert info["row"] == 0
            assert info["column"] == 0

        # The new buttons are placed in the new frame at row 0, columns 0..3
        new_frame = frames[1]
        for button, col in zip(new_buttons, range(4)):
            binfo = button.grid_info()
            assert binfo["in"] is new_frame
            assert binfo["row"] == 0
            assert binfo["column"] == col
    finally:
        win.destroy()
