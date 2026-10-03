"""Tests for ``gui.main_window.MainWindow._setup_ui``.

Specification: docs/00-Architecture/gui_main_window__MainWindow__setup_ui.yaml

``_setup_ui`` builds the main window UI: it sets the grid weights
(``self.grid_columnconfigure(0, weight=1)`` and ``self.grid_rowconfigure(2,
weight=1)``), creates the toolbar / main content / status bar via
``_create_toolbar()`` / ``_create_main_content()`` / ``_create_status_bar()``,
and registers the close handler via
``self.protocol("WM_DELETE_WINDOW", self._on_close)``. It returns ``None``.

Mocked / stand-in dependencies (per the test-generation rules): a real
``MainWindow`` instance is constructed (``__init__`` runs for real, so the
documented "call from ``__init__``" is exercised on a real CTk window backed
by a real Tcl toplevel, created with a retry helper against unstable Tcl
environments and destroyed at the end of the test). Only the file/GUI-builder
dependencies are mocked: ``_load_settings`` (settings-file read) returns ``{}``,
and ``_create_toolbar`` / ``_create_main_content`` / ``_create_status_bar`` /
``_on_close`` are class-level mocks so widget creation is observed without
building real sub-widgets. ``_setup_ui`` itself, and the ``grid_*`` /
``protocol`` Tk methods it calls, run real.

``errors`` section of the spec (documented here only, NOT tested, per the
test-generation rules):

- condition: _create_toolbar / _create_main_content / _create_status_bar のいずれかが例外を投げる
  behavior: 例外が外へ伝播する。プロトコル登録は最終ステップのため、登録前に例外が起きるとクローズハンドラは登録されない
"""

import time
from unittest.mock import patch

import tkinter as tk

from gui.main_window import MainWindow


def _grid_config_value(cfg, option):
    """Extract ``option``'s value from a ``grid_*configure`` query result,
    whether it comes back as a dict or as an alternating option/value
    sequence."""
    if isinstance(cfg, dict):
        return cfg.get(option)
    seq = list(cfg)
    for i, item in enumerate(seq):
        if item == option and i + 1 < len(seq):
            return seq[i + 1]
    return None


def _build_window():
    """Construct a real ``MainWindow`` (``__init__`` runs for real) with only
    the file/GUI-builder dependencies mocked; retry up to 5 times at 0.25 s
    intervals against unstable Tcl environments; re-raise the original
    ``TclError`` if every attempt fails."""
    last_error = None
    for attempt in range(5):
        if attempt:
            time.sleep(0.25)
        try:
            with patch.object(MainWindow, "_load_settings",
                             return_value={}), \
                 patch.object(MainWindow, "_create_toolbar") as toolbar_mock, \
                 patch.object(MainWindow, "_create_main_content") as content_mock, \
                 patch.object(MainWindow, "_create_status_bar") as statusbar_mock, \
                 patch.object(MainWindow, "_on_close") as onclose_mock:
                win = MainWindow()
            return win, toolbar_mock, content_mock, statusbar_mock, onclose_mock
        except tk.TclError as exc:
            last_error = exc
    raise last_error


def test_edge_01():
    """
    input: メインウィンドウ作成後、__init__ からの通常呼び出し
    expected: 例外なし。weight が設定され、3 つのサブ UI が作成され、クローズハンドラが登録される
    """
    win, toolbar_mock, content_mock, statusbar_mock, onclose_mock = _build_window()
    try:
        # postcondition: 列 0 の weight が 1、行 2 の weight が 1
        assert _grid_config_value(win.grid_columnconfigure(0), "weight") == 1
        assert _grid_config_value(win.grid_rowconfigure(2), "weight") == 1
        # postcondition: 3 つのサブ UI が各 1 回呼び出される
        toolbar_mock.assert_called_once()
        content_mock.assert_called_once()
        statusbar_mock.assert_called_once()
        # postcondition: WM_DELETE_WINDOW のハンドラとして _on_close が登録される
        handler = win.protocol("WM_DELETE_WINDOW")
        assert handler
        # postcondition: 戻り値は None
        ret = win._setup_ui()
        assert ret is None
    finally:
        try:
            win.destroy()
        except tk.TclError:
            pass
