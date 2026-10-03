"""Tests for ``gui.main_window.MainWindow._open_settings``.

Specification: docs/00-Architecture/gui_main_window__MainWindow__open_settings.yaml

Per the spec's ``purpose``, ``_open_settings`` shows the settings dialog
(``SettingsDialog``) modally, and after it is closed, re-reads the settings
file from disk via ``_load_settings`` and updates ``self.settings``:

- creates ``SettingsDialog(self)`` and stores it in a local ``dialog``
- calls ``dialog.wait_window()`` (blocking until the dialog window is closed)
- assigns the result of ``self._load_settings()`` to ``self.settings``
  (``data_path("config/settings.yaml")`` parsed by yaml.safe_load if it
  exists, ``{}`` if the parse result is None or the file is absent)
- returns None

The ``MainWindow`` instance is built per the spec's ``preconditions``
(``__init__``-completed state, ``self.settings`` already set by
``_load_settings``); the constructor and the ``SettingsDialog``
implementation are out of the reading scope (spec ``missing``), so
``SettingsDialog`` and ``_load_settings`` (the disk I/O dependency) are
mocked with ``unittest.mock``. A minimal ``tkinter.Tk()`` root window is
created for each test and destroyed at the end.

``errors`` section of the spec (documented here only, NOT tested, per the
test-generation rules):

- condition: 'data_path("config/settings.yaml") が存在するが有効な YAML ではない'
  behavior: 'yaml.safe_load による yaml.YAMLError 系例外が呼び出し元へ送出される'
- condition: 'data_path("config/settings.yaml") が存在するが開けない（権限など）'
  behavior: 'open() による PermissionError / OSError 系例外が呼び出し元へ送出される'
- condition: 'SettingsDialog のコンストラクタまたは wait_window が例外を送出'
  behavior: '例外が呼び出し元へ送出され、self.settings は更新されない'
"""

import tkinter as tk
import unittest.mock

from gui.main_window import MainWindow


def test_edge_01():
    """
    input: self（MainWindow）。ユーザーがダイアログを閉じた後、設定ファイルが存在し有効な YAML マッピングである
    expected: wait_window 復帰後、self.settings は当該ファイルのパース済みマッピングと等しい
    """
    root = tk.Tk()
    root.withdraw()
    try:
        window = MainWindow.__new__(MainWindow)
        window.settings = {}
        parsed = {"threshold": 0.5, "output_dir": "out"}
        dialog_cls = unittest.mock.Mock(name="SettingsDialog")
        dialog_cls.return_value.wait_window.return_value = None
        with unittest.mock.patch("gui.main_window.SettingsDialog", dialog_cls):
            window._load_settings = unittest.mock.Mock(
                name="_load_settings", return_value=parsed
            )
            result = window._open_settings()
        assert result is None
        dialog_cls.assert_called_once_with(window)
        dialog_cls.return_value.wait_window.assert_called_once()
        assert window.settings == parsed
    finally:
        root.destroy()


def test_edge_02():
    """
    input: self（MainWindow）。設定ファイルが存在しない
    expected: self.settings は {} となる
    """
    root = tk.Tk()
    root.withdraw()
    try:
        window = MainWindow.__new__(MainWindow)
        window.settings = {"stale": True}
        dialog_cls = unittest.mock.Mock(name="SettingsDialog")
        dialog_cls.return_value.wait_window.return_value = None
        with unittest.mock.patch("gui.main_window.SettingsDialog", dialog_cls):
            window._load_settings = unittest.mock.Mock(
                name="_load_settings", return_value={}
            )
            result = window._open_settings()
        assert result is None
        dialog_cls.assert_called_once_with(window)
        dialog_cls.return_value.wait_window.assert_called_once()
        assert window.settings == {}
    finally:
        root.destroy()


def test_edge_03():
    """
    input: self（MainWindow）。設定ファイルが存在するが YAML パース結果が None（空ファイルまたは null）
    expected: self.settings は {} となる（or {} により変換）
    """
    root = tk.Tk()
    root.withdraw()
    try:
        window = MainWindow.__new__(MainWindow)
        window.settings = {"stale": True}
        dialog_cls = unittest.mock.Mock(name="SettingsDialog")
        dialog_cls.return_value.wait_window.return_value = None
        with unittest.mock.patch("gui.main_window.SettingsDialog", dialog_cls):
            window._load_settings = unittest.mock.Mock(
                name="_load_settings", return_value={}
            )
            result = window._open_settings()
        assert result is None
        dialog_cls.assert_called_once_with(window)
        dialog_cls.return_value.wait_window.assert_called_once()
        assert window.settings == {}
    finally:
        root.destroy()
