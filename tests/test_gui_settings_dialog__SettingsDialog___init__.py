"""Tests for ``gui.settings_dialog.SettingsDialog.__init__``.

Specification: docs/00-Architecture/gui_settings_dialog__SettingsDialog___init__.yaml

``__init__`` creates the top-level settings dialog window
(``SettingsDialog``), sets the settings file path, loads the settings
file, and builds the UI:

- Step 1: ``super().__init__(master)`` initializes the CTkToplevel base
  class
- Step 2: window title set to "設定", size 500x900, not resizable
  (``resizable(False, False)``)
- Step 3: ``self.settings_file`` set to ``data_path("config/settings.yaml")``
- Step 4: ``self.config`` set to the return value of
  ``self._load_settings()`` (``{}`` when the file is missing or the
  parse result is falsy; otherwise no dict type is guaranteed)
- Step 5: ``self._setup_ui()`` called to build the UI

``errors`` section of the spec: empty (no error conditions documented).
"""

import contextlib
import tempfile
import time
from pathlib import Path
from unittest import mock

import customtkinter as ctk
import pytest

from gui.settings_dialog import SettingsDialog


@contextlib.contextmanager
def _dialog():
    """Construct a real ``SettingsDialog()`` window, retrying up to 5
    times at 0.25 s intervals against unstable Tcl environments; destroy
    it at the end."""
    last_error = None
    dlg = None
    for attempt in range(5):
        if attempt:
            time.sleep(0.25)
        try:
            dlg = SettingsDialog()
            break
        except Exception as exc:
            last_error = exc
    if dlg is None:
        raise last_error
    try:
        yield dlg
    finally:
        try:
            dlg.destroy()
        except Exception:
            pass


def _assert_window(dlg):
    """Assert the observable window postconditions of ``__init__``."""
    assert isinstance(dlg, ctk.CTkToplevel)
    assert dlg.title() == "設定"
    assert dlg.geometry().startswith("500x900")
    assert dlg.wm_resizable() == (0, 0)


def test_edge_01():
    """
    input: master=None（デフォルト）
    expected: master=None でウィンドウが作成され、None が CTkToplevel.__init__ にそのまま渡される
    """
    recorded = []
    real_init = ctk.CTkToplevel.__init__

    def recording_init(self, master=None, **kwargs):
        recorded.append(master)
        real_init(self, master, **kwargs)

    setup_calls = []

    def recording_setup(self):
        # __init__ の検証のみ行うため、実際の UI 構築は実行しない
        # （self.config が dict 型を保証されない場合、_setup_ui が例外を送出し得る）。
        setup_calls.append(1)

    tmp = Path(tempfile.mkdtemp())
    cfg = tmp / "settings.yaml"  # 存在しない
    with mock.patch.object(ctk.CTkToplevel, "__init__", recording_init), \
            mock.patch.object(SettingsDialog, "_setup_ui", recording_setup), \
            mock.patch("gui.settings_dialog.data_path", return_value=cfg):
        with _dialog() as dlg:
            _assert_window(dlg)
            # None が CTkToplevel.__init__ にそのまま渡される
            assert recorded == [None]
            assert dlg.settings_file == cfg
            assert dlg.config == {}
            assert setup_calls == [1]


def test_edge_02():
    """
    input: 設定ファイルが存在しない
    expected: self.config == {}（ファイルは読まれず、例外は起きない）
    """
    setup_calls = []

    def recording_setup(self):
        # __init__ の検証のみ行うため、実際の UI 構築は実行しない
        # （self.config が dict 型を保証されない場合、_setup_ui が例外を送出し得る）。
        setup_calls.append(1)

    tmp = Path(tempfile.mkdtemp())
    cfg = tmp / "settings.yaml"  # 存在しない
    with mock.patch.object(SettingsDialog, "_setup_ui", recording_setup), \
            mock.patch("gui.settings_dialog.data_path", return_value=cfg):
        with _dialog() as dlg:
            _assert_window(dlg)
            assert dlg.settings_file == cfg
            assert dlg.config == {}
            assert not cfg.exists()
            assert setup_calls == [1]


def test_edge_03():
    """
    input: 設定ファイルが存在し 0 バイト（空ファイル）
    expected: self.config == {}（yaml.safe_load が None を返すため {} に置換される）
    """
    setup_calls = []

    def recording_setup(self):
        # __init__ の検証のみ行うため、実際の UI 構築は実行しない
        # （self.config が dict 型を保証されない場合、_setup_ui が例外を送出し得る）。
        setup_calls.append(1)

    tmp = Path(tempfile.mkdtemp())
    cfg = tmp / "settings.yaml"
    cfg.write_bytes(b"")
    with mock.patch.object(SettingsDialog, "_setup_ui", recording_setup), \
            mock.patch("gui.settings_dialog.data_path", return_value=cfg):
        with _dialog() as dlg:
            _assert_window(dlg)
            assert dlg.settings_file == cfg
            assert dlg.config == {}
            assert setup_calls == [1]


def test_edge_04():
    """
    input: 設定ファイルの内容が truthy な非 mapping（例 hello というスカラー）
    expected: self.config が 'hello' に設定された後、_setup_ui 内の self.config.get(...) で AttributeError が送出され、__init__ は捕捉せず呼び出し側に伝播する
    """
    tmp = Path(tempfile.mkdtemp())
    cfg = tmp / "settings.yaml"
    cfg.write_text("hello", encoding="utf-8")
    with mock.patch("gui.settings_dialog.data_path", return_value=cfg):
        with pytest.raises(AttributeError):
            SettingsDialog()
