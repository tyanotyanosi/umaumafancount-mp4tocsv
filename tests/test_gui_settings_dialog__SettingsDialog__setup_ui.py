"""Tests for ``gui.settings_dialog.SettingsDialog._setup_ui``.

Specification: docs/00-Architecture/gui_settings_dialog__SettingsDialog__setup_ui.yaml

``_setup_ui`` builds the grid layout of the settings dialog window
(CTkToplevel), places the main CTkFrame, and then calls the five
section-generation methods in order with ``main_frame`` as the argument:

- Step 1: set the window's column 0 (grid_columnconfigure) and row 0
  (grid_rowconfigure) weight to 1
- Step 2: create ``main_frame`` (ctk.CTkFrame) with ``self`` as parent and
  place it at row=0, column=0 (sticky="nsew", padx=20, pady=20)
- Step 3: set main_frame's column 0 weight to 1
- Step 4: ``self._create_video_settings(main_frame)``
- Step 5: ``self._create_ocr_settings(main_frame)``
- Step 6: ``self._create_text_region_settings(main_frame)``
- Step 7: ``self._create_name_mapping_settings(main_frame)``
- Step 8: ``self._create_buttons(main_frame)``

The method has no return statement and returns ``None``.

Mocked / stand-in dependencies (per the test-generation rules): a real
``SettingsDialog()`` window is constructed (retry helper against unstable
Tcl environments, destroyed in finally); ``gui.settings_dialog.data_path``
is patched to a temporary path so the settings file never touches the real
project config. ``self.config`` is set per test and the real
``_setup_ui`` is exercised (the five section methods run real).

``errors`` section of the spec (documented here only, NOT tested, per the
test-generation rules):

- condition: 'selfが破棄済みまたはtkinterウィンドウが未初期化'
  behavior: 'grid_columnconfigure / grid_rowconfigure / ctk.CTkFrame生成でエラー（TclError系）が発生し、_setup_uiは捕捉せず伝播する。正確な例外種別は未確認。'
"""

import contextlib
import tempfile
import time
from pathlib import Path
from unittest import mock

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


def test_edge_01():
    """
    input: self.config == {} （設定ファイル不存在またはyaml空の場合）
    expected: _setup_ui()はNoneを返し、返却後 self.interval_var.get() == "1.0" かつ self.diff_var.get() == True かつ self.diff_threshold_var.get() == "0.1" かつ self.diff_only_var.get() == False かつ self.ocr_engine_var.get() == "meiki" となる。
    """
    tmp = Path(tempfile.mkdtemp())
    cfg = tmp / "settings.yaml"  # 存在しない
    with mock.patch("gui.settings_dialog.data_path", return_value=cfg):
        with _dialog():
            dlg = SettingsDialog()
            assert dlg.config == {}
            ret = dlg._setup_ui()
            assert ret is None
            assert dlg.interval_var.get() == "1.0"
            assert dlg.diff_var.get() is True
            assert dlg.diff_threshold_var.get() == "0.1"
            assert dlg.diff_only_var.get() is False
            assert dlg.ocr_engine_var.get() == "meiki"


def test_edge_02():
    """
    input: self.config == {"video": {"frame_interval": 2.0}, "ocr": {"engine": "gemma4"}}
    expected: _setup_ui()はNoneを返し、self.interval_var.get() == "2.0" かつ self.ocr_engine_var.get() == "gemma4" となる。gemma4のメニュー項目は無効化されるが、変数の値はconfigの値をそのまま受取る。
    """
    tmp = Path(tempfile.mkdtemp())
    cfg = tmp / "settings.yaml"  # 存在しない
    with mock.patch("gui.settings_dialog.data_path", return_value=cfg):
        with _dialog():
            dlg = SettingsDialog()
            dlg.config = {"video": {"frame_interval": 2.0},
                          "ocr": {"engine": "gemma4"}}
            ret = dlg._setup_ui()
            assert ret is None
            assert dlg.interval_var.get() == "2.0"
            assert dlg.ocr_engine_var.get() == "gemma4"


def test_edge_03():
    """
    input: self.config == {"video": None}
    expected: _setup_ui()は返らず、_create_video_settings呼び出し中にAttributeError（'NoneType' object has no attribute 'get'）が発生し、捕捉されずに呼び出し側（__init__）へ伝播する。
    """
    tmp = Path(tempfile.mkdtemp())
    cfg = tmp / "settings.yaml"  # 存在しない
    with mock.patch("gui.settings_dialog.data_path", return_value=cfg):
        with _dialog():
            dlg = SettingsDialog()
            dlg.config = {"video": None}
            with pytest.raises(AttributeError):
                dlg._setup_ui()
