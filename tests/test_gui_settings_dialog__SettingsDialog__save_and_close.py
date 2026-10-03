"""Tests for ``gui.settings_dialog.SettingsDialog._save_and_close``.

Specification: docs/00-Architecture/gui_settings_dialog__SettingsDialog__save_and_close.yaml

``_save_and_close`` calls ``_save_settings`` to save the settings and
closes this dialog window (destroy) regardless of the save result:

- Call ``self._save_settings()``
- Call ``self.destroy()`` regardless of the result of ``_save_settings``
  (success or early return)
- End the function without a return value (None)

Mocked / stand-in dependencies (per the test-generation rules): a real
``SettingsDialog()`` window is constructed (retry helper against unstable
Tcl environments, destroyed in finally); ``gui.settings_dialog.data_path``
is patched to a temporary path so the settings file never touches the real
project config; ``gui.settings_dialog.messagebox`` is patched to capture
error dialogs. The real ``_save_settings`` and the real ``destroy`` are
exercised.

``errors`` section of the spec (documented here only, NOT tested, per the
test-generation rules):

- condition: "本関数自体にエラー処理はない。_save_settings または destroy が例外を送出した場合"
  behavior: "例外は捕捉されず呼び出し元（ボタンの command コールバック等）へそのまま伝播する"
"""

import contextlib
import tempfile
import time
from pathlib import Path
from unittest import mock

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
    input: interval_var（または diff_threshold_var、name_mapping_threshold_var）の値が非数値文字列（例 abc）
    expected: _save_settings が「フレーム間隔・差分閾値・編集距離閾値には有効な数値を入力してください。」というエラーを表示し、設定を書き換えず早期 return する。それでも _save_and_close は destroy を呼び出しダイアログを閉じる
    """
    tmp = Path(tempfile.mkdtemp())
    cfg = tmp / "settings.yaml"
    with mock.patch("gui.settings_dialog.data_path", return_value=cfg), \
            mock.patch("gui.settings_dialog.messagebox") as mock_mb:
        with _dialog():
            dlg = SettingsDialog()
            assert dlg.settings_file == cfg
            dlg.interval_var.set("abc")
            ret = dlg._save_and_close()
            assert ret is None
            # ダイアログが閉じられる
            assert dlg.winfo_exists() == 0
            # エラー表示され、設定を書き換えず早期 return
            mock_mb.showerror.assert_called_once()
            args = mock_mb.showerror.call_args[0]
            assert ("フレーム間隔・差分閾値・編集距離閾値には有効な数値を入力してください。"
                    in args)
            assert not cfg.exists()


def test_edge_02():
    """
    input: text_region_var が truthy で、region_left_var（または top・right・bottom いずれか）の値が非数値文字列
    expected: _save_settings が「テキスト領域の座標には有効な数値を入力してください。」というエラーを表示し、設定を書き換えず早期 return する。それでも _save_and_close はダイアログを閉じる
    """
    tmp = Path(tempfile.mkdtemp())
    cfg = tmp / "settings.yaml"
    with mock.patch("gui.settings_dialog.data_path", return_value=cfg), \
            mock.patch("gui.settings_dialog.messagebox") as mock_mb:
        with _dialog():
            dlg = SettingsDialog()
            assert dlg.settings_file == cfg
            dlg.text_region_var.set(True)
            dlg.region_left_var.set("abc")
            ret = dlg._save_and_close()
            assert ret is None
            # ダイアログが閉じられる
            assert dlg.winfo_exists() == 0
            # エラー表示され、設定を書き換えず早期 return
            mock_mb.showerror.assert_called_once()
            args = mock_mb.showerror.call_args[0]
            assert ("テキスト領域の座標には有効な数値を入力してください。" in args)
            assert not cfg.exists()


def test_edge_03():
    """
    input: 全数値入力が有効で、text_region_var が falsy（領域無効）の場合
    expected: _save_settings が可視部分（44〜60行）で領域座標を 0.0 に設定したまま後続の処理（書き込み部分。読込範囲外）へ進み、その後ダイアログが閉じる
    """
    tmp = Path(tempfile.mkdtemp())
    cfg = tmp / "settings.yaml"
    with mock.patch("gui.settings_dialog.data_path", return_value=cfg), \
            mock.patch("gui.settings_dialog.messagebox") as mock_mb:
        with _dialog():
            dlg = SettingsDialog()
            assert dlg.settings_file == cfg
            # デフォルト値: 全数値入力が有効で text_region_var が falsy
            assert dlg.text_region_var.get() is False
            ret = dlg._save_and_close()
            assert ret is None
            # ダイアログが閉じられる
            assert dlg.winfo_exists() == 0
            # エラー表示なし（書き込み部分へ進む）
            mock_mb.showerror.assert_not_called()
            assert cfg.exists()
