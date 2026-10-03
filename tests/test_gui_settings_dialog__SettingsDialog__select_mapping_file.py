"""Tests for ``gui.settings_dialog.SettingsDialog._select_mapping_file``.

Specification: docs/00-Architecture/gui_settings_dialog__SettingsDialog__select_mapping_file.yaml

``_select_mapping_file`` opens the file-selection dialog for the
name-mapping definition file and, if the user selects a file, stores its
path in ``name_mapping_file_var``:

- Call ``tkinter.filedialog.askopenfilename`` with
  title="名前マッピング定義ファイルを選択" and
  filetypes=[("JSONファイル", "*.json"), ("全ファイル", "*.*")]
- If the return value (path) is truthy, pass that value as-is to
  ``self.name_mapping_file_var.set``
- If the return value is falsy, return without doing anything

The function has no return value and returns ``None``.

Mocked / stand-in dependencies (per the test-generation rules): the
``SettingsDialog`` instance is created without running ``__init__``
(``object.__new__``); ``self.name_mapping_file_var`` is a real
``ctk.StringVar`` on a real Tk root (retry helper against unstable Tcl
environments, destroyed at the end of the test);
``gui.settings_dialog.filedialog`` is mocked so no real dialog is shown.
The function under test itself runs real.

``errors`` section of the spec (documented here only, NOT tested, per the
test-generation rules):

- condition: "コードに明示的なエラー分岐はない。askopenfilename が例外を送出した場合（表示環境不可用の TclError など）"
  behavior: "例外は捕捉されず呼び出し元へそのまま伝播する"
"""

import contextlib
import time
from unittest import mock

import customtkinter as ctk
import tkinter as tk

from gui.settings_dialog import SettingsDialog


@contextlib.contextmanager
def _root_window():
    """Create a real Tk root window (withdrawn), retrying up to 5 times
    at 0.25 s intervals against unstable Tcl environments; destroy it at
    the end."""
    last_error = None
    root = None
    for attempt in range(5):
        if attempt:
            time.sleep(0.25)
        try:
            root = tk.Tk()
            root.withdraw()
            break
        except Exception as exc:
            last_error = exc
    if root is None:
        raise last_error
    try:
        yield root
    finally:
        try:
            root.destroy()
        except Exception:
            pass


def _make_self(root):
    """Build a ``SettingsDialog`` instance (without running ``__init__``)
    with a real ``name_mapping_file_var``."""
    self = object.__new__(SettingsDialog)
    self.name_mapping_file_var = ctk.StringVar(root,
                                               value="config/name_mapping.json")
    return self


def test_edge_01():
    """
    input: ユーザーが選択ダイアログをキャンセルした（askopenfilename の戻り値が空）
    expected: name_mapping_file_var の値は変更されず、関数は None を返す
    """
    with _root_window() as root:
        self = _make_self(root)
        with mock.patch("gui.settings_dialog.filedialog") as mock_fd:
            mock_fd.askopenfilename.return_value = ""
            ret = self._select_mapping_file()
            assert ret is None
            # 値は変更されない
            assert self.name_mapping_file_var.get() == "config/name_mapping.json"
            # ダイアログ呼び出しの確認
            mock_fd.askopenfilename.assert_called_once()
            args, kwargs = mock_fd.askopenfilename.call_args
            assert ("名前マッピング定義ファイルを選択" in args
                    or "名前マッピング定義ファイルを選択" in kwargs.values())
            assert ([("JSONファイル", "*.json"), ("全ファイル", "*.*")] in args
                    or [("JSONファイル", "*.json"), ("全ファイル", "*.*")]
                    in kwargs.values())


def test_edge_02():
    """
    input: ユーザーがファイル C:/data/mapping.json を選択した（askopenfilename の戻り値がそのパス）
    expected: name_mapping_file_var の値が選択されたパスに等しくなる
    """
    with _root_window() as root:
        self = _make_self(root)
        with mock.patch("gui.settings_dialog.filedialog") as mock_fd:
            mock_fd.askopenfilename.return_value = "C:/data/mapping.json"
            ret = self._select_mapping_file()
            assert ret is None
            assert self.name_mapping_file_var.get() == "C:/data/mapping.json"
