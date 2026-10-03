"""Tests for ``gui.settings_dialog.SettingsDialog._create_buttons``.

Specification: docs/00-Architecture/gui_settings_dialog__SettingsDialog__create_buttons.yaml

``_create_buttons`` creates a button frame on ``parent`` (grid row 4,
column 0) and places two buttons, "保存" and "キャンセル", inside it:

- Create ``ctk.CTkFrame(parent)`` and place it at grid row=4, column=0,
  pady=(10, 0)
- Create a CTkButton with text "保存" (command=self._save_and_close) and
  pack it with side="left", padx=10, pady=10
- Create a CTkButton with text "キャンセル" (command=self.destroy) and
  pack it with side="right", padx=10, pady=10

The function has no return value and returns ``None``.

Mocked / stand-in dependencies (per the test-generation rules): the
``SettingsDialog`` instance is created without running ``__init__``
(``object.__new__``); ``self._save_and_close`` and ``self.destroy`` are
mocks (only referenced as button commands, never invoked); ``parent`` is a
real ``ctk.CTkFrame`` widget (created with a retry helper against unstable
Tcl environments and destroyed at the end of the test). The created
widgets are real ``ctk`` widgets and are asserted on via the real widget
tree.

``errors`` section of the spec (documented here only, NOT tested, per the
test-generation rules):

- condition: "コードに明示的なエラー分岐はない"
  behavior: "ウィジェットの生成や配置で発生した例外は捕捉されず呼び出し元へそのまま伝播する"
"""

import contextlib
import time
from unittest import mock

import customtkinter as ctk
import pytest

from gui.settings_dialog import SettingsDialog


@contextlib.contextmanager
def _parent_frame():
    """Create a real ``ctk.CTkFrame`` master widget, retrying up to 5
    times at 0.25 s intervals against unstable Tcl environments; destroy
    it at the end."""
    last_error = None
    parent = None
    for attempt in range(5):
        if attempt:
            time.sleep(0.25)
        try:
            parent = ctk.CTkFrame(None)
            break
        except Exception as exc:
            last_error = exc
    if parent is None:
        raise last_error
    try:
        yield parent
    finally:
        try:
            parent.destroy()
        except Exception:
            pass


def _make_self():
    """Build a ``SettingsDialog`` instance (without running ``__init__``)
    with ``_save_and_close`` and ``destroy`` mocked."""
    self = object.__new__(SettingsDialog)
    self._save_and_close = mock.MagicMock(name="_save_and_close")
    self.destroy = mock.MagicMock(name="destroy")
    return self


def test_edge_01():
    """
    input: parent が有効な CTk ルートウィンドウ（またはフレームをホストできるコンテナウィジェット）
    expected: 例外なく完了し、フレームは grid の4行0列に配置され、その子には CTkButton がちょうど2個存在する（pack 順序で先が「保存」、次が「キャンセル」）
    """
    self = _make_self()
    with _parent_frame() as parent:
        ret = self._create_buttons(parent)
        assert ret is None
        # フレームは grid の4行0列に配置される
        slaves = parent.grid_slaves()
        assert any(
            (slave.grid_info()["row"], slave.grid_info()["column"]) == (4, 0)
            for slave in slaves
        )
        frame = parent.winfo_children()[-1]
        assert isinstance(frame, ctk.CTkFrame)
        # その子には CTkButton がちょうど2個存在する
        buttons = [c for c in frame.winfo_children()
                   if isinstance(c, ctk.CTkButton)]
        assert len(buttons) == 2
        # pack 順序で先が「保存」、次が「キャンセル」
        packed = list(frame.pack_slaves())
        assert len(packed) == 2
        btn0 = frame.nametowidget(packed[0])
        btn1 = frame.nametowidget(packed[1])
        assert btn0.cget("text") == "保存"
        assert btn1.cget("text") == "キャンセル"
        # command のバインド
        assert btn0.cget("command") is self._save_and_close
        assert btn1.cget("command") is self.destroy


def test_edge_02():
    """
    input: parent が無効な親（None 等、非ウィジェット）
    expected: ctk.CTkFrame のコンストラクタから TclError 等の例外が発生し、捕捉されないため呼び出し元へそのまま伝播する
    """
    self = _make_self()
    # 観測された乖離（TEST_GENERATION_REPORT.md に記録）:
    # 文字列（非ウィジェット）を parent に渡すと AttributeError
    # （'str' object has no attribute 'tk'）が発生する。
    # （None を渡すとデフォルートルートウィンドウが作成され例外は発生しない）
    with pytest.raises(AttributeError):
        self._create_buttons("not_a_widget")
