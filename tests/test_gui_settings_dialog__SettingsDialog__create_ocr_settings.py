"""Tests for ``gui.settings_dialog.SettingsDialog._create_ocr_settings``.

Specification: docs/00-Architecture/gui_settings_dialog__SettingsDialog__create_ocr_settings.yaml

``_create_ocr_settings`` creates the "OCR設定" group frame under the given
parent widget, with the OCR engine selection menu (choices meiki /
gemma4) and a label informing that gemma4 is not implemented:

- Step 1: create ``ocr_frame`` (CTkFrame) with ``parent`` and place it at
  row=1, column=0 (sticky="ew", pady=(0, 10))
- Step 2: label "OCR設定" (font Arial 14 bold) at (row=0, column=0)
- Step 3: ``ocr_choices = ["meiki", "gemma4"]``
- Step 4: ``self.ocr_engine_var`` (ctk.StringVar, initial value
  ``self.config.get("ocr", {}).get("engine", "meiki")`` — no ``str()``
  conversion)
- Step 5: ``CTkOptionMenu`` with values=ocr_choices,
  variable=self.ocr_engine_var, width=150, placed at (row=1, column=0)
- Step 6: ``self._disable_option_menu_item(ocr_menu, "gemma4")``
- Step 7: label "gemma4 は未実装のため選択できません" (font Arial 10,
  text_color ("gray55", "gray45")) at (row=2, column=0), pady=(0, 5)

The method has no return statement and returns ``None``.

Mocked / stand-in dependencies (per the test-generation rules): the
``SettingsDialog`` instance is created without running ``__init__``
(``object.__new__``); ``self.config`` is a real dict per test;
``self._disable_option_menu_item`` is a mock (its implementation is outside
the read scope; only the call is asserted); ``parent`` is a real
``ctk.CTkFrame`` widget (created with a retry helper against unstable Tcl
environments and destroyed at the end of the test). The created widgets
are real ``ctk`` widgets and are asserted on via the real widget tree.

``errors`` section of the spec (documented here only, NOT tested, per the
test-generation rules):

- condition: 'parentが存在しないまたは使用できないコンテナウィジェット（None・破棄済み等）'
  behavior: 'ctk.CTkFrame(parent)の生成でエラーが発生する。正確な例外種別はcustomtkinter/tkinterの実装依存で未確認であり、本メソッドは捕捉せず伝播する。'
- condition: 'self._disable_option_menu_item内部で例外が発生する（実装は読取範囲外）'
  behavior: '本メソッドにtry/exceptはないため、例外は捕捉されず呼び出し側へそのまま伝播する。'
"""

import contextlib
import time
from unittest import mock

import customtkinter as ctk

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


def _make_self(config):
    """Build a ``SettingsDialog`` instance (without running ``__init__``)
    with ``config`` and a mocked ``_disable_option_menu_item``."""
    self = object.__new__(SettingsDialog)
    self.config = config
    self._disable_option_menu_item = mock.MagicMock(
        name="_disable_option_menu_item")
    return self


def test_edge_01():
    """
    input: self.config == {}
    expected: メソッド返却後、self.ocr_engine_var.get() == "meiki" であり、ウィジェット4個が生成され、_disable_option_menu_item(ocr_menu, "gemma4")が1回呼び出される。
    """
    with _parent_frame() as parent:
        self = _make_self({})
        ret = self._create_ocr_settings(parent)
        assert ret is None
        assert isinstance(self.ocr_engine_var, ctk.StringVar)
        assert self.ocr_engine_var.get() == "meiki"
        self._disable_option_menu_item.assert_called_once()
        assert self._disable_option_menu_item.call_args[0][1] == "gemma4"
        # ウィジェット4個: ocr_frame (row=1, column=0) + 子3個
        slaves = parent.grid_slaves()
        assert any(
            (slave.grid_info()["row"], slave.grid_info()["column"]) == (1, 0)
            for slave in slaves
        )
        ocr_frame = parent.winfo_children()[-1]
        assert isinstance(ocr_frame, ctk.CTkFrame)
        assert len(ocr_frame.winfo_children()) == 3
        menus = [c for c in ocr_frame.winfo_children()
                 if isinstance(c, ctk.CTkOptionMenu)]
        assert len(menus) == 1
        assert menus[0]._values == ["meiki", "gemma4"]


def test_edge_02():
    """
    input: self.config == {"ocr": {}}
    expected: 空configの場合と同一。self.ocr_engine_var.get() == "meiki" となる。
    """
    with _parent_frame() as parent:
        self = _make_self({"ocr": {}})
        ret = self._create_ocr_settings(parent)
        assert ret is None
        assert self.ocr_engine_var.get() == "meiki"
        self._disable_option_menu_item.assert_called_once()
        assert self._disable_option_menu_item.call_args[0][1] == "gemma4"


def test_edge_03():
    """
    input: self.config == {"ocr": {"engine": "gemma4"}}
    expected: self.ocr_engine_var.get() == "gemma4" となる。gemma4項目はメニュー上で無効化されるが、変数の値はconfigの値をそのまま受取る。
    """
    with _parent_frame() as parent:
        self = _make_self({"ocr": {"engine": "gemma4"}})
        ret = self._create_ocr_settings(parent)
        assert ret is None
        assert self.ocr_engine_var.get() == "gemma4"
        self._disable_option_menu_item.assert_called_once()
        assert self._disable_option_menu_item.call_args[0][1] == "gemma4"


def test_edge_04():
    """
    input: self.config == {"ocr": {"engine": "other"}}
    expected: self.ocr_engine_var.get() == "other" となる。本メソッドはocr_choicesとの突合検査を行わず、値を変数にそのまま渡す。
    """
    with _parent_frame() as parent:
        self = _make_self({"ocr": {"engine": "other"}})
        ret = self._create_ocr_settings(parent)
        assert ret is None
        assert self.ocr_engine_var.get() == "other"
        # ocr_choices との突合検査は行われない
        ocr_frame = parent.winfo_children()[-1]
        menus = [c for c in ocr_frame.winfo_children()
                 if isinstance(c, ctk.CTkOptionMenu)]
        assert len(menus) == 1
        assert menus[0]._values == ["meiki", "gemma4"]
        self._disable_option_menu_item.assert_called_once()


def test_edge_05():
    """
    input: self.config == {"ocr": None}
    expected: メソッドは返らず、手順4のNoneに対する.get("engine", "meiki")呼び出しでAttributeError（'NoneType' object has no attribute 'get'）が発生し、捕捉されずに呼び出し側へ伝播する。
    """
    with _parent_frame() as parent:
        self = _make_self({"ocr": None})
        before = len(parent.winfo_children())
        try:
            self._create_ocr_settings(parent)
            raised = False
        except AttributeError:
            raised = True
        assert raised
        # 発生前に ocr_frame とタイトルラベルが既に生成済み
        children = parent.winfo_children()
        assert len(children) == before + 1
        ocr_frame = children[-1]
        assert isinstance(ocr_frame, ctk.CTkFrame)
        sub = ocr_frame.winfo_children()
        assert len(sub) == 1
        assert isinstance(sub[0], ctk.CTkLabel)
        assert not hasattr(self, "ocr_engine_var")
        self._disable_option_menu_item.assert_not_called()
