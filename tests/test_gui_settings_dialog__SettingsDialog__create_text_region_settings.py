"""Tests for ``gui.settings_dialog.SettingsDialog._create_text_region_settings``.

Specification: docs/00-Architecture/gui_settings_dialog__SettingsDialog__create_text_region_settings.yaml

``_create_text_region_settings`` creates the text-region settings UI
inside the settings dialog (enable checkbox, unit radio buttons, 4
coordinate entries) and registers the bound variables on ``self``:

- Step 1: a ``CTkFrame`` under ``parent`` at grid row=2, column=0
  (sticky=ew, pady=(0,10)); a bold title label (text=文字領域設定,
  font=Arial 14 bold) at row=0
- Step 2: the ``text_region`` sub-dict (empty dict when the key is
  missing); ``self.text_region_var`` (BooleanVar) from 'enabled'
  (default False); a checkbox bound to it (text=文字領域を指定) at row=1
- Step 3: ``self.region_unit_var`` (StringVar) set to 'percent' if
  'use_percent' (default False) is truthy, else 'pixel'; two radio
  buttons 'パーセント指定 (%)' (value=percent) at row=2 and
  'ピクセル指定 (px)' (value=pixel) at row=3 bound to the same variable
- Step 4: 'left', 'top', 'right', 'bottom' (defaults 22.0, 5.0, 58.0,
  92.0) read and stored as ``str()`` values in
  ``self.region_left_var`` / ``self.region_top_var`` /
  ``self.region_right_var`` / ``self.region_bottom_var`` (each StringVar)
- Step 5: 4 labels (左/上/右/下) and 4 ``CTkEntry`` (width=60) bound to
  the StringVars, placed at rows 4/5 columns 0-3

The function returns ``None``.

Mocked / stand-in dependencies (per the test-generation rules): the
``SettingsDialog`` instance is created without running ``__init__``
(``object.__new__``); ``self.config`` is a real dict per test; ``parent``
is a real ``ctk.CTkFrame`` widget (created with a retry helper against
unstable Tcl environments and destroyed at the end of the test). The
created widgets are real ``ctk`` widgets and are asserted on via the real
widget tree.

``errors`` section of the spec (documented here only, NOT tested, per the
test-generation rules):

- condition: "self.config['text_region'] が dict 以外の値（str, list 等）"
  behavior: "最初の .get(...) 呼び出しで AttributeError が送出され、呼び出し側に伝播する"
"""

import contextlib
import time

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
    with ``config`` set per test."""
    self = object.__new__(SettingsDialog)
    self.config = config
    return self


def test_edge_01():
    """
    input: self.config = {}（text_region キー無し）、parent は有効な master
    expected: text_region_var.get() は False、region_unit_var.get() は 'pixel'、座標エントリ変数は '22.0', '5.0', '58.0', '92.0'。UI は通常通り生成され、戻り値 None
    """
    with _parent_frame() as parent:
        self = _make_self({})
        ret = self._create_text_region_settings(parent)
        assert ret is None
        assert self.text_region_var.get() is False
        assert self.region_unit_var.get() == "pixel"
        assert self.region_left_var.get() == "22.0"
        assert self.region_top_var.get() == "5.0"
        assert self.region_right_var.get() == "58.0"
        assert self.region_bottom_var.get() == "92.0"
        # CTkFrame が parent の grid row=2, column=0 に配置され、
        # 子ウィジェット 12 個が生成される
        slaves = parent.grid_slaves()
        assert any(
            (slave.grid_info()["row"], slave.grid_info()["column"]) == (2, 0)
            for slave in slaves
        )
        new_frame = parent.winfo_children()[-1]
        assert isinstance(new_frame, ctk.CTkFrame)
        assert len(new_frame.winfo_children()) == 12


def test_edge_02():
    """
    input: self.config = {"text_region": None}
    expected: text_region キーが None の値で存在するため、最初の .get('enabled', False) 呼び出しで AttributeError（NoneType に get 属性が無い）が発生する。発生前に CTkFrame とタイトルラベルは既に生成済み
    """
    with _parent_frame() as parent:
        self = _make_self({"text_region": None})
        before = len(parent.winfo_children())
        try:
            self._create_text_region_settings(parent)
            raised = False
        except AttributeError:
            raised = True
        assert raised
        children = parent.winfo_children()
        assert len(children) == before + 1
        new_frame = children[-1]
        assert isinstance(new_frame, ctk.CTkFrame)
        # 発生前に CTkFrame とタイトルラベルが既に生成済み
        sub = new_frame.winfo_children()
        assert len(sub) == 1
        assert isinstance(sub[0], ctk.CTkLabel)
        assert not hasattr(self, "text_region_var")
        assert not hasattr(self, "region_unit_var")


def test_edge_03():
    """
    input: self.config = {"text_region": {"enabled": 1, "use_percent": 1, "left": 10, "top": 2, "right": 60, "bottom": 90}}
    expected: text_region_var.get() は True（真値の非 bool は BooleanVar が変換）、region_unit_var.get() は 'percent'、座標エントリ変数は '10', '2', '60', '90'
    """
    with _parent_frame() as parent:
        self = _make_self({
            "text_region": {
                "enabled": 1,
                "use_percent": 1,
                "left": 10,
                "top": 2,
                "right": 60,
                "bottom": 90,
            }
        })
        ret = self._create_text_region_settings(parent)
        assert ret is None
        assert self.text_region_var.get() is True
        assert self.region_unit_var.get() == "percent"
        assert self.region_left_var.get() == "10"
        assert self.region_top_var.get() == "2"
        assert self.region_right_var.get() == "60"
        assert self.region_bottom_var.get() == "90"


def test_edge_04():
    """
    input: self.config = {"text_region": "x"}（dict 以外）
    expected: 最初の .get 呼び出しで AttributeError（'str' object has no attribute 'get'）が発生する。発生前に CTkFrame とタイトルラベルは既に生成済み
    """
    with _parent_frame() as parent:
        self = _make_self({"text_region": "x"})
        before = len(parent.winfo_children())
        try:
            self._create_text_region_settings(parent)
            raised = False
        except AttributeError:
            raised = True
        assert raised
        children = parent.winfo_children()
        assert len(children) == before + 1
        new_frame = children[-1]
        assert isinstance(new_frame, ctk.CTkFrame)
        # 発生前に CTkFrame とタイトルラベルが既に生成済み
        sub = new_frame.winfo_children()
        assert len(sub) == 1
        assert isinstance(sub[0], ctk.CTkLabel)
        assert not hasattr(self, "text_region_var")
        assert not hasattr(self, "region_unit_var")


def test_edge_05():
    """
    input: self.config = {"text_region": {"enabled": False}}（enabled のみ存在、他キー欠落）
    expected: text_region_var.get() は False、region_unit_var.get() は 'pixel'、座標エントリ変数はデフォルトの '22.0', '5.0', '58.0', '92.0'
    """
    with _parent_frame() as parent:
        self = _make_self({"text_region": {"enabled": False}})
        ret = self._create_text_region_settings(parent)
        assert ret is None
        assert self.text_region_var.get() is False
        assert self.region_unit_var.get() == "pixel"
        assert self.region_left_var.get() == "22.0"
        assert self.region_top_var.get() == "5.0"
        assert self.region_right_var.get() == "58.0"
        assert self.region_bottom_var.get() == "92.0"
