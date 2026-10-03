"""Tests for ``gui.settings_dialog.SettingsDialog._create_name_mapping_settings``.

Specification: docs/00-Architecture/gui_settings_dialog__SettingsDialog__create_name_mapping_settings.yaml

``_create_name_mapping_settings`` creates the name-mapping settings UI
inside the settings dialog (definition-file entry + select button,
enable/warn checkboxes, approximate-match threshold entry,
unmapped-action dropdown) and registers the bound variables on ``self``:

- A ``CTkFrame`` is created under ``parent`` and placed at grid
  row=3, column=0 (sticky=ew, pady=(0,10)); a bold title label
  (text=名前マッピング設定, columnspan=3) at row=0
- The ``name_mapping`` sub-dict is taken from ``self.config`` (empty dict
  when the key is missing)
- row=1: label (text=定義ファイル:), ``CTkEntry`` (width=250) bound to
  ``self.name_mapping_file_var`` at column=1, "選択..." button with
  ``command=self._select_mapping_file`` at column=2
- row=2 (columnspan=3): checkbox bound to ``self.name_mapping_enable_var``
  (text=名前マッピング有効)
- row=3: label (text=近似一致閾値:) and ``CTkEntry`` (width=100) bound to
  ``self.name_mapping_threshold_var`` at column=1
- row=4 (columnspan=3): checkbox bound to ``self.name_mapping_warn_var``
  (text=近似一致時に警告)
- row=5: label (text=未マッピング扱い:) and ``CTkOptionMenu`` with
  values=['suggest', 'keep', 'drop], width=120 bound to
  ``self.name_mapping_action_var`` at column=1

The bound variables take their values from the ``name_mapping`` sub-dict
with defaults: file 'config/name_mapping.json' (str), enable False (bool),
edit_distance_threshold 2 (str), warn_on_approx True (bool),
unmapped_action 'suggest' (str). The function returns ``None``.

Mocked / stand-in dependencies (per the test-generation rules): the
``SettingsDialog`` instance is created without running ``__init__``
(``object.__new__``); ``self.config`` is a real dict per test;
``self._select_mapping_file`` is a plain function (only referenced as the
button command, never invoked); ``parent`` is a real ``ctk.CTkFrame``
widget (created with a retry helper against unstable Tcl environments and
destroyed at the end of the test). The created widgets are real
``ctk`` widgets and are asserted on via the real widget tree.

``errors`` section of the spec (documented here only, NOT tested, per the
test-generation rules):

- condition: "self.config['name_mapping'] が dict 以外の値（str, list 等）"
  behavior: "最初の nm.get(...) 呼び出しで AttributeError が送出され、呼び出し側に伝播する"
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
    with ``config`` and ``_select_mapping_file`` set per test."""
    self = object.__new__(SettingsDialog)
    self.config = config
    self._select_mapping_file = lambda: None
    return self


def test_edge_01():
    """
    input: self.config = {}（name_mapping キー無し）、parent は有効な master
    expected: self.name_mapping_file_var が 'config/name_mapping.json'、self.name_mapping_enable_var.get() が False、self.name_mapping_threshold_var が '2'、self.name_mapping_warn_var.get() が True、self.name_mapping_action_var が 'suggest'。UI は通常通り生成され、戻り値 None
    """
    with _parent_frame() as parent:
        self = _make_self({})
        ret = self._create_name_mapping_settings(parent)
        assert ret is None
        assert self.name_mapping_file_var.get() == "config/name_mapping.json"
        assert self.name_mapping_enable_var.get() is False
        assert self.name_mapping_threshold_var.get() == "2"
        assert self.name_mapping_warn_var.get() is True
        assert self.name_mapping_action_var.get() == "suggest"
        # CTkFrame が parent の grid row=3, column=0 に配置され、
        # 子ウィジェット 10 個が生成される
        slaves = parent.grid_slaves()
        assert any(
            (slave.grid_info()["row"], slave.grid_info()["column"]) == (3, 0)
            for slave in slaves
        )
        new_frame = parent.winfo_children()[-1]
        assert isinstance(new_frame, ctk.CTkFrame)
        assert len(new_frame.winfo_children()) == 10


def test_edge_02():
    """
    input: self.config = {"name_mapping": None}
    expected: name_mapping キーが None の値で存在するため、nm.get('file', 'config/name_mapping.json') の呼び出しで AttributeError（NoneType に get 属性が無い）が発生する。発生前に CTkFrame とタイトルラベルは既に生成済み
    """
    with _parent_frame() as parent:
        self = _make_self({"name_mapping": None})
        before = len(parent.winfo_children())
        try:
            self._create_name_mapping_settings(parent)
            raised = False
        except AttributeError:
            raised = True
        assert raised
        children = parent.winfo_children()
        assert len(children) == before + 1
        new_frame = children[-1]
        assert isinstance(new_frame, ctk.CTkFrame)
        # 発生前に CTkFrame とタイトルラベル、「定義ファイル:」ラベルが既に生成済み
        # （nm.get("file", ...) は「定義ファイル:」ラベルの生成の後に呼ばれる）
        sub = new_frame.winfo_children()
        assert len(sub) == 2
        assert isinstance(sub[0], ctk.CTkLabel)
        assert isinstance(sub[1], ctk.CTkLabel)


def test_edge_03():
    """
    input: self.config = {"name_mapping": {"enable": 1, "warn_on_approx": 0, "edit_distance_threshold": 3, "file": 5, "unmapped_action": "drop"}}
    expected: self.name_mapping_enable_var.get() が True（bool(1)）、self.name_mapping_warn_var.get() が False（bool(0)）、self.name_mapping_threshold_var が '3'、self.name_mapping_file_var が '5'（str 変換）、self.name_mapping_action_var が 'drop'
    """
    with _parent_frame() as parent:
        self = _make_self({
            "name_mapping": {
                "enable": 1,
                "warn_on_approx": 0,
                "edit_distance_threshold": 3,
                "file": 5,
                "unmapped_action": "drop",
            }
        })
        ret = self._create_name_mapping_settings(parent)
        assert ret is None
        assert self.name_mapping_enable_var.get() is True
        assert self.name_mapping_warn_var.get() is False
        assert self.name_mapping_threshold_var.get() == "3"
        assert self.name_mapping_file_var.get() == "5"
        assert self.name_mapping_action_var.get() == "drop"


def test_edge_04():
    """
    input: self.config = {"name_mapping": {"unmapped_action": "other"}}
    expected: self.name_mapping_action_var が 'other' になる。CTkOptionMenu の values は ['suggest', 'keep', 'drop'] のままで、変数の値が values に含まれない状態になる（'other' が表示されるかは CTkOptionMenu の実装が見えるコードの外のため未確認）
    """
    with _parent_frame() as parent:
        self = _make_self({"name_mapping": {"unmapped_action": "other"}})
        ret = self._create_name_mapping_settings(parent)
        assert ret is None
        assert self.name_mapping_action_var.get() == "other"
        # CTkOptionMenu の values は ['suggest', 'keep', 'drop'] のまま
        new_frame = parent.winfo_children()[-1]
        menus = [c for c in new_frame.winfo_children()
                 if isinstance(c, ctk.CTkOptionMenu)]
        assert len(menus) == 1
        assert menus[0]._values == ["suggest", "keep", "drop"]


def test_edge_05():
    """
    input: self.config = {"name_mapping": {"enable": True}}（enable のみ存在、他キー欠落）
    expected: self.name_mapping_enable_var.get() が True、self.name_mapping_warn_var.get() が True（デフォルト）、self.name_mapping_threshold_var が '2'（デフォルト）、self.name_mapping_file_var が 'config/name_mapping.json'（デフォルト）、self.name_mapping_action_var が 'suggest'（デフォルト）
    """
    with _parent_frame() as parent:
        self = _make_self({"name_mapping": {"enable": True}})
        ret = self._create_name_mapping_settings(parent)
        assert ret is None
        assert self.name_mapping_enable_var.get() is True
        assert self.name_mapping_warn_var.get() is True
        assert self.name_mapping_threshold_var.get() == "2"
        assert self.name_mapping_file_var.get() == "config/name_mapping.json"
        assert self.name_mapping_action_var.get() == "suggest"
