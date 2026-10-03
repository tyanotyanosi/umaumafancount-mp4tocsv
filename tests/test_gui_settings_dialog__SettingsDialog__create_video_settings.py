"""Tests for ``gui.settings_dialog.SettingsDialog._create_video_settings``.

Specification: docs/00-Architecture/gui_settings_dialog__SettingsDialog__create_video_settings.yaml

``_create_video_settings`` creates the "動画設定" group frame under the
given parent widget, with 4 items bound to values from ``self.config``:
extraction-interval entry, diff-check enable checkbox, diff-threshold
entry, and diff-only checkbox:

- Step 1: create ``video_frame`` (CTkFrame) with ``parent`` and place it
  at row=0, column=0 (sticky="ew", pady=(0, 10))
- Step 2: label "動画設定" (font Arial 14 bold) at (row=0, column=0) and
  label "抽出間隔（秒）:" at (row=1, column=0)
- Step 3: ``self.interval_var`` (ctk.StringVar, initial value
  ``str(self.config.get("video", {}).get("frame_interval", 1.0))``) and
  a ``CTkEntry`` (width=100, bound to it) at (row=1, column=1)
- Step 4: ``self.diff_var`` (ctk.BooleanVar, initial value
  ``self.config.get("video", {}).get("enable_diff_check", True)``) and a
  checkbox "差分判定有効" (bound) at (row=2, column=0)
- Step 5: label "差分閾値:" at (row=3, column=0)
- Step 6: ``self.diff_threshold_var`` (ctk.StringVar, initial value
  ``str(self.config.get("video", {}).get("diff_threshold", 0.1))``) and a
  ``CTkEntry`` (width=100, bound) at (row=3, column=1)
- Step 7: ``self.diff_only_var`` (ctk.BooleanVar, initial value
  ``self.config.get("video", {}).get("diff_only", False)``) and a
  checkbox "差分判定オンリー（間隔サンプリングなし）" (bound) at
  (row=4, column=0)

The method has no return statement and returns ``None``.

Mocked / stand-in dependencies (per the test-generation rules): the
``SettingsDialog`` instance is created without running ``__init__``
(``object.__new__``); ``self.config`` is a real dict per test; ``parent``
is a real ``ctk.CTkFrame`` widget (created with a retry helper against
unstable Tcl environments and destroyed at the end of the test). The
created widgets are real ``ctk`` widgets and are asserted on via the real
widget tree.

``errors`` section of the spec (documented here only, NOT tested, per the
test-generation rules):

- condition: 'parentが存在しないまたは使用できないコンテナウィジェット（None・破棄済み等）'
  behavior: 'ctk.CTkFrame(parent)の生成でエラーが発生する。正確な例外種別はcustomtkinter/tkinterの実装依存で未確認であり、本メソッドは捕捉せず伝播する。'
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
    input: self.config == {}
    expected: メソッド返却後、self.interval_var.get() == "1.0" かつ self.diff_var.get() == True かつ self.diff_threshold_var.get() == "0.1" かつ self.diff_only_var.get() == False であり、parent配下にウィジェット8個が生成される。
    """
    with _parent_frame() as parent:
        self = _make_self({})
        ret = self._create_video_settings(parent)
        assert ret is None
        assert isinstance(self.interval_var, ctk.StringVar)
        assert self.interval_var.get() == "1.0"
        assert isinstance(self.diff_var, ctk.BooleanVar)
        assert self.diff_var.get() is True
        assert isinstance(self.diff_threshold_var, ctk.StringVar)
        assert self.diff_threshold_var.get() == "0.1"
        assert isinstance(self.diff_only_var, ctk.BooleanVar)
        assert self.diff_only_var.get() is False
        # ウィジェット8個: video_frame (row=0, column=0) + 子7個
        slaves = parent.grid_slaves()
        assert any(
            (slave.grid_info()["row"], slave.grid_info()["column"]) == (0, 0)
            for slave in slaves
        )
        video_frame = parent.winfo_children()[-1]
        assert isinstance(video_frame, ctk.CTkFrame)
        assert len(video_frame.winfo_children()) == 7


def test_edge_02():
    """
    input: self.config == {"video": {}}
    expected: 空configの場合と同一。4つの状態変数が既定値 "1.0" / True / "0.1" / False を取る。
    """
    with _parent_frame() as parent:
        self = _make_self({"video": {}})
        ret = self._create_video_settings(parent)
        assert ret is None
        assert self.interval_var.get() == "1.0"
        assert self.diff_var.get() is True
        assert self.diff_threshold_var.get() == "0.1"
        assert self.diff_only_var.get() is False


def test_edge_03():
    """
    input: self.config == {"video": {"frame_interval": 2.5, "enable_diff_check": False, "diff_threshold": 0.35, "diff_only": True}}
    expected: self.interval_var.get() == "2.5" かつ self.diff_var.get() == False かつ self.diff_threshold_var.get() == "0.35" かつ self.diff_only_var.get() == True。
    """
    with _parent_frame() as parent:
        self = _make_self({
            "video": {
                "frame_interval": 2.5,
                "enable_diff_check": False,
                "diff_threshold": 0.35,
                "diff_only": True,
            }
        })
        ret = self._create_video_settings(parent)
        assert ret is None
        assert self.interval_var.get() == "2.5"
        assert self.diff_var.get() is False
        assert self.diff_threshold_var.get() == "0.35"
        assert self.diff_only_var.get() is True


def test_edge_04():
    """
    input: self.config == {"video": None}
    expected: メソッドは返らず、手順3のNoneに対する.get("frame_interval", 1.0)呼び出しでAttributeError（'NoneType' object has no attribute 'get'）が発生し、捕捉されずに呼び出し側へ伝播する。
    """
    with _parent_frame() as parent:
        self = _make_self({"video": None})
        before = len(parent.winfo_children())
        try:
            self._create_video_settings(parent)
            raised = False
        except AttributeError:
            raised = True
        assert raised
        # 発生前に video_frame とラベル2個が既に生成済み
        children = parent.winfo_children()
        assert len(children) == before + 1
        video_frame = children[-1]
        assert isinstance(video_frame, ctk.CTkFrame)
        sub = video_frame.winfo_children()
        assert len(sub) == 2
        assert all(isinstance(c, ctk.CTkLabel) for c in sub)
        assert not hasattr(self, "interval_var")
        assert not hasattr(self, "diff_var")


def test_edge_05():
    """
    input: self.config == {"video": {"frame_interval": "abc"}}
    expected: self.interval_var.get() == "abc" となる。本メソッドは数値検証を行わず、値をstr()で文字列化してウィジェットを正常生成する。
    """
    with _parent_frame() as parent:
        self = _make_self({"video": {"frame_interval": "abc"}})
        ret = self._create_video_settings(parent)
        assert ret is None
        assert self.interval_var.get() == "abc"
        # 他キーはデフォルト
        assert self.diff_var.get() is True
        assert self.diff_threshold_var.get() == "0.1"
        assert self.diff_only_var.get() is False
