"""Tests for ``gui.video_player.VideoPreviewFrame._setup_ui``.

Specification: docs/00-Architecture/gui_video_player__VideoPreviewFrame__setup_ui.yaml

``_setup_ui`` creates the 5 UI child widgets (preview label, progress bar,
process-start / play / stop buttons) inside the frame and places them in a
grid layout:

- Step 1: grid row 0/1/2 weights are set to 1/0/0 (``grid_rowconfigure``)
- Step 2: ``self.video_label`` (text='動画プレビュー', anchor='center') at
  row=0, column=0, sticky='nsew', padx=5, pady=5
- Step 3: ``self.progress`` created, value set to 0, at row=1, column=0,
  sticky='ew', padx=5, pady=5
- Step 4: ``self.btn_process`` (text='処理開始', command=self._on_process,
  state='disabled') at row=2, column=0, columnspan=2, sticky='ew', padx=5,
  pady=5
- Step 5: ``self.btn_play`` (text='再生', command=self._toggle_play) at
  row=3, column=0, sticky='ew', padx=5, pady=5
- Step 6: ``self.btn_stop`` (text='停止', command=self._stop) at row=3,
  column=1, sticky='ew', padx=5, pady=5

The function returns ``None``.

Mocked / stand-in dependencies (per the test-generation rules): for edge
case 1 a real ``ctk.CTkFrame`` widget is created (a
``VideoPreviewFrame`` instance built with ``object.__new__`` and
``ctk.CTkFrame.__init__`` run directly, so the widget exists but the
``__init__`` attribute initialization does not; a retry helper guards
against unstable Tcl environments and the widget is destroyed at the end);
``self._on_process`` / ``self._toggle_play`` / ``self._stop`` are plain
functions assigned as button commands (only referenced, not invoked). For
edge case 2 the real ``_setup_ui`` is called unbound on a plain
``ctk.CTkFrame`` instance that does not have the ``_on_process`` method,
so the ``self._on_process`` reference in step 4 raises
``AttributeError``.

``errors`` section of the spec (documented here only, NOT tested, per the
test-generation rules):

- condition: "self._on_process / self._toggle_play / _stop 属性が self に存在しない"
  behavior: "該当ボタンの生成時（手順4/5/6）に属性参照で AttributeError が発生し、メソッドは異常終了する"
- condition: "ウィジェットの生成や grid 配置で例外が発生する（例: self が有効な親ウィジェットでない）"
  behavior: "try/except なし。tkinter/customtkinter 由来の例外が呼び出し元へそのまま伝播する（例外種別は未確認）"
"""

import contextlib
import time

import customtkinter as ctk

from gui.video_player import VideoPreviewFrame


@contextlib.contextmanager
def _video_frame_instance():
    """Build a real ``ctk.CTkFrame`` widget carrying the
    ``VideoPreviewFrame`` methods (without running its ``__init__``),
    retrying up to 5 times at 0.25 s intervals against unstable Tcl
    environments, and destroy it at the end."""
    last_error = None
    self = None
    for attempt in range(5):
        if attempt:
            time.sleep(0.25)
        try:
            self = object.__new__(VideoPreviewFrame)
            ctk.CTkFrame.__init__(self, None)
            break
        except Exception as exc:
            last_error = exc
    if self is None:
        raise last_error
    try:
        yield self
    finally:
        try:
            self.destroy()
        except Exception:
            pass


@contextlib.contextmanager
def _plain_frame():
    """Build a plain ``ctk.CTkFrame`` widget (which does not have the
    ``VideoPreviewFrame`` methods such as ``_on_process``), retrying up to
    5 times at 0.25 s intervals against unstable Tcl environments, and
    destroy it at the end."""
    last_error = None
    frame = None
    for attempt in range(5):
        if attempt:
            time.sleep(0.25)
        try:
            frame = ctk.CTkFrame(None)
            break
        except Exception as exc:
            last_error = exc
    if frame is None:
        raise last_error
    try:
        yield frame
    finally:
        try:
            frame.destroy()
        except Exception:
            pass


def _row_weight(frame, row):
    """Read the grid row weight, handling both dict and sequence return
    forms of ``grid_rowconfigure``."""
    info = frame.grid_rowconfigure(row)
    if isinstance(info, dict):
        return info.get("weight")
    return info[1]


def test_edge_01():
    """
    input: self が有効なインスタンス（フレーム生成済み）かつ _on_process/_toggle_play/_stop 属性が存在する
    expected: 5つのウィジェットが postconditions 通り作成・配置され、progress の値は 0、btn_process は disabled、btn_play と btn_stop は state 未指定（コンストラクタ既定値）で生成され、戻り値 None
    """
    with _video_frame_instance() as self:
        self._on_process = lambda path: None
        self._toggle_play = lambda: None
        self._stop = lambda: None
        ret = self._setup_ui()
        assert ret is None
        # ウィジェットの型
        assert isinstance(self.video_label, ctk.CTkLabel)
        assert isinstance(self.progress, ctk.CTkProgressBar)
        assert isinstance(self.btn_process, ctk.CTkButton)
        assert isinstance(self.btn_play, ctk.CTkButton)
        assert isinstance(self.btn_stop, ctk.CTkButton)
        # ウィジェットの内容
        assert self.video_label.cget("text") == "動画プレビュー"
        assert self.video_label.cget("anchor") == "center"
        assert self.progress.get() == 0
        assert self.btn_process.cget("text") == "処理開始"
        assert self.btn_process.cget("state") == "disabled"
        assert self.btn_process.cget("command") is self._on_process
        assert self.btn_play.cget("text") == "再生"
        assert self.btn_play.cget("command") is self._toggle_play
        assert self.btn_stop.cget("text") == "停止"
        assert self.btn_stop.cget("command") is self._stop
        # grid の行 weight
        assert _row_weight(self, 0) == 1
        assert _row_weight(self, 1) == 0
        assert _row_weight(self, 2) == 0
        # grid 配置（row, column）
        assert self.video_label.grid_info()["row"] == 0
        assert self.video_label.grid_info()["column"] == 0
        assert self.progress.grid_info()["row"] == 1
        assert self.progress.grid_info()["column"] == 0
        assert self.btn_process.grid_info()["row"] == 2
        assert self.btn_process.grid_info()["column"] == 0
        assert self.btn_process.grid_info()["columnspan"] == 2
        assert self.btn_play.grid_info()["row"] == 3
        assert self.btn_play.grid_info()["column"] == 0
        assert self.btn_stop.grid_info()["row"] == 3
        assert self.btn_stop.grid_info()["column"] == 1


def test_edge_02():
    """
    input: self に _on_process 属性が存在しない場合（メソッドを持たないオブジェクトに対して直接呼び出した場合）
    expected: video_label と progress は作成済みだが、手順4 で btn_process を生成する際 self._on_process 参照時に AttributeError が発生し、btn_play と btn_stop は未生成のまま異常終了
    """
    with _plain_frame() as self:
        # メソッドを持たないオブジェクトに対して直接呼び出した場合を模すため、
        # _setup_ui をバインドなしで呼び出す。
        try:
            VideoPreviewFrame._setup_ui(self)
            raised = False
        except AttributeError:
            raised = True
        assert raised
        assert isinstance(self.video_label, ctk.CTkLabel)
        assert isinstance(self.progress, ctk.CTkProgressBar)
        assert not hasattr(self, "btn_process")
        assert not hasattr(self, "btn_play")
        assert not hasattr(self, "btn_stop")
