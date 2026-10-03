"""Tests for ``gui.video_player.VideoPreviewFrame._draw_frame``.

Specification: docs/00-Architecture/gui_video_player__VideoPreviewFrame__draw_frame.yaml

``_draw_frame`` converts the given frame (BGR) to RGB, resizes it to the
preview display size, and shows it on ``self.video_label`` (no seek,
drawing only, per its docstring):

- ``cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)`` converts BGR to RGB
- ``cv2.resize(frame_rgb, (self.disp_width, self.disp_height))`` resizes
  (tuple order: width, height)
- ``Image.fromarray(frame_resized)`` creates a PIL image and
  ``ImageTk.PhotoImage`` creates the Tk photo object
- ``self.video_label.configure(image=photo, text="")`` replaces the label
  image and clears the text
- ``self.video_label.image = photo`` keeps a reference so the photo is not
  garbage-collected

The function returns ``None``.

Mocked / stand-in dependencies (per the test-generation rules): the
``VideoPreviewFrame`` instance is created without running ``__init__``
(``object.__new__``); ``self.video_label`` is a minimal fake widget
recording ``configure`` calls and the ``image`` assignment;
``self.disp_width`` / ``self.disp_height`` are set per test (absent for edge
case 4); ``frame`` is a real numpy array. ``cv2.cvtColor`` / ``cv2.resize`` /
``Image.fromarray`` / ``ImageTk.PhotoImage`` run real; a minimal
``tkinter.Tk()`` default root window (created with a retry helper against
unstable Tcl environments and destroyed at the end of each test) is required
for ``ImageTk.PhotoImage``.

``errors`` section of the spec (documented here only, NOT tested, per the
test-generation rules):

- condition: 'frame が None 或非画像配列'
  behavior: 'cv2.cvtColor のエラーが送出され、捕捉されず呼び出し元に伝播する（具体的な例外型は OpenCV 依存で未確認）'
- condition: 'frame のチャネル数が cv2.COLOR_BGR2RGB と互換でない（1 チャンネル等）'
  behavior: 'cv2 バージョン依存。本環境 cv2 5.0.0 では 1ch / 4ch 入力も 3ch に変換され例外は送出されない（旧バージョンでは cv2.error を送出し得る）'
- condition: 'self.disp_width / self.disp_height 属性が存在しない'
  behavior: 'AttributeError が送出され、捕捉されず呼び出し元に伝播する'
- condition: 'frame の形状が cv2.resize の処理に不可である（例 空配列）'
  behavior: 'cv2.resize のエラーが送出され、捕捉されず呼び出し元に伝播する'
"""

import contextlib
import time

import cv2
import numpy as np
import tkinter as tk
from PIL import ImageTk
import pytest

from gui.video_player import VideoPreviewFrame


class _FakeLabel:
    """Minimal stand-in for a Tk label: records ``configure`` keyword calls
    and attribute assignments so tests can assert on ``image`` / ``text``."""

    def __init__(self):
        self.image = None
        self.text = None
        self.configure_calls = []

    def configure(self, **kwargs):
        self.configure_calls.append(kwargs)
        if "image" in kwargs:
            self.image = kwargs["image"]
        if "text" in kwargs:
            self.text = kwargs["text"]


def _make_self(disp_width=None, disp_height=None):
    """Build a ``VideoPreviewFrame`` instance (without running ``__init__``)
    with the dependencies ``_draw_frame`` touches faked. ``disp_width`` /
    ``disp_height`` of ``None`` leave the attributes absent (edge case 4)."""
    self = object.__new__(VideoPreviewFrame)
    self.video_label = _FakeLabel()
    if disp_width is not None:
        self.disp_width = disp_width
    if disp_height is not None:
        self.disp_height = disp_height
    return self


def _create_root():
    """Create a minimal ``tkinter.Tk()`` default root window, retrying up to
    5 times at 0.25 s intervals against unstable Tcl environments; re-raise
    the original ``TclError`` if every attempt fails."""
    last_error = None
    for attempt in range(5):
        if attempt:
            time.sleep(0.25)
        try:
            return tk.Tk()
        except tk.TclError as exc:
            last_error = exc
    raise last_error


@contextlib.contextmanager
def _root_window():
    root = _create_root()
    try:
        yield root
    finally:
        try:
            root.destroy()
        except tk.TclError:
            pass


def test_edge_01():
    """
    input: frame = None
    expected: 112行目の cv2.cvtColor でエラーが送出され関数は完了しない。video_label は更新されない
    """
    with _root_window():
        self = _make_self(disp_width=640, disp_height=360)
        try:
            self._draw_frame(None)
            raised = False
        except Exception:
            raised = True
        assert raised
        assert self.video_label.configure_calls == []
        assert self.video_label.image is None
        assert self.video_label.text is None


def test_edge_02():
    """
    input: frame が 2 次元（グレースケール 1 チャンネル）ndarray
    expected: 例外は送出されない（cv2 5.0.0: COLOR_BGR2RGB は 1ch 入力を 3ch に変換）。frame が (disp_width, disp_height) にリサイズされ、video_label が新しい PhotoImage と text="" で更新される
    """
    with _root_window():
        self = _make_self(disp_width=640, disp_height=360)
        frame = np.zeros((1080, 1920), dtype=np.uint8)
        ret = self._draw_frame(frame)
        assert ret is None
        assert self.video_label.text == ""
        photo = self.video_label.image
        assert isinstance(photo, ImageTk.PhotoImage)
        assert photo.width() == 640
        assert photo.height() == 360


def test_edge_03():
    """
    input: frame が 4 チャンネル ndarray
    expected: 例外は送出されない（cv2 5.0.0: COLOR_BGR2RGB は 4ch 入力を 3ch に変換）。frame が (disp_width, disp_height) にリサイズされ、video_label が新しい PhotoImage と text="" で更新される
    """
    with _root_window():
        self = _make_self(disp_width=640, disp_height=360)
        frame = np.zeros((1080, 1920, 4), dtype=np.uint8)
        ret = self._draw_frame(frame)
        assert ret is None
        assert self.video_label.text == ""
        photo = self.video_label.image
        assert isinstance(photo, ImageTk.PhotoImage)
        assert photo.width() == 640
        assert photo.height() == 360


def test_edge_04():
    """
    input: self.disp_width / self.disp_height 属性が未作成、frame は有効な BGR 3 チャンネル画像
    expected: 113行目の属性アクセスで AttributeError が発生（cvtColor が成功した後）
    """
    with _root_window():
        self = _make_self()
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        with pytest.raises(AttributeError):
            self._draw_frame(frame)
        # cvtColor は成功した後に属性アクセスで失敗するため、ラベルは未更新
        assert self.video_label.configure_calls == []
        assert self.video_label.image is None
        assert self.video_label.text is None


def test_edge_05():
    """
    input: 有効な BGR 3 チャンネルフレーム（例 1920x1080）、disp_width = 640、disp_height = 360 設定済み
    expected: None を返し、video_label のテキストが "" になり video_label.image が 640x360 の新しい PhotoImage になる
    """
    with _root_window():
        self = _make_self(disp_width=640, disp_height=360)
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        ret = self._draw_frame(frame)
        assert ret is None
        assert self.video_label.text == ""
        photo = self.video_label.image
        assert isinstance(photo, ImageTk.PhotoImage)
        assert photo.width() == 640
        assert photo.height() == 360
