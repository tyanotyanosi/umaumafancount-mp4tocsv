"""Tests for ``gui.video_player.VideoPreviewFrame._show_frame``.

Specification: docs/00-Architecture/gui_video_player__VideoPreviewFrame__show_frame.yaml

``_show_frame`` seeks the loaded video to ``frame_idx`` and draws the read
frame into the preview label, updating the progress bar:

- if ``self.cap`` is falsy (None), it returns immediately
- ``self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)`` seeks to the frame
  (the return value is not checked)
- ``self.cap.read()`` reads one frame, yielding ``ret`` and ``frame``
- if ``ret`` is False, it returns immediately (no preview / progress update)
- on success it calls ``self._draw_frame(frame)`` and then
  ``self.progress.set(frame_idx / self.frame_count if self.frame_count > 0
  else 0)``
- ``self.current_frame_idx`` is not modified; the function returns ``None``

Mocked / stand-in dependencies (per the test-generation rules): the
``VideoPreviewFrame`` instance is created without running ``__init__``
(``object.__new__``); ``self.cap`` is a ``MagicMock`` stand-in for an open
``cv2.VideoCapture`` (``set`` / ``read`` controlled per test) or ``None``;
``self.frame_count`` is set per test (absent for edge case 5);
``self._draw_frame`` is a mock; ``self.progress`` is a minimal fake
recording ``set`` calls; ``self.current_frame_idx`` is a sentinel value that
must remain unchanged.

``errors`` section of the spec (documented here only, NOT tested, per the
test-generation rules):

- condition: 'self.frame_count 属性が存在せず、self.cap が真値で読み取り成功'
  behavior: 'self.frame_count 参照で AttributeError が発生し、捕捉されず呼び出し元に伝播する'
- condition: 'cv2.VideoCapture.set か .read 自体が例外を送出する（状態が不正な場合等、OpenCV 依存）'
  behavior: 'この関数には try/except がないため、例外は捕捉されず呼び出し元に伝播する'
"""

from unittest import mock

import cv2
import pytest

from gui.video_player import VideoPreviewFrame


class _FakeProgress:
    """Minimal stand-in for a CTkProgressBar: records ``set`` calls."""

    def __init__(self):
        self.value = None
        self.set_calls = []

    def set(self, value):
        self.set_calls.append(value)
        self.value = value


def _make_self(cap, frame_count):
    """Build a ``VideoPreviewFrame`` instance (without running ``__init__``)
    with the dependencies ``_show_frame`` touches mocked / faked.
    ``frame_count=None`` leaves the attribute absent (edge case 5)."""
    self = object.__new__(VideoPreviewFrame)
    self.cap = cap
    if frame_count is not None:
        self.frame_count = frame_count
    self.current_frame_idx = 7  # sentinel: must remain unchanged
    self._draw_frame = mock.MagicMock(name="_draw_frame")
    self.progress = _FakeProgress()
    return self


def test_edge_01():
    """
    input: self.cap が None（動画未ロード）、frame_idx は任意
    expected: None を返す。cap.set/read・_draw_frame・progress.set が一切呼び出されない
    """
    self = _make_self(None, frame_count=100)
    ret = self._show_frame(5)
    assert ret is None
    self._draw_frame.assert_not_called()
    assert self.progress.set_calls == []
    assert self.current_frame_idx == 7


def test_edge_02():
    """
    input: self.cap 開状態、self.frame_count = 100、frame_idx = 30、cap.read() が ret=True を返す
    expected: _draw_frame が1回呼び出され、progress.set(0.3) が呼び出される（0.3 は真算除法の浮動小数点）
    """
    cap = mock.MagicMock(name="cap")
    cap.read.return_value = (True, "frame")
    self = _make_self(cap, frame_count=100)
    ret = self._show_frame(30)
    assert ret is None
    cap.set.assert_called_once_with(cv2.CAP_PROP_POS_FRAMES, 30)
    self._draw_frame.assert_called_once_with("frame")
    assert self.progress.set_calls == [0.3]
    assert self.current_frame_idx == 7


def test_edge_03():
    """
    input: self.cap 開状態、self.frame_count = 0、frame_idx = 0、読み取り成功
    expected: _draw_frame が1回呼び出され、progress.set(0) が呼び出される（三項演算子の else 分）
    """
    cap = mock.MagicMock(name="cap")
    cap.read.return_value = (True, "frame")
    self = _make_self(cap, frame_count=0)
    ret = self._show_frame(0)
    assert ret is None
    self._draw_frame.assert_called_once_with("frame")
    assert self.progress.set_calls == [0]
    assert self.current_frame_idx == 7


def test_edge_04():
    """
    input: self.cap 開状態、self.frame_count = 10、frame_idx = 15、読み取り成功
    expected: progress.set(1.5) が呼び出される（1 を超過。CTkProgressBar 側の受入挙動は実装依存で未確認）
    """
    cap = mock.MagicMock(name="cap")
    cap.read.return_value = (True, "frame")
    self = _make_self(cap, frame_count=10)
    ret = self._show_frame(15)
    assert ret is None
    self._draw_frame.assert_called_once_with("frame")
    assert self.progress.set_calls == [1.5]
    assert self.current_frame_idx == 7


def test_edge_05():
    """
    input: self.cap が真値、読み取り成功、ただし self.frame_count 属性が未作成
    expected: 108行目の self.frame_count 参照で AttributeError が発生し、関数は完了しない
    """
    cap = mock.MagicMock(name="cap")
    cap.read.return_value = (True, "frame")
    self = _make_self(cap, frame_count=None)
    with pytest.raises(AttributeError):
        self._show_frame(30)
    # _draw_frame は progress 設定より先に実行済み（behavior の順序）
    self._draw_frame.assert_called_once_with("frame")
    assert self.progress.set_calls == []
    assert self.current_frame_idx == 7


def test_edge_06():
    """
    input: self.cap が真値、cap.read() が ret=False を返す（OpenCV の範囲外シーク時等の動作）
    expected: None を返す。_draw_frame は呼び出されずプログレスバーも更新されない
    """
    cap = mock.MagicMock(name="cap")
    cap.read.return_value = (False, None)
    self = _make_self(cap, frame_count=100)
    ret = self._show_frame(30)
    assert ret is None
    cap.set.assert_called_once_with(cv2.CAP_PROP_POS_FRAMES, 30)
    self._draw_frame.assert_not_called()
    assert self.progress.set_calls == []
    assert self.current_frame_idx == 7
