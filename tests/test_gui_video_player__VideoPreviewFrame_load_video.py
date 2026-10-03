"""Tests for ``gui.video_player.VideoPreviewFrame.load_video``.

Specification: docs/00-Architecture/gui_video_player__VideoPreviewFrame_load_video.yaml

``load_video`` opens ``video_path`` with ``cv2.VideoCapture``, reads fps /
frame count / native size, updates the preview UI (frame 0 display,
progress reset, button states); if the capture cannot be opened it resets
the state and shows an error label:

- Step 1: ``self._cancel_play_tick()`` cancels any pending play tick
- Step 2: ``is_playing=False`` and ``btn_play``'s text is set to "再生"
- Step 3: ``video_path`` is stored in ``self.video_path``
- Step 4: if ``self.cap`` exists it is released (``self.cap.release()``)
- Step 5: ``self.cap = cv2.VideoCapture(video_path)``
- Step 6: if ``cap`` is None or ``isOpened()`` is False, the new cap is
  released (when not None) and the state is reset: cap=None,
  video_path=None, fps=0.0, frame_count=0, native_width=0,
  native_height=0, current_frame_idx=0, progress=0, btn_process=disabled,
  video_label image='' and text='動画を読み込めませんでした'
- Step 7 (success): fps / frame_count / native width/height are read from
  ``cap.get`` into ``self.fps`` / ``self.frame_count`` /
  ``self.native_width`` / ``self.native_height``; ``current_frame_idx=0``
- Step 8 (success): ``self._compute_display_size()``
- Step 9 (success): ``self._show_frame(0)``
- Step 10 (success): ``btn_process``'s state is set to 'normal'

The function returns ``None``.

Mocked / stand-in dependencies (per the test-generation rules): the
``VideoPreviewFrame`` instance is created without running ``__init__``
(``object.__new__``); ``self._cancel_play_tick`` / ``self._show_frame`` /
``self._compute_display_size`` are mocks; ``self.btn_play`` /
``self.btn_process`` are minimal fake widgets recording ``configure``
calls; ``self.progress`` is a minimal fake recording ``set`` calls;
``self.video_label`` is a minimal fake label recording ``configure`` calls
and the ``image`` assignment. ``cv2.VideoCapture`` runs real: the failure
case uses a non-existent path, and the success cases use a small MJPG video
created with ``cv2.VideoWriter`` in a temporary directory (320x240, 10
frames, 30 fps).

``errors`` section of the spec (documented here only, NOT tested, per the
test-generation rules):

- condition: "cv2.VideoCapture(video_path) または isOpened() が例外を送出する"
  behavior: "try/except なし。例外は呼び出し元へそのまま伝播する。その時点で self.video_path は video_path に設定済みであり、旧 self.cap（存在時）は解放済みだが self.cap は解放済み旧ハンドルを指し続けたまま None にリセットされない（代入は評価前に失敗するため）"
- condition: "_cancel_play_tick / cap.release / cap.get / _compute_display_size / _show_frame / progress.set / configure のいずれかで例外が発生する"
  behavior: "try/except なし。例外は呼び出し元へそのまま伝播しメソッドは異常終了する。例外発生時点でインスタンス状態は部分的に更新済みのまま"
"""

import tempfile
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from gui.video_player import VideoPreviewFrame


class _FakeWidget:
    """Minimal stand-in for a Tk widget: records ``configure`` keyword calls
    and item assignments so tests can assert on ``text`` / ``state``."""

    def __init__(self):
        self.text = None
        self.state = None
        self.configure_calls = []

    def configure(self, **kwargs):
        self.configure_calls.append(kwargs)
        if "text" in kwargs:
            self.text = kwargs["text"]
        if "state" in kwargs:
            self.state = kwargs["state"]

    def __setitem__(self, key, value):
        if key == "text":
            self.text = value
        elif key == "state":
            self.state = value


class _FakeProgress:
    """Minimal stand-in for a CTkProgressBar: records ``set`` calls."""

    def __init__(self):
        self.value = None
        self.set_calls = []

    def set(self, value):
        self.set_calls.append(value)
        self.value = value


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


def _make_video(path: Path) -> None:
    """Create a small deterministic MJPG video (320x240, 30 fps, 10 frames)
    at ``path`` using ``cv2.VideoWriter``."""
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, 30.0, (320, 240))
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    for _ in range(10):
        writer.write(frame)
    writer.release()


def _make_self(cap=None, is_playing=False):
    """Build a ``VideoPreviewFrame`` instance (without running ``__init__``)
    with the dependencies ``load_video`` touches mocked / faked."""
    self = object.__new__(VideoPreviewFrame)
    self.cap = cap
    self.video_path = None
    self.fps = 0.0
    self.frame_count = 0
    self.native_width = 0
    self.native_height = 0
    self.current_frame_idx = 0
    self.is_playing = is_playing
    self._cancel_play_tick = mock.MagicMock(name="_cancel_play_tick")
    self._show_frame = mock.MagicMock(name="_show_frame")
    self._compute_display_size = mock.MagicMock(name="_compute_display_size")
    self.btn_play = _FakeWidget()
    self.btn_process = _FakeWidget()
    self.progress = _FakeProgress()
    self.video_label = _FakeLabel()
    return self


def test_edge_01():
    """
    input: cv2.VideoCapture(video_path) の isOpened() が False になる video_path（例: 存在しないファイル）
    expected: 失敗分岐 — self.cap==None, self.video_path==None, self.fps==0.0, self.frame_count==0, self.native_width==0, self.native_height==0, self.current_frame_idx==0, progress の値 0, btn_process は disabled, video_label は text '動画を読み込めませんでした' と image ''、戻り値 None
    """
    self = _make_self()
    missing = Path(tempfile.mkdtemp()) / "no_such_file.mp4"
    ret = self.load_video(str(missing))
    assert ret is None
    assert self.cap is None
    assert self.video_path is None
    assert self.fps == 0.0
    assert self.frame_count == 0
    assert self.native_width == 0
    assert self.native_height == 0
    assert self.current_frame_idx == 0
    assert self.progress.set_calls == [0]
    assert self.btn_process.state == "disabled"
    assert self.video_label.text == "動画を読み込めませんでした"
    assert self.video_label.image == ""


def test_edge_02():
    """
    input: cv2.VideoCapture の isOpened() が True になる有効な動画ファイルのパス
    expected: 成功分岐 — self.cap は開いたキャプチャを保持、video_path / fps / frame_count / native 幅高が cap の値に設定、current_frame_idx==0、_compute_display_size() と _show_frame(0) が呼び出され、btn_process は normal、戻り値 None
    """
    tmp = Path(tempfile.mkdtemp())
    video = tmp / "sample.mp4"
    _make_video(video)
    self = _make_self()
    ret = self.load_video(str(video))
    assert ret is None
    assert self.cap is not None
    assert self.cap.isOpened()
    assert self.video_path == str(video)
    assert self.fps == self.cap.get(cv2.CAP_PROP_FPS)
    assert self.frame_count == int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    assert self.native_width == int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    assert self.native_height == int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    assert self.current_frame_idx == 0
    self._compute_display_size.assert_called_once()
    self._show_frame.assert_called_once_with(0)
    assert self.btn_process.state == "normal"
    assert self.btn_play.text == "再生"
    assert self.is_playing is False
    self.cap.release()


def test_edge_03():
    """
    input: self.cap が None（__init__ 直後）の初回ロードで video_path が有効
    expected: 手順4 の解放はスキップされ（if self.cap が偽）、残りは成功分岐の動作どおり
    """
    tmp = Path(tempfile.mkdtemp())
    video = tmp / "sample.mp4"
    _make_video(video)
    self = _make_self(cap=None)
    ret = self.load_video(str(video))
    assert ret is None
    # 手順4: 旧 cap が偽のため解放はスキップ（呼ぶ対象が存在しない）
    assert self.cap is not None
    assert self.cap.isOpened()
    assert self.video_path == str(video)
    assert self.frame_count == int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    self._compute_display_size.assert_called_once()
    self._show_frame.assert_called_once_with(0)
    assert self.btn_process.state == "normal"
    self.cap.release()


def test_edge_04():
    """
    input: self.cap が既に開いたキャプチャを保持している状態での再ロード
    expected: 手順4 で旧ハンドルの self.cap.release() が呼び出され、その後新しいハンドルが作成される
    """
    tmp = Path(tempfile.mkdtemp())
    video = tmp / "sample.mp4"
    _make_video(video)
    old_cap = mock.MagicMock(name="old_cap")
    self = _make_self(cap=old_cap)
    ret = self.load_video(str(video))
    assert ret is None
    old_cap.release.assert_called_once()
    assert self.cap is not old_cap
    assert self.cap.isOpened()
    assert self.video_path == str(video)
    self._compute_display_size.assert_called_once()
    self._show_frame.assert_called_once_with(0)
    assert self.btn_process.state == "normal"
    self.cap.release()


def test_edge_05():
    """
    input: 再生中（self.is_playing が True、after() tick が保留中）の呼び出し
    expected: 手順1 で self._cancel_play_tick() が呼び出され、is_playing は False になり、btn_play の text は '再生' に戻る（tick 取り消しの具体的効果は _cancel_play_tick 実装依存で未確認）
    """
    tmp = Path(tempfile.mkdtemp())
    video = tmp / "sample.mp4"
    _make_video(video)
    self = _make_self(is_playing=True)
    ret = self.load_video(str(video))
    assert ret is None
    self._cancel_play_tick.assert_called_once()
    assert self.is_playing is False
    assert self.btn_play.text == "再生"
    self.cap.release()
