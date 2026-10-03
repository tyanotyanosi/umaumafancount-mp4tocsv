"""Tests for ``gui.video_player.VideoPreviewFrame.close``.

Specification: docs/00-Architecture/gui_video_player__VideoPreviewFrame_close.yaml

``close`` releases the ``VideoPreviewFrame``'s resources: it stops the
playback state, cancels the play tick, and releases the
``cv2.VideoCapture``:

- Step 1: ``self.is_playing = False``
- Step 2: ``self._cancel_play_tick()`` cancels any pending play tick
  (via which ``self._play_job_id`` becomes ``None``)
- Step 3: ``self.btn_play``'s text is reset to "再生"
- Step 4: if ``self.cap`` is truthy: ``self.cap.release()`` is called and
  ``self.cap = None``; if None, the step is skipped

The function does not modify ``self.video_path`` or
``self.current_frame_idx`` (no reset logic in the function), does not
destroy the Tk widget itself, and returns ``None``.

Mocked / stand-in dependencies (per the test-generation rules): the
``VideoPreviewFrame`` instance is created without running ``__init__``
(``object.__new__``); ``self.cap`` is a ``MagicMock`` stand-in for an open
``cv2.VideoCapture`` (truthy) or ``None``; ``self.is_playing`` /
``self._play_job_id`` / ``self.video_path`` / ``self.current_frame_idx``
are set per test; ``self.btn_play`` is a minimal fake widget recording
``configure`` calls; ``self.after_cancel`` is a mock (the real
``_cancel_play_tick`` runs and uses it); ``self._cancel_play_tick`` runs
real per its spec (cancels the job and clears ``_play_job_id`` only when
the job id is not None).

``errors`` section of the spec (documented here only, NOT tested, per the
test-generation rules):

- condition: "コードに明示的なエラー処理はない。cap.release() の戻り値は使用されていない。"
  behavior: "本関数には例外送出パスはない（Tk / cv2 内部の失敗は関数外の範疇）"
"""

from unittest import mock

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


def _make_self(cap, is_playing, play_job_id, video_path=None,
               current_frame_idx=0):
    """Build a ``VideoPreviewFrame`` instance (without running ``__init__``)
    with the dependencies ``close`` touches mocked / faked."""
    self = object.__new__(VideoPreviewFrame)
    self.cap = cap
    self.is_playing = is_playing
    self._play_job_id = play_job_id
    self.video_path = video_path
    self.current_frame_idx = current_frame_idx
    self.btn_play = _FakeWidget()
    self.after_cancel = mock.MagicMock(name="after_cancel")
    return self


def test_edge_01():
    """
    input: self.cap=None, self._play_job_id=None, self.is_playing=True（動画未ロード）
    expected: is_playing==False、_play_job_id は None、btn_play テキスト==再生、cap.release は呼び出されず、cap は None のまま、戻り値 None
    """
    self = _make_self(None, True, None, video_path="sample.mp4",
                      current_frame_idx=3)
    ret = self.close()
    assert ret is None
    assert self.is_playing is False
    assert self._play_job_id is None
    assert self.btn_play.text == "再生"
    assert self.cap is None
    self.after_cancel.assert_not_called()
    # video_path / current_frame_idx は変更されない
    assert self.video_path == "sample.mp4"
    assert self.current_frame_idx == 3


def test_edge_02():
    """
    input: self.cap=truthy な cv2.VideoCapture、self._play_job_id=有効なジョブID、self.is_playing=True（動画ロード済み・tick進行中）
    expected: is_playing==False、_play_job_id は None、btn_play テキスト==再生、cap.release() が1回呼び出され、cap は None となり、戻り値 None
    """
    cap = mock.MagicMock(name="cap")
    self = _make_self(cap, True, 12345, video_path="sample.mp4",
                      current_frame_idx=10)
    ret = self.close()
    assert ret is None
    assert self.is_playing is False
    assert self._play_job_id is None
    assert self.btn_play.text == "再生"
    cap.release.assert_called_once()
    assert self.cap is None
    self.after_cancel.assert_called_once_with(12345)
    # video_path / current_frame_idx は変更されない
    assert self.video_path == "sample.mp4"
    assert self.current_frame_idx == 10


def test_edge_03():
    """
    input: self.cap=None、close を2回連続で呼び出した2回目の呼び出し
    expected: idempotent: 動画未ロードの場合と同様に例外なく完了し、戻り値 None
    """
    self = _make_self(None, True, None, video_path="sample.mp4",
                      current_frame_idx=3)
    first = self.close()
    assert first is None
    second = self.close()
    assert second is None
    # 2回目で例外なく完了し、状態は動画未ロードの場合と同様
    assert self.is_playing is False
    assert self._play_job_id is None
    assert self.btn_play.text == "再生"
    assert self.cap is None
    self.after_cancel.assert_not_called()
