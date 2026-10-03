"""Tests for ``gui.video_player.VideoPreviewFrame._cancel_play_tick``.

Specification: docs/00-Architecture/gui_video_player__VideoPreviewFrame__cancel_play_tick.yaml

``_cancel_play_tick`` cancels a reserved video play tick (a Tk ``after()``
job) if one exists, and clears the recorded job id:

- Step 1: check whether ``self._play_job_id`` is None
- Step 2: if None, return immediately without cancelling anything
  (no attribute change)
- Step 3: if not None: ``self.after_cancel(self._play_job_id)`` cancels the
  reserved play tick
- Step 4: ``self._play_job_id = None``

The function returns ``None``.

Mocked / stand-in dependencies (per the test-generation rules): the
``VideoPreviewFrame`` instance is created without running ``__init__``
(``object.__new__``); ``self._play_job_id`` is set per test;
``self.after_cancel`` is a mock (the Tk ``after_cancel`` method is not
exercised).

``errors`` section of the spec: none documented (empty).
"""

from unittest import mock

from gui.video_player import VideoPreviewFrame


def _make_self(play_job_id):
    """Build a ``VideoPreviewFrame`` instance (without running ``__init__``)
    with the dependencies ``_cancel_play_tick`` touches mocked / faked."""
    self = object.__new__(VideoPreviewFrame)
    self._play_job_id = play_job_id
    self.after_cancel = mock.MagicMock(name="after_cancel")
    return self


def test_edge_01():
    """
    input: self._play_job_id が None（構築直後、または既にtickをキャンセルした後に再度呼び出した場合）
    expected: after_cancel は呼び出されず、_play_job_id は None のまま、戻り値 None
    """
    self = _make_self(None)
    ret = self._cancel_play_tick()
    assert ret is None
    assert self._play_job_id is None
    self.after_cancel.assert_not_called()


def test_edge_02():
    """
    input: self._play_job_id が有効な after() ジョブID（再生tickが予約済み）
    expected: self.after_cancel(当該ジョブID) が1回呼び出され、_play_job_id は None となり、戻り値 None
    """
    self = _make_self(424242)
    ret = self._cancel_play_tick()
    assert ret is None
    assert self._play_job_id is None
    self.after_cancel.assert_called_once_with(424242)
