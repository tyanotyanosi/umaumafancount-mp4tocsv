"""Tests for ``gui.video_player.VideoPreviewFrame._on_process``.

Specification: docs/00-Architecture/gui_video_player__VideoPreviewFrame__on_process.yaml

``_on_process`` is the "process start" button callback: if a configured
external process callback exists, it calls it once with the current video
path as the sole positional argument:

- Step 1: check whether ``self.on_process`` is truthy (not None)
- Step 2: if None, return without doing anything
- Step 3: if set, call ``self.on_process(self.video_path)`` (``video_path``
  is ``None`` when ``load_video`` was never called, and that is passed
  through as-is)

The function does not modify ``self``'s state and returns ``None``.

Mocked / stand-in dependencies (per the test-generation rules): the
``VideoPreviewFrame`` instance is created without running ``__init__``
(``object.__new__``); ``self.on_process`` is a mock callable (or a plain
function raising ``ValueError`` for edge case 4, or ``None``);
``self.video_path`` is set per test. No other dependencies are touched.

``errors`` section of the spec (documented here only, NOT tested, per the
test-generation rules):

- condition: "self.on_process に設定済みのコールバックが例外を送出した場合。"
  behavior: "例外は _on_process で捕捉されず、呼び出し元（ボタンの command コールバックとしての Tk イベント処理）へ伝播する"
- condition: "関数本体に明示的なエラー処理・raise がない。"
  behavior: "コールバック起因の例外を除き、_on_process 自体は例外を送出しない"
"""

from unittest import mock

import pytest

from gui.video_player import VideoPreviewFrame


def _make_self(on_process, video_path):
    """Build a ``VideoPreviewFrame`` instance (without running ``__init__``)
    with ``on_process`` and ``video_path`` set per test."""
    self = object.__new__(VideoPreviewFrame)
    self.on_process = on_process
    self.video_path = video_path
    return self


def test_edge_01():
    """
    input: self.on_process=None, self.video_path='sample.mp4'（コールバック未設定）
    expected: コールバックは呼び出されず、状態は不変、戻り値 None
    """
    self = _make_self(None, "sample.mp4")
    ret = self._on_process()
    assert ret is None
    assert self.on_process is None
    assert self.video_path == "sample.mp4"


def test_edge_02():
    """
    input: self.on_process=mock の callable、self.video_path='sample.mp4'（load_video 済み）
    expected: self.on_process が引数 ('sample.mp4',) で正確に1回呼び出され、戻り値 None
    """
    cb = mock.MagicMock(name="on_process")
    self = _make_self(cb, "sample.mp4")
    ret = self._on_process()
    assert ret is None
    cb.assert_called_once_with("sample.mp4")


def test_edge_03():
    """
    input: self.on_process=mock の callable、self.video_path=None（動画未ロード）
    expected: self.on_process が引数 (None,) で正確に1回呼び出され（コードに None ガードは無く、None がそのまま渡される）、戻り値 None
    """
    cb = mock.MagicMock(name="on_process")
    self = _make_self(cb, None)
    ret = self._on_process()
    assert ret is None
    cb.assert_called_once_with(None)


def test_edge_04():
    """
    input: self.on_process=ValueError を送出する callable、self.video_path='sample.mp4'
    expected: _on_process は例外を捕捉せず、ValueError が呼び出し元へ伝播する
    """
    def cb(path):
        raise ValueError("boom")

    self = _make_self(cb, "sample.mp4")
    with pytest.raises(ValueError):
        self._on_process()
