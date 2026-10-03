"""Tests for ``gui.main_window.MainWindow._process_video``.

Specification: docs/00-Architecture/gui_main_window__MainWindow__process_video.yaml

Per the spec's ``purpose``, ``_process_video`` shows a video file selection
dialog, and if a file is selected, records it in ``self.video_path``,
updates the status bar, and loads the video into the preview frame:

- calls ``ctk.filedialog.askopenfilename(title="動画を選択", filetypes=[("MP4ファイル", "*.mp4"), ("すべてのファイル", "*.*")])``
  and stores the result in ``video_path``
- if ``video_path`` is falsy (empty string, i.e. the dialog was cancelled),
  it returns immediately
- otherwise: assigns ``video_path`` to ``self.video_path``, sets the text of
  ``self.status_label`` to ``f"動画選択: {video_path}"``, and calls
  ``self.video_preview.load_video(video_path)``
- returns None

The ``MainWindow`` instance is built per the spec's ``preconditions``
(``_setup_ui`` executed: ``self.video_path`` is None, ``self.status_label``
and ``self.video_preview`` exist); ``ctk.filedialog.askopenfilename``
(the modal dialog dependency) is mocked with ``unittest.mock``, and
``self.video_preview`` is mocked (the VideoPreviewFrame implementation is
out of the reading scope, per the spec's ``missing``). ``self.status_label``
is a real ``tkinter.Label`` on a minimal ``tkinter.Tk()`` root window,
which is destroyed at the end of each test.

``errors`` section of the spec (documented here only, NOT tested, per the
test-generation rules):

- condition: 'self.status_label または self.video_preview が存在せず、ユーザーがファイルを選択した場合'
  behavior: '参照時点の AttributeError'
- condition: 'self.video_preview.load_video が例外を送出（ファイルが開けない・デコード不能など）'
  behavior: '例外が呼び出し元（tkinter のボタンのコマンドチェーン）へそのまま送出される'
"""

import tkinter as tk
import unittest.mock

from gui.main_window import MainWindow


def test_edge_01():
    r"""
    input: self（video_path が None）。ユーザーがダイアログをキャンセル
    expected: 戻り値 None。self.video_path は None のまま、ステータスラベルのテキストは更新されず、load_video は呼び出されない
    """
    root = tk.Tk()
    root.withdraw()
    try:
        window = MainWindow.__new__(MainWindow)
        window.video_path = None
        window.status_label = tk.Label(root, text="")
        window.video_preview = unittest.mock.Mock(name="video_preview")
        with unittest.mock.patch(
            "gui.main_window.ctk.filedialog.askopenfilename", return_value=""
        ) as ask:
            result = window._process_video()
        assert result is None
        ask.assert_called_once_with(
            title="動画を選択",
            filetypes=[("MP4ファイル", "*.mp4"), ("すべてのファイル", "*.*")],
        )
        assert window.video_path is None
        assert window.status_label.cget("text") == ""
        window.video_preview.load_video.assert_not_called()
    finally:
        root.destroy()


def test_edge_02():
    r"""
    input: self（video_path が "C:\old\a.mp4"）。ユーザーがダイアログで "C:\videos\b.mp4" を選択
    expected: self.video_path が "C:\videos\b.mp4" となり、ステータスラベルのテキストが "動画選択: C:\videos\b.mp4" となり、load_video が "C:\videos\b.mp4" を引数に1回呼び出される
    """
    root = tk.Tk()
    root.withdraw()
    try:
        window = MainWindow.__new__(MainWindow)
        window.video_path = "C:\\old\\a.mp4"
        window.status_label = tk.Label(root, text="")
        window.video_preview = unittest.mock.Mock(name="video_preview")
        with unittest.mock.patch(
            "gui.main_window.ctk.filedialog.askopenfilename",
            return_value="C:\\videos\\b.mp4",
        ) as ask:
            result = window._process_video()
        assert result is None
        ask.assert_called_once_with(
            title="動画を選択",
            filetypes=[("MP4ファイル", "*.mp4"), ("すべてのファイル", "*.*")],
        )
        assert window.video_path == "C:\\videos\\b.mp4"
        assert window.status_label.cget("text") == "動画選択: C:\\videos\\b.mp4"
        window.video_preview.load_video.assert_called_once_with("C:\\videos\\b.mp4")
    finally:
        root.destroy()


def test_edge_03():
    r"""
    input: self。ユーザーが「すべてのファイル」ビューから .mp4 以外のファイル（例: a.txt）を選択
    expected: このメソッドは拡張子を検証しない。self.video_path が "a.txt" になり、ステータスラベルは "動画選択: a.txt" に更新され、load_video が "a.txt" を引数に呼び出される
    """
    root = tk.Tk()
    root.withdraw()
    try:
        window = MainWindow.__new__(MainWindow)
        window.video_path = None
        window.status_label = tk.Label(root, text="")
        window.video_preview = unittest.mock.Mock(name="video_preview")
        with unittest.mock.patch(
            "gui.main_window.ctk.filedialog.askopenfilename", return_value="a.txt"
        ) as ask:
            result = window._process_video()
        assert result is None
        ask.assert_called_once_with(
            title="動画を選択",
            filetypes=[("MP4ファイル", "*.mp4"), ("すべてのファイル", "*.*")],
        )
        assert window.video_path == "a.txt"
        assert window.status_label.cget("text") == "動画選択: a.txt"
        window.video_preview.load_video.assert_called_once_with("a.txt")
    finally:
        root.destroy()
