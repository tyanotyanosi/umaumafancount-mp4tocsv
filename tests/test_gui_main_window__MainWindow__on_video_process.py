"""Tests for ``gui.main_window.MainWindow._on_video_process``.

Specification: docs/00-Architecture/gui_main_window__MainWindow__on_video_process.yaml

``MainWindow._on_video_process`` is the video-processing-start callback
(``def _on_video_process(self, video_path: str)``). Per the spec's
``purpose`` / ``behavior`` fields it:

- sets ``self.status_label`` text to 「処理中...」
- sets ``self.video_preview.btn_process`` state to ``"disabled"``
- reads ``video.diff_only`` from ``self.settings`` (default ``False``)
  and converts it with ``bool()`` into ``diff_only``
- ``interval = 0.0`` if ``diff_only`` is true, otherwise
  ``interval = float(self.settings`` の ``video.frame_interval``、既定 1.0``)``
- ``ocr_engine = str(self.settings`` の ``ocr.engine``、既定 "meiki")``
- imports ``threading`` locally; the remaining body (line 130 onwards)
  is unconfirmed per the spec, so no return value / final UI state is
  guaranteed

The method touches GUI widgets and ``threading``, so the tests below
build a bare instance with ``MainWindow.__new__(MainWindow)`` (skipping
``__init__``), attach only the attributes the spec's preconditions
require (``settings`` / ``status_label`` / ``video_preview.btn_process``)
as minimal fakes, and patch ``threading.Thread`` with
``unittest.mock`` so no real thread is started. The local variables
``diff_only`` / ``interval`` / ``ocr_engine`` are not observable from
outside the method, so each test asserts the observable postconditions
(status text, button state) or the exception the spec documents.

``errors`` section of the spec (documented here only, NOT tested, per the
test-generation rules):

- condition: 'self.settings が dict ではない（例: 設定ファイルのトップレベルがリストやスカラー）'
  behavior: AttributeError（非 dict への .get 呼び出し）
- condition: self.settings["video"] が存在するが None または非 dict
  behavior: AttributeError（非 dict への .get 呼び出し）
- condition: diff_only が偽で、self.settings["video"]["frame_interval"] が float 変換不能（非数値文字列・None など）
  behavior: ValueError または TypeError（float() 変換で送出）
- condition: self.status_label または self.video_preview.btn_process が存在しない
  behavior: 参照時点の AttributeError
"""

import pytest
from unittest import mock

from gui.main_window import MainWindow


class _FakeStatusLabel:
    """Minimal stand-in for the window status label.

    Accepts a PyQt-style ``setText(...)`` call, a tkinter-style
    ``configure(text=...)`` call, or a direct ``text`` attribute
    assignment; all are recorded on ``self.text``.
    """

    def __init__(self):
        self.text = ""

    def setText(self, text):
        self.text = str(text)

    def configure(self, **kwargs):
        if "text" in kwargs:
            self.text = str(kwargs["text"])


class _FakeProcessButton:
    """Minimal stand-in for ``VideoPreviewFrame.btn_process``.

    Accepts either a PyQt-style ``setEnabled(False)`` call or a direct
    ``state`` assignment, normalizing both to ``state == "disabled"``
    per the spec's postcondition.
    """

    def __init__(self):
        self.state = "normal"

    def setEnabled(self, enabled):
        self.state = "normal" if enabled else "disabled"

    def configure(self, **kwargs):
        if "state" in kwargs:
            self.state = str(kwargs["state"])


def _make_window(settings):
    """Return a bare ``MainWindow`` (``__init__`` not executed) carrying
    only the attributes the spec's preconditions require."""
    window = MainWindow.__new__(MainWindow)
    window.settings = settings
    window.status_label = _FakeStatusLabel()
    window.video_preview = mock.MagicMock(btn_process=_FakeProcessButton())
    return window


def test_edge_01():
    """
    input: self.settings == {}。video_path が "v.mp4"
    expected: ステータスラベルが「処理中...」になり btn_process が disabled になる。interval == 1.0、ocr_engine == "meiki"（diff_only は偽）
    """
    window = _make_window({})
    with mock.patch("threading.Thread"):
        window._on_video_process("v.mp4")
    assert window.status_label.text == "処理中..."
    assert window.video_preview.btn_process.state == "disabled"


def test_edge_02():
    """
    input: self.settings == {"video": {"diff_only": True}}
    expected: interval == 0.0（frame_interval は参照されない）、ocr_engine == "meiki"
    """
    window = _make_window({"video": {"diff_only": True}})
    with mock.patch("threading.Thread"):
        window._on_video_process("v.mp4")
    assert window.status_label.text == "処理中..."
    assert window.video_preview.btn_process.state == "disabled"


def test_edge_03():
    """
    input: self.settings == {"video": {"frame_interval": 2.5}, "ocr": {"engine": "tesseract"}}
    expected: interval == 2.5、ocr_engine == "tesseract"
    """
    window = _make_window({"video": {"frame_interval": 2.5}, "ocr": {"engine": "tesseract"}})
    with mock.patch("threading.Thread"):
        window._on_video_process("v.mp4")
    assert window.status_label.text == "処理中..."
    assert window.video_preview.btn_process.state == "disabled"


def test_edge_04():
    """
    input: self.settings == {"ocr": {"engine": 3}}
    expected: ocr_engine == "3"（str 変換される）
    """
    window = _make_window({"ocr": {"engine": 3}})
    with mock.patch("threading.Thread"):
        window._on_video_process("v.mp4")
    assert window.status_label.text == "処理中..."
    assert window.video_preview.btn_process.state == "disabled"


def test_edge_05():
    """
    input: self.settings == {"video": {"diff_only": "off"}}（空でない文字列）
    expected: bool("off") は真のため diff_only が真となり interval == 0.0
    """
    window = _make_window({"video": {"diff_only": "off"}})
    with mock.patch("threading.Thread"):
        window._on_video_process("v.mp4")
    assert window.status_label.text == "処理中..."
    assert window.video_preview.btn_process.state == "disabled"


def test_edge_06():
    """
    input: self.settings == {"video": None}
    expected: AttributeError（None への .get 呼び出し）
    """
    window = _make_window({"video": None})
    with pytest.raises(AttributeError):
        window._on_video_process("v.mp4")


def test_edge_07():
    """
    input: self.settings == {"video": {"frame_interval": "abc"}}（diff_only は偽）
    expected: ValueError（float("abc") の変換失敗）
    """
    window = _make_window({"video": {"frame_interval": "abc"}})
    with pytest.raises(ValueError):
        window._on_video_process("v.mp4")
