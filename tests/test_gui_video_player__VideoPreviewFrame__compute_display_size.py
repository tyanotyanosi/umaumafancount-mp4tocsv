"""Tests for ``gui.video_player.VideoPreviewFrame._compute_display_size``.

Specification: docs/00-Architecture/gui_video_player__VideoPreviewFrame__compute_display_size.yaml

``_compute_display_size`` computes the preview display size that fits the
video's native resolution inside the current widget size while preserving
the aspect ratio, and stores it in ``self.disp_width`` /
``self.disp_height``:

- Step 1: ``base_w = self.winfo_width()`` when > 0, otherwise 640
- Step 2: ``base_h = self.winfo_height()`` when > 0, otherwise 360
- Step 3: when ``native_width <= 0`` or ``native_height <= 0``, set
  ``disp_width = base_w``, ``disp_height = base_h`` and return
- Step 4: ``ratio = min(base_w/native_width, base_h/native_height)``
- Step 5: ``disp_width = max(1, int(native_width * ratio))`` and
  ``disp_height = max(1, int(native_height * ratio))``

``self.winfo_width()`` and ``self.winfo_height()`` are called twice when the
return value is > 0 (condition part and value part), but only once when the
return value is 0 (short-circuit evaluation skips the value part). The
function returns ``None``.

Mocked / stand-in dependencies (per the test-generation rules): the
``VideoPreviewFrame`` instance is created without running ``__init__``
(``object.__new__``); ``self.native_width`` / ``self.native_height`` are set
per test; ``self.winfo_width`` / ``self.winfo_height`` are replaced by plain
functions returning the controlled values and counting their calls (no Tk
window is involved, so no display is required).

``errors`` section of the spec (documented here only, NOT tested, per the
test-generation rules):

- condition: "self に native_width / native_height 属性が存在しない（load_video 未呼び出し、または属性が削除済み）"
  behavior: "手順3 の self.native_width（または native_height）参照で AttributeError が発生し、disp_width / disp_height は設定されない"
- condition: "self.winfo_width() / winfo_height() が例外を送出する（例: ウィジェットがすでに破棄済み）"
  behavior: "try/except なし。tkinter 由来の例外が呼び出し元へそのまま伝播する（具体的な例外種別は未確認）"
- condition: "native_width または native_height が 0 と比較できない型（例: None）である"
  behavior: "手順3 の <= 比較で TypeError が発生し、disp_width / disp_height は設定されない"
"""

from gui.video_player import VideoPreviewFrame


def _make_self(native_width, native_height, winfo_w, winfo_h):
    """Build a ``VideoPreviewFrame`` instance (without running ``__init__``)
    with the native size and winfo values controlled per test; returns the
    instance and the winfo call counters."""
    self = object.__new__(VideoPreviewFrame)
    self.native_width = native_width
    self.native_height = native_height
    calls = {"w": 0, "h": 0}

    def winfo_width():
        calls["w"] += 1
        return winfo_w

    def winfo_height():
        calls["h"] += 1
        return winfo_h

    self.winfo_width = winfo_width
    self.winfo_height = winfo_height
    return self, calls


def test_edge_01():
    """
    input: native_width=0, native_height=720; winfo_width()=1000, winfo_height()=600
    expected: 手順3のフォールバック: disp_width==1000, disp_height==600
    """
    self, calls = _make_self(0, 720, 1000, 600)
    ret = self._compute_display_size()
    assert ret is None
    assert self.disp_width == 1000
    assert self.disp_height == 600
    assert calls == {"w": 2, "h": 2}


def test_edge_02():
    """
    input: native_width=0, native_height=0（load_video 失敗後の値）; winfo_width()=0, winfo_height()=0
    expected: disp_width==640, disp_height==360（フォールバックとデフォルト値の両方が適用）
    """
    self, calls = _make_self(0, 0, 0, 0)
    ret = self._compute_display_size()
    assert ret is None
    assert self.disp_width == 640
    assert self.disp_height == 360
    # winfo が 0 を返すため短絡評価で値取得部がスキップされ、各 1 回のみ
    assert calls == {"w": 1, "h": 1}


def test_edge_03():
    """
    input: native_width=1920, native_height=1080; winfo_width()=960, winfo_height()=540
    expected: ratio==min(0.5, 0.5)==0.5 として disp_width==960, disp_height==540
    """
    self, calls = _make_self(1920, 1080, 960, 540)
    ret = self._compute_display_size()
    assert ret is None
    assert self.disp_width == 960
    assert self.disp_height == 540
    assert calls == {"w": 2, "h": 2}


def test_edge_04():
    """
    input: native_width=640, native_height=480; winfo_width()=0, winfo_height()=0（未マップ状態など）
    expected: base_w==640, base_h==360 として ratio==min(1.0, 0.75)==0.75、disp_width==480, disp_height==360
    """
    self, calls = _make_self(640, 480, 0, 0)
    ret = self._compute_display_size()
    assert ret is None
    assert self.disp_width == 480
    assert self.disp_height == 360
    # winfo が 0 を返すため短絡評価で値取得部がスキップされ、各 1 回のみ
    assert calls == {"w": 1, "h": 1}


def test_edge_05():
    """
    input: native_width=320, native_height=240; winfo_width()=640, winfo_height()=480
    expected: ratio==min(2.0, 2.0)==2.0 として disp_width==640, disp_height==480（ベースサイズより小さい動画は拡大して収まる）
    """
    self, calls = _make_self(320, 240, 640, 480)
    ret = self._compute_display_size()
    assert ret is None
    assert self.disp_width == 640
    assert self.disp_height == 480
    assert calls == {"w": 2, "h": 2}


def test_edge_06():
    """
    input: native_width=1, native_height=2; winfo_width()=640, winfo_height()=1
    expected: ratio==min(640.0, 0.5)==0.5 として disp_width==max(1, int(0.5))==1, disp_height==max(1, int(1.0))==1（max(1, ...) の下限が有効になる）
    """
    self, calls = _make_self(1, 2, 640, 1)
    ret = self._compute_display_size()
    assert ret is None
    assert self.disp_width == 1
    assert self.disp_height == 1
    assert calls == {"w": 2, "h": 2}
