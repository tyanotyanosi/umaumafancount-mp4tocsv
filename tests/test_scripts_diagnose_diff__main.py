"""Tests for ``scripts.diagnose_diff.main``.

Specification: docs/00-Architecture/scripts_diagnose_diff__main.yaml

``main`` loads the fixed video file ``VIDEO``, measures the diff rate
against the previous frame for all frames, and prints the simulation
results of two keep/skip decisions (previous-frame comparison and
last-kept comparison, both with threshold 0.1):

- ``cap = cv2.VideoCapture(VIDEO)`` and
  ``fps = cap.get(cv2.CAP_PROP_FPS)``
- loop reading frames with ``cap.read()``; stop when ret is False,
  otherwise grayscale via ``gray(frame)``
- path A (from frame 2 on, when prev_gray is not None):
  ``r = diff_rate(g, prev_gray)``; record ``(idx, r)`` in
  rates_prev; if ``r >= 0.1`` append idx to kept_prev
- path B (from frame 3 on, when last_kept_gray is not None):
  ``rl = diff_rate(g, last_kept_gray)``; if ``rl >= 0.1`` append idx
  to kept_lastkept and update ``last_kept_gray = g.copy()``;
  otherwise keep the old reference frame
- per frame: ``prev_gray = g`` and ``idx += 1``
- ``cap.release()``, then print the total frame count and fps
- if rates_prev is non-empty: print the [直前フレーム差分]
  distribution (min/max/avg to 4 decimals) and the frame counts at or
  above the thresholds 0.01, 0.02, 0.05, 0.1, 0.2, 0.5 (6 lines)
- print the kept_lastkept count and full list, the kept_prev count,
  and, if kept_prev is non-empty, its first 10 entries

The function returns None (no explicit return statement).

Mocked / stand-in dependencies (per the test-generation rules):
``cv2.VideoCapture`` is patched with ``mock.patch`` to return a
stand-in capture object (``_FakeCap``) whose ``get`` returns a fixed
fps, whose ``read`` yields a scripted sequence of BGR numpy frames
(and ``(False, None)`` at the end), and whose ``release`` is
counted; the real ``VIDEO`` file is never opened. ``gray`` and
``diff_rate`` run on the real cv2/numpy.

``errors`` section of the spec (documented here only, NOT tested, per
the test-generation rules):

- condition: VIDEOファイルが存在しない、または読込不能
  behavior: 例外は送出されず、cap.read()が(False, ...)を返すためループが即座に終了し "全フレーム: 0" と両keep=0が出力される
- condition: fpsを取得できない（cap.getが0を返す）
  behavior: 例外は送出されず、出力は {fps:.2f} で書式化されるため0の場合は "(fps=0.00)" となる
"""

import numpy as np
from unittest import mock

from scripts.diagnose_diff import main


class _FakeCap:
    """Stand-in cv2.VideoCapture handle with a scripted frame
    sequence and call counters."""

    def __init__(self, fps, frames):
        self.fps = fps
        self.frames = frames
        self._pos = 0
        self.read_count = 0
        self.released = False

    def get(self, prop):
        return self.fps

    def read(self):
        if self._pos >= len(self.frames):
            return (False, None)
        self._pos += 1
        self.read_count += 1
        return (True, self.frames[self._pos - 1])

    def release(self):
        self.released = True


def _run_main(fps, frames):
    """Run ``main`` with a stand-in capture holding the given frames
    and return the capture handle."""
    cap = _FakeCap(fps, frames)
    with mock.patch("scripts.diagnose_diff.cv2.VideoCapture",
                    return_value=cap):
        main()
    return cap


def _frame(value):
    """Build an 8x8 BGR frame with every pixel set to ``value``."""
    return np.full((8, 8, 3), value, dtype=np.uint8)


def test_edge_01(capsys):
    """
    input: 'VIDEOパスの動画が0フレーム（cap.read()が即座にret=Falseを返す）'
    expected: '"全フレーム: 0  (fps=<CAP_PROP_FPSの取得値>)" を出力し、[直前フレーム差分] 分布ブロックは出力されず、[現行ロジック] と [直前フレーム比較] はともにkeep=0を出力し、(先頭10) 行は出力されない'
    """
    cap = _run_main(27.5, [])
    out = capsys.readouterr().out
    assert out == (
        "全フレーム: 0  (fps=27.50)\n"
        "\n[現行ロジック: last-kept 比較, 閾値0.1] keep=0 件\n"
        "  keep フレーム: []\n"
        "\n[直前フレーム比較, 閾値0.1] keep=0 件\n"
    )
    assert cap.released is True
    assert cap.read_count == 0


def test_edge_02(capsys):
    """
    input: 'VIDEOパスの動画が1フレームのみ'
    expected: '"全フレーム: 1  (fps=...)" を出力し、フレーム0は比較対象外のためrates_prev・kept_prevが空で[直前フレーム差分] 分布ブロックは出力されず、両keep=0'
    """
    cap = _run_main(27.5, [_frame(0)])
    out = capsys.readouterr().out
    assert out == (
        "全フレーム: 1  (fps=27.50)\n"
        "\n[現行ロジック: last-kept 比較, 閾値0.1] keep=0 件\n"
        "  keep フレーム: []\n"
        "\n[直前フレーム比較, 閾値0.1] keep=0 件\n"
    )
    assert cap.released is True
    assert cap.read_count == 1


def test_edge_03(capsys):
    """
    input: '3フレームの動画で、フレーム0が全ピクセル0、フレーム1が全ピクセル255、フレーム2が全ピクセル0（単色グレースケール）'
    expected: 'kept_prev=[1, 2] かつ kept_lastkept=[]（フレーム1はパスBで比較されない。フレーム2は基準フレーム0との差分が0.0 < 0.1）となり、[現行ロジック] keep=0、[直前フレーム比較] keep=2 で (先頭10)=[1, 2] を出力する'
    """
    cap = _run_main(27.5, [_frame(0), _frame(255), _frame(0)])
    out = capsys.readouterr().out
    assert out == (
        "全フレーム: 3  (fps=27.50)\n"
        "\n[直前フレーム差分] 分布:\n"
        "  min=1.0000  max=1.0000  avg=1.0000\n"
        "  閾値 0.01 以上: 2 フレーム\n"
        "  閾値 0.02 以上: 2 フレーム\n"
        "  閾値 0.05 以上: 2 フレーム\n"
        "  閾値 0.10 以上: 2 フレーム\n"
        "  閾値 0.20 以上: 2 フレーム\n"
        "  閾値 0.50 以上: 2 フレーム\n"
        "\n[現行ロジック: last-kept 比較, 閾値0.1] keep=0 件\n"
        "  keep フレーム: []\n"
        "\n[直前フレーム比較, 閾値0.1] keep=2 件\n"
        "  (先頭10): [1, 2]\n"
    )
    assert cap.released is True
    assert cap.read_count == 3


def test_edge_04(capsys):
    """
    input: '2フレーム以上で全フレームの連続フレーム間差分率が0.1未満（例: 静止画動画）'
    expected: 'kept_prev=[] かつ kept_lastkept=[] で両keep=0。rates_prevは非空のため[直前フレーム差分] 分布ブロックは出力され、各閾値以上のフレーム数は0'
    """
    cap = _run_main(27.5, [_frame(128), _frame(128), _frame(128)])
    out = capsys.readouterr().out
    assert out == (
        "全フレーム: 3  (fps=27.50)\n"
        "\n[直前フレーム差分] 分布:\n"
        "  min=0.0000  max=0.0000  avg=0.0000\n"
        "  閾値 0.01 以上: 0 フレーム\n"
        "  閾値 0.02 以上: 0 フレーム\n"
        "  閾値 0.05 以上: 0 フレーム\n"
        "  閾値 0.10 以上: 0 フレーム\n"
        "  閾値 0.20 以上: 0 フレーム\n"
        "  閾値 0.50 以上: 0 フレーム\n"
        "\n[現行ロジック: last-kept 比較, 閾値0.1] keep=0 件\n"
        "  keep フレーム: []\n"
        "\n[直前フレーム比較, 閾値0.1] keep=0 件\n"
    )
    assert cap.released is True
    assert cap.read_count == 3
