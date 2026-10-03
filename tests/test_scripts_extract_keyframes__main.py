"""Tests for ``scripts.extract_keyframes.main``.

Specification: docs/00-Architecture/scripts_extract_keyframes__main.yaml

``main`` decides keep frames with the current pipeline (comparison
against the last-kept grayscale, threshold 0.1), saves the first 40
kept frames to ``OUTDIR`` as PNG, and returns the 0-based index list
of the kept frames:

- 1. ``cap = cv2.VideoCapture(VIDEO)``; initialize idx=0,
  last_kept_gray=None, prev_gray=None, kept=[]
- 2. loop: ``ret, frame = cap.read()``; break when ret is False
- 3. grayscale ``g = gray(frame)`` (cv2.cvtColor, COLOR_BGR2GRAY)
- 4. if last_kept_gray is None then is_diff=True, otherwise
  ``is_diff = diff_rate(g, last_kept_gray) >= 0.1``
- 5. if is_diff: append idx to kept, ``last_kept_gray = g.copy()``;
  additionally, if ``len(kept) <= 40``, save the color frame with
  ``cv2.imwrite(f'{OUTDIR}\\kept_{idx:04d}.png', frame)``
- 6. assign ``prev_gray = g`` (never read afterwards)
- 7. ``idx += 1`` and return to step 2
- 8. after the loop, ``cap.release()``, print the two lines
  ``current-pipeline kept frames (N):`` and the kept list, and
  return kept

Mocked / stand-in dependencies (per the test-generation rules):
``cv2.VideoCapture`` is patched with ``mock.patch`` to return a
stand-in capture object (``_FakeCap``) whose ``get`` returns a fixed
fps, whose ``read`` yields a scripted sequence of BGR numpy frames
(and ``(False, None)`` at the end), and whose ``release`` is
counted; the real ``VIDEO`` file is never opened. ``OUTDIR`` is
patched to a fresh temporary directory (created per test, removed in
finally) so no PNG is written into the real
``output\\debug_check\\kept_current`` directory. ``gray`` and
``diff_rate`` run on the real cv2/numpy. Frames are monochrome BGR
(R=G=B), which converts to the exact same grayscale value.

``errors`` section of the spec (documented here only, NOT tested, per
the test-generation rules):

- condition: cv2.imwrite が失敗する（OUTDIR パスが不正、権限なし、ディスク満杯など）
  behavior: imwrite の戻り値は未チェックのため例外は発生せず、処理は黙して継続する。kept リストと後続処理には影響しない
- condition: cap.read() が ret=True を返すが frame が None（OpenCV 読み取り結果の異常）
  behavior: gray(frame) 内の cv2.cvtColor が例外を送出する（OpenCV の標準例外は cv2.error。具体的な例外種別は未検証）
"""

import os
import shutil
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np

from scripts.extract_keyframes import main


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


def _frame(value, size=10):
    """Build a ``size``x``size`` monochrome BGR frame (R=G=B=value),
    which converts to exactly ``value`` in grayscale."""
    return np.full((size, size, 3), value, dtype=np.uint8)


def _run_main(frames, outdir):
    """Run ``main`` with a stand-in capture holding the given frames
    and ``OUTDIR`` patched to ``outdir``; return (kept, cap)."""
    cap = _FakeCap(27.5, frames)
    with mock.patch("scripts.extract_keyframes.cv2.VideoCapture",
                    return_value=cap), \
         mock.patch("scripts.extract_keyframes.OUTDIR", str(outdir)):
        kept = main()
    return kept, cap


def _tmp_dir():
    """Create a temporary directory inside the workspace root."""
    return Path(tempfile.mkdtemp(dir=Path(__file__).resolve().parent.parent))


def test_edge_01(capsys):
    """
    input: '読み取り可能なフレームが0枚（cap.read() が初回に ret=False を返す、例: VIDEO ファイル欠損）'
    expected: 'kept は空リストで [] が返る。PNGは1枚も書かれない。stdout は current-pipeline kept frames (0): と空リストの2行が出力される'
    """
    tmp = _tmp_dir()
    try:
        kept, cap = _run_main([], tmp)
        out = capsys.readouterr().out
        assert kept == []
        # print(" ", kept) joins its two arguments with a space, so the
        # list line starts with two spaces.
        assert out == "current-pipeline kept frames (0):\n  []\n"
        assert sorted(os.listdir(tmp)) == []
        assert cap.released is True
        assert cap.read_count == 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_edge_02(capsys):
    """
    input: '読み取り可能なフレームが1枚のみ'
    expected: 'kept = [0]、OUTDIR\\kept_0000.png が1枚書かれ、[0] が返る'
    """
    tmp = _tmp_dir()
    try:
        kept, cap = _run_main([_frame(0)], tmp)
        out = capsys.readouterr().out
        assert kept == [0]
        assert out == "current-pipeline kept frames (1):\n  [0]\n"
        assert sorted(os.listdir(tmp)) == ["kept_0000.png"]
        assert cap.released is True
        assert cap.read_count == 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_edge_03(capsys):
    """
    input: '読み取り可能なフレームが100枚で、各フレームの直前last-keptとの diff_rate が全て 0.1 以上'
    expected: 'kept = [0..99] の全100個。PNGは先頭40個のフレーム分のみ kept_{idx:04d}.png として書かれ、41個目以降はファイルなし'
    """
    tmp = _tmp_dir()
    try:
        frames = [_frame(0 if i % 2 == 0 else 255) for i in range(100)]
        kept, cap = _run_main(frames, tmp)
        out = capsys.readouterr().out
        assert kept == list(range(100))
        assert out == f"current-pipeline kept frames (100):\n  {kept}\n"
        files = sorted(os.listdir(tmp))
        assert files == sorted(f"kept_{i:04d}.png" for i in range(40))
        assert "kept_0040.png" not in files
        assert cap.released is True
        assert cap.read_count == 100
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_edge_04(capsys):
    """
    input: '全フレームのグレースケールが同一（後続フレームの diff_rate が全て 0）'
    expected: 'kept = [0]（先頭フレームのみkeep）、OUTDIR\\kept_0000.png 以外のファイルは書かれない'
    """
    tmp = _tmp_dir()
    try:
        frames = [_frame(128) for _ in range(5)]
        kept, cap = _run_main(frames, tmp)
        out = capsys.readouterr().out
        assert kept == [0]
        assert out == "current-pipeline kept frames (1):\n  [0]\n"
        assert sorted(os.listdir(tmp)) == ["kept_0000.png"]
        assert cap.released is True
        assert cap.read_count == 5
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_edge_05(capsys):
    """
    input: 'あるフレームのグレースケールと直前last-keptの diff_rate がちょうど 0.1'
    expected: 'そのフレームは keep 判定される（比較は >= であり 0.1 でも keep）'
    """
    tmp = _tmp_dir()
    try:
        # frame1: 10 of 100 pixels differ from frame0 -> 10/100 == 0.1
        f1 = np.zeros((10, 10, 3), dtype=np.uint8)
        f1[:2, :5] = 255
        kept, cap = _run_main([_frame(0), f1], tmp)
        out = capsys.readouterr().out
        assert kept == [0, 1]
        assert out == "current-pipeline kept frames (2):\n  [0, 1]\n"
        assert cap.released is True
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_edge_06(capsys):
    """
    input: 'keep 判定されたフレームが40個目の場合'
    expected: 'そのフレームの kept_{idx:04d}.png が書き出される（len(kept) == 40 が <= 40 を満たす）'
    """
    tmp = _tmp_dir()
    try:
        frames = [_frame(0 if i % 2 == 0 else 255) for i in range(40)]
        kept, cap = _run_main(frames, tmp)
        out = capsys.readouterr().out
        assert kept == list(range(40))
        assert out == f"current-pipeline kept frames (40):\n  {kept}\n"
        files = sorted(os.listdir(tmp))
        assert files == sorted(f"kept_{i:04d}.png" for i in range(40))
        assert "kept_0039.png" in files
        assert cap.released is True
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_edge_07(capsys):
    """
    input: 'keep 判定されたフレームが41個目の場合'
    expected: 'idx は kept に追加されるが PNG ファイルは書き出されない（len(kept) == 41 で <= 40 を満たさない）'
    """
    tmp = _tmp_dir()
    try:
        frames = [_frame(0 if i % 2 == 0 else 255) for i in range(41)]
        kept, cap = _run_main(frames, tmp)
        out = capsys.readouterr().out
        assert kept == list(range(41))
        assert out == f"current-pipeline kept frames (41):\n  {kept}\n"
        files = sorted(os.listdir(tmp))
        assert files == sorted(f"kept_{i:04d}.png" for i in range(40))
        assert "kept_0040.png" not in files
        assert cap.released is True
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_edge_08(capsys):
    """
    input: 'keep 判定されたフレームの idx が 10000 以上'
    expected: 'ファイル名は kept_10000.png のように5桁になる（:04d は最小幅指定であり切り詰めではない）'
    """
    tmp = _tmp_dir()
    try:
        # frame0 kept; frames 1..9999 identical to frame0 (skipped);
        # frame 10000 differs -> kept = [0, 10000]
        frames = [_frame(0)] + [_frame(0) for _ in range(9999)] + [_frame(255)]
        kept, cap = _run_main(frames, tmp)
        out = capsys.readouterr().out
        assert kept == [0, 10000]
        assert out == "current-pipeline kept frames (2):\n  [0, 10000]\n"
        files = sorted(os.listdir(tmp))
        assert files == ["kept_0000.png", "kept_10000.png"]
        assert cap.released is True
        assert cap.read_count == 10001
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
