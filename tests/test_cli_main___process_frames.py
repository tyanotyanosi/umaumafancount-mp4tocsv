"""Tests for ``cli.main._process_frames``.

Specification: docs/00-Architecture/cli_main___process_frames.yaml

``_process_frames`` processes a list of frames in order (card detection +
OCR of each card's name / fan count) and returns a per-frame result list:

- Step 1: initialize ``all_results = []`` and compute
  ``debug_dir = Path(output_dir)/debug``
- Step 2: if debug is True, create the directory with
  ``mkdir(parents=True, exist_ok=True)`` and print
  「デバッグモード: {debug_dir} に画像を保存」 (independent of quiet)
- Step 3: iterate frames as ``(i, frame)``; if quiet is False, print
  「フレーム {i+1}/{len(frames)} 処理中...」
- Step 4: if debug is True, save the frame as ``frame_{i:04d}.png``
- Step 5: get the card list via ``cards = detector.detect(frame)`` and
  initialize ``frame_result = {"cards": []}``
- Step 6: for each card ``(c, card)``: if ``card.name_box`` is truthy,
  crop via ``_extract_crop`` and ``name_raw = name_ocr.recognize(crop)``;
  if name_raw is falsy and ``name_ocr_low`` is not None, overwrite with
  ``name_raw = name_ocr_low.recognize(crop)`` (the second result
  overwrites unconditionally, even if None or falsy)
- Step 7: if ``card.fan_box`` is truthy, crop and
  ``fans_raw = fan_ocr.recognize(crop)``; if falsy, ``fans_raw = None``
- Step 8: append ``{role, name_raw, fans_raw, name_box, fan_box}`` to
  ``frame_result["cards"]``; if quiet is False and ``i < 5``, print
  「  カード{c} ({card.role}): name=..., fans=...」
- Step 9: append frame_result to all_results
- Step 10: return all_results

Mocked / stand-in dependencies (per the test-generation rules):
``detector`` is a mock whose ``detect`` returns a list of stand-in card
objects (with ``role`` / ``name_box`` / ``fan_box`` attributes);
``name_ocr`` / ``name_ocr_low`` / ``fan_ocr`` are mocks whose
``recognize`` returns the configured value; ``cli.main._extract_crop``
is a mock returning a real small numpy image. Frames are real numpy
arrays (cv2-compatible BGR images); ``cv2.imwrite`` and the OCR text
``open(...)`` writes run real in a temporary directory.

``errors`` section of the spec (documented here only, NOT tested, per the
test-generation rules):

- condition: "debug=True の場合で、output_dir/debug が作成できない（親パスがファイル、権限不足等）"
  behavior: "Path.mkdir の例外（FileNotFoundError / PermissionError 等）がそのまま伝播する（関数内に try/except はない）"
- condition: "detector.detect、_extract_crop、3つのOCRエンジンの .recognize、cv2.imwrite、open(...) のいずれかが例外を送出"
  behavior: "例外が呼び出し側にそのまま伝播する（関数内に try/except はない）"
"""

import shutil
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np

from cli.main import _process_frames


def _tmp_dir():
    """Create a temporary directory under the session workspace (the
    platform temp area is not writable for subdirectory creation under
    the file sandbox; the workspace is)."""
    workspace = Path(__file__).resolve().parent.parent
    return Path(tempfile.mkdtemp(dir=workspace))


class _FakeCard:
    """Stand-in for a detected card: exposes ``role`` / ``name_box`` /
    ``fan_box`` attributes."""

    def __init__(self, role="card", name_box=None, fan_box=None):
        self.role = role
        self.name_box = name_box
        self.fan_box = fan_box


def _frame():
    """Create a real small BGR numpy frame."""
    return np.zeros((240, 320, 3), dtype=np.uint8)


def _crop():
    """Create a real small BGR numpy crop."""
    return np.zeros((50, 100, 3), dtype=np.uint8)


def _ocr_mock(result):
    """Create an OCR engine mock whose ``recognize`` returns ``result``."""
    m = mock.MagicMock(name="ocr")
    m.recognize.return_value = result
    return m


def test_edge_01(capsys):
    """
    input: frames=[]（debug=False、quiet=False）
    expected: [] を返す; ループ本体は実行されず、ファイル書き出しも stdout 出力もしない
    """
    detector = mock.MagicMock(name="detector")
    ret = _process_frames([], detector, _ocr_mock("x"), None,
                          _ocr_mock("x"), "output")
    assert ret == []
    assert capsys.readouterr().out == ""


def test_edge_02(capsys):
    """
    input: frames=[F] かつ detector.detect(F) が []（カード0枚）を返す
    expected: [{"cards": []}] を返す
    """
    detector = mock.MagicMock(name="detector")
    detector.detect.return_value = []
    ret = _process_frames([_frame()], detector, _ocr_mock("x"), None,
                          _ocr_mock("x"), "output")
    assert ret == [{"cards": []}]
    assert "フレーム 1/1 処理中..." in capsys.readouterr().out


def test_edge_03(capsys):
    """
    input: name_box=None かつ fan_box=None のカード
    expected: 対応する要素は name_raw=None、fans_raw=None; name_ocr.recognize も fan_ocr.recognize も呼ばれない
    """
    card = _FakeCard(role="card", name_box=None, fan_box=None)
    detector = mock.MagicMock(name="detector")
    detector.detect.return_value = [card]
    name_ocr = _ocr_mock("x")
    fan_ocr = _ocr_mock("x")
    with mock.patch("cli.main._extract_crop", return_value=_crop()):
        ret = _process_frames([_frame()], detector, name_ocr, None,
                              fan_ocr, "output")
    assert ret == [{"cards": [{"role": "card", "name_raw": None,
                              "fans_raw": None, "name_box": None,
                              "fan_box": None}]}]
    name_ocr.recognize.assert_not_called()
    fan_ocr.recognize.assert_not_called()


def test_edge_04(capsys):
    """
    input: name_ocr.recognize が None を返し、name_ocr_low=None
    expected: name_raw は None のまま（フォールバック条件 name_ocr_low is not None を満たさないため name_ocr_low は呼ばれない）
    """
    card = _FakeCard(role="card", name_box=(10, 10, 100, 50),
                     fan_box=None)
    detector = mock.MagicMock(name="detector")
    detector.detect.return_value = [card]
    name_ocr = _ocr_mock(None)
    fan_ocr = _ocr_mock("x")
    with mock.patch("cli.main._extract_crop", return_value=_crop()):
        ret = _process_frames([_frame()], detector, name_ocr, None,
                              fan_ocr, "output")
    assert ret[0]["cards"][0]["name_raw"] is None
    assert ret[0]["cards"][0]["fans_raw"] is None
    name_ocr.recognize.assert_called_once()


def test_edge_05(capsys):
    """
    input: name_ocr.recognize が "" を返し、name_ocr_low が None でないかつその recognize が "Sakura" を返す
    expected: name_raw == "Sakura"
    """
    card = _FakeCard(role="card", name_box=(10, 10, 100, 50),
                     fan_box=None)
    detector = mock.MagicMock(name="detector")
    detector.detect.return_value = [card]
    name_ocr = _ocr_mock("")
    name_ocr_low = _ocr_mock("Sakura")
    fan_ocr = _ocr_mock("x")
    with mock.patch("cli.main._extract_crop", return_value=_crop()):
        ret = _process_frames([_frame()], detector, name_ocr, name_ocr_low,
                              fan_ocr, "output")
    assert ret[0]["cards"][0]["name_raw"] == "Sakura"
    name_ocr.recognize.assert_called_once()
    name_ocr_low.recognize.assert_called_once()


def test_edge_06(capsys):
    """
    input: name_ocr.recognize が "" を返し、name_ocr_low が None でないかつその recognize が None を返す
    expected: name_raw は None（第2 recognize の結果が前値を無条件に上書きするため）
    """
    card = _FakeCard(role="card", name_box=(10, 10, 100, 50),
                     fan_box=None)
    detector = mock.MagicMock(name="detector")
    detector.detect.return_value = [card]
    name_ocr = _ocr_mock("")
    name_ocr_low = _ocr_mock(None)
    fan_ocr = _ocr_mock("x")
    with mock.patch("cli.main._extract_crop", return_value=_crop()):
        ret = _process_frames([_frame()], detector, name_ocr, name_ocr_low,
                              fan_ocr, "output")
    assert ret[0]["cards"][0]["name_raw"] is None
    name_ocr_low.recognize.assert_called_once()


def test_edge_07(capsys):
    """
    input: debug=True、frames=[F]、F のカード1枚に name_box と fan_box が両方あり
    expected: output_dir/debug が作成され、frame_0000.png、frame_0000_card0_name.png、frame_0000_card0_fan.png、frame_0000_card0_name_ocr.txt、frame_0000_card0_fan_ocr.txt が書き出される
    """
    tmp = _tmp_dir()
    try:
        output_dir = str(tmp)
        card = _FakeCard(role="card", name_box=(10, 10, 100, 50),
                         fan_box=(20, 20, 200, 80))
        detector = mock.MagicMock(name="detector")
        detector.detect.return_value = [card]
        name_ocr = _ocr_mock("Sakura")
        fan_ocr = _ocr_mock("123")
        with mock.patch("cli.main._extract_crop", return_value=_crop()):
            ret = _process_frames([_frame()], detector, name_ocr, None,
                                  fan_ocr, output_dir, debug=True)
        debug_dir = tmp / "debug"
        assert debug_dir.is_dir()
        for name in ("frame_0000.png", "frame_0000_card0_name.png",
                     "frame_0000_card0_fan.png",
                     "frame_0000_card0_name_ocr.txt",
                     "frame_0000_card0_fan_ocr.txt"):
            assert (debug_dir / name).is_file()
        assert "デバッグモード: " in capsys.readouterr().out
        assert ret[0]["cards"][0]["name_raw"] == "Sakura"
        assert ret[0]["cards"][0]["fans_raw"] == "123"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_edge_08(capsys):
    """
    input: quiet=True、frames が7枚以上
    expected: 「フレーム ... 処理中...」も行カード詳細行も一切出力されない（カード詳細行は quiet=False でも i<5 に限定される）
    """
    frames = [_frame() for _ in range(7)]
    card = _FakeCard(role="card", name_box=(10, 10, 100, 50),
                     fan_box=None)
    detector = mock.MagicMock(name="detector")
    detector.detect.return_value = [card]
    name_ocr = _ocr_mock("X")
    fan_ocr = _ocr_mock("x")
    with mock.patch("cli.main._extract_crop", return_value=_crop()):
        ret = _process_frames(frames, detector, name_ocr, None,
                              fan_ocr, "output", quiet=True)
    out = capsys.readouterr().out
    assert "処理中" not in out
    assert "カード" not in out
    assert len(ret) == 7
    assert all(r == {"cards": [{"role": "card", "name_raw": "X",
                               "fans_raw": None,
                               "name_box": (10, 10, 100, 50),
                               "fan_box": None}]} for r in ret)
