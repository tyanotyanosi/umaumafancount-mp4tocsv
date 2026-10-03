"""Tests for ``scripts.make_golden.main``.

Specification: docs/00-Architecture/scripts_make_golden__main.yaml

``main`` detects sample frames with CardDetector and regenerates the
golden expectation file ``tests/golden/detect_expected.json``
(display only with ``--dry-run``):

- parse the command line with argparse; the only option is
  ``--dry-run`` (action=store_true)
- ``settings = load_settings()`` and
  ``detector = CardDetector(str(TEMPLATE_DIR), settings)``
- glob ``sample_*.png`` in FRAMES_DIR and sort by path; if the result
  is empty, raise SystemExit with the message「サンプルフレームが
  {FRAMES_DIR} に存在しません」
- for each frame path (sorted): read BGR with
  ``cv2.imread(frame_path, cv2.IMREAD_COLOR)``; if None, raise
  SystemExit with「フレームを読み込めません: {frame_path}」
- ``detector.detect(frame)`` -> cards; append
  ``{"file": frame_path.name, "shape": [int(h), int(w), 3],
  "cards": to_golden_cards(cards)}`` to golden
- print「{frame_path.name}: {n} カード」per frame
- if ``--dry-run``: print the notice line and the whole golden as
  JSON (json.dumps, ensure_ascii=False, indent=1) and return
- otherwise: if detect_expected.json exists, load the old content and
  print「注意: 既存のゴールデンと検出結果が異なる（上書きします）。」
  when different, or「ゴールデンは不変（検出結果が一致）」when equal
- write the new result JSON (ensure_ascii=False, indent=1, encoding
  utf-8) to detect_expected.json and print「書き込み完了: {EXPECTED}」

Mocked / stand-in dependencies (per the test-generation rules):
- ``sys.argv`` is patched for argparse (``--dry-run`` / ``--foo``)
- ``load_settings`` is patched to return a fixed dict (the real
  config/settings.yaml is not read)
- ``CardDetector`` is patched with a ``_FakeDetector`` stand-in whose
  ``detect`` returns scripted card lists (the real detector
  implementation is out of read scope per the spec's ``missing``
  section)
- ``cv2.imread`` is patched with a scripted result (no real PNG
  decoding); frames are plain numpy uint8 arrays
- ``FRAMES_DIR`` and ``EXPECTED`` are patched to fresh temporary
  paths (created per test, removed in finally) so the real
  ``tests/golden`` files are never touched
- card objects are ``types.SimpleNamespace`` stand-ins

``errors`` section of the spec (documented here only, NOT tested, per
the test-generation rules):

- condition: サンプルフレームが0件（glob 結果が空）
  behavior: メッセージ「サンプルフレームが {FRAMES_DIR} に存在しません」で SystemExit。
- condition: サンプルフレームの cv2.imread が None を返す
  behavior: メッセージ「フレームを読み込めません: {frame_path}」で SystemExit。
- condition: config/settings.yaml が存在しない、または YAML エラー
  behavior: load_settings からの例外（FileNotFoundError、yaml.YAMLError 等）がフレーム処理の前に伝播する。
- condition: 既存 detect_expected.json が無効な JSON
  behavior: 全フレームの検出の後・新ファイル書き込みの前に json.JSONDecodeError が伝播する。既存ファイルは変更されない。
- condition: 未知のコマンドラインオプション
  behavior: argparse による SystemExit(2)。
"""

import json
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

import scripts.make_golden as m
from scripts.make_golden import main


class _FakeDetector:
    """Stand-in CardDetector returning scripted card lists."""

    def __init__(self, template_dir, settings, card_lists):
        self.template_dir = template_dir
        self.settings = settings
        self.card_lists = list(card_lists)
        self.detect_calls = []

    def detect(self, frame):
        self.detect_calls.append(frame)
        return self.card_lists.pop(0)


def _card(role, badge_box, name_box, fan_box):
    """Build a SimpleNamespace stand-in card object."""
    return SimpleNamespace(role=role, badge_box=badge_box,
                           name_box=name_box, fan_box=fan_box)


def _fake_frame():
    """A 12x16 BGR frame (h=12, w=16)."""
    return np.zeros((12, 16, 3), dtype=np.uint8)


def _one_card():
    return _card("r", (1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12))


def _golden_for(card_lists, frame_names):
    """Build the golden structure main() would produce for the given
    frames (all 12x16)."""
    golden = []
    for name, cards in zip(frame_names, card_lists):
        golden.append({
            "file": name,
            "shape": [12, 16, 3],
            "cards": m.to_golden_cards(cards),
        })
    return golden


@contextmanager
def _patched(argv, frames_dir, expected_path, imread_map, card_lists):
    """Patch sys.argv, load_settings, CardDetector, FRAMES_DIR,
    EXPECTED and cv2.imread; yield the fake detector."""
    settings = {"cfg": 1}
    detector = _FakeDetector("template-dir", settings, card_lists)
    with mock.patch("sys.argv", argv), \
         mock.patch("scripts.make_golden.load_settings",
                    return_value=settings), \
         mock.patch("scripts.make_golden.CardDetector",
                    return_value=detector), \
         mock.patch("scripts.make_golden.FRAMES_DIR", frames_dir), \
         mock.patch("scripts.make_golden.EXPECTED", expected_path), \
         mock.patch.object(m.cv2, "imread",
                           side_effect=lambda path, flag: imread_map[path]):
        yield detector


def _tmp_dir():
    """Create a temporary directory inside the workspace root."""
    return Path(tempfile.mkdtemp(dir=Path(__file__).resolve().parent.parent))


def _make_frames(frames_dir, names):
    """Create empty sample_*.png files in frames_dir."""
    frames_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in names:
        p = frames_dir / name
        p.write_bytes(b"")
        paths.append(p)
    return paths


def test_edge_01(capsys):
    """
    input: 'FRAMES_DIR に sample_*.png が0件'
    expected: '「サンプルフレームが {FRAMES_DIR} に存在しません」（実際には絶対パスが埋め込まれた）メッセージで SystemExit。ファイルは一切書かれない。'
    """
    tmp = _tmp_dir()
    try:
        frames_dir = tmp / "frames"
        expected_path = tmp / "detect_expected.json"
        with _patched(["make_golden"], frames_dir, expected_path, {}, []):
            with pytest.raises(SystemExit) as excinfo:
                main()
        assert str(excinfo.value) == (
            f"サンプルフレームが {frames_dir} に存在しません"
        )
        assert not expected_path.exists()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_edge_02(capsys):
    """
    input: 'cv2.imread が読み込めないサンプル PNG（破損など）'
    expected: '「フレームを読み込めません: {frame_path}」で SystemExit。処理は中断し、ファイルは書かれない（以降のフレームは未処理）。'
    """
    tmp = _tmp_dir()
    try:
        frames_dir = tmp / "frames"
        expected_path = tmp / "detect_expected.json"
        paths = _make_frames(frames_dir, ["sample_a.png", "sample_b.png"])
        imread_map = {str(paths[0]): _fake_frame(), str(paths[1]): None}
        with _patched(["make_golden"], frames_dir, expected_path,
                       imread_map, [[_one_card()], []]):
            with pytest.raises(SystemExit) as excinfo:
                main()
        assert str(excinfo.value) == f"フレームを読み込めません: {paths[1]}"
        assert not expected_path.exists()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_edge_03(capsys):
    """
    input: '--dry-run で全フレームを読み込める場合'
    expected: '通知行と golden の JSON が標準出力に出る。detect_expected.json は作成も変更もしない（既存ファイルがあっても触れない）。'
    """
    tmp = _tmp_dir()
    try:
        frames_dir = tmp / "frames"
        expected_path = tmp / "detect_expected.json"
        expected_path.write_text("SENTINEL", encoding="utf-8")
        paths = _make_frames(frames_dir, ["sample_a.png"])
        imread_map = {str(paths[0]): _fake_frame()}
        with _patched(["make_golden", "--dry-run"], frames_dir,
                       expected_path, imread_map, [[_one_card()]]):
            main()
        out = capsys.readouterr().out
        golden = _golden_for([[_one_card()]], ["sample_a.png"])
        # print("\n--dry-run: ...。\n") ends with an embedded newline,
        # and print adds one more -> "\n\n" before the JSON.
        expected_out = (
            "sample_a.png: 1 カード\n"
            "\n--dry-run: ファイルを書き込みません。\n\n"
            + json.dumps(golden, ensure_ascii=False, indent=1)
            + "\n"
        )
        assert out == expected_out
        assert expected_path.read_text(encoding="utf-8") == "SENTINEL"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_edge_04(capsys):
    """
    input: '非 dry-run で detect_expected.json が存在しない'
    expected: '差分レポートの手順をスキップし、新結果を detect_expected.json に直接書き込む。'
    """
    tmp = _tmp_dir()
    try:
        frames_dir = tmp / "frames"
        expected_path = tmp / "detect_expected.json"
        paths = _make_frames(frames_dir, ["sample_a.png"])
        imread_map = {str(paths[0]): _fake_frame()}
        with _patched(["make_golden"], frames_dir, expected_path,
                       imread_map, [[_one_card()]]):
            main()
        out = capsys.readouterr().out
        golden = _golden_for([[_one_card()]], ["sample_a.png"])
        assert out == (
            "sample_a.png: 1 カード\n"
            f"書き込み完了: {expected_path}\n"
        )
        assert "注意:" not in out
        assert "ゴールデンは不変" not in out
        assert json.loads(expected_path.read_text(encoding="utf-8")) \
            == golden
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_edge_05(capsys):
    """
    input: '既存ファイルの内容と新結果が等しい'
    expected: '「ゴールデンは不変（検出結果が一致）」を出力しつつ、ファイルは依然として上書きされる（書き込みは無条件）。'
    """
    tmp = _tmp_dir()
    try:
        frames_dir = tmp / "frames"
        expected_path = tmp / "detect_expected.json"
        golden = _golden_for([[_one_card()]], ["sample_a.png"])
        expected_path.write_text(
            json.dumps(golden, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
        paths = _make_frames(frames_dir, ["sample_a.png"])
        imread_map = {str(paths[0]): _fake_frame()}
        with _patched(["make_golden"], frames_dir, expected_path,
                       imread_map, [[_one_card()]]):
            main()
        out = capsys.readouterr().out
        assert out == (
            "sample_a.png: 1 カード\n"
            "\nゴールデンは不変（検出結果が一致）\n"
            f"書き込み完了: {expected_path}\n"
        )
        assert json.loads(expected_path.read_text(encoding="utf-8")) \
            == golden
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_edge_06(capsys):
    """
    input: '既存ファイルの内容と新結果が異なる'
    expected: '「注意: 既存のゴールデンと検出結果が異なる（上書きします）。」を出力し、ファイルを上書きする。'
    """
    tmp = _tmp_dir()
    try:
        frames_dir = tmp / "frames"
        expected_path = tmp / "detect_expected.json"
        expected_path.write_text("[1, 2, 3]", encoding="utf-8")
        paths = _make_frames(frames_dir, ["sample_a.png"])
        imread_map = {str(paths[0]): _fake_frame()}
        with _patched(["make_golden"], frames_dir, expected_path,
                       imread_map, [[_one_card()]]):
            main()
        out = capsys.readouterr().out
        golden = _golden_for([[_one_card()]], ["sample_a.png"])
        assert out == (
            "sample_a.png: 1 カード\n"
            "\n注意: 既存のゴールデンと検出結果が異なる（上書きします）。\n"
            f"書き込み完了: {expected_path}\n"
        )
        assert json.loads(expected_path.read_text(encoding="utf-8")) \
            == golden
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_edge_07(capsys):
    """
    input: '検出カードが0件のフレーム'
    expected: 'そのフレームのエントリの cards が [] となり、「ファイル名: 0 カード」が出力され、処理は正常に継続する。'
    """
    tmp = _tmp_dir()
    try:
        frames_dir = tmp / "frames"
        expected_path = tmp / "detect_expected.json"
        paths = _make_frames(frames_dir, ["sample_a.png", "sample_b.png"])
        imread_map = {str(paths[0]): _fake_frame(),
                      str(paths[1]): _fake_frame()}
        with _patched(["make_golden"], frames_dir, expected_path,
                       imread_map, [[], [_one_card()]]):
            main()
        out = capsys.readouterr().out
        golden = _golden_for([[], [_one_card()]],
                             ["sample_a.png", "sample_b.png"])
        assert out == (
            "sample_a.png: 0 カード\n"
            "sample_b.png: 1 カード\n"
            f"書き込み完了: {expected_path}\n"
        )
        assert json.loads(expected_path.read_text(encoding="utf-8")) \
            == golden
        assert golden[0]["cards"] == []
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_edge_08(capsys):
    """
    input: '未知のコマンドラインオプション（例: --foo）'
    expected: 'argparse がエラーを標準エラーに出して SystemExit(2) で終了する。'
    """
    tmp = _tmp_dir()
    try:
        frames_dir = tmp / "frames"
        expected_path = tmp / "detect_expected.json"
        with _patched(["make_golden", "--foo"], frames_dir,
                       expected_path, {}, []):
            with pytest.raises(SystemExit) as excinfo:
                main()
        assert excinfo.value.code == 2
        err = capsys.readouterr().err
        assert "unrecognized arguments: --foo" in err
        assert not expected_path.exists()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
