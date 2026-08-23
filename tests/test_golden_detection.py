"""ゴールデン回帰テスト: 9 サンプルフレームで CardDetector の検出が安定していることを保証する。

- テンプレートマッチングのみ（OCR なし）のため高速・決定論的。
- 本番と同様に単一の CardDetector を全フレームで再利用（スケールキャッシュ有効）。
- role / badge_box / name_box / fan_box を整数で厳密比較。

ゴールデン由来:
  tests/golden/frames/sample_*.png
  tests/golden/detect_expected.json  (output/refactor_check/detect_before.json と同内容、
  再掲組前後一致を確認済み)
"""

import json
from pathlib import Path

import cv2
import pytest

from src.video.card_detector import CardDetector

ROOT = Path(__file__).parent.parent
TEMPLATE_DIR = ROOT / "template"
GOLDEN_DIR = ROOT / "tests" / "golden"
FRAMES_DIR = GOLDEN_DIR / "frames"
EXPECTED = GOLDEN_DIR / "detect_expected.json"

# config/settings.yaml の card_detection セクションと同一のパラメータ（hermetic に定義）
DETECTOR_SETTINGS = {
    "card_detection": {
        "badge_match_threshold": 0.6,
        "label_match_threshold": 0.7,
        "icon_match_threshold": 0.7,
        "max_cards": 3,
        "edge_margin": 8,
        "name_margin": 8,
        "name_v_margin": 10,
        "max_name_width": 400,
        "fan_width": 280,
        "reference_width": 2560,
        "multi_scale": True,
        "scale_window_low": 0.8,
        "scale_window_high": 1.3,
        "scale_step": 0.05,
        "coarse_to_fine": True,
        "coarse_scale": 0.5,
        "refine_radius": 200,
        "scale_cache": True,
    }
}


def _golden():
    return json.load(open(EXPECTED, encoding="utf-8"))


def _frames():
    files = sorted(FRAMES_DIR.glob("sample_*.png"))
    assert files, f"サンプルフレームが {FRAMES_DIR} に存在しない"
    return files


def _box(box):
    return None if box is None else [int(box[0]), int(box[1]), int(box[2]), int(box[3])]


def _as_golden(cards):
    out = []
    for c in cards:
        out.append([c.role, _box(c.badge_box), _box(c.name_box), _box(c.fan_box)])
    return out


@pytest.fixture(scope="module")
def detector():
    return CardDetector(str(TEMPLATE_DIR), DETECTOR_SETTINGS)


@pytest.mark.parametrize("frame_path", _frames(), ids=lambda p: p.name)
def test_detection_matches_golden(detector, frame_path):
    exp_by_file = {e["file"]: e for e in _golden()}
    exp = exp_by_file.get(frame_path.name)
    assert exp is not None, f"ゴールデンに {frame_path.name} の期待値が無い"

    frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    assert frame is not None, f"フレームを読み込めません: {frame_path}"
    assert list(frame.shape) == exp["shape"], "フレームサイズがゴールデンと異なる"

    cards = detector.detect(frame)
    got = _as_golden(cards)
    assert got == exp["cards"], (
        f"{frame_path.name} の検出結果がゴールデンと異なる:\n"
        f"  期待: {exp['cards']}\n"
        f"  実際: {got}"
    )


def test_all_golden_frames_covered():
    """ゴールデンの全ファイルがフレームディレクトリに揃っていることを保証する。"""
    golden_files = {e["file"] for e in _golden()}
    frame_files = {p.name for p in _frames()}
    assert golden_files == frame_files, (
        f"ゴールデンとフレームが不一致: 余分={golden_files - frame_files}, "
        f"不足={frame_files - golden_files}"
    )
