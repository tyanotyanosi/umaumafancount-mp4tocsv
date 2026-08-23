"""ゴールデン検出結果 (tests/golden/detect_expected.json) を再生成する。

サンプルフレーム (tests/golden/frames/sample_*.png) を CardDetector で検出し、
期待値 JSON を上書きする。検出ロジック（テンプレート／閾値）を意図的に変更した後、
あるいはゴールデンの差分確認に使う。

使い方:
    python scripts/make_golden.py            # 現在の検出器で detect_expected.json を再生成
    python scripts/make_golden.py --dry-run  # 再生成結果を表示のみ（ファイルを書かない）

パラメータは config/settings.yaml の card_detection セクションをそのまま読む。
"""

import argparse
import json
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
GOLDEN_DIR = ROOT / "tests" / "golden"
FRAMES_DIR = GOLDEN_DIR / "frames"
EXPECTED = GOLDEN_DIR / "detect_expected.json"
TEMPLATE_DIR = ROOT / "template"

import yaml
from src.video.card_detector import CardDetector


def load_settings() -> dict:
    with open(ROOT / "config" / "settings.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def to_golden_cards(cards):
    out = []
    for c in cards:
        fb = None
        if c.fan_box is not None:
            fb = [int(c.fan_box[0]), int(c.fan_box[1]), int(c.fan_box[2]), int(c.fan_box[3])]
        out.append([
            c.role,
            [int(c.badge_box[0]), int(c.badge_box[1]), int(c.badge_box[2]), int(c.badge_box[3])],
            [int(c.name_box[0]), int(c.name_box[1]), int(c.name_box[2]), int(c.name_box[3])],
            fb,
        ])
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="ファイルを書き込まず結果を表示する")
    args = parser.parse_args()

    settings = load_settings()
    # 本番と同様に単一検出器を全フレームで再利用（スケールキャッシュ有効）
    detector = CardDetector(str(TEMPLATE_DIR), settings)

    frames = sorted(FRAMES_DIR.glob("sample_*.png"))
    if not frames:
        raise SystemExit(f"サンプルフレームが {FRAMES_DIR} に存在しません")

    golden = []
    for frame_path in frames:
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise SystemExit(f"フレームを読み込めません: {frame_path}")
        h, w = frame.shape[:2]
        cards = detector.detect(frame)
        golden.append({
            "file": frame_path.name,
            "shape": [int(h), int(w), 3],
            "cards": to_golden_cards(cards),
        })
        n = len(cards)
        print(f"{frame_path.name}: {n} カード")

    if args.dry_run:
        print("\n--dry-run: ファイルを書き込みません。\n")
        print(json.dumps(golden, ensure_ascii=False, indent=1))
        return

    # 旧ゴールデンとの差分を報告（あれば）
    if EXPECTED.exists():
        old = json.load(open(EXPECTED, encoding="utf-8"))
        if old != golden:
            print(f"\n注意: 既存のゴールデンと検出結果が異なる（上書きします）。")
        else:
            print("\nゴールデンは不変（検出結果が一致）")

    EXPECTED.write_text(json.dumps(golden, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"書き込み完了: {EXPECTED}")


if __name__ == "__main__":
    main()
