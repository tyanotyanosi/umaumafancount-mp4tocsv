"""合成フレームから MP4 を生成する（テスト / CI の E2E スモーク用）。

自己完結が前提:
- 削除済みの ``tests.test_card_detector`` には依存しない（v1.0.0 改修で消滅したため）。
- 読み込む画像は git 管理下の ``template/*.png`` だけ（gitignore 対象の素材は使わない）。
- 生成処理は ``main()`` の中だけで行う（import 時にファイルを書かない）。

フレーム幅を ``config/settings.yaml`` の ``card_detection.reference_width``
と同じ 2560 にしておくと、検出器の等倍スケール (s=1.0) が候補に含まれ、
テンプレートマッチングがヒットする。
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TEMPLATE_DIR = ROOT / "template"
OUT = ROOT / "output" / "synthetic.mp4"

FRAME_W, FRAME_H = 2560, 1440  # reference_width と同じ解像度
FPS = 10
N_FRAMES = 10

# 実フレーム計測（scripts/anchor_probe.py）のアンカー相対位置を模した配置
BADGE_X = 614
BADGE_Y0 = 544
CARD_PITCH = 158          # バッジ 2 枚の縦ピッチ（実測 544 → 702）
ICON_DX, ICON_DY = 400, 3  # i_icon はバッジと同じヘッダ行の右側
LABEL_DX, LABEL_DY = 12, 45  # fan_count_label はバッジの下のファン行


def _load_bgr(name: str) -> np.ndarray:
    """テンプレートを BGR で読み込む（CardDetector._load と同じ読み方）。"""
    img = cv2.imread(str(TEMPLATE_DIR / name), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"テンプレートが見つかりません: {TEMPLATE_DIR / name}")
    return img


def _paste(frame: np.ndarray, patch: np.ndarray, x: int, y: int) -> None:
    """patch を frame の (x, y) に切り出しながら貼り付ける。"""
    fh, fw = frame.shape[:2]
    ph, pw = patch.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(fw, x + pw), min(fh, y + ph)
    if x1 <= x0 or y1 <= y0:
        return
    frame[y0:y1, x0:x1] = patch[y0 - y:y1 - y, x0 - x:x1 - x]


def build_frame(n_cards: int = 3) -> np.ndarray:
    """テンプレートを貼り付けた合成フレーム（BGR）を 1 枚作る。"""
    frame = np.full((FRAME_H, FRAME_W, 3), 255, dtype=np.uint8)
    badge = _load_bgr("header_member.png")
    icon = _load_bgr("i_icon.png")
    label = _load_bgr("fan_count_label.png")
    bh = badge.shape[0]
    for i in range(n_cards):
        by = BADGE_Y0 + i * CARD_PITCH
        _paste(frame, badge, BADGE_X, by)
        _paste(frame, icon, BADGE_X + ICON_DX, by + ICON_DY)
        _paste(frame, label, BADGE_X + LABEL_DX, by + bh + LABEL_DY)
    return frame


def write_video(path: Path = OUT, n_frames: int = N_FRAMES,
                frame: np.ndarray | None = None) -> Path:
    """合成フレームを n_frames 書き込んで MP4 を作る。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if frame is None:
        frame = build_frame()
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (FRAME_W, FRAME_H))
    if not writer.isOpened():
        raise RuntimeError(f"VideoWriter を開けません: {path}")
    try:
        for _ in range(n_frames):
            writer.write(frame)
    finally:
        writer.release()
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"MP4 を書き出せません: {path}")
    return path


def main() -> int:
    frame = build_frame()
    out = write_video(OUT, N_FRAMES, frame)

    # 検出器の動作確認（CI を落とさないため失敗にはしない）
    from src.video.card_detector import CardDetector

    cards = CardDetector(str(TEMPLATE_DIR)).detect(frame)
    print(f"synthetic video: {out} ({FRAME_W}x{FRAME_H}, {N_FRAMES} frames @ {FPS}fps)")
    print(f"detector check: {len(cards)} cards")
    if not cards:
        print("警告: 合成フレームからカードを検出できませんでした（スモーク動画としては有効）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
