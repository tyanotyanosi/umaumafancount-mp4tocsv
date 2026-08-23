"""合成フレームから MP4 を生成する（Phase 3 検証用）"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np

from tests.test_card_detector import build_frame, FRAME_W, FRAME_H, _det

OUT = Path(__file__).parent.parent / "output" / "synthetic.mp4"
OUT.parent.mkdir(parents=True, exist_ok=True)

writer = cv2.VideoWriter(str(OUT), cv2.VideoWriter_fourcc(*"mp4v"), 10, (FRAME_W, FRAME_H))
n_frames = 10
for i in range(n_frames):
    frame = build_frame(n_cards=3)
    writer.write(frame)
writer.release()

# 検出器の動作も確認
cards = _det().detect(build_frame(n_cards=3))
print(f"synthetic video: {OUT}")
print(f"detector check: {len(cards)} cards")
