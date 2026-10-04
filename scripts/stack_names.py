# -*- coding: utf-8 -*-
"""スクロール区間の keep フレームからメンバーカード領域を切り出し、
縦積みしてどのメンバーがどのフレームで見えるか確認する"""
import cv2
import numpy as np
import os

VIDEO = r"inputmov\UmamusumePrettyDerby_Jpn 2026-02-15 15-15-34.mp4"
OUT = r"output\debug_check\stack_names.png"

# スクロール区間の keep フレーム
KEEP = [240, 249, 270, 300, 330, 339, 349, 360, 390, 420, 428, 450, 510, 518, 540]

# メンバーカード領域（フル解像度 2560x1072 内の座標）
# 名前+バッジ+ファン数が読める範囲
X0, X1 = 560, 1900
Y0, Y1 = 470, 970
SCALE = 0.55


def read_frames(cap, want):
    """want のフレーム番号のみ読み返す"""
    for idx in want:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, f = cap.read()
        if ret:
            yield idx, f


def main():
    cap = cv2.VideoCapture(VIDEO)
    crops = []
    for idx, f in read_frames(cap, KEEP):
        crop = f[Y0:Y1, X0:X1]
        crop = cv2.resize(crop, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_LINEAR)
        # フレーム番号ラベル
        h, w = crop.shape[:2]
        cv2.putText(crop, f"#{idx}", (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        crops.append(crop)
    cap.release()

    # 縦積み（4列グリッドに）
    cols = 3
    w = max(c.shape[1] for c in crops)
    h = max(c.shape[0] for c in crops)
    rows = (len(crops) + cols - 1) // cols
    gap = 10
    sheet = np.full((rows * h + (rows + 1) * gap, cols * w + (cols + 1) * gap, 3), 40, dtype=np.uint8)
    for i, c in enumerate(crops):
        r, col = divmod(i, cols)
        x = gap + col * (w + gap)
        y = gap + r * (h + gap)
        sheet[y:y + h, x:x + w] = c
    cv2.imwrite(OUT, sheet)
    print(f"saved {OUT} frames={len(crops)}")


if __name__ == "__main__":
    main()
