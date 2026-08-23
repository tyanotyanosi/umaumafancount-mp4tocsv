# -*- coding: utf-8 -*-
"""keep フレームの左右半分を切り出して、スクロール内容を確認"""
import cv2
import os

INDIR = r"output\debug_check\kept_current"
OUTDIR = r"output\debug_check\halves"
os.makedirs(OUTDIR, exist_ok=True)

for fn in sorted(os.listdir(INDIR)):
    if not fn.endswith(".png"):
        continue
    f = cv2.imread(os.path.join(INDIR, fn))
    if f is None:
        continue
    h, w = f.shape[:2]
    left = f[:, : w // 2]
    right = f[:, w // 2:]
    # 縮小して可読サイズに
    scale = 0.5
    cv2.imwrite(os.path.join(OUTDIR, fn.replace(".png", "_left.png")),
                cv2.resize(left, (int(w // 2 * scale), int(h * scale))))
    cv2.imwrite(os.path.join(OUTDIR, fn.replace(".png", "_right.png")),
                cv2.resize(right, (int(w // 2 * scale), int(h * scale))))
print("done")
