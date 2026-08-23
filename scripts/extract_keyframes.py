# -*- coding: utf-8 -*-
"""現行パイプライン（全フレーム読込→直前last-kept比較・閾値0.1）を正確に再現し、
keep フレームを保存。さらに直前フレーム比較（常にupdate）でも keep 判定。
"""
import cv2
import numpy as np
import os

VIDEO = r"inputmov\UmamusumePrettyDerby_Jpn 2026-02-15 15-15-34.mp4"
OUTDIR = r"output\debug_check\kept_current"
os.makedirs(OUTDIR, exist_ok=True)


def gray(f):
    return cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)


def diff_rate(a, b):
    d = cv2.absdiff(a, b)
    return np.count_nonzero(d) / d.size


def main():
    cap = cv2.VideoCapture(VIDEO)
    idx = 0
    last_kept_gray = None
    prev_gray = None
    kept = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        g = gray(frame)
        # 現行 DiffChecker: last_frame は None なら True、update は keep 時のみ
        if last_kept_gray is None:
            is_diff = True
        else:
            is_diff = diff_rate(g, last_kept_gray) >= 0.1
        if is_diff:
            kept.append(idx)
            last_kept_gray = g.copy()
            if len(kept) <= 40:
                cv2.imwrite(f"{OUTDIR}\\kept_{idx:04d}.png", frame)
        prev_gray = g
        idx += 1
    cap.release()
    print(f"current-pipeline kept frames ({len(kept)}):")
    print(" ", kept)
    return kept


if __name__ == "__main__":
    main()
