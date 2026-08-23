# -*- coding: utf-8 -*-
"""スクロール区間（フレーム240〜540）の全フレームで、直前フレーム差分の分布を計測。
last-kept 比較と、直前比較の keep 数を比較する。"""
import cv2
import numpy as np

VIDEO = r"inputmov\UmamusumePrettyDerby_Jpn 2026-02-15 15-15-34.mp4"


def gray(f):
    return cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)


def dr(a, b):
    d = cv2.absdiff(a, b)
    return np.count_nonzero(d) / d.size


def main():
    cap = cv2.VideoCapture(VIDEO)
    idx = 0
    prev = None
    last_kept = None
    keep_lastkept = []
    keep_prev = []
    diffs = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        g = gray(f)
        if 240 <= idx <= 540:
            if prev is not None:
                r = dr(g, prev)
                diffs.append((idx, r))
                if r >= 0.1:
                    keep_prev.append(idx)
            # last-kept 比較（現行）
            if last_kept is None:
                is_diff = True
            else:
                is_diff = dr(g, last_kept) >= 0.1
            if is_diff:
                keep_lastkept.append(idx)
                last_kept = g.copy()
        prev = g
        idx += 1
    cap.release()

    vals = [r for _, r in diffs]
    print(f"scroll region frames: {len(vals)}")
    print(f"consecutive diff: min={min(vals):.4f} max={max(vals):.4f} avg={np.mean(vals):.4f}")
    print(f"  >=0.10: {sum(1 for v in vals if v >= 0.10)}")
    print(f"  >=0.05: {sum(1 for v in vals if v >= 0.05)}")
    print(f"  >=0.03: {sum(1 for v in vals if v >= 0.03)}")
    print(f"  >=0.02: {sum(1 for v in vals if v >= 0.02)}")
    print(f"last-kept kept: {len(keep_lastkept)} -> {keep_lastkept}")
    print(f"prev-frame kept (thr 0.1): {len(keep_prev)} -> {keep_prev}")


if __name__ == "__main__":
    main()
