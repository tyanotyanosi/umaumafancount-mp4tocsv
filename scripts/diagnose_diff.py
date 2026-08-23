# -*- coding: utf-8 -*-
"""差分フィルタ診断スクリプト
- 全フレームを逐次読み込み、直前フレームとの差分率を計測
- 現行の DiffChecker（keep 時のみ update）をシミュレートし、
  どのフレームがスキップされるか、スキップされたフレームの内容を調べる
"""
import cv2
import numpy as np
import sys

VIDEO = r"inputmov\UmamusumePrettyDerby_Jpn 2026-02-15 15-15-34.mp4"


def gray(f):
    return cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)


def diff_rate(a_gray, b_gray):
    d = cv2.absdiff(a_gray, b_gray)
    return np.count_nonzero(d) / d.size


def main():
    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # --- パスA: 直前フレーム（常に update）との差分 ---
    # --- パスB: 現行ロジック（keep 時のみ update、閾値 0.1） ---
    prev_gray = None
    last_kept_gray = None
    kept_prev = []
    kept_lastkept = []
    rates_prev = []
    rates_lastkept = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        g = gray(frame)
        if prev_gray is not None:
            r = diff_rate(g, prev_gray)
            rates_prev.append((idx, r))
            if r >= 0.1:
                kept_prev.append(idx)
        if last_kept_gray is not None:
            rl = diff_rate(g, last_kept_gray)
            rates_lastkept.append((idx, rl))
            if rl >= 0.1:
                kept_lastkept.append(idx)
                last_kept_gray = g.copy()
        prev_gray = g
        idx += 1
    cap.release()

    print(f"全フレーム: {idx}  (fps={fps:.2f})")

    # 直前差分の分布
    if rates_prev:
        rs = [r for _, r in rates_prev]
        print("\n[直前フレーム差分] 分布:")
        print(f"  min={min(rs):.4f}  max={max(rs):.4f}  avg={sum(rs)/len(rs):.4f}")
        for t in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]:
            c = sum(1 for r in rs if r >= t)
            print(f"  閾値 {t:.2f} 以上: {c} フレーム")

    # 現行ロジックのシミュレーション
    print(f"\n[現行ロジック: last-kept 比較, 閾値0.1] keep={len(kept_lastkept)} 件")
    print("  keep フレーム:", kept_lastkept)

    print(f"\n[直前フレーム比較, 閾値0.1] keep={len(kept_prev)} 件")
    if kept_prev:
        print("  (先頭10):", kept_prev[:10])


if __name__ == "__main__":
    main()
