# -*- coding: utf-8 -*-
"""リスト領域の垂直シフトを連続フレーム間で測定し、
スクロール速度（カード/フレーム）を算出。バースト区間の速度を確認する。"""
import cv2
import numpy as np

VIDEO = r"inputmov\UmamusumePrettyDerby_Jpn 2026-02-15 15-15-34.mp4"

# リスト領域（メンバーカードがスクロールする部分）
X0, X1 = 580, 1500
Y0, Y1 = 470, 960


def crop_gray(f):
    return cv2.cvtColor(f[Y0:Y1, X0:X1], cv2.COLOR_BGR2GRAY)


def shift(a, b):
    """a から b への垂直シフト（px、正=下方向にスクロール）を phaseCorrelate で推定"""
    try:
        result = cv2.phaseCorrelate(a.astype(np.float64), b.astype(np.float64),
                                     np.hanning(a.shape[1]), np.hanning(a.shape[0]))
        return float(result[1])  # dy
    except Exception:
        return None


def main():
    cap = cv2.VideoCapture(VIDEO)
    idx = 0
    prev_crop = None
    shifts = {}
    while True:
        ret, f = cap.read()
        if not ret:
            break
        c = crop_gray(f)
        if prev_crop is not None and 240 <= idx <= 540:
            s = shift(prev_crop, c)
            shifts[idx] = s
        prev_crop = c
        idx += 1
    cap.release()

    fps = 27.517928
    valid = [(i, s) for i, s in shifts.items() if s is not None]
    print(f"measured shifts: {len(valid)}")
    # 区間ごとにシフトを合計し、スクロール量（px）と速度を算出
    for lo, hi, label in [(240, 300, "early"), (300, 450, "mid"), (450, 540, "burst")]:
        ss = [s for i, s in valid if lo <= i < hi]
        if ss:
            total = sum(ss)
            n = hi - lo
            print(f"[{label}] frames {lo}-{hi}: total shift={total:.1f}px, "
                  f"avg/frame={total/n:.3f}px/f, per 28f(1s)={total/n*28:.1f}px")
    # バーストの詳細
    print("\nburst region (440-540) per-frame shift:")
    for i, s in sorted(shifts.items()):
        if 440 <= i <= 540:
            flag = "  <<<" if (s is not None and s > 5) else ""
            print(f"  #{i} ({i/fps:.1f}s): {s if s is None else round(s,2)}{flag}")


if __name__ == "__main__":
    main()
