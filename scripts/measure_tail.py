# -*- coding: utf-8 -*-
"""動画後半（フレーム500以降）の動きをストリーミングで精密計測（フル解像度）"""
import cv2
import numpy as np

VIDEO = r"inputmov\UmamusumePrettyDerby_Jpn 2026-02-15 15-15-34.mp4"


def gray(f):
    return cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)


def dr(a, b):
    d = cv2.absdiff(a, b)
    return np.count_nonzero(d) / d.size


def main():
    # Pass 1: #540 の参照フレームを取得
    cap = cv2.VideoCapture(VIDEO)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 540)
    ret, ref_frame = cap.read()
    cap.release()
    ref_gray = gray(ref_frame)
    fps = 27.517928

    # Pass 2: 全フレームをストリーミング、prev と ref を比較
    cap = cv2.VideoCapture(VIDEO)
    idx = 0
    prev_gray = None
    last_prev_diff = 0.0
    while True:
        ret, f = cap.read()
        if not ret:
            break
        g = gray(f)
        if prev_gray is not None:
            r = dr(g, prev_gray)
            last_prev_diff = r
            if idx >= 500 and idx % 10 == 0:
                flag = "  <<<" if r >= 0.1 else ""
                print(f"  consecutive #{idx} ({idx/fps:.1f}s): {r:.4f}{flag}")
            if idx >= 540 and idx % 10 == 0:
                rc = dr(g, ref_gray)
                flag = "  <<<" if rc >= 0.1 else ""
                print(f"  cum-from-540 #{idx} ({idx/fps:.1f}s): {rc:.4f}{flag}")
        prev_gray = g
        idx += 1
    cap.release()

    print(f"\nlast frame #{idx-1}: consecutive diff={last_prev_diff:.4f}")
    print(f"#540 vs last(#{idx-1}): {dr(prev_gray, ref_gray):.4f}")


if __name__ == "__main__":
    main()
