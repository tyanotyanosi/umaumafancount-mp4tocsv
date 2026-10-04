# -*- coding: utf-8 -*-
"""1秒間隔でフレームをサンプリングしコンタクトシートを生成。
動画の内容（スクロールの様子）を一目で確認する。
"""
import cv2
import numpy as np

VIDEO = r"inputmov\UmamusumePrettyDerby_Jpn 2026-02-15 15-15-34.mp4"
OUT = r"output\debug_check\contact_sheet.png"


def main():
    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS)
    step = max(1, int(round(fps * 1.0)))  # 1秒間隔
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # サムネイルサイズ
    tw, th = 320, int(h * 320 / w)
    cols = 4
    margin = 8
    label_h = 28

    cells = []
    labels = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            ts = idx / fps
            thumb = cv2.resize(frame, (tw, th))
            cells.append(thumb)
            labels.append(f"#{idx}  {ts:.1f}s")
        idx += 1
    cap.release()

    rows = (len(cells) + cols - 1) // cols
    sheet_w = cols * tw + (cols + 1) * margin
    sheet_h = rows * (th + label_h) + (rows + 1) * margin
    sheet = np.zeros((sheet_h, sheet_w, 3), dtype=np.uint8)

    for i, c in enumerate(cells):
        r, col = divmod(i, cols)
        x = margin + col * (tw + margin)
        y = margin + r * (th + label_h + margin)
        sheet[y:y + th, x:x + tw] = c
        ly = y + th + 2
        cv2.putText(sheet, labels[i], (x + 4, ly + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imwrite(OUT, sheet)
    print(f"saved {OUT}  frames={len(cells)} step={step} grid={cols}x{rows}")


if __name__ == "__main__":
    main()
