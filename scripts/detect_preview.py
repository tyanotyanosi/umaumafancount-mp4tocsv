import cv2
import numpy as np
import sys

frame_path = r"output/preview/f01_0182.png"
frame = cv2.imread(frame_path)
g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
vis = frame.copy()

names = {
    "header_member": "M",
    "header_leader": "L",
    "i_icon": "I",
    "fan_count_label": "F",
}

for name, tag in names.items():
    tpl = cv2.imread(f"template/{name}.png")
    gt = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)
    th, tw = gt.shape[:2]
    res = cv2.matchTemplate(g, gt, cv2.TM_CCOEFF_NORMED)
    flat = res.copy()
    found = []
    while True:
        minv, maxv, minl, maxl = cv2.minMaxLoc(flat)
        if maxv < 0.88:
            break
        found.append((maxl[0], maxl[1], maxv))
        x0, y0 = maxl
        flat[max(0, y0 - 25): y0 + th + 25, max(0, x0 - 25): x0 + tw + 25] = -1
    for x, y, v in found:
        cv2.rectangle(vis, (x, y), (x + tw, y + th), (0, 255, 0), 2)
        cv2.putText(vis, f"{tag} {v:.2f}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        print(f"{name}: x={x} y={y} score={v:.3f}")

cv2.imwrite("output/preview/detect_boxes.png", vis)
print("saved")
