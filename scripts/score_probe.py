import cv2
import numpy as np

frame = cv2.imread("output/preview/f01_0182.png")
print("frame shape:", frame.shape)
g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
m_t = cv2.cvtColor(cv2.imread("template/header_member.png"), cv2.COLOR_BGR2GRAY)
l_t = cv2.cvtColor(cv2.imread("template/header_leader.png"), cv2.COLOR_BGR2GRAY)
mm = cv2.matchTemplate(g, m_t, cv2.TM_CCOEFF_NORMED)
ml = cv2.matchTemplate(g, l_t, cv2.TM_CCOEFF_NORMED)

print("mm shape:", mm.shape, "ml shape:", ml.shape)
print("mm global max:", cv2.minMaxLoc(mm)[3])
print("ml global max:", cv2.minMaxLoc(ml)[3])
print("mm[702,614] =", float(mm[702, 614]))
print("ml[544,614] =", float(ml[544, 614]))
print("mm[544,614] =", float(mm[544, 614]))
print("mm[860:880, 614] =", [round(float(v), 3) for v in mm[860:880, 614]])
