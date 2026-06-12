import cv2
import numpy as np
from io import BytesIO
from PIL import Image

def crop_roi(frame: np.ndarray, y_start: float, y_end: float, x_start: float, x_end: float) -> np.ndarray:
    """numpy array から ROI を切り抜く"""
    h, w = frame.shape[:2]
    y1 = int(h * y_start)
    y2 = int(h * y_end)
    x1 = int(w * x_start)
    x2 = int(w * x_end)
    return frame[y1:y2, x1:x2]

def to_gray(frame: np.ndarray) -> np.ndarray:
    """グレースケール変換"""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def img_to_bytes(frame: np.ndarray, is_gray: bool) -> bytes:
    """
    numpy array → PNG bytes 変換
    OCRに渡す画像はRGBに変換が必要
    """
    if is_gray:
        # グレースケールの場合は直接uint8として処理
        img_for_ocr = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    else:
        img_for_ocr = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    buf = BytesIO()
    Image.fromarray(img_for_ocr).save(buf, format="PNG")
    return buf.getvalue()
