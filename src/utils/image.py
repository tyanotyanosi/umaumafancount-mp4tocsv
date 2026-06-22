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


def check_motion(last_frame_gray, current_frame, threshold: float) -> tuple[bool, np.ndarray | None]:
    """
    動的检测を実行し、フレームをスキップする必要があるか判定する。

    Args:
        last_frame_gray: 前フレームのグレースケール画像 (np.ndarray or None)
        current_frame: 現在のフレーム (BGR or grayscale np.ndarray)
        threshold: 動检测の閾値 (0.0〜1.0)

    Returns:
        (skip_flag, updated_last_frame)
        - skip_flag: Trueならフレームをスキップ
        - updated_last_frame: 更新後のグレースケール画像（スキップしない場合のみ）
    """
    if last_frame_gray is None:
        # 初回フレームは比較できないので処理継続
        if len(current_frame.shape) == 3:
            gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = current_frame
        return False, gray

    # グレースケール化
    if len(current_frame.shape) == 3:
        gray_current = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    else:
        gray_current = current_frame

    diff = cv2.absdiff(last_frame_gray, gray_current)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    motion_score = np.count_nonzero(thresh) / thresh.size

    if motion_score < threshold:
        # 動检测なし → スキップ（last_frame_grayは更新しない）
        return True, None
    else:
        # 動检测あり → 処理継続（last_frame_grayを更新）
        return False, gray_current
