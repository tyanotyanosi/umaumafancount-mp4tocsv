import cv2
import numpy as np
from typing import Optional


class DiffChecker:
    """前フレームとの差分を判定"""

    def __init__(self, threshold: float = 0.1):
        self.threshold = threshold
        self.last_frame: Optional[object] = None

    def is_different(self, current_frame: object) -> bool:
        """
        前フレームとの差分が閾値以上か判定
        True: 差分あり（処理必要）
        False: 差分なし（スキップ可能）
        """
        if self.last_frame is None:
            return True

        gray_current = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        gray_last = cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2GRAY)

        diff = cv2.absdiff(gray_current, gray_last)
        diff_rate = np.count_nonzero(diff) / (gray_current.shape[0] * gray_current.shape[1])

        return diff_rate >= self.threshold

    def update(self, frame: object):
        """現在フレームを保持"""
        self.last_frame = frame.copy()
