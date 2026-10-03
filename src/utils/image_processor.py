import cv2
import numpy as np


class ImageProcessor:
    """画像処理ユーティリティ"""

    @staticmethod
    def to_grayscale(image: np.ndarray) -> np.ndarray:
        """BGR画像をグレースケールに変換"""
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def crop_region(image: np.ndarray, region: tuple, use_percent: bool = False) -> np.ndarray:
        """
        指定領域をクロップ

        Args:
            image: 入力画像
            region: (left, top, right, bottom) if use_percent else (x, y, w, h)
            use_percent: Trueの場合、regionをパーセント指定として扱う
        """
        img_height, img_width = image.shape[:2]

        if use_percent:
            left = int(region[0] / 100 * img_width)
            top = int(region[1] / 100 * img_height)
            right = int(region[2] / 100 * img_width)
            bottom = int(region[3] / 100 * img_height)
            # 画像範囲へクランプ（負 / 100% 超過でもクラッシュしない）
            left = max(0, min(left, img_width))
            top = max(0, min(top, img_height))
            right = max(0, min(right, img_width))
            bottom = max(0, min(bottom, img_height))
            return image[top:bottom, left:right]
        else:
            x, y, w, h = region
            x = max(0, min(int(x), img_width))
            y = max(0, min(int(y), img_height))
            w = max(0, min(int(w), img_width - x))
            h = max(0, min(int(h), img_height - y))
            return image[y:y+h, x:x+w]

    @staticmethod
    def resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
        """画像をリサイズ"""
        return cv2.resize(image, (width, height))
