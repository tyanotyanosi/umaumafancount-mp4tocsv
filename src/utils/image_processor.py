import cv2
import numpy as np


class ImageProcessor:
    """画像処理ユーティリティ"""

    @staticmethod
    def to_grayscale(image) -> object:
        """BGR画像をグレースケールに変換"""
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def crop_region(image, region: tuple, use_percent: bool = False) -> object:
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
            return image[top:bottom, left:right]
        else:
            x, y, w, h = region
            return image[y:y+h, x:x+w]

    @staticmethod
    def resize(image, width: int, height: int) -> object:
        """画像をリサイズ"""
        return cv2.resize(image, (width, height))
