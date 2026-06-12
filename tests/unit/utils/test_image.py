"""ユニットテスト: src.utils.image

  テスト対象:
    - crop_roi
    - to_gray
    - img_to_bytes
"""

from __future__ import annotations

from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from src.utils.image import crop_roi, to_gray, img_to_bytes


class TestCropRoi:
    """crop_roi() の座標計算テスト"""

    def test_basic_crop(self):
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        cropped = crop_roi(frame, y_start=0.5, y_end=0.8,
                           x_start=0.1, x_end=0.3)
        # y: 50..80 (30), x: 20..60 (40)
        assert cropped.shape == (30, 40, 3)

    def test_full_frame(self):
        frame = np.ones((50, 100, 3), dtype=np.uint8)
        cropped = crop_roi(frame, y_start=0.0, y_end=1.0,
                           x_start=0.0, x_end=1.0)
        assert cropped.shape == (50, 100, 3)

    def test_single_pixel(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        cropped = crop_roi(frame, y_start=0.0, y_end=0.01,
                           x_start=0.0, x_end=0.01)
        assert cropped.shape[0] >= 1
        assert cropped.shape[1] >= 1


class TestToGray:
    """to_gray() のテスト（cv2.cvtColor はモック）"""

    def test_disabled_by_default(self):
        """デフォルトでは変換しない（使う側で制御）"""
        # to_gray は常に cv2.cvtColor を呼ぶ独立関数
        frame = np.zeros((10, 20, 3), dtype=np.uint8)
        with patch("src.utils.image.cv2.cvtColor",
                   return_value=np.zeros((10, 20), dtype=np.uint8)):
            result = to_gray(frame)
            assert result.ndim == 2


class TestImgToBytes:
    """img_to_bytes() のテスト"""

    def test_color_image(self):
        frame = np.zeros((5, 5, 3), dtype=np.uint8)
        with patch("src.utils.image.Image.fromarray") as mock_fromarray:
            mock_img = MagicMock()
            mock_fromarray.return_value = mock_img
            result = img_to_bytes(frame, is_gray=False)
            assert isinstance(result, bytes)
            mock_img.save.assert_called_once()

    def test_gray_image(self):
        frame = np.zeros((5, 5), dtype=np.uint8)
        with patch("src.utils.image.Image.fromarray") as mock_fromarray:
            mock_img = MagicMock()
            mock_fromarray.return_value = mock_img
            result = img_to_bytes(frame, is_gray=True)
            assert isinstance(result, bytes)
            mock_img.save.assert_called_once()
