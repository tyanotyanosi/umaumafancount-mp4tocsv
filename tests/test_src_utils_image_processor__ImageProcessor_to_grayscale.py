"""src.utils.image_processor.ImageProcessor.to_grayscale の仕様書駆動 pytest テスト。

spec: docs/00-Architecture/src_utils_image_processor__ImageProcessor_to_grayscale.yaml
function: src.utils.image_processor.ImageProcessor.to_grayscale

検証する行動:
  cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) を呼び出し、その戻り値（np.ndarray）をそのまま返す。

モックした依存: なし（実 cv2 を使用）。

errors セクション（記録のみ・テストしない）:
  - image が COLOR_BGR2GRAY フラグの cv2.cvtColor に受け入れられない入力 →
    cv2.cvtColor が送出する例外がそのまま伝播する（例外種別は cv2 の実装で決まる）。
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.utils.image_processor import ImageProcessor


def test_edge_01():
    """input: 3チャネル BGR 画像（例: shape (1080, 1920, 3)、dtype uint8）
    expected: cv2.cvtColor の COLOR_BGR2GRAY 挙向に従うグレースケール画像（2次元配列）を返す。
    """
    # 等灰度 BGR（R=G=B=128）で、グレースケール変換後も 128 になることを確認。
    img = np.full((10, 20, 3), 128, dtype=np.uint8)
    result = ImageProcessor.to_grayscale(img)
    assert result.shape == (10, 20)
    assert result.ndim == 2
    assert np.all(result == 128)


def test_edge_02():
    """input: COLOR_BGR2GRAY フラグでは cv2.cvtColor がサポートしない入力（例: 1次元配列、チャネル数の合わない配列）
    expected: cv2.cvtColor が送出する例外がそのまま伝播する（例外の種類・メッセージは cv2 の実装で決まる、読込対象コードで確定できない）。
    """
    with pytest.raises(cv2.error):
        ImageProcessor.to_grayscale(np.zeros((5,), dtype=np.uint8))
