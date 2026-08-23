from abc import ABC, abstractmethod
from typing import Optional


class OCRBase(ABC):
    """OCRエンジン共通インターフェース"""

    @abstractmethod
    def recognize(self, image) -> str:
        """
        画像から文字を認識

        Args:
            image: 入力画像 (numpy配列)

        Returns:
            認識結果文字列
        """
        pass

    @abstractmethod
    def recognize_with_confidence(self, image) -> list:
        """
        信頼度付きで文字を認識

        Args:
            image: 入力画像

        Returns:
            認識結果のリスト [{"text": "...", "confidence": 0.95}, ...]
        """
        pass
