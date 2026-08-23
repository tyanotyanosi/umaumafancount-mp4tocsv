from meikiocr import MeikiOCR as _MeikiOCREngine
from .base import OCRBase


class MeikiOCRWrapper(OCRBase):
    """meikiOCRの実装ラッパー"""

    def __init__(self, det_threshold: float = 0.8, rec_threshold: float = 0.2):
        self.engine = _MeikiOCREngine()
        self.det_threshold = det_threshold
        self.rec_threshold = rec_threshold

    def recognize(self, image) -> str:
        """画像から文字を認識"""
        results = self.engine.run_ocr(image, det_threshold=self.det_threshold, rec_threshold=self.rec_threshold)
        return '\n'.join(line['text'] for line in results if line['text'])

    def recognize_with_confidence(self, image) -> list:
        """信頼度付きで文字を認識"""
        results = self.engine.run_ocr(image, det_threshold=self.det_threshold, rec_threshold=self.rec_threshold)
        return [
            {"text": line['text'], "confidence": line.get('confidence', 0.0)}
            for line in results if line['text']
        ]
