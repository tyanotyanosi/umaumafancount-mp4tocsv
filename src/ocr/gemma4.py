from .base import OCRBase


class Gemma4Wrapper(OCRBase):
    """Gemma4の実装ラッパー（オプション）"""

    def __init__(self, model_path: str = "./models"):
        self.model_path = model_path
        self._model = None

    def _ensure_model(self):
        """モデルが読み込まれていれば読み込む（初回時）"""
        if self._model is not None:
            return

        try:
            from transformers import AutoModelForCausalLM, AutoProcessor
            import torch

            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            self._processor = AutoProcessor.from_pretrained(self.model_path)
        except ImportError:
            raise ImportError(
                "Gemma4を使用するには `pip install torch transformers` が必要です"
            )

    def recognize(self, image) -> str:
        """画像から文字を認識"""
        self._ensure_model()
        # TODO: 実際の推論処理を実装
        return ""

    def recognize_with_confidence(self, image) -> list:
        """信頼度付きで文字を認識"""
        self._ensure_model()
        # TODO: 実際の推論処理を実装
        return []
