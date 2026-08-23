"""crop ハッシュに基づく OCR キャッシュ。

同じ crop 画像が繰り返し OCR されるのを防ぎ、以下を改善する:
- 実行時間（同一 crop の再認識を省く）
- 認識の揺らぎ（同一 crop に対しては初回結果を返す）

キーは crop の ``tobytes()`` の md5。空文字の結果もキャッシュする
（空結果の再計算回避のため）。
"""

import hashlib


class CachingOCR:
    """OCR エンジンをラップし、crop をキーに結果をキャッシュする。

    ``recognize`` / ``recognize_with_confidence`` をラップ。内部エンジンが
    持たないメソッドはそのまま委譲する。
    """

    def __init__(self, engine, name: str = "ocr"):
        self.engine = engine
        self.name = name
        self._cache = {}
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _key(image) -> str:
        return hashlib.md5(image.tobytes()).hexdigest()

    def recognize(self, image) -> str:
        key = self._key(image)
        cached = self._cache.get(key)
        if cached is not None:
            self.hits += 1
            return cached
        self.misses += 1
        result = self.engine.recognize(image)
        self._cache[key] = result
        return result

    def recognize_with_confidence(self, image) -> list:
        key = self._key(image)
        cached = self._cache.get(key)
        if cached is not None:
            self.hits += 1
            return cached
        self.misses += 1
        result = self.engine.recognize_with_confidence(image)
        self._cache[key] = result
        return result

    def __getattr__(self, attr):
        # その他のメソッド / 属性は内部エンジンへ委譲する
        return getattr(self.engine, attr)

    @property
    def stats(self) -> dict:
        return {
            "name": self.name,
            "hits": self.hits,
            "misses": self.misses,
            "unique": len(self._cache),
        }
