"""crop ハッシュに基づく OCR キャッシュ。

同じ crop 画像が繰り返し OCR されるのを防ぎ、以下を改善する:
- 実行時間（同一 crop の再認識を省く）
- 認識の揺らぎ（同一 crop に対しては初回結果を返す）

キーは crop の ``tobytes()`` の md5 にメソッド名を付加した名前空間キー
（``recognize`` は文字列、``recognize_with_confidence`` は辞書リストを返すため、
共用キーでは型が混在し得る）。空文字・``None`` の結果もキャッシュする
（空結果・None 結果の再計算回避のため）。
"""

import hashlib

_MISSING = object()  # dict.get のデフォルト（キャッシュに未登録）
_NONE = object()  # エンジンの None 結果をキャッシュするためのプレースホルダ


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
        key = ("recognize", self._key(image))
        cached = self._cache.get(key, _MISSING)
        if cached is not _MISSING:
            self.hits += 1
            return None if cached is _NONE else cached
        self.misses += 1
        result = self.engine.recognize(image)
        self._cache[key] = _NONE if result is None else result
        return result

    def recognize_with_confidence(self, image) -> list:
        key = ("recognize_with_confidence", self._key(image))
        cached = self._cache.get(key, _MISSING)
        if cached is not _MISSING:
            self.hits += 1
            return None if cached is _NONE else cached
        self.misses += 1
        result = self.engine.recognize_with_confidence(image)
        self._cache[key] = _NONE if result is None else result
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
