"""Tests for ``src.ocr.cache.CachingOCR.stats``.

Specification: docs/00-Architecture/src_ocr_cache__CachingOCR_stats.yaml

``stats`` is a property (``@property``) that returns the current
cache statistics (cache name, hit count, miss count, registered key
count) as a new dict:

- because it is a property, it is invoked by attribute access
  (``cache.stats`` without parentheses)
- generates and returns
  ``{'name': self.name, 'hits': self.hits, 'misses': self.misses,
  'unique': len(self._cache)}``
- ``'unique'`` is the number of keys registered in ``self._cache``;
  keys include the method name, so the same image cached by both
  ``recognize`` and ``recognize_with_confidence`` counts as 2

Mocked / stand-in dependencies (per the test-generation rules):
the ``engine`` argument is a mock whose ``recognize`` /
``recognize_with_confidence`` return the specified values; the
``image`` argument is a mock whose ``tobytes()`` returns the
specified bytes (the real engine implementation is outside the read
scope).

``errors`` section of the spec: empty (no error cases documented).
"""

from unittest import mock

from src.ocr.cache import CachingOCR


def _image(payload):
    """Create a mock image whose ``tobytes()`` returns ``payload``."""
    img = mock.Mock()
    img.tobytes.return_value = payload
    return img


def test_edge_01():
    """
    input: 認識メソッドを一度も呼ばないばかりに作ったばかりのインスタンス（CachingOCR(engine)）の cache.stats
    expected: {'name': 'ocr', 'hits': 0, 'misses': 0, 'unique': 0} が返る
    """
    inst = CachingOCR(mock.Mock(), name="ocr")
    assert inst.stats == {"name": "ocr", "hits": 0,
                          "misses": 0, "unique": 0}


def test_edge_02():
    """
    input: 同一画像での recognize(img) 2回呼び出し（1回目がミス、2回目がヒット）後の cache.stats
    expected: {'name': 'ocr', 'hits': 1, 'misses': 1, 'unique': 1} が返る
    """
    engine = mock.Mock()
    engine.recognize.return_value = "abc"
    inst = CachingOCR(engine, name="ocr")
    img = _image(b"img-bytes")
    inst.recognize(img)
    inst.recognize(img)
    assert inst.stats == {"name": "ocr", "hits": 1,
                          "misses": 1, "unique": 1}


def test_edge_03():
    """
    input: 同一画像で recognize(img) 1回と recognize_with_confidence(img) 1回を呼び出した後の cache.stats
    expected: stats['unique'] == 2 となる（2メソッドのキー名前空間が異なるため2つのキーとして登録される）
    """
    engine = mock.Mock()
    engine.recognize.return_value = "abc"
    engine.recognize_with_confidence.return_value = [{"text": "a"}]
    inst = CachingOCR(engine, name="ocr")
    img = _image(b"img-bytes")
    inst.recognize(img)
    inst.recognize_with_confidence(img)
    stats = inst.stats
    assert stats["unique"] == 2


def test_edge_04():
    """
    input: 一度取得した dict を変更した（例: d = cache.stats; d['hits'] = 999）後に再度 cache.stats を読み取る
    expected: インスタンスのカウントに影響しない。元の hits の値を持つ新しい dict が返る
    """
    engine = mock.Mock()
    engine.recognize.return_value = "abc"
    inst = CachingOCR(engine, name="ocr")
    img = _image(b"img-bytes")
    inst.recognize(img)
    inst.recognize(img)
    d = inst.stats
    d["hits"] = 999
    again = inst.stats
    assert again is not d
    assert again["hits"] == 1
    assert inst.hits == 1
