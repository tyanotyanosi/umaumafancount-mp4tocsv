"""Tests for ``src.ocr.cache.CachingOCR.recognize_with_confidence``.

Specification: docs/00-Architecture/src_ocr_cache__CachingOCR_recognize_with_confidence.yaml

``recognize_with_confidence`` calls
``self.engine.recognize_with_confidence(image)`` and caches the
result keyed by the MD5 of the image bytes; a second and later call
with the same image does not call the engine and returns the cached
result:

- compute the key as the tuple
  ``('recognize_with_confidence', CachingOCR._key(image))``
- on a cache hit, increment ``self.hits`` by 1 and return the cached
  value (the same list object; None when the cached value is the
  placeholder ``_NONE``)
- on a miss, increment ``self.misses`` by 1 and call
  ``self.engine.recognize_with_confidence(image)``
- store the engine's return value in ``self._cache`` (a None result
  is stored as the placeholder ``_NONE``) and return it as-is
- the cache key is namespaced by method name, so the same image does
  not share a key with ``recognize``

Mocked / stand-in dependencies (per the test-generation rules):
the ``engine`` argument is a mock whose
``recognize_with_confidence`` returns (or raises) the specified
value; the ``image`` argument is a mock whose ``tobytes()`` returns
the specified bytes (the real engine implementation is outside the
read scope).

``errors`` section of the spec (documented here only, NOT tested, per
the test-generation rules):

- condition: "キャッシュミス時で、self.engine に recognize_with_confidence 属性が存在しない"
  behavior: "AttributeError（self.engine.recognize_with_confidence の参照時に送出）"
- condition: "image に tobytes 属性がない、または tobytes() が bytes 以外の型を返す"
  behavior: "AttributeError / TypeError（キャッシュ参照より前の _key 計算時に送出）"
- condition: "self.engine.recognize_with_confidence(image) が例外を送出する"
  behavior: "例外は呼び出し側にそのまま伝播し、結果はキャッシュされない"
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
    input: engine.recognize_with_confidence が [{"text": "a", "confidence": 0.9}] を返すスタブエンジンでの初回呼び出し
    expected: 同一リストを返す。呼び出し後 instance.hits == 0、instance.misses == 1、len(instance._cache) == 1 となる
    """
    result = [{"text": "a", "confidence": 0.9}]
    engine = mock.Mock()
    engine.recognize_with_confidence.return_value = result
    inst = CachingOCR(engine, name="ocr")
    img = _image(b"img-bytes")
    ret = inst.recognize_with_confidence(img)
    assert ret is result
    assert inst.hits == 0
    assert inst.misses == 1
    assert len(inst._cache) == 1


def test_edge_02():
    """
    input: 同一画像での2回目の recognize_with_confidence 呼び出し
    expected: 同一リストオブジェクト（キャッシュされたオブジェクトの同一性）が返る。instance.hits == 1、instance.misses == 1。engine.recognize_with_confidence は通算1回のみ呼び出される
    """
    result = [{"text": "a", "confidence": 0.9}]
    engine = mock.Mock()
    engine.recognize_with_confidence.return_value = result
    inst = CachingOCR(engine, name="ocr")
    img = _image(b"img-bytes")
    first = inst.recognize_with_confidence(img)
    second = inst.recognize_with_confidence(img)
    assert first is result
    assert second is first
    assert inst.hits == 1
    assert inst.misses == 1
    assert engine.recognize_with_confidence.call_count == 1


def test_edge_03():
    """
    input: engine.recognize_with_confidence が None を返すスタブエンジンでの呼び出し
    expected: 初回呼び出しは None を返し misses が 1 増える。2回目はキャッシュから None を返し hits が 1 増える。engine は通算1回のみ呼び出される
    """
    engine = mock.Mock()
    engine.recognize_with_confidence.return_value = None
    inst = CachingOCR(engine, name="ocr")
    img = _image(b"img-bytes")
    first = inst.recognize_with_confidence(img)
    second = inst.recognize_with_confidence(img)
    assert first is None
    assert second is None
    assert inst.hits == 1
    assert inst.misses == 1
    assert engine.recognize_with_confidence.call_count == 1


def test_edge_04():
    """
    input: engine.recognize_with_confidence が []（空リスト）を返すスタブエンジンでの呼び出し
    expected: 空リストはキャッシュに格納され、2回目はキャッシュヒットで [] が返る（空リストはミス扱いにならない）
    """
    result = []
    engine = mock.Mock()
    engine.recognize_with_confidence.return_value = result
    inst = CachingOCR(engine, name="ocr")
    img = _image(b"img-bytes")
    first = inst.recognize_with_confidence(img)
    second = inst.recognize_with_confidence(img)
    assert first == []
    assert second is first
    assert inst.hits == 1
    assert inst.misses == 1
    assert engine.recognize_with_confidence.call_count == 1


def test_edge_05():
    """
    input: engine.recognize_with_confidence が KeyError を送出するスタブエンジンでの呼び出し
    expected: KeyError が呼び出し側に伝播する。self.misses は増加済み。self._cache にはキーが登録されないため、次に同一画像を呼び出すと engine が再び呼び出される
    """
    engine = mock.Mock()
    engine.recognize_with_confidence.side_effect = KeyError("boom")
    inst = CachingOCR(engine, name="ocr")
    img = _image(b"img-bytes")
    try:
        inst.recognize_with_confidence(img)
        assert False, "KeyError が送出されなかった"
    except KeyError:
        pass
    assert inst.misses == 1
    assert inst.hits == 0
    assert len(inst._cache) == 0
    try:
        inst.recognize_with_confidence(img)
        assert False, "KeyError が送出されなかった"
    except KeyError:
        pass
    assert engine.recognize_with_confidence.call_count == 2
