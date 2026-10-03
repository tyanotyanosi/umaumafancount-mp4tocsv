"""Tests for ``src.ocr.cache.CachingOCR.recognize``.

Specification: docs/00-Architecture/src_ocr_cache__CachingOCR_recognize.yaml

``recognize`` calls ``self.engine.recognize(image)`` and caches the
result keyed by the MD5 of the image bytes; a second and later call
with the same image does not call the engine and returns the cached
result:

- compute the key as the tuple
  ``('recognize', CachingOCR._key(image))``
- on a cache hit, increment ``self.hits`` by 1 and return the cached
  value (None when the cached value is the placeholder ``_NONE``)
- on a miss, increment ``self.misses`` by 1 and call
  ``self.engine.recognize(image)``
- store the engine's return value in ``self._cache`` (a None result
  is stored as the placeholder ``_NONE``) and return it as-is
- the cache key is namespaced by method name, so the same image does
  not share a key with ``recognize_with_confidence``

Mocked / stand-in dependencies (per the test-generation rules):
the ``engine`` argument is a mock whose ``recognize`` returns (or
raises) the specified value; the ``image`` argument is a mock whose
``tobytes()`` returns the specified bytes (the real engine
implementation is outside the read scope).

``errors`` section of the spec (documented here only, NOT tested, per
the test-generation rules):

- condition: "キャッシュミス時で、self.engine に recognize 属性が存在しない"
  behavior: "AttributeError（self.engine.recognize の参照時に送出）"
- condition: "image に tobytes 属性がない、または tobytes() が bytes 以外の型を返す"
  behavior: "AttributeError / TypeError（キャッシュ参照より前の _key 計算時に送出）"
- condition: "self.engine.recognize(image) が例外を送出する"
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
    input: engine.recognize が 'abc' を返すスタブエンジンでの recognize(img) 初回呼び出し
    expected: 'abc' を返す。呼び出し後 instance.hits == 0、instance.misses == 1、len(instance._cache) == 1 となる
    """
    engine = mock.Mock()
    engine.recognize.return_value = "abc"
    inst = CachingOCR(engine, name="ocr")
    img = _image(b"img-bytes")
    ret = inst.recognize(img)
    assert ret == "abc"
    assert inst.hits == 0
    assert inst.misses == 1
    assert len(inst._cache) == 1


def test_edge_02():
    """
    input: 同一 img での2回目の recognize(img) 呼び出し
    expected: 'abc' を返す。instance.hits == 1、instance.misses == 1（増加しない）。engine.recognize は通算1回のみ呼び出される
    """
    engine = mock.Mock()
    engine.recognize.return_value = "abc"
    inst = CachingOCR(engine, name="ocr")
    img = _image(b"img-bytes")
    first = inst.recognize(img)
    second = inst.recognize(img)
    assert first == "abc"
    assert second == "abc"
    assert inst.hits == 1
    assert inst.misses == 1
    assert engine.recognize.call_count == 1


def test_edge_03():
    """
    input: engine.recognize が None を返すスタブエンジンでの recognize(img) 呼び出し
    expected: 初回呼び出しは None を返し misses が 1 増える。2回目はキャッシュから None を返し hits が 1 増える。engine.recognize は通算1回のみ呼び出される
    """
    engine = mock.Mock()
    engine.recognize.return_value = None
    inst = CachingOCR(engine, name="ocr")
    img = _image(b"img-bytes")
    first = inst.recognize(img)
    second = inst.recognize(img)
    assert first is None
    assert second is None
    assert inst.hits == 1
    assert inst.misses == 1
    assert engine.recognize.call_count == 1


def test_edge_04():
    """
    input: engine.recognize が ''（空文字列）を返すスタブエンジンでの recognize(img) 呼び出し
    expected: 初回は '' を返す。2回目はキャッシュヒットで '' が返る。空文字列は None プレースホルダとは区別されキャッシュされる
    """
    engine = mock.Mock()
    engine.recognize.return_value = ""
    inst = CachingOCR(engine, name="ocr")
    img = _image(b"img-bytes")
    first = inst.recognize(img)
    second = inst.recognize(img)
    assert first == ""
    assert second == ""
    assert inst.hits == 1
    assert inst.misses == 1
    assert engine.recognize.call_count == 1


def test_edge_05():
    """
    input: tobytes() が同一の別 image オブジェクト A, B に対する recognize(A) 後の recognize(B) 呼び出し
    expected: 2回目がキャッシュヒットとなる（hits == 1、misses == 1）。engine.recognize は通算1回のみ呼び出される
    """
    engine = mock.Mock()
    engine.recognize.return_value = "abc"
    inst = CachingOCR(engine, name="ocr")
    img_a = _image(b"same-bytes")
    img_b = _image(b"same-bytes")
    first = inst.recognize(img_a)
    second = inst.recognize(img_b)
    assert first == "abc"
    assert second == "abc"
    assert inst.hits == 1
    assert inst.misses == 1
    assert engine.recognize.call_count == 1


def test_edge_06():
    """
    input: engine.recognize が ValueError を送出するスタブエンジンでの recognize(img) 呼び出し
    expected: ValueError が呼び出し側に伝播する。self.misses は増加済み。self._cache にはキーが登録されないため、次に同一画像を呼び出すと engine.recognize が再び呼び出される
    """
    engine = mock.Mock()
    engine.recognize.side_effect = ValueError("boom")
    inst = CachingOCR(engine, name="ocr")
    img = _image(b"img-bytes")
    try:
        inst.recognize(img)
        assert False, "ValueError が送出されなかった"
    except ValueError:
        pass
    assert inst.misses == 1
    assert inst.hits == 0
    assert len(inst._cache) == 0
    try:
        inst.recognize(img)
        assert False, "ValueError が送出されなかった"
    except ValueError:
        pass
    assert engine.recognize.call_count == 2
