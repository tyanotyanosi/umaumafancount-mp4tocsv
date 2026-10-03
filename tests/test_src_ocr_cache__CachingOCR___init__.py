"""Tests for ``src.ocr.cache.CachingOCR.__init__``.

Specification: docs/00-Architecture/src_ocr_cache__CachingOCR___init__.yaml

``__init__`` initializes a CachingOCR instance: it keeps the wrapped
OCR engine and sets the cache name, cache dict and hit/miss counters:

- store ``engine`` in ``self.engine`` and ``name`` in ``self.name``
- create a new empty dict as ``self._cache``
- initialize ``self.hits`` and ``self.misses`` to 0
- no type / None check / interface validation of ``engine`` is
  performed; any object is stored

Mocked / stand-in dependencies (per the test-generation rules):
the ``engine`` argument is a mock (the real engine implementation is
outside the read scope); ``test_edge_03`` uses ``None`` directly.

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
    input: CachingOCR(engine)（name を省略）
    expected: 例外なくインスタンスが生成され、instance.name == 'ocr' となる
    """
    engine = mock.Mock()
    inst = CachingOCR(engine)
    assert inst.name == "ocr"
    assert inst.engine is engine
    assert inst._cache == {}
    assert inst.hits == 0
    assert inst.misses == 0


def test_edge_02():
    """
    input: CachingOCR(engine, name='c2')
    expected: 例外なくインスタンスが生成され、instance.name == 'c2' となる
    """
    engine = mock.Mock()
    inst = CachingOCR(engine, name="c2")
    assert inst.name == "c2"
    assert inst.engine is engine
    assert inst._cache == {}
    assert inst.hits == 0
    assert inst.misses == 0


def test_edge_03():
    """
    input: engine=None（OCR エンジン以外のオブジェクト）
    expected: __init__ 自体は例外を送出せず instance.engine == None のインスタンスが生成される。問題はその後の recognize / recognize_with_confidence 呼び出し時に初めて現れる（AttributeError）
    """
    inst = CachingOCR(None)
    assert inst.engine is None
    try:
        inst.recognize(_image(b"img-bytes"))
        assert False, "AttributeError が送出されなかった"
    except AttributeError:
        pass
