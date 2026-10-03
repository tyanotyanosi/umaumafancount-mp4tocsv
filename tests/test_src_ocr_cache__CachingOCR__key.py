"""Tests for ``src.ocr.cache.CachingOCR._key``.

Specification: docs/00-Architecture/src_ocr_cache__CachingOCR__key.yaml

``_key`` generates the 16-hexadecimal digest string of the MD5 of the
byte string from ``image.tobytes()`` and uses it as the cache key
(staticmethod):

- a static method (``@staticmethod``) that does not take a ``self``
  argument
- computes and returns ``hashlib.md5(image.tobytes()).hexdigest()``

Mocked / stand-in dependencies (per the test-generation rules):
the ``image`` argument is a mock (or a bare object) whose
``tobytes()`` returns the specified value (the real ``tobytes``
implementation, e.g. numpy, is outside the read scope).

``errors`` section of the spec (documented here only, NOT tested, per
the test-generation rules):

- condition: "image に tobytes 属性が存在しない"
  behavior: "AttributeError"
- condition: "image.tobytes() が bytes 以外の型を返す"
  behavior: "TypeError（hashlib.md5 が送出）"
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
    input: tobytes() が b''（空バイト列）を返す image オブジェクト
    expected: 'd41d8cd98f00b204e9800998ecf8427e'（空バイト列の MD5）が返る
    """
    img = _image(b"")
    assert CachingOCR._key(img) == "d41d8cd98f00b204e9800998ecf8427e"


def test_edge_02():
    """
    input: tobytes() の結果が同一の別オブジェクト image A, B
    expected: _key(A) == _key(B) となる（キーはオブジェクト同一性ではなくバイト列で決まる）
    """
    img_a = _image(b"same-bytes")
    img_b = _image(b"same-bytes")
    assert CachingOCR._key(img_a) == CachingOCR._key(img_b)


def test_edge_03():
    """
    input: tobytes 属性を持たないオブジェクト
    expected: AttributeError が送出される（image.tobytes() の参照時）
    """
    class _NoTobytes:
        pass

    obj = _NoTobytes()
    try:
        CachingOCR._key(obj)
        assert False, "AttributeError が送出されなかった"
    except AttributeError:
        pass


def test_edge_04():
    """
    input: tobytes() が bytes 以外の型（例: str）を返すオブジェクト
    expected: hashlib.md5 が TypeError を送出する
    """
    img = _image("not-bytes")
    try:
        CachingOCR._key(img)
        assert False, "TypeError が送出されなかった"
    except TypeError:
        pass
