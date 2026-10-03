"""Tests for ``src.ocr.cache.CachingOCR.__getattr__``.

Specification: docs/00-Architecture/src_ocr_cache__CachingOCR___getattr__.yaml

``__getattr__`` forwards access to attributes that do not exist on
the instance to the wrapped engine, so that CachingOCR can
transparently use the engine's methods and attributes:

- the Python interpreter only calls this method with the ``attr``
  argument when normal attribute lookup (instance ``__dict__``,
  class MRO) fails
- attributes that CachingOCR itself has (engine, name, _cache, hits,
  misses, stats, recognize, recognize_with_confidence, _key,
  ``__init__`` etc.) are found by normal lookup, so this method is
  not reached for them
- executes ``getattr(self.engine, attr)`` and returns its result

Mocked / stand-in dependencies (per the test-generation rules):
the ``engine`` argument is a mock (the real engine implementation is
outside the read scope); ``test_edge_04`` uses a bare
``object.__new__(CachingOCR)`` instance with no attributes at all.

``errors`` section of the spec (documented here only, NOT tested, per
the test-generation rules):

- condition: "属性がインスタンス上に不存在で、かつ self.engine 上にも存在しない"
  behavior: "AttributeError（getattr(self.engine, attr) が送出）"
- condition: "self.engine が未設定の状態（初期化不完全なインスタンス）で属性検索失敗が起きる"
  behavior: "__getattr__ による無限再帰 → RecursionError"
"""

from unittest import mock

from src.ocr.cache import CachingOCR


class _PlainEngine:
    """Stand-in engine with no attributes at all (a plain ``mock.Mock``
    auto-creates attributes and would not raise AttributeError)."""

    pass


def test_edge_01():
    """
    input: engine に存在する属性へのアクセス（例: engine に version 属性がある場合の cache.version）
    expected: engine.version が返る。engine に対して直接アクセスした場合と同一の値（同一のバインドメソッド）が得られる
    """
    engine = mock.Mock()
    engine.version = "1.0"
    inst = CachingOCR(engine, name="ocr")
    assert inst.version == engine.version
    assert inst.version is engine.version


def test_edge_02():
    """
    input: インスタンスにも engine にも存在しない属性へのアクセス（例: cache.nonexistent）
    expected: AttributeError が送出される（getattr(self.engine, attr) が送出）
    """
    engine = _PlainEngine()
    inst = CachingOCR(engine, name="ocr")
    try:
        inst.nonexistent
        assert False, "AttributeError が送出されなかった"
    except AttributeError:
        pass


def test_edge_03():
    """
    input: インスタンスと engine の両方に存在する属性へのアクセス（例: cache.engine）
    expected: インスタンス自身の属性が返る。__getattr__ は呼び出されない
    """
    engine = mock.Mock()
    inst = CachingOCR(engine, name="ocr")
    assert inst.engine is engine


def test_edge_04():
    """
    input: self.engine 未設定のインスタンス（例: object.__new__(CachingOCR) で作ったもの）に対し、存在しない属性をアクセスする
    expected: __getattr__ 内部の self.engine 参照が再度属性検索失敗となり __getattr__ が再帰的に呼び出され、無限再帰となる（RecursionError）
    """
    inst = object.__new__(CachingOCR)
    try:
        inst.nonexistent
        assert False, "RecursionError が送出されなかった"
    except RecursionError:
        pass
