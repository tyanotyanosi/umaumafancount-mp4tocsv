"""Tests for ``src.parser.name_mapper.NameMapper.map``.

Specification: docs/00-Architecture/src_parser_name_mapper__NameMapper_map.yaml

``map`` decides a single detected name by the priority
exact (user_names aliases) -> exact (raw_to_user) -> approximate
(distance <= threshold) -> no match, and returns a ``MappedName``:

1. exact (user_names aliases): scan ``self.user_names`` in dict
   insertion order as (user_name, entry); for the first user_name
   whose ``entry.get("aliases", [])`` contains raw_name, return
   ``MappedName(user_name, raw_name, True, 'exact', raw_name, None)``
2. exact (raw_to_user): if step 1 did not return and
   ``raw_name in self.raw_to_user``, return
   ``MappedName(self.raw_to_user[raw_name], raw_name, True, 'exact',
   raw_name, None)``
3. approximate (user_names aliases): for every (user_name, alias)
   pair compute ``dist = levenshtein_distance(raw_name, alias)``;
   skip ``dist == 0`` and ``dist > self.edit_distance_threshold``;
   keep the best candidate (smallest dist; on a tie the smaller
   user_name by Unicode code point order)
4. if a best candidate exists: with
   ``warning = f"近似一致 (edit_distance={dist}): '{raw_name}' ->
   '{user_name}' (alias '{alias}')"`` when ``self.warn_on_approx`` is
   true, else None, return
   ``MappedName(user_name, raw_name, True, 'approx', alias, warning)``
5. if no best candidate exists: return
   ``MappedName(None, raw_name, False, None, None, None)``

``self.unmapped_names`` and ``self.warnings`` are not modified by
``map`` (asserted in each test).

Mocked / stand-in dependencies (per the test-generation rules):
none — ``NameMapper`` is a plain class and ``levenshtein_distance`` is
a pure function; instances are constructed directly with the
``mapping`` dict from each spec edge case.

``errors`` section of the spec (documented here only, NOT tested, per
the test-generation rules):

- condition: "self.user_names の値 entry が dict でない（例: {'x': 'str'}）"
  behavior: "entry.get(\"aliases\", []) で AttributeError が送出される。"
- condition: "aliases が list 以外の反復可能型（例: str）"
  behavior: "例外は送出されない。1) の in 判定は部分文字列一致になり、3) は alias を1文字ずつの単位として距離計算する。"
- condition: "raw_to_user の値が非文字列（例: dict）"
  behavior: "例外は送出されず、MappedName の user_name フィールドとしてその値がそのまま返る（MappedName は型を強制しない）。"
"""

from src.parser.name_mapper import MappedName, NameMapper


def _mapper(user_names, raw_to_user=None, edit_distance_threshold=2,
            warn_on_approx=True):
    """Build a NameMapper from the spec edge-case state."""
    mapping = {"user_names": user_names, "raw_to_user": raw_to_user or {}}
    return NameMapper(mapping=mapping,
                      edit_distance_threshold=edit_distance_threshold,
                      warn_on_approx=warn_on_approx)


def _fields(result):
    """Return the six MappedName fields as a tuple."""
    assert isinstance(result, MappedName)
    return (result.user_name, result.raw_name, result.matched,
            result.match_type, result.alias_hit, result.warning)


def test_edge_01():
    """
    input: "user_names={'alice': {'aliases': ['A', 'AA']}}, raw_to_user={}, raw_name='AA'"
    expected: "MappedName('alice', 'AA', True, 'exact', 'AA', None) が返る。"
    """
    mapper = _mapper({"alice": {"aliases": ["A", "AA"]}})
    result = mapper.map("AA")
    assert _fields(result) == ("alice", "AA", True, "exact", "AA", None)
    assert mapper.unmapped_names == []
    assert mapper.warnings == []


def test_edge_02():
    """
    input: "user_names={}, raw_to_user={'X': 'bob'}, raw_name='X'"
    expected: "MappedName('bob', 'X', True, 'exact', 'X', None) が返る。"
    """
    mapper = _mapper({}, raw_to_user={"X": "bob"})
    result = mapper.map("X")
    assert _fields(result) == ("bob", "X", True, "exact", "X", None)
    assert mapper.unmapped_names == []
    assert mapper.warnings == []


def test_edge_03():
    """
    input: "user_names={'alice': {'aliases': ['A']}}, raw_to_user={'A': 'bob'}, raw_name='A'"
    expected: "user_names 側が優先され MappedName('alice', 'A', True, 'exact', 'A', None) が返る。"
    """
    mapper = _mapper({"alice": {"aliases": ["A"]}}, raw_to_user={"A": "bob"})
    result = mapper.map("A")
    assert _fields(result) == ("alice", "A", True, "exact", "A", None)
    assert mapper.unmapped_names == []
    assert mapper.warnings == []


def test_edge_04():
    """
    input: "user_names={'alice': {'aliases': ['alic']}}, edit_distance_threshold=2, warn_on_approx=True, raw_name='alice'"
    expected: "dist=1 であり MappedName('alice', 'alice', True, 'approx', 'alic', \"近似一致 (edit_distance=1): 'alice' -> 'alice' (alias 'alic')\") が返る。"
    """
    mapper = _mapper({"alice": {"aliases": ["alic"]}},
                     warn_on_approx=True)
    result = mapper.map("alice")
    assert _fields(result) == ("alice", "alice", True, "approx", "alic",
                               "近似一致 (edit_distance=1): 'alice' -> 'alice' (alias 'alic')")
    assert mapper.unmapped_names == []
    assert mapper.warnings == []


def test_edge_05():
    """
    input: "user_names={'alice': {'aliases': ['alic']}}, edit_distance_threshold=2, warn_on_approx=False, raw_name='alice'"
    expected: "MappedName('alice', 'alice', True, 'approx', 'alic', None) が返る。"
    """
    mapper = _mapper({"alice": {"aliases": ["alic"]}},
                     warn_on_approx=False)
    result = mapper.map("alice")
    assert _fields(result) == ("alice", "alice", True, "approx", "alic", None)
    assert mapper.unmapped_names == []
    assert mapper.warnings == []


def test_edge_06():
    """
    input: "user_names={'bob': {'aliases': ['xyz']}}, edit_distance_threshold=2, raw_name='alice'"
    expected: "dist('alice','xyz')=5 が閾値超過のため MappedName(None, 'alice', False, None, None, None) が返る。"
    """
    mapper = _mapper({"bob": {"aliases": ["xyz"]}})
    result = mapper.map("alice")
    assert _fields(result) == (None, "alice", False, None, None, None)
    assert mapper.unmapped_names == []
    assert mapper.warnings == []


def test_edge_07():
    """
    input: "user_names={'b_user': {'aliases': ['ayc']}, 'a_user': {'aliases': ['axb']}}, edit_distance_threshold=2, warn_on_approx=False, raw_name='axc'"
    expected: "両 user が dist=1 で同値のため user_name 文字列比較で小さい 'a_user' が採用され、MappedName('a_user', 'axc', True, 'approx', 'axb', None) が返る。"
    """
    mapper = _mapper({"b_user": {"aliases": ["ayc"]},
                      "a_user": {"aliases": ["axb"]}},
                     warn_on_approx=False)
    result = mapper.map("axc")
    assert _fields(result) == ("a_user", "axc", True, "approx", "axb", None)
    assert mapper.unmapped_names == []
    assert mapper.warnings == []


def test_edge_08():
    """
    input: "user_names={'u': {'aliases': ['ad', 'bc']}}, edit_distance_threshold=2, warn_on_approx=False, raw_name='ac'"
    expected: "同一 user の2つの alias が等距離 (dist=1) のため走査順の先頭 alias 'ad' が採用され、MappedName('u', 'ac', True, 'approx', 'ad', None) が返る。"
    """
    mapper = _mapper({"u": {"aliases": ["ad", "bc"]}},
                     warn_on_approx=False)
    result = mapper.map("ac")
    assert _fields(result) == ("u", "ac", True, "approx", "ad", None)
    assert mapper.unmapped_names == []
    assert mapper.warnings == []
