"""Tests for ``src.parser.name_mapper.NameMapper.__init__``.

Specification: docs/00-Architecture/src_parser_name_mapper__NameMapper___init__.yaml

``__init__`` initializes a NameMapper instance holding the mapping
definition dict and the decision parameters:

- set ``self.mapping = mapping or {}`` (a falsy mapping such as None
  or an empty dict yields a new empty dict)
- assign ``edit_distance_threshold``, ``unmapped_action`` and
  ``warn_on_approx`` to instance attributes as-is
- set ``self.user_names = self.mapping.get("user_names", {})``
- set ``self.raw_to_user = self.mapping.get("raw_to_user", {})``
- initialize ``self.unmapped_names = []`` and ``self.warnings = []``

There is no validation of any argument in the code.

Mocked / stand-in dependencies (per the test-generation rules):
none — ``NameMapper`` is instantiated directly with the spec
edge-case arguments.

``errors`` section of the spec (documented here only, NOT tested, per
the test-generation rules):

- condition: "mapping が dict 以外の truthy なオブジェクト"
  behavior: ".get メソッドを持たない場合、self.mapping の代入直後 self.mapping.get(\"user_names\", {}) で AttributeError が送出される。.get を持つ場合、この初期子自体は完了し得る（問題は map() 実行時に出る）。"
"""

from src.parser.name_mapper import NameMapper


def test_edge_01():
    """
    input: "mapping=None, edit_distance_threshold=2, unmapped_action='suggest', warn_on_approx=True（デフォルト）"
    expected: "self.mapping == {} かつ self.user_names == {} かつ self.raw_to_user == {} かつ self.unmapped_names == [] かつ self.warnings == [] になる。"
    """
    inst = NameMapper()
    assert inst.mapping == {}
    assert inst.user_names == {}
    assert inst.raw_to_user == {}
    assert inst.unmapped_names == []
    assert inst.warnings == []
    # The parameters are kept as-is (spec postconditions).
    assert inst.edit_distance_threshold == 2
    assert inst.unmapped_action == "suggest"
    assert inst.warn_on_approx is True


def test_edge_02():
    """
    input: "mapping={'user_names': {'alice': {'aliases': ['A']}}, 'raw_to_user': {'X': 'bob'}}"
    expected: "self.user_names == {'alice': {'aliases': ['A']}} かつ self.raw_to_user == {'X': 'bob'} になる。"
    """
    mapping = {"user_names": {"alice": {"aliases": ["A"]}},
               "raw_to_user": {"X": "bob"}}
    inst = NameMapper(mapping=mapping)
    assert inst.user_names == {"alice": {"aliases": ["A"]}}
    assert inst.raw_to_user == {"X": "bob"}
    # A truthy dict mapping is referenced as-is (not copied).
    assert inst.mapping is mapping
    assert inst.unmapped_names == []
    assert inst.warnings == []


def test_edge_03():
    """
    input: "mapping={}"
    expected: "self.mapping == {}（source 側の空辞書ではなく新しい空辞書）、self.user_names == {} かつ self.raw_to_user == {} になる。"
    """
    mapping = {}
    inst = NameMapper(mapping=mapping)
    assert inst.mapping == {}
    # A new empty dict, not the caller's one.
    assert inst.mapping is not mapping
    assert inst.user_names == {}
    assert inst.raw_to_user == {}
    assert inst.unmapped_names == []
    assert inst.warnings == []
