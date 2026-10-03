"""Tests for ``src.parser.result_parser.ResultParser.parse_batch._add``.

Specification: docs/00-Architecture/src_parser_result_parser__ResultParser_parse_batch__add.yaml

``_add`` is an inner function defined inside ``parse_batch``: it
appends one fan count to the candidate list for the given user name,
accumulating counts across multiple frames:

- ``candidates.setdefault(name, [])`` obtains (or creates) the list
  for key ``name`` and appends ``count`` to its end; no
  de-duplication, sorting or validation is performed
- returns None (implicitly)

Nested-function call-path policy (per the test-generation rules):
``_add`` is a closure that exists only during a ``parse_batch`` call
and cannot be invoked directly, so it is tested through the outer
function's call path: cards fed to ``parse_batch`` drive ``_add``
calls, and the observable effect is the returned result dict (whose
values are the per-name mode of the accumulated candidate lists, with
tie-breaking by digit count and then first-occurrence order).

Mocked / stand-in dependencies (per the test-generation rules):
in ``test_edge_03`` a stand-in mapper (``_FakeMapper``) is used
because the matched branch passes ``mapped.user_name`` to ``_add``
without validation, which is the only reachable way to call ``_add``
with an empty name (``parse_batch`` itself skips empty cleaned names
before calling ``_add``).

``errors`` section of the spec (documented here only, NOT tested, per
the test-generation rules):

- condition: "name が list / dict 等のハッシュ可能でない型（宣言型 str を満たさない）"
  behavior: "TypeError（dict の参照 / setdefault 時に発生）"
"""

from src.parser.result_parser import ResultParser


class _Mapped:
    """Stand-in for the return value of ``mapper.map``."""

    def __init__(self, warning=None, matched=False,
                 user_name=None, raw_name=None):
        self.warning = warning
        self.matched = matched
        self.user_name = user_name
        self.raw_name = raw_name


class _FakeMapper:
    """Stand-in for NameMapper (see the parse_batch spec)."""

    def __init__(self, map_result, unmapped_action="suggest"):
        self.map = lambda name: map_result
        self.unmapped_action = unmapped_action
        self.unmapped_names = []
        self.warnings = []


def test_edge_01():
    """
    input: _add("a", 1) のあと _add("a", 2)
    expected: candidates["a"] == [1, 2]（呼び出し順に追加され、重複排除されない）
    """
    # Two cards with the same name "a" drive _add("a", 1) then
    # _add("a", 2): the accumulated list is [1, 2] in call order.
    # The returned value is the mode of [1, 2]: a 1-1 tie, equal
    # digit counts, so the first-occurrence value (1) is chosen —
    # which is only correct if the list is [1, 2] in append order.
    ret = ResultParser().parse_batch(
        [{"cards": [{"name": "a", "fans": "001"},
                    {"name": "a", "fans": "002"}]}])
    assert ret == {"a": 1}


def test_edge_02():
    """
    input: _add("new", 100)（"new" が candidates にまだ存在しない場合）
    expected: candidates["new"] == [100]（新しいリストが作成される）
    """
    # A single card for a name not seen before drives
    # _add("new", 100): a new list [100] is created and its single
    # element is the mode.
    ret = ResultParser().parse_batch(
        [{"cards": [{"name": "new", "fans": "100"}]}])
    assert ret == {"new": 100}


def test_edge_03():
    """
    input: _add("", 0)
    expected: candidates[""] == [0]（空の名前も0も拒否されない）
    """
    # parse_batch skips empty cleaned names before calling _add, so
    # the reachable call path for _add("", 0) goes through the
    # matched branch: mapper.map returns matched=True with
    # user_name="" (passed to _add without validation), and the fan
    # text "000" int-ifies to 0. The result dict then contains the
    # empty-string key with the mode of [0].
    mapper = _FakeMapper(_Mapped(matched=True, user_name="",
                                 raw_name="A"))
    ret = ResultParser().parse_batch(
        [{"cards": [{"name": "A", "fans": "000"}]}], mapper)
    assert ret == {"": 0}
