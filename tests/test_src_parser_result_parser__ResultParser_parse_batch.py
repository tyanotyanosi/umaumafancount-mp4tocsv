"""Tests for ``src.parser.result_parser.ResultParser.parse_batch``.

Specification: docs/00-Architecture/src_parser_result_parser__ResultParser_parse_batch.yaml

``parse_batch`` merges per-frame card OCR results by majority vote and
returns a dict of user name -> fan count; when a mapper is passed,
names are mapped before counting, and unmapped names and warnings are
collected:

- initialize a local dict ``candidates`` (user name -> list of fan
  counts) and ``unmapped_action = "suggest"``
- if mapper is not None, reset ``mapper.unmapped_names`` and
  ``mapper.warnings`` to ``[]`` and read ``unmapped_action`` from
  ``mapper.unmapped_action`` (getattr, default "suggest")
- iterate frame_results in order; skip elements that are not dicts,
  lack the "cards" key, or whose value is falsy
- iterate cards in order; skip non-dict elements
- take the name: ``card["name"]`` (fall back to ``card["name_raw"]``
  only when the key is absent); None is treated as an empty string and
  cleaned by ``self._clean_user_name``; skip the card when the result
  is empty
- take the fan text: ``card["fans"]`` (fall back to
  ``card["fans_raw"]`` only when the key is absent); None is treated as
  an empty string and int-ified by ``self._parse_fan_text``; skip the
  card when the result is None
- if mapper is None: append the count to the list for that name in
  candidates
- if mapper is not None: ``mapped = mapper.map(name)``; if
  ``mapped.warning`` is not None and not already in
  ``mapper.warnings``, append it; if ``mapped.matched`` is truthy,
  append the count to ``mapped.user_name``; otherwise branch on
  ``unmapped_action``: "drop" -> skip; "suggest" -> append
  ``mapped.raw_name`` to ``mapper.unmapped_names`` if absent (not
  counted); any other value ("keep" etc.) -> treat the cleaned name as
  the real user name and append the count
- after all frames, count occurrences with Counter for each name and
  store the mode (most frequent value); on a tie, prefer the value
  with more digits (longer str representation); on a further tie,
  prefer the value that appeared first in processing order
- return the result dict

Mocked / stand-in dependencies (per the test-generation rules):
the ``mapper`` argument is a stand-in object (``_FakeMapper``)
exposing ``map`` / ``unmapped_names`` / ``warnings`` /
``unmapped_action`` (the real NameMapper implementation is outside the
read scope); ``ResultParser`` and its internal helpers
``_clean_user_name`` / ``_parse_fan_text`` run real.

``errors`` section of the spec (documented here only, NOT tested, per
the test-generation rules):

- condition: "frame_results が反復可能でない（例 int）"
  behavior: "TypeError（for frame in frame_results ループで発生）"
- condition: "mapper が None でないが map 属性が欠落・非呼出し可能であり、かつ有効な名前とファン数を持つカードが存在する"
  behavior: "AttributeError（mapper.map(name) の呼び出し箇所）"
- condition: "mapper.map の戻り値に warning / matched / user_name / raw_name のいずれかの属性が欠落している（かつ該当属性が参照される）"
  behavior: "AttributeError（属性参照箇所）"
- condition: "card["name"]/name_raw の値が int 等の非 str 型"
  behavior: "TypeError（_clean_user_name 内の re.sub が非 str に適用されず発生）"
- condition: "card["fans"]/fans_raw の値が int 等の非 str 型"
  behavior: "AttributeError（_parse_fan_text 内の str.replace が int に適用されず発生）"
"""

from src.parser.result_parser import ResultParser


class _Mapped:
    """Stand-in for the return value of ``mapper.map`` (attributes
    ``warning`` / ``matched`` / ``user_name`` / ``raw_name``)."""

    def __init__(self, warning=None, matched=False,
                 user_name=None, raw_name=None):
        self.warning = warning
        self.matched = matched
        self.user_name = user_name
        self.raw_name = raw_name


class _FakeMapper:
    """Stand-in for NameMapper: a callable ``map`` plus the
    ``unmapped_names`` / ``warnings`` / ``unmapped_action`` attributes."""

    def __init__(self, map_result, unmapped_action="suggest",
                 unmapped_names=None, warnings=None):
        self.map = lambda name: map_result
        self.unmapped_action = unmapped_action
        self.unmapped_names = list(unmapped_names or [])
        self.warnings = list(warnings or [])


def test_edge_01():
    """
    input: parse_batch([])
    expected: 空dict {} が返る
    """
    assert ResultParser().parse_batch([]) == {}


def test_edge_02():
    """
    input: parse_batch([{"cards": []}, {"cards": None}, {"other": 1}])
    expected: 空dict {} が返る（cards なし・偽値のフレームはスキップ）
    """
    assert ResultParser().parse_batch(
        [{"cards": []}, {"cards": None}, {"other": 1}]) == {}


def test_edge_03():
    """
    input: parse_batch([42, "frame", None])
    expected: 空dict {} が返る（dictでないフレームはスキップ、例外は起きない）
    """
    assert ResultParser().parse_batch([42, "frame", None]) == {}


def test_edge_04():
    """
    input: parse_batch([{"cards": ["not a card", 7, None]}])
    expected: 空dict {} が返る（dictでないカードはスキップ、例外は起きない）
    """
    assert ResultParser().parse_batch(
        [{"cards": ["not a card", 7, None]}]) == {}


def test_edge_05():
    """
    input: parse_batch([{"cards": [{"fans": "1234"}]}])
    expected: 空dict {} が返る（name/name_raw キー欠如 → 名前が空 → スキップ）
    """
    assert ResultParser().parse_batch(
        [{"cards": [{"fans": "1234"}]}]) == {}


def test_edge_06():
    """
    input: parse_batch([{"cards": [{"name": "（A）", "fans": "1234"}]}])
    expected: 空dict {} が返る（クリーニング後の名前が空 → スキップ）
    """
    assert ResultParser().parse_batch(
        [{"cards": [{"name": "（A）", "fans": "1234"}]}]) == {}


def test_edge_07():
    """
    input: parse_batch([{"cards": [{"name": "太郎", "fans": None}]}])
    expected: 空dict {} が返る（fans が None → 空文字列 → ファン数が None → スキップ）
    """
    assert ResultParser().parse_batch(
        [{"cards": [{"name": "太郎", "fans": None}]}]) == {}


def test_edge_08():
    """
    input: parse_batch([{"cards": [{"name": "太郎", "fans": "1234人"}]}])
    expected: {"太郎": 1234} が返る
    """
    assert ResultParser().parse_batch(
        [{"cards": [{"name": "太郎", "fans": "1234人"}]}]) == {"太郎": 1234}


def test_edge_09():
    """
    input: parse_batch([{"cards": [{"name_raw": "太郎", "fans_raw": "1,234人"}]}])
    expected: {"太郎": 1234} が返る（name/name_raw・fans/fans_raw へのフォールバック、カンマは除去）
    """
    assert ResultParser().parse_batch(
        [{"cards": [{"name_raw": "太郎", "fans_raw": "1,234人"}]}]) == {
            "太郎": 1234}


def test_edge_10():
    """
    input: parse_batch([{"cards": [{"name": "A", "fans": "O234"}]}])
    expected: {"A": 234} が返る（O が混同表で 0 に置換され "0234" → int 234）
    """
    assert ResultParser().parse_batch(
        [{"cards": [{"name": "A", "fans": "O234"}]}]) == {"A": 234}


def test_edge_11():
    """
    input: parse_batch([{"cards": [{"name": "A", "fans": "x234"}]}])
    expected: 空dict {} が返る（x は混同表に無く補正後も非数字が残るためファン数が None）
    """
    assert ResultParser().parse_batch(
        [{"cards": [{"name": "A", "fans": "x234"}]}]) == {}


def test_edge_12():
    """
    input: parse_batch([{"cards": [{"name": "太郎", "fans": "123"}, {"name": "太郎", "fans": "1234"}, {"name": "太郎", "fans": "1234"}]}])
    expected: {"太郎": 1234} が返る（多数決。1234 が2回で 123 の1回を上回る）
    """
    assert ResultParser().parse_batch(
        [{"cards": [{"name": "太郎", "fans": "123"},
                    {"name": "太郎", "fans": "1234"},
                    {"name": "太郎", "fans": "1234"}]}]) == {"太郎": 1234}


def test_edge_13():
    """
    input: parse_batch([{"cards": [{"name": "A", "fans": "123"}, {"name": "A", "fans": "9999"}]}])
    expected: {"A": 9999} が返る（同頻度1回の同票 → 桁数が多い方が採用）
    """
    assert ResultParser().parse_batch(
        [{"cards": [{"name": "A", "fans": "123"},
                    {"name": "A", "fans": "9999"}]}]) == {"A": 9999}


def test_edge_14():
    """
    input: parse_batch([{"cards": [{"name": "A", "fans": "111"}, {"name": "A", "fans": "999"}]}])
    expected: {"A": 111} が返る（同頻度・同桁数の同票 → 処理順で最初に現れた方が採用）
    """
    assert ResultParser().parse_batch(
        [{"cards": [{"name": "A", "fans": "111"},
                    {"name": "A", "fans": "999"}]}]) == {"A": 111}


def test_edge_15():
    """
    input: parse_batch([{"cards": [{"name": "A", "fans": "99"}, {"name": "A", "fans": "12345678901"}]}])
    expected: 空dict {} が返る（99 は3桁未満、12345678901 は 10000000000 超過で双方棄却）
    """
    assert ResultParser().parse_batch(
        [{"cards": [{"name": "A", "fans": "99"},
                    {"name": "A", "fans": "12345678901"}]}]) == {}


def test_edge_16():
    """
    input: parse_batch([{"cards": [{"name": "A", "fans": "10000000000"}]}])
    expected: {"A": 10000000000} が返る（上限 10000000000 は含む）
    """
    assert ResultParser().parse_batch(
        [{"cards": [{"name": "A", "fans": "10000000000"}]}]) == {
            "A": 10000000000}


def test_edge_17():
    """
    input: parse_batch([], mapper)（mapper に事前へ unmapped_names=["x"]、warnings=["y"] が設定済み）
    expected: 空dict {} が返り、呼び出し後 mapper.unmapped_names == [] かつ mapper.warnings == []（有効データがなくてもリセットは起きる）
    """
    mapper = _FakeMapper(_Mapped(), unmapped_names=["x"],
                         warnings=["y"])
    ret = ResultParser().parse_batch([], mapper)
    assert ret == {}
    assert mapper.unmapped_names == []
    assert mapper.warnings == []


def test_edge_18():
    """
    input: parse_batch([{"cards": [{"name": "検知名", "fans": "1000"}]}], mapper)（mapper.map は warning=None、matched=True、user_name="実名" を返す）
    expected: {"実名": 1000} が返る（マッピング後の名前で集計）
    """
    mapper = _FakeMapper(_Mapped(matched=True, user_name="実名",
                                 raw_name="検知名"))
    ret = ResultParser().parse_batch(
        [{"cards": [{"name": "検知名", "fans": "1000"}]}], mapper)
    assert ret == {"実名": 1000}


def test_edge_19():
    """
    input: parse_batch([{"cards": [{"name": "検知名", "fans": "1000"}]}], mapper)（mapper.unmapped_action="suggest" 又は属性欠如。map は matched=False、raw_name="検知名"、warning=None を返す）
    expected: 空dict {} が返り、mapper.unmapped_names == ["検知名"]（未マッピングは記録のみで集計されない）
    """
    mapper = _FakeMapper(_Mapped(matched=False, raw_name="検知名"),
                         unmapped_action="suggest")
    ret = ResultParser().parse_batch(
        [{"cards": [{"name": "検知名", "fans": "1000"}]}], mapper)
    assert ret == {}
    assert mapper.unmapped_names == ["検知名"]


def test_edge_20():
    """
    input: parse_batch([{"cards": [{"name": "検知名", "fans": "1000"}]}], mapper)（mapper.unmapped_action="drop"。map は matched=False、raw_name="検知名" を返す）
    expected: 空dict {} が返り、mapper.unmapped_names == []（drop は集計・記録双方に含めない）
    """
    mapper = _FakeMapper(_Mapped(matched=False, raw_name="検知名"),
                         unmapped_action="drop")
    ret = ResultParser().parse_batch(
        [{"cards": [{"name": "検知名", "fans": "1000"}]}], mapper)
    assert ret == {}
    assert mapper.unmapped_names == []


def test_edge_21():
    """
    input: parse_batch([{"cards": [{"name": "検知名", "fans": "1000"}]}], mapper)（mapper.unmapped_action="keep"。map は matched=False を返す）
    expected: {"検知名": 1000} が返る（keep はクリーニング後の名前で集計）
    """
    mapper = _FakeMapper(_Mapped(matched=False, raw_name="検知名"),
                         unmapped_action="keep")
    ret = ResultParser().parse_batch(
        [{"cards": [{"name": "検知名", "fans": "1000"}]}], mapper)
    assert ret == {"検知名": 1000}


def test_edge_22():
    """
    input: parse_batch([{"cards": [{"name": "検知名", "fans": "1000"}]}], mapper)（map は matched=True、user_name="実名"、warning="近似一致" を返す）
    expected: {"実名": 1000} が返り、mapper.warnings == ["近似一致"]（matched 時でも警告は蓄積される）
    """
    mapper = _FakeMapper(_Mapped(warning="近似一致", matched=True,
                                 user_name="実名", raw_name="検知名"))
    ret = ResultParser().parse_batch(
        [{"cards": [{"name": "検知名", "fans": "1000"}]}], mapper)
    assert ret == {"実名": 1000}
    assert mapper.warnings == ["近似一致"]


def test_edge_23():
    """
    input: parse_batch([{"cards": [{"name": "B", "fans": "1000"}, {"name": "C", "fans": "1000"}]}], mapper)（unmapped_action="suggest"。map は双方 matched=False、warning="近似一致"、raw_name は各々 "B" と "C" を返す）
    expected: 空dict {} が返り、mapper.warnings == ["近似一致"]（同一文字列は1件のみ）かつ mapper.unmapped_names == ["B", "C"]
    """
    mapper = _FakeMapper(None, unmapped_action="suggest")
    mapper.map = lambda name: _Mapped(warning="近似一致",
                                      matched=False, raw_name=name)
    ret = ResultParser().parse_batch(
        [{"cards": [{"name": "B", "fans": "1000"},
                    {"name": "C", "fans": "1000"}]}], mapper)
    assert ret == {}
    assert mapper.warnings == ["近似一致"]
    assert mapper.unmapped_names == ["B", "C"]


def test_edge_24():
    """
    input: parse_batch([{"cards": [{"name": "B", "fans": "1000"}, {"name": "B", "fans": "1000"}]}], mapper)（map は常に matched=False、raw_name="B"、warning="w" を返す）
    expected: 空dict {} が返り、mapper.unmapped_names == ["B"]（raw_name で重複排除）かつ mapper.warnings == ["w"]
    """
    mapper = _FakeMapper(_Mapped(warning="w", matched=False,
                                 raw_name="B"),
                         unmapped_action="suggest")
    ret = ResultParser().parse_batch(
        [{"cards": [{"name": "B", "fans": "1000"},
                    {"name": "B", "fans": "1000"}]}], mapper)
    assert ret == {}
    assert mapper.unmapped_names == ["B"]
    assert mapper.warnings == ["w"]
