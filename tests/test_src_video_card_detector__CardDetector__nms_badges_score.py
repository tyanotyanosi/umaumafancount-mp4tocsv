"""Tests for ``src.video.card_detector.CardDetector._nms_badges.score``.

Specification: docs/00-Architecture/src_video_card_detector__CardDetector__nms_badges_score.yaml

``score`` is the nested function ``def score(c):`` inside
``CardDetector._nms_badges`` (per the spec's ``function`` field). For a badge
candidate ``c`` — a 5-element list ``[x, y, s, score_m, score_l]`` per the
parent's docstring, of which the code reads only ``c[3]`` and ``c[4]`` — it
returns the larger of the member score ``c[3]`` and the leader score
``c[4]``:

- ``max(c[3], c[4])`` is evaluated and its result is returned as-is (the
  spec's ``behavior``), so the return value is the larger of ``c[3]`` and
  ``c[4]`` (the value itself when the two are equal)
- it has no side effects and no external dependencies (no files, GUI,
  network, time, randomness, or CV2/NumPy use), so the tests below invoke
  ``score`` directly with no mocks

Because ``score`` is a local function of ``_nms_badges`` it cannot be
imported by name; the module-level ``score`` object below is recovered from
the nested code object named ``score`` inside
``CardDetector._nms_badges.__code__.co_consts`` (searched recursively) and
rebound as a plain function via ``types.FunctionType`` with the module's
globals.

``errors`` section of the spec (documented here only, NOT tested, per the
test-generation rules):

- condition: 'c の長さが 5 未満'
  behavior: 'IndexError'
- condition: 'c[3] と c[4] が比較できない（例: None と数値の混在）'
  behavior: 'TypeError'
- condition: 'c がインデックスアクセス不能な型（例: int）'
  behavior: 'TypeError'
"""

import types

from src.video.card_detector import CardDetector

# Recover the nested ``score`` function object of ``CardDetector._nms_badges``
# (see the module docstring for the rationale).
_score_code = None
_stack = [CardDetector._nms_badges.__code__]
while _score_code is None and _stack:
    _code = _stack.pop()
    for _const in _code.co_consts:
        if not isinstance(_const, types.CodeType):
            continue
        if _const.co_name == "score" and _const.co_argcount == 1 and _const.co_varnames[0] == "c":
            _score_code = _const
            break
        _stack.append(_const)
if _score_code is None:
    raise AssertionError(
        "nested function 'score' (per the spec's function field) was not "
        "found inside CardDetector._nms_badges"
    )
score = types.FunctionType(_score_code, CardDetector._nms_badges.__globals__)


def test_edge_01():
    """
    input: c = [0, 0, 1.0, 0.9, 0.3]
    expected: 0.9 を返す
    """
    assert score([0, 0, 1.0, 0.9, 0.3]) == 0.9


def test_edge_02():
    """
    input: c = [0, 0, 1.0, 0.3, 0.9]
    expected: 0.9 を返す
    """
    assert score([0, 0, 1.0, 0.3, 0.9]) == 0.9


def test_edge_03():
    """
    input: c = [0, 0, 1.0, 0.5, 0.5]
    expected: 0.5 を返す（同値）
    """
    assert score([0, 0, 1.0, 0.5, 0.5]) == 0.5


def test_edge_04():
    """
    input: c = [0, 0, 1.0, 0.9]（長さが 4）
    expected: c[4] アクセス時に IndexError が送出される
    """
    try:
        score([0, 0, 1.0, 0.9])
    except IndexError:
        raised = True
    else:
        raised = False
    assert raised


def test_edge_05():
    """
    input: c = [0, 0, 1.0, None, 0.5]
    expected: max() 内で TypeError が送出される（None と float は比較できないため）
    """
    try:
        score([0, 0, 1.0, None, 0.5])
    except TypeError:
        raised = True
    else:
        raised = False
    assert raised
