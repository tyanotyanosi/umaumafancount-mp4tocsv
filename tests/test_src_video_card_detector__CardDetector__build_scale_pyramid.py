"""Tests for ``src.video.card_detector.CardDetector._build_scale_pyramid``.

Specification: docs/00-Architecture/src_video_card_detector__CardDetector__build_scale_pyramid.yaml

``CardDetector._build_scale_pyramid(self, S)`` caches resized versions of all
templates into the instance attribute ``self._pyramid`` (a dict keyed by
scale) and returns the pyramid dict. Per the spec's ``behavior`` section, for
each ``s`` in ``S`` (in ``S`` order):

- If ``s`` is already a key of ``self._pyramid``, it is skipped entirely
  (no recomputation, no replacement of the existing entry).
- Otherwise ``self._pyramid[s] = {}`` is created and, for each ``(name, tpl)``
  in ``self.templates.items()``, ``w = max(1, int(tpl["w"] * s))`` and
  ``h = max(1, int(tpl["h"] * s))`` are computed and
  ``cv2.resize(tpl["gray"], (w, h), interpolation=cv2.INTER_AREA)`` is stored
  in ``self._pyramid[s][name]``.
- The returned object is ``self._pyramid`` itself (not a copy) and contains
  all previously cached scales, not only ``S``.

Per the spec's side effects the function depends on OpenCV (``cv2.resize``
with ``interpolation=cv2.INTER_AREA``), so a mock ``cv2`` module is installed
in ``sys.modules`` (before the module under test is imported, unless a real
``cv2`` module is already present) and each test patches ``cv2.resize`` to
record the calls and to return a stand-in image whose ``.shape`` is
``(height, width)``. The template state is constructed directly from the
spec's preconditions (the 4 template names "member", "leader", "i_icon",
"label", each a dict with a "gray" 2-D grayscale array and int "w"/"h"), so
``__init__`` side effects are not relied upon.

``errors`` section of the spec (documented here only, NOT tested, per the
test-generation rules):

- condition: 'S の要素 s がハッシュ可能でない型（list / dict 等）である'
  behavior: 'TypeError（s not in self._pyramid の membership チェックで発生）'
- condition: 'S の要素 s が数値以外（str 等）である'
  behavior: 'TypeError（tpl["w"] * s の掛け算で発生）'
"""

import sys
from unittest import mock

# The spec lists cv2.resize / cv2.INTER_AREA as external dependencies
# (side_effects). Use the already-imported cv2 module if one exists;
# otherwise install a mock so the module under test can be imported and its
# resize calls intercepted.
if "cv2" in sys.modules:
    _cv2 = sys.modules["cv2"]
else:
    _cv2 = mock.MagicMock(name="cv2")
    sys.modules["cv2"] = _cv2

from src.video.card_detector import CardDetector  # noqa: E402
import src.video.card_detector as _detector_module  # noqa: E402

# If the module under test was already imported earlier with a different cv2
# bound to it, rebind the reference so the patched object is always used.
if getattr(_detector_module, "cv2", None) is not _cv2:
    _detector_module.cv2 = _cv2


class _FakeImage:
    """Stand-in for a resized grayscale array; ``.shape`` is (height, width)."""

    def __init__(self, shape, source):
        self.shape = shape
        self.source = source


def _fake_resize(image, dsize, interpolation=None, **_kwargs):
    """Emulate cv2.resize: dsize is (width, height), result shape is (h, w)."""
    width, height = dsize
    return _FakeImage((height, width), image)


# Template dimensions chosen per the spec's preconditions: each template is a
# dict with a "gray" 2-D grayscale array and int "w" (width) / "h" (height).
_TEMPLATE_DIMS = {
    "member": (10, 20),
    "leader": (8, 16),
    "i_icon": (4, 4),
    "label": (12, 6),
}


def _make_detector():
    """Build a CardDetector that satisfies the spec's preconditions only."""
    detector = object.__new__(CardDetector)
    detector.templates = {
        name: {"gray": [[0]], "w": width, "h": height}
        for name, (width, height) in _TEMPLATE_DIMS.items()
    }
    detector._pyramid = {}
    return detector


def test_edge_01():
    """
    input: S = []
    expected: self._pyramid は変更されず、返り値は現在の self._pyramid そのもの（それまで呼び出しがなければ空 dict {}）
    """
    detector = _make_detector()
    with mock.patch.object(_cv2, "resize", side_effect=_fake_resize) as mresize:
        result = detector._build_scale_pyramid([])
    assert result == {}
    assert result is detector._pyramid
    assert mresize.call_count == 0


def test_edge_02():
    """
    input: S = [1.0] を同じインスタンスで 2 回呼び出す（初回キャッシュなし）
    expected: 2 回目の呼び出しでは 1.0 が既に self._pyramid に存在するためリサイズ計算は行われず、返り値は 1 回目と同一の dict オブジェクト（1.0 エントリを含む）
    """
    detector = _make_detector()
    with mock.patch.object(_cv2, "resize", side_effect=_fake_resize) as mresize:
        first = detector._build_scale_pyramid([1.0])
        calls_after_first = mresize.call_count
        second = detector._build_scale_pyramid([1.0])
        calls_after_second = mresize.call_count
    assert second is first
    assert calls_after_first == 4
    assert calls_after_second == 4
    assert 1.0 in second
    assert set(second[1.0].keys()) == set(_TEMPLATE_DIMS)


def test_edge_03():
    """
    input: S = [0.0]
    expected: int(tpl["w"] * 0.0) = 0 となるため max(1, ...) により w = 1, h = 1 にクランプされ、self._pyramid[0.0] へ各テンプレート名に対して 1x1 グレースケール配列が格納される
    """
    detector = _make_detector()
    with mock.patch.object(_cv2, "resize", side_effect=_fake_resize) as mresize:
        result = detector._build_scale_pyramid([0.0])
    assert result is detector._pyramid
    assert list(result.keys()) == [0.0]
    assert set(result[0.0].keys()) == set(_TEMPLATE_DIMS)
    for name in _TEMPLATE_DIMS:
        assert result[0.0][name].shape == (1, 1)
    assert mresize.call_count == 4
    for call in mresize.call_args_list:
        args, kwargs = call
        assert (args[1][0], args[1][1]) == (1, 1)
        assert kwargs.get("interpolation") is _cv2.INTER_AREA


def test_edge_04():
    """
    input: S = [-1.0]
    expected: int(tpl["w"] * s) が負になるが max(1, ...) により w = 1, h = 1 にクランプされ、self._pyramid[-1.0] へ各テンプレート名に対して 1x1 配列が格納される
    """
    detector = _make_detector()
    with mock.patch.object(_cv2, "resize", side_effect=_fake_resize) as mresize:
        result = detector._build_scale_pyramid([-1.0])
    assert list(result.keys()) == [-1.0]
    assert set(result[-1.0].keys()) == set(_TEMPLATE_DIMS)
    for name in _TEMPLATE_DIMS:
        assert result[-1.0][name].shape == (1, 1)
    assert mresize.call_count == 4
    for call in mresize.call_args_list:
        assert (call.args[1][0], call.args[1][1]) == (1, 1)


def test_edge_05():
    """
    input: S = [1, 1.0]（初回呼び出し、キャッシュなし）
    expected: 1 と 1.0 は dict 上同一キー（1 == 1.0 で hash も同一）であるため、self._pyramid にはキー 1 のエントリが 1 つのみ作成され、self._pyramid[1.0] で同一 dict へアクセスできる
    """
    detector = _make_detector()
    with mock.patch.object(_cv2, "resize", side_effect=_fake_resize) as mresize:
        result = detector._build_scale_pyramid([1, 1.0])
    assert list(result.keys()) == [1]
    assert result[1.0] is result[1]
    assert mresize.call_count == 4
    assert result[1]["member"].shape == (20, 10)
    assert result[1]["i_icon"].shape == (4, 4)
