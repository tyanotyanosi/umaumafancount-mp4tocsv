"""Tests for ``src.video.card_detector.CardDetector._refine_fine``.

Specification: docs/00-Architecture/src_video_card_detector__CardDetector__refine_fine.yaml

``CardDetector._refine_fine(gray, approx, S, s0)`` refines the coarse badge
positions produced by ``_detect_coarse``. For each coarse hit ``[ax, ay, as_,
role]`` (all four values are unpacked, but ``as_`` and ``role`` are never
referenced afterwards), it:

- Step 1: starts with ``cands = []``.
- Step 2a: computes ``r = int(self.refine_radius)``.
- Step 2b: clamps the local window to the image bounds:
  ``x0 = max(0, int(ax) - r)``, ``y0 = max(0, int(ay) - r)``,
  ``x1 = min(gray.shape[1], int(ax) + r)``, ``y1 = min(gray.shape[0], int(ay) + r)``
  (``int()`` truncates fractional coordinates).
- Step 2c: skips the hit when ``x1 - x0 < 2`` or ``y1 - y0 < 2``.
- Step 2d: takes the window view ``sub = gray[y0:y1, x0:x1]``.
- Step 2e: keeps only the scales ``s`` from ``S`` that fit the window, using
  the member template size only (``sub`` width >= ``max(1, int(member_w * s))``
  and ``sub`` height >= ``max(1, int(member_h * s))``).
- Step 2f: extends ``cands`` with ``self._collect_scale_cands(sub, valid,
  ox=x0, oy=y0)``, converting window-local coordinates to global ``gray``
  coordinates via ``ox``/``oy``.
- Step 3: returns ``([], None)`` when no candidate was collected.
- Step 4: otherwise picks the winning scale
  ``s_w = self._pick_winning_scale(cands, s0)`` (per-scale sum of
  ``max(sm, sl)``; on ties the scale closer to ``s0`` wins).
- Step 5: keeps only the candidates whose scale equals ``s_w`` and applies
  NMS via ``self._nms_badges``.
- Step 6: returns ``(self._cands_to_badges(cands_w), s_w)`` where each badge
  is the 5-tuple ``(x, y, bw, bh, role)`` with ``bw = int(124 * s_w)``,
  ``bh = int(37 * s_w)`` and role ``"member"`` when ``sm >= sl`` else
  ``"leader"``.

Per the spec's ``side_effects``, ``_collect_scale_cands`` builds a cv2 resize
pyramid and calls ``cv2.matchTemplate`` (member and leader per scale), so the
tests mock it with ``unittest.mock``: tests where the window is skipped before
the call assert it is never invoked, and the others assert on the recorded
call arguments and the returned result. ``_pick_winning_scale``,
``_nms_badges`` and ``_cands_to_badges`` run as-is; per the spec's
``unconfirmed`` note, the mock-based cases assume ``_nms_badges`` returns a
single candidate unchanged. The detector instances are created with
``CardDetector.__new__`` (the real ``__init__`` is out of spec scope) and
given exactly the state the spec requires to exist: ``refine_radius`` and
``self.templates`` (member 123x37, leader 124x37, per the spec's note on the
``_sample`` document).

``errors`` section of the spec (documented here only, NOT tested, per the
test-generation rules):

- condition: 'approx の要素が4値アンパッキング不能（例: 3要素または5要素のリスト）'
  behavior: 'for ax, ay, as_, role in approx で ValueError（unpack 件数不一致）。捕捉されず呼び出し元へ伝播。'
- condition: 'ax または ay が int() で変換不能（例: 数値でない文字列）'
  behavior: 'int(ax) / int(ay) で ValueError または TypeError。捕捉されず呼び出し元へ伝播。'
- condition: 'gray が1次元（shape が2要素未満）'
  behavior: 'gray.shape[1] の参照で IndexError。捕捉されず呼び出し元へ伝播。'
"""

import unittest.mock as mock

import numpy as np

from src.video.card_detector import CardDetector


def _make_detector(refine_radius):
    """Build a bare ``CardDetector`` instance with only the state the spec requires."""
    inst = CardDetector.__new__(CardDetector)
    inst.refine_radius = refine_radius
    inst.templates = {
        "member": {"w": 123, "h": 37},
        "leader": {"w": 124, "h": 37},
    }
    return inst


def test_edge_01():
    """
    input: approx = []（gray, S, s0 は任意）
    expected: ループ体は実行されず cands が空のまま。([], None) を返す。
    """
    inst = _make_detector(200.0)
    gray = np.zeros((4, 4), dtype=np.uint8)
    with mock.patch.object(
        inst, "_collect_scale_cands", mock.MagicMock(return_value=[])
    ) as collect:
        result = inst._refine_fine(gray, [], [1.0], 1.0)
    assert result == ([], None)
    collect.assert_not_called()


def test_edge_02():
    """
    input: gray が 4x4（h=4, w=4）、refine_radius = 1、approx = [[0.0, 0.0, 1.0, "member"]]
    expected: x0 = 0, y0 = 0, x1 = min(4, 1) = 1, y1 = min(4, 1) = 1 となり x1 - x0 = 1 < 2 のためスキップ。cands が空で ([], None) を返す。
    """
    inst = _make_detector(1)
    gray = np.zeros((4, 4), dtype=np.uint8)
    with mock.patch.object(
        inst, "_collect_scale_cands", mock.MagicMock(return_value=[])
    ) as collect:
        result = inst._refine_fine(gray, [[0.0, 0.0, 1.0, "member"]], [1.0], 1.0)
    assert result == ([], None)
    collect.assert_not_called()


def test_edge_03():
    """
    input: refine_radius = 0、gray が 100x100、approx = [[50.0, 50.0, 1.0, "member"]]
    expected: x0 = max(0, 50) = 50, x1 = min(100, 50) = 50 となり x1 - x0 = 0 < 2 のため全候補がスキップ。([], None) を返す。
    """
    inst = _make_detector(0)
    gray = np.zeros((100, 100), dtype=np.uint8)
    with mock.patch.object(
        inst, "_collect_scale_cands", mock.MagicMock(return_value=[])
    ) as collect:
        result = inst._refine_fine(gray, [[50.0, 50.0, 1.0, "member"]], [1.0], 1.0)
    assert result == ([], None)
    collect.assert_not_called()


def test_edge_04():
    """
    input: gray が 10x10、refine_radius = 100、approx = [[5.0, 5.0, 1.0, "member"]]、S = [1000.0]
    expected: 窓は画像全体 (10x10) になる。テンプレートの w, h が 1 以上なら int(w * 1000.0) >= 1000 > 10 のため valid = [] となり、_collect_scale_cands は候補なしを返す。cands が空で ([], None) を返す。
    """
    inst = _make_detector(100)
    gray = np.zeros((10, 10), dtype=np.uint8)
    with mock.patch.object(
        inst, "_collect_scale_cands", mock.MagicMock(return_value=[])
    ) as collect:
        result = inst._refine_fine(gray, [[5.0, 5.0, 1.0, "member"]], [1000.0], 1.0)
    assert result == ([], None)
    collect.assert_called_once()
    sub, valid = collect.call_args.args
    # The window spans the whole 10x10 image and the scale must be rejected.
    assert sub.shape == (10, 10)
    assert valid == []


def test_edge_05():
    """
    input: gray が 100x100、refine_radius = 30、approx = [[-5.0, -5.0, 1.0, "leader"]]、_collect_scale_cands をモックスパイ
    expected: x0 = max(0, -35) = 0, y0 = 0, x1 = min(100, 25) = 25, y1 = min(100, 25) = 25。窓は 2x2 以上なのでスキップされず、_collect_scale_cands は shape (25, 25) の sub、ox=0, oy=0 で1回呼ばれる（負座標は画像境界でクランプされる）。
    """
    inst = _make_detector(30)
    gray = np.zeros((100, 100), dtype=np.uint8)
    with mock.patch.object(
        inst, "_collect_scale_cands", mock.MagicMock(return_value=[])
    ) as collect:
        result = inst._refine_fine(gray, [[-5.0, -5.0, 1.0, "leader"]], [1.0], 1.0)
    assert result == ([], None)
    collect.assert_called_once()
    sub, _valid = collect.call_args.args
    assert sub.shape == (25, 25)
    assert collect.call_args.kwargs == {"ox": 0, "oy": 0}


def test_edge_06():
    """
    input: 有効な窓1つの _collect_scale_cands を [[10, 10, 1.0, 0.9, 0.5]] を返すようモック（引数を無視）、_nms_badges を同一関数のまま、S = [1.0]、s0 = 1.0
    expected: (badges, s_w) = ([(10, 10, 124, 37, "member")], 1.0) を返す。s_w = 1.0、bw = int(124 * 1.0) = 124、bh = int(37 * 1.0) = 37、sm = 0.9 >= sl = 0.5 なので role = "member"。
    """
    inst = _make_detector(30)
    gray = np.zeros((100, 100), dtype=np.uint8)
    with mock.patch.object(
        inst,
        "_collect_scale_cands",
        mock.MagicMock(return_value=[[10, 10, 1.0, 0.9, 0.5]]),
    ) as collect:
        result = inst._refine_fine(gray, [[50.0, 50.0, 1.0, "member"]], [1.0], 1.0)
    assert result == ([(10, 10, 124, 37, "member")], 1.0)
    collect.assert_called_once()


def test_edge_07():
    """
    input: approx 候補2つの _collect_scale_cands を窓1で A = [10, 10, 1.0, 0.9, 0.1]、窓2で B = [50, 50, 1.1, 0.95, 0.2] を返すようモック、_nms_badges を同一関数のまま、S = [1.0, 1.1]、s0 = 1.0
    expected: 勝者は全窓の候補を合計して決まる。scores = {1.0: 0.9, 1.1: 0.95} なので s_w = 1.1 となり (badges, s_w) = ([(50, 50, 136, 40, "member")], 1.1) を返す（bw = int(124 * 1.1) = 136、bh = int(37 * 1.1) = 40、sm = 0.95 >= sl = 0.2 なので role = "member"）。窓ごとに独立に決まらない。
    """
    inst = _make_detector(30)
    gray = np.zeros((100, 100), dtype=np.uint8)
    collect = mock.MagicMock(
        side_effect=[
            [[10, 10, 1.0, 0.9, 0.1]],
            [[50, 50, 1.1, 0.95, 0.2]],
        ]
    )
    with mock.patch.object(inst, "_collect_scale_cands", collect):
        result = inst._refine_fine(
            gray,
            [[10.0, 10.0, 1.0, "member"], [70.0, 70.0, 1.0, "member"]],
            [1.0, 1.1],
            1.0,
        )
    assert result == ([(50, 50, 136, 40, "member")], 1.1)
    assert collect.call_count == 2
