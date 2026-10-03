"""Tests for ``src.video.card_detector.CardDetector._detect_badges_multiscale``.

Specification: docs/00-Architecture/src_video_card_detector__CardDetector__detect_badges_multiscale.yaml

``CardDetector._detect_badges_multiscale(self, gray, S, s0)`` collects badge
candidates (matchTemplate + peak search) for every scale ``s`` in ``S``,
selects the winning scale ``s_w``, and returns the NMS-filtered badge list
of the winning scale, ``(x, y, bw, bh, role)``, as the 2-tuple
``(badges, s_w)``. Per the spec's ``behavior`` steps:

1. ``cands = self._collect_scale_cands(gray, S)`` — internally, for each
   ``s`` in ``S``, ``cv2.matchTemplate`` (TM_CCOEFF_NORMED) is run twice per
   scale (resized member / leader templates) and peaks are searched
2. if ``cands`` is empty, return ``([], None)`` and stop
3. ``s_w = self._pick_winning_scale(cands, s0)``
4. ``cands_w = [c for c in cands if c[2] == s_w]``
5. ``cands_w = self._nms_badges(cands_w)``
6. return ``(self._cands_to_badges(cands_w), s_w)``

External dependencies indicated by the spec are mocked with
``unittest.mock``; the tests themselves use no files, GUI, network, time, or
randomness and never call OpenCV/CV2 or NumPy directly:

- cv2/NumPy-backed template matching: every ``cv2.matchTemplate`` call
  happens inside ``self._collect_scale_cands`` (per the spec's
  ``side_effects``), so it is replaced by a ``Mock`` that returns the
  candidate list for each edge case;
- the template file loading in ``__init__`` (per the spec's preconditions):
  each instance is created with ``CardDetector.__new__(CardDetector)`` so no
  template files are read, and the minimal state the spec names
  (``templates``, ``_pyramid``, ``badge_threshold``) is set on the instance;
- the spec's own collaborators ``_pick_winning_scale``, ``_nms_badges`` and
  ``_cands_to_badges`` (called per the spec's ``side_effects``; their
  internal rules are unconfirmed/missing per the spec's
  ``unconfirmed``/``missing`` sections) are replaced by ``Mock`` objects
  implementing the contracts the spec itself states, so the tests assert the
  orchestration of the function under test: the 2-tuple shape, the
  ``([], None)`` early return, the ``c[2] == s_w`` filtering, and the call
  wiring between the steps.

``errors`` section of the spec (documented here only, NOT tested, per the
test-generation rules):

- condition: 'S のある s においてリサイズ後テンプレートが gray より大きい（幅または高さの片方でも超える）状態で cv2.matchTemplate が実行される'
  behavior: 'cv2.matchTemplate（OpenCV）による cv2.error'
- condition: 'gray が 2 次元単一チャネル配列でない（例。3 チャネル BGR 画像）'
  behavior: 'cv2.matchTemplate（OpenCV）による cv2.error'
"""

from unittest.mock import Mock

from src.video.card_detector import CardDetector


def test_edge_01():
    """
    input: S = [], gray は任意の有効な 2 次元画像, s0 = 1.0
    expected: スケールが走査されないため cands = [] となり、([], None) を返す
    """
    detector = CardDetector.__new__(CardDetector)
    detector.templates = {}
    detector._pyramid = {}
    detector.badge_threshold = 0.8
    # cv2/NumPy-backed collection is mocked: S is empty, so no scale is
    # scanned and no candidate is collected.
    detector._collect_scale_cands = Mock(return_value=[])
    detector._pick_winning_scale = Mock(return_value=None)
    detector._nms_badges = Mock(return_value=[])
    detector._cands_to_badges = Mock(return_value=[])

    gray = [[128] * 4 for _ in range(4)]  # any valid 2D image
    result = detector._detect_badges_multiscale(gray, [], 1.0)

    assert result == ([], None)
    assert result[1] is None
    detector._collect_scale_cands.assert_called_once_with(gray, [])
    detector._pick_winning_scale.assert_not_called()
    detector._nms_badges.assert_not_called()
    detector._cands_to_badges.assert_not_called()


def test_edge_02():
    """
    input: gray は 640x480 の単一値グレースケール画像（全ピクセル 128）, S = [0.8, 1.0, 1.2], s0 = 1.0
    expected: いずれの s でも self.badge_threshold 以上のピークが存在せず、([], None) を返す
    """
    detector = CardDetector.__new__(CardDetector)
    detector.templates = {}
    detector._pyramid = {}
    detector.badge_threshold = 0.8
    # A single-valued image has no peak above self.badge_threshold at any
    # scale, so the cv2/NumPy-backed collection yields no candidates.
    detector._collect_scale_cands = Mock(return_value=[])
    detector._pick_winning_scale = Mock(return_value=None)
    detector._nms_badges = Mock(return_value=[])
    detector._cands_to_badges = Mock(return_value=[])

    gray = [[128] * 640 for _ in range(480)]
    result = detector._detect_badges_multiscale(gray, [0.8, 1.0, 1.2], 1.0)

    assert result == ([], None)
    detector._collect_scale_cands.assert_called_once_with(gray, [0.8, 1.0, 1.2])
    detector._pick_winning_scale.assert_not_called()
    detector._nms_badges.assert_not_called()
    detector._cands_to_badges.assert_not_called()


def test_edge_03():
    """
    input: gray は 640x480 の単一値グレースケール画像（全ピクセル 128）に member テンプレートを (100, 100) に貼り付けたもの, S = [1.0], s0 = 1.0
    expected: 返り値の第 2 要素 s_w は 1.0。第 1 要素は 1 件の候補に対する _nms_badges の結果であり、_nms_badges が単独候補を保持する場合は [(x, y, 124, 37, "member")]（x, y の正確な値は _find_peaks の定義が未読のため不定）
    """
    detector = CardDetector.__new__(CardDetector)
    detector.templates = {}
    detector._pyramid = {}
    detector.badge_threshold = 0.8
    # One member peak at scale 1.0: candidate [x, y, s, sm, sl] with sm >= sl
    # (the pasted member template dominates over the leader score).
    candidate = [100, 100, 1.0, 0.9, 0.5]
    detector._collect_scale_cands = Mock(return_value=[list(candidate)])
    # With a single scale in S, the winner must be that scale (spec
    # postconditions: s_w is one of the candidates' c[2] values).
    detector._pick_winning_scale = Mock(return_value=1.0)
    # Per the edge-case expectation, _nms_badges keeps the lone candidate.
    detector._nms_badges = Mock(side_effect=lambda cands: [list(c) for c in cands])
    # Spec postconditions: badge = (x, y, bw, bh, role) with bw = int(124 * s),
    # bh = int(37 * s), role = "member" if sm >= sl else "leader".
    detector._cands_to_badges = Mock(
        side_effect=lambda cands: [
            (c[0], c[1], int(124 * c[2]), int(37 * c[2]),
             "member" if c[3] >= c[4] else "leader")
            for c in cands
        ]
    )

    gray = [[128] * 640 for _ in range(480)]
    for y in range(100, 100 + 37):  # member template pasted at (100, 100), 124 x 37 at s = 1.0
        for x in range(100, 100 + 124):
            gray[y][x] = 200

    result = detector._detect_badges_multiscale(gray, [1.0], 1.0)

    assert result == ([(100, 100, 124, 37, "member")], 1.0)
    assert result[1] == 1.0
    detector._pick_winning_scale.assert_called_once_with([candidate], 1.0)
    detector._nms_badges.assert_called_once_with([candidate])
    detector._cands_to_badges.assert_called_once_with([candidate])


def test_edge_04():
    """
    input: gray は s = 0.9 と s = 1.1 の両方で自己のテンプレートと強い一致領域を持つ画像（例。各スケールサイズでテンプレートを貼付したもの）, S = [0.9, 1.1], s0 = 1.0 で、score(0.9) > score(1.1) となる貼付配置
    expected: s_w = 0.9 となり、badges は s == 0.9 の候補のみ（NMS 後）から構成され、返り値の第 2 要素は 0.9（より s0 に近い 1.1 でも score が低いため選定されない）
    """
    detector = CardDetector.__new__(CardDetector)
    detector.templates = {}
    detector._pyramid = {}
    detector.badge_threshold = 0.8
    # Candidates at both scales: score(s) = sum(max(sm, sl)) is 0.9 at
    # s = 0.9 and 0.7 at s = 1.1, so score(0.9) > score(1.1) even though
    # s = 1.1 is closer to s0 = 1.0.
    cand_09 = [10, 10, 0.9, 0.9, 0.4]
    cand_11 = [20, 20, 1.1, 0.7, 0.3]
    cands = [cand_09, cand_11]
    detector._collect_scale_cands = Mock(return_value=[list(c) for c in cands])
    # Spec postconditions: s_w maximizes score(s) = sum(max(sm, sl))
    # (ties: smaller abs(s - s0)) -> 0.9 here.
    detector._pick_winning_scale = Mock(return_value=0.9)
    detector._nms_badges = Mock(side_effect=lambda cands: [list(c) for c in cands])
    detector._cands_to_badges = Mock(
        side_effect=lambda cands: [
            (c[0], c[1], int(124 * c[2]), int(37 * c[2]),
             "member" if c[3] >= c[4] else "leader")
            for c in cands
        ]
    )

    gray = [[128] * 640 for _ in range(480)]
    for y in range(10, 10 + 33):  # strong-match region at s = 0.9 (int(124 * 0.9) x int(37 * 0.9))
        for x in range(10, 10 + 111):
            gray[y][x] = 200
    for y in range(300, 300 + 40):  # strong-match region at s = 1.1 (int(124 * 1.1) x int(37 * 1.1))
        for x in range(300, 300 + 136):
            gray[y][x] = 200

    result = detector._detect_badges_multiscale(gray, [0.9, 1.1], 1.0)

    assert result == ([(10, 10, 111, 33, "member")], 0.9)
    assert result[1] == 0.9
    detector._pick_winning_scale.assert_called_once_with(cands, 1.0)
    # Only the winning scale's candidates (c[2] == 0.9) reach _nms_badges.
    detector._nms_badges.assert_called_once_with([cand_09])
    detector._cands_to_badges.assert_called_once_with([cand_09])
