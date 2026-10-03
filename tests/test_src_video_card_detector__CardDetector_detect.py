"""src.video.card_detector.CardDetector.detect のテスト。

仕様書: docs/00-Architecture/src_video_card_detector__CardDetector_detect.yaml
対象関数: src.video.card_detector.CardDetector.detect

errors セクション（文書化のみ・テスト化しない）:
- frame が None の場合: cv2.cvtColor が cv2.error を送出（detect は捕捉せず呼び出し側に伝播）。
- frame の深度が cv2.cvtColor で扱えない（例: float32 / float64 の画像。COLOR_BGR2GRAY は 8U / 16U を前提）場合: cv2.cvtColor が cv2.error を送出（捕捉せず伝播）。
- frame が cv2 が画像に変換できない Python オブジェクト（例: str）の場合: cv2.cvtColor が cv2.error を送出（捕捉せず伝播）。

実装メモ:
- CardDetector は __new__ で構築する（実 __init__ はテンプレート 4 ファイルをディスクから読み込む
  外部ファイル依存を持つ。detect のプライベートメソッドはすべてモックされるためテンプレート内容は無関係）。
  属性は __init__ 仕様書のデフォルト値で設定する。
- cv2 は仕様書が言及する外部依存のため unittest.mock（src.video.card_detector.cv2）でモックし、
  cv2.cvtColor が shape (100, 200) の決定的なグレースケール画像を返す。
- プライベートメソッド（_detect_badges / _detect_at_scale / _estimate_scale_prior /
  _detect_coarse / _refine_fine / _detect_badges_multiscale / _not_touching_edge /
  _find_anchor / _clamp）は仕様書に従い unittest.mock でモックする。
"""

import unittest.mock as mock

import numpy as np

from src.video.card_detector import CardDetector

# (fh, fw) = (100, 200): 100 行 x 200 列の黒 BGR フレーム（全 edge テスト共通）
_FRAME = np.zeros((100, 200, 3), dtype=np.uint8)
# cv2.cvtColor が返すグレースケール画像（shape (100, 200)）
_GRAY = np.zeros((100, 200), dtype=np.uint8)
# フルスウィープのス케ール集合 S_full（仕様書 behavior ステップ 7。scale_step=0.05 デフォルトで 0.5〜2.0 の 31 要素）
_S_FULL = list(np.round(np.arange(0.5, 2.0 + 1e-9, 0.05), 6))

# detect が呼び出すプライベートメソッド（仕様書 side_effects / behavior）
_PRIVATE_METHODS = (
    "_detect_badges",
    "_detect_at_scale",
    "_estimate_scale_prior",
    "_detect_coarse",
    "_refine_fine",
    "_detect_badges_multiscale",
    "_not_touching_edge",
    "_find_anchor",
    "_clamp",
)


def _make_detector(**overrides):
    """__init__ 済み状態の CardDetector インスタンスを構築する。

    実 __init__ はテンプレートファイルのディスク読み込み（外部ファイル依存）を行うため
    スキップし、detect が読み取る属性（仕様書 inputs.constraints）を __init__ 仕様書の
    デフォルト値で設定する。overrides で各 edge の設定を置換する。
    """
    det = CardDetector.__new__(CardDetector)
    det.multi_scale = True
    det.scale_cache = True
    det.coarse_to_fine = True
    det.icon_threshold = 0.7
    det.label_threshold = 0.7
    det.max_cards = 3
    det.name_margin = 8
    det.name_v_margin = 10
    det.max_name_width = 400
    det.fan_width = 280
    det.scale_step = 0.05
    det._scale = None
    det._scale_fw = None
    det._scale_fh = None
    for name, value in overrides.items():
        setattr(det, name, value)
    return det


def _mock_privates(det, **config):
    """detect のプライベートメソッドを unittest.mock でモックし、モックの dict を返す。

    config の形: {メソッド名: {"return_value": ...} または {"side_effect": ...}}
    """
    mocks = {}
    for name in _PRIVATE_METHODS:
        m = mock.MagicMock()
        if name in config:
            m.configure_mock(**config[name])
        setattr(det, name, m)
        mocks[name] = m
    return mocks


def test_edge_01():
    """
    input: "frame = np.zeros((100, 200, 3), dtype=np.uint8)（100 行 x 200 列の黒 BGR 画像）、self.multi_scale=False、max_cards=3。mock: _detect_badges は [] を返す"
    expected: "fh=100、fw=200 が gray.shape[:2] から得られる。返り値は空リスト []。self._scale / _scale_fw / _scale_fh は None のまま（変更されない）。"
    """
    det = _make_detector(multi_scale=False, max_cards=3)
    m = _mock_privates(det, _detect_badges={"return_value": []})
    with mock.patch("src.video.card_detector.cv2") as mcv2:
        mcv2.cvtColor.return_value = _GRAY
        result = det.detect(_FRAME)
    assert result == []
    assert det._scale is None
    assert det._scale_fw is None
    assert det._scale_fh is None
    assert m["_detect_badges"].call_count == 1


def test_edge_02():
    """
    input: "同上 frame。multi_scale=True、scale_cache=True、self._scale=1.2、_scale_fw=200、_scale_fh=100（キャッシュが今回のサイズと一致）。mock: _detect_at_scale(gray, 1.2) は [(10, 20, 30, 12, 'member')] を返し、_not_touching_edge は True、_find_anchor は全て None"
    expected: "_detect_at_scale が 1 回のみ呼ばれ、_estimate_scale_prior / _detect_coarse / _detect_badges_multiscale は呼ばれない。self._scale=1.2 / _scale_fw=200 / _scale_fh=100 は変更されない。返り値は 1 件の Card リストで role='member'、badge_box=(10, 20, 30, 12)、fan_box=None。name_box は _clamp(49, 8, 471, 36, 200, 100) が呼ばれる（name_x=10+30+int(8*1.2)=49、name_right=10+30+int(400*1.2)=520、幅=471、y=20-int(10*1.2)=8、h=12+2*int(10*1.2)=36）。"
    """
    det = _make_detector(
        multi_scale=True,
        scale_cache=True,
        _scale=1.2,
        _scale_fw=200,
        _scale_fh=100,
    )
    m = _mock_privates(
        det,
        _detect_at_scale={"return_value": [(10, 20, 30, 12, "member")]},
        _not_touching_edge={"return_value": True},
        _find_anchor={"return_value": None},
        _clamp={"return_value": (99, 99, 99, 99)},
    )
    with mock.patch("src.video.card_detector.cv2") as mcv2:
        mcv2.cvtColor.return_value = _GRAY
        result = det.detect(_FRAME)
    assert len(result) == 1
    card = result[0]
    assert card.role == "member"
    assert card.badge_box == (10, 20, 30, 12)
    assert card.fan_box is None
    assert card.name_box == (99, 99, 99, 99)
    assert m["_detect_at_scale"].call_count == 1
    assert m["_estimate_scale_prior"].call_count == 0
    assert m["_detect_coarse"].call_count == 0
    assert m["_detect_badges_multiscale"].call_count == 0
    assert det._scale == 1.2
    assert det._scale_fw == 200
    assert det._scale_fh == 100
    assert m["_clamp"].call_count == 1
    assert m["_clamp"].call_args == mock.call(49, 8, 471, 36, 200, 100)


def test_edge_03():
    """
    input: "同上 frame。multi_scale=True、scale_cache=True、self._scale=None（初回）、coarse_to_fine=True。mock: _estimate_scale_prior(200) は (1.0, [0.9, 1.0, 1.1]) を返し、_detect_coarse(gray, [0.9, 1.0, 1.1], 1.0) は [(10, 20, 30, 12)]（非空）を返し、_refine_fine(gray, [(10, 20, 30, 12)], [0.9, 1.0, 1.1], 1.0) は ([(10, 20, 30, 12, 'member')], 1.05) を返し、_not_touching_edge は True、_find_anchor は全て None"
    expected: "approx が非空のため _refine_fine が呼ばれ、_detect_badges_multiscale は呼ばれない。s=1.05 としてバッジ 1 件が採用される（ユーザ名・ファン数領域の int() 計算も s=1.05 基準）。self._scale=1.05、_scale_fw=200、_scale_fh=100 に更新される。返り値は 1 件の Card リスト。"
    """
    det = _make_detector(multi_scale=True, coarse_to_fine=True)
    m = _mock_privates(
        det,
        _estimate_scale_prior={"return_value": (1.0, [0.9, 1.0, 1.1])},
        _detect_coarse={"return_value": [(10, 20, 30, 12)]},
        _refine_fine={"return_value": ([(10, 20, 30, 12, "member")], 1.05)},
        _not_touching_edge={"return_value": True},
        _find_anchor={"return_value": None},
        _clamp={"return_value": (77, 77, 77, 77)},
    )
    with mock.patch("src.video.card_detector.cv2") as mcv2:
        mcv2.cvtColor.return_value = _GRAY
        result = det.detect(_FRAME)
    assert len(result) == 1
    card = result[0]
    assert card.role == "member"
    assert card.badge_box == (10, 20, 30, 12)
    assert card.fan_box is None
    assert card.name_box == (77, 77, 77, 77)
    assert m["_detect_coarse"].call_count == 1
    assert m["_refine_fine"].call_count == 1
    assert m["_detect_badges_multiscale"].call_count == 0
    assert det._scale == 1.05
    assert det._scale_fw == 200
    assert det._scale_fh == 100


def test_edge_04():
    """
    input: "同上 frame。multi_scale=True、self._scale=None、coarse_to_fine=True。mock: _detect_coarse は []（空）を返し、_detect_badges_multiscale(gray, [0.9, 1.0, 1.1], 1.0) は ([], 1.0) を返し、フルスウィープの _detect_badges_multiscale(gray, S_full, 1.0) も ([], 1.0) を返し、_detect_badges(gray) は [(10, 20, 30, 12, 'leader')] を返し、_not_touching_edge は True"
    expected: "粗探索が空のため _detect_badges_multiscale が直接呼ばれ、空の結果に続くフルスウィープ（S_full は scale_step=0.05 デフォルトで [0.5, 0.55, ..., 2.0] の 31 要素、中心 1.0）も空のため、s=1.0 として _detect_badges が呼ばれる。self._scale=1.0 に更新される。返り値は role='leader' の 1 件の Card リスト。"
    """
    det = _make_detector(multi_scale=True, coarse_to_fine=True)
    m = _mock_privates(
        det,
        _estimate_scale_prior={"return_value": (1.0, [0.9, 1.0, 1.1])},
        _detect_coarse={"return_value": []},
        _detect_badges_multiscale={"return_value": ([], 1.0)},
        _detect_badges={"return_value": [(10, 20, 30, 12, "leader")]},
        _not_touching_edge={"return_value": True},
        _find_anchor={"return_value": None},
        _clamp={"return_value": (66, 66, 66, 66)},
    )
    with mock.patch("src.video.card_detector.cv2") as mcv2:
        mcv2.cvtColor.return_value = _GRAY
        result = det.detect(_FRAME)
    assert len(result) == 1
    card = result[0]
    assert card.role == "leader"
    assert card.badge_box == (10, 20, 30, 12)
    assert m["_detect_coarse"].call_count == 1
    assert m["_detect_badges_multiscale"].call_count == 2
    assert m["_detect_badges_multiscale"].call_args_list[0] == mock.call(
        _GRAY, [0.9, 1.0, 1.1], 1.0
    )
    assert m["_detect_badges_multiscale"].call_args_list[1] == mock.call(
        _GRAY, _S_FULL, 1.0
    )
    assert len(_S_FULL) == 31
    assert m["_detect_badges"].call_count == 1
    assert det._scale == 1.0
    assert det._scale_fw == 200
    assert det._scale_fh == 100


def test_edge_05():
    """
    input: "同上 frame。multi_scale=True、self._scale=None、coarse_to_fine=False。mock: _detect_badges_multiscale(gray, [0.9, 1.0, 1.1], 1.0) は ([], 1.0) を返し、フルスウィープの _detect_badges_multiscale(gray, S_full, 1.0) は ([(10, 20, 30, 12, 'member')], 0.7) を返し、_not_touching_edge は True、_find_anchor は全て None"
    expected: "coarse_to_fine=False のため _detect_coarse / _refine_fine は呼ばれない。限定スケール集合での検出が空のためフルスウィープにフォールバックし、badges_full / s_w_full が採用される（s=0.7）。self._scale=0.7 に更新される。返り値は 1 件の Card リスト（領域計算は s=0.7 基準）。"
    """
    det = _make_detector(multi_scale=True, coarse_to_fine=False)
    m = _mock_privates(
        det,
        _estimate_scale_prior={"return_value": (1.0, [0.9, 1.0, 1.1])},
        _detect_badges_multiscale={
            "side_effect": [([], 1.0), ([(10, 20, 30, 12, "member")], 0.7)]
        },
        _not_touching_edge={"return_value": True},
        _find_anchor={"return_value": None},
        _clamp={"return_value": (55, 55, 55, 55)},
    )
    with mock.patch("src.video.card_detector.cv2") as mcv2:
        mcv2.cvtColor.return_value = _GRAY
        result = det.detect(_FRAME)
    assert len(result) == 1
    card = result[0]
    assert card.role == "member"
    assert card.badge_box == (10, 20, 30, 12)
    assert card.fan_box is None
    assert m["_detect_coarse"].call_count == 0
    assert m["_refine_fine"].call_count == 0
    assert m["_detect_badges_multiscale"].call_count == 2
    assert m["_detect_badges_multiscale"].call_args_list[0] == mock.call(
        _GRAY, [0.9, 1.0, 1.1], 1.0
    )
    assert m["_detect_badges_multiscale"].call_args_list[1] == mock.call(
        _GRAY, _S_FULL, 1.0
    )
    assert m["_detect_badges"].call_count == 0
    assert det._scale == 0.7
    assert det._scale_fw == 200
    assert det._scale_fh == 100


def test_edge_06():
    """
    input: "multi_scale=False、frame は 100 行 x 200 列の BGR 画像。mock: _detect_badges は [(0, 50, 20, 10, 'member')]（x=0 で左端接）を返し、_not_touching_edge((0, 50, 20, 10, 'member'), 200, 100) は False"
    expected: "端接バッジは除外され、返り値は空リスト []。Card は構築されない。"
    """
    det = _make_detector(multi_scale=False)
    m = _mock_privates(
        det,
        _detect_badges={"return_value": [(0, 50, 20, 10, "member")]},
        _not_touching_edge={"return_value": False},
        _find_anchor={"return_value": None},
        _clamp={"return_value": (44, 44, 44, 44)},
    )
    with mock.patch("src.video.card_detector.cv2") as mcv2:
        mcv2.cvtColor.return_value = _GRAY
        result = det.detect(_FRAME)
    assert result == []
    assert m["_detect_badges"].call_count == 1
    assert m["_not_touching_edge"].call_count == 1
    assert m["_not_touching_edge"].call_args == mock.call((0, 50, 20, 10, "member"), 200, 100)
    assert m["_find_anchor"].call_count == 0
    assert m["_clamp"].call_count == 0


def test_edge_07():
    """
    input: "multi_scale=False、max_cards=3。mock: _detect_badges は [(10, 40, 20, 10, 'member'), (20, 20, 20, 10, 'leader'), (30, 40, 20, 10, 'member'), (40, 10, 20, 10, 'leader')]（4 件）を返し、_not_touching_edge は全て True、_find_anchor は全て None"
    expected: "y 座標昇順ソート（キー b[1]）後の並びは (40, 10, ...), (20, 20, ...), (10, 40, ...), (30, 40, ...) で、先頭 3 件に截断される。返り値は 3 件の Card: 1 件目 badge_box=(40, 10, 20, 10) role='leader'、2 件目 badge_box=(20, 20, 20, 10) role='leader'、3 件目 badge_box=(10, 40, 20, 10) role='member'（(30, 40, ...) は除外）。y=40 の同点 2 件はソート前の検出順（Python の stable sort で (10, 40, ...) が先）。各 Card の fan_box は None。"
    """
    det = _make_detector(multi_scale=False, max_cards=3)
    m = _mock_privates(
        det,
        _detect_badges={
            "return_value": [
                (10, 40, 20, 10, "member"),
                (20, 20, 20, 10, "leader"),
                (30, 40, 20, 10, "member"),
                (40, 10, 20, 10, "leader"),
            ]
        },
        _not_touching_edge={"return_value": True},
        _find_anchor={"return_value": None},
        _clamp={"return_value": (33, 33, 33, 33)},
    )
    with mock.patch("src.video.card_detector.cv2") as mcv2:
        mcv2.cvtColor.return_value = _GRAY
        result = det.detect(_FRAME)
    assert len(result) == 3
    assert result[0].badge_box == (40, 10, 20, 10)
    assert result[0].role == "leader"
    assert result[1].badge_box == (20, 20, 20, 10)
    assert result[1].role == "leader"
    assert result[2].badge_box == (10, 40, 20, 10)
    assert result[2].role == "member"
    assert all(card.fan_box is None for card in result)
    assert m["_not_touching_edge"].call_count == 4
    assert m["_find_anchor"].call_count == 6


def test_edge_08():
    """
    input: "multi_scale=False、デフォルト設定（s=1.0、name_margin=8、name_v_margin=10、max_name_width=400、fan_width=280）。mock: _detect_badges は [(10, 50, 30, 12, 'member')] を返し、_not_touching_edge は True、_find_anchor('i_icon', ...) は (200, 52, 16, 16) を返し、_find_anchor('label', ...) は (40, 68, 30, 8) を返す"
    expected: "icon 検出のため name_right=icon[0]=200。_clamp は name_box として (48, 40, 152, 32, 200, 100)（name_x=10+30+8=48、幅=200-48=152、y=50-10=40、h=12+20=32）が呼ばれる。label 検出のため fan_h=max(1, int(8*1.2))=9、fan_y=68+8//2-9//2=68、fan_box として _clamp(78, 68, 280, 9, 200, 100)（x=40+30+8=78、幅=int(280*1.0)=280）が呼ばれる。Card.fan_box は None ではなく 4 要素 tuple である（_clamp 本体は未読のため、クランプ後の最終値までは断言できない。_find_anchor の探索範囲は icon: x0=10, y0=40, x1=610, y1=72、label: x0=10, y0=52, x1=410, y1=200）。"
    """
    det = _make_detector(
        multi_scale=False,
        icon_threshold=0.7,
        label_threshold=0.7,
        name_margin=8,
        name_v_margin=10,
        max_name_width=400,
        fan_width=280,
    )
    m = _mock_privates(
        det,
        _detect_badges={"return_value": [(10, 50, 30, 12, "member")]},
        _not_touching_edge={"return_value": True},
        _find_anchor={
            "side_effect": [(200, 52, 16, 16), (40, 68, 30, 8)]
        },
        _clamp={
            "side_effect": [(111, 111, 111, 111), (222, 222, 222, 222)]
        },
    )
    with mock.patch("src.video.card_detector.cv2") as mcv2:
        mcv2.cvtColor.return_value = _GRAY
        result = det.detect(_FRAME)
    assert len(result) == 1
    card = result[0]
    assert card.role == "member"
    assert card.badge_box == (10, 50, 30, 12)
    icon_call = m["_find_anchor"].call_args_list[0]
    label_call = m["_find_anchor"].call_args_list[1]
    assert icon_call == mock.call(
        _GRAY, "i_icon", 0.7, x0=10, y0=40, x1=610, y1=72, s=1.0
    )
    assert label_call == mock.call(
        _GRAY, "label", 0.7, x0=10, y0=52, x1=410, y1=200, s=1.0
    )
    assert m["_clamp"].call_args_list == [
        mock.call(48, 40, 152, 32, 200, 100),
        mock.call(78, 68, 280, 9, 200, 100),
    ]
    assert card.name_box == (111, 111, 111, 111)
    assert card.fan_box is not None
    assert isinstance(card.fan_box, tuple)
    assert len(card.fan_box) == 4
    assert card.fan_box == (222, 222, 222, 222)
