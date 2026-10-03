"""Tests for ``src.video.card_detector.CardDetector.__init__``.

Spec: docs/00-Architecture/src_video_card_detector__CardDetector___init__.yaml

function    : src.video.card_detector.CardDetector.__init__
signature   : def __init__(self, template_dir: str, settings: Optional[dict] = None):
purpose     : CardDetector を初期化する。settings の 'card_detection' 以下の設定を
              読み込む（未指定のキーはすべてデフォルト値）、マルチスケール検出の
              キャッシュ状態を初期化し、4 種のテンプレート画像をロードする。

The template-file dependency the spec mentions (side_effects:
"ファイルシステムから 4 ファイルを読む（self._load 経由で cv2.imread …）") is
mocked with unittest.mock: ``CardDetector._load`` stands in for the actual
cv2.imread-based file reads, so no real files, images or OpenCV state are
needed and the tests stay deterministic.

errors (documented only per workflow rules; intentionally not tested):
- condition: "4 テンプレートのいずれかが不存在、または cv2.imread が None を返す（デコード不能等）"
  behavior: "FileNotFoundError。メッセージは 'テンプレートが見つかりません: <path>'。__init__ は捕捉せず呼び出し側に伝播する。"
- condition: "'card_detection' キーが存在するが値が dict 以外（例: str, int）"
  behavior: "AttributeError（非 dict オブジェクトへの .get 呼び出し）。"
- condition: "数値へキャストされる設定値が数値変換不能な str（例: 'abc'）"
  behavior: "ValueError（float()/int() 変換失敗）。"
- condition: "設定値が明示的に None（例: {'max_cards': None}）"
  behavior: "TypeError（int(None)/float(None) 変換失敗。キーが存在すると .get のデフォルト値は使われない）。"
"""

from unittest.mock import MagicMock, patch

import pytest

from src.video.card_detector import CardDetector

# template_dir はコード上検証されない（仕様 inputs）。_load へそのまま渡される
# だけで、_load はモックされるため、固定文字列で十分（決定論的）。
TEMPLATE_DIR = "templates"

# self.templates のキー（仕様 postconditions）。
_TEMPLATE_KEYS = ("member", "leader", "i_icon", "label")


def _load_result() -> dict:
    """テンプレート読込結果1件：キー 'gray'/'w'/'h' を持つ dict（仕様 postconditions）。"""
    return {"gray": b"tpl", "w": 64, "h": 32}


def _assert_default_settings(detector) -> None:
    """18 個の設定属性がすべてデフォルト値であることを検証（仕様 behavior 3）。"""
    # float 系
    assert detector.badge_threshold == 0.6
    assert detector.label_threshold == 0.7
    assert detector.icon_threshold == 0.7
    assert detector.scale_window_low == 0.8
    assert detector.scale_window_high == 1.3
    assert detector.scale_step == 0.05
    assert detector.coarse_scale == 0.5
    assert detector.refine_radius == 200.0
    # int 系
    assert detector.max_cards == 3
    assert detector.edge_margin == 8
    assert detector.name_margin == 8
    assert detector.name_v_margin == 10
    assert detector.max_name_width == 400
    assert detector.fan_width == 280
    assert detector.reference_width == 2560
    # bool 系
    assert detector.multi_scale is True
    assert detector.coarse_to_fine is True
    assert detector.scale_cache is True


def test_edge_01():
    """Edge case 1.

    input: "settings=None（テンプレート 4 ファイルが存在）"
    expected: "全属性がデフォルト値になる（badge_threshold=0.6, label_threshold=0.7, icon_threshold=0.7, max_cards=3, edge_margin=8, name_margin=8, name_v_margin=10, max_name_width=400, fan_width=280, reference_width=2560, multi_scale=True, scale_window_low=0.8, scale_window_high=1.3, scale_step=0.05, coarse_to_fine=True, coarse_scale=0.5, refine_radius=200.0, scale_cache=True）。_pyramid は空 dict、self.templates は 4 キーを持つ。"
    """
    with patch(
        "src.video.card_detector.CardDetector._load",
        return_value=_load_result(),
    ) as mock_load:
        detector = CardDetector(TEMPLATE_DIR, None)
        assert mock_load.call_count == 4
    _assert_default_settings(detector)
    assert detector._pyramid == {}
    assert set(detector.templates) == set(_TEMPLATE_KEYS)
    assert detector._scale is None
    assert detector._scale_fw is None
    assert detector._scale_fh is None


def test_edge_02():
    """Edge case 2.

    input: "settings={}"
    expected: "settings=None と同一（falsy なので {} に置換され、全属性がデフォルト値）。"
    """
    with patch(
        "src.video.card_detector.CardDetector._load",
        return_value=_load_result(),
    ) as mock_load:
        detector = CardDetector(TEMPLATE_DIR, {})
        assert mock_load.call_count == 4
    _assert_default_settings(detector)
    assert detector._pyramid == {}
    assert set(detector.templates) == set(_TEMPLATE_KEYS)


def test_edge_03():
    """Edge case 3.

    input: "settings={'card_detection': {}}"
    expected: "全属性がデフォルト値（サブ dict は存在するが空のため）。"
    """
    with patch(
        "src.video.card_detector.CardDetector._load",
        return_value=_load_result(),
    ) as mock_load:
        detector = CardDetector(TEMPLATE_DIR, {"card_detection": {}})
        assert mock_load.call_count == 4
    _assert_default_settings(detector)
    assert detector._pyramid == {}
    assert set(detector.templates) == set(_TEMPLATE_KEYS)


def test_edge_04():
    """Edge case 4.

    input: "settings={'unrelated_key': {'max_cards': 9}}"
    expected: "全属性がデフォルト値。'card_detection' キーが欠落するため 'unrelated_key' は参照されず max_cards=3。"
    """
    with patch(
        "src.video.card_detector.CardDetector._load",
        return_value=_load_result(),
    ) as mock_load:
        detector = CardDetector(TEMPLATE_DIR, {"unrelated_key": {"max_cards": 9}})
        assert mock_load.call_count == 4
    _assert_default_settings(detector)
    assert detector.max_cards == 3
    assert set(detector.templates) == set(_TEMPLATE_KEYS)


def test_edge_05():
    """Edge case 5.

    input: "settings={'card_detection': {'badge_match_threshold': 0.9, 'max_cards': '5'}}"
    expected: "badge_threshold=0.9 かつ max_cards=5（int('5') のキャストは成功）。残りのキーはデフォルト値。"
    """
    settings = {"card_detection": {"badge_match_threshold": 0.9, "max_cards": "5"}}
    with patch(
        "src.video.card_detector.CardDetector._load",
        return_value=_load_result(),
    ) as mock_load:
        detector = CardDetector(TEMPLATE_DIR, settings)
        assert mock_load.call_count == 4
    # 指定された 2 キー
    assert detector.badge_threshold == 0.9
    assert detector.max_cards == 5
    # 残りはデフォルト値
    assert detector.label_threshold == 0.7
    assert detector.icon_threshold == 0.7
    assert detector.edge_margin == 8
    assert detector.name_margin == 8
    assert detector.name_v_margin == 10
    assert detector.max_name_width == 400
    assert detector.fan_width == 280
    assert detector.reference_width == 2560
    assert detector.multi_scale is True
    assert detector.scale_window_low == 0.8
    assert detector.scale_window_high == 1.3
    assert detector.scale_step == 0.05
    assert detector.coarse_to_fine is True
    assert detector.coarse_scale == 0.5
    assert detector.refine_radius == 200.0
    assert detector.scale_cache is True
    assert set(detector.templates) == set(_TEMPLATE_KEYS)


def test_edge_06():
    """Edge case 6.

    input: "settings={'card_detection': {'multi_scale': '0'}}"
    expected: "multi_scale=True になる（非空文字列は truthy なので bool('0') は True）。"
    """
    settings = {"card_detection": {"multi_scale": "0"}}
    with patch(
        "src.video.card_detector.CardDetector._load",
        return_value=_load_result(),
    ) as mock_load:
        detector = CardDetector(TEMPLATE_DIR, settings)
        assert mock_load.call_count == 4
    assert detector.multi_scale is True
    # 残りはデフォルト値（multi_scale もデフォルト値と同一）
    _assert_default_settings(detector)
    assert set(detector.templates) == set(_TEMPLATE_KEYS)


def test_edge_07():
    """Edge case 7.

    input: "settings={'card_detection': {'max_cards': None}}"
    expected: "int(None) により TypeError。キーが存在するため .get は None を返し、デフォルト値 3 は使われない。"
    """
    settings = {"card_detection": {"max_cards": None}}
    with patch(
        "src.video.card_detector.CardDetector._load",
        return_value=_load_result(),
    ) as mock_load:
        with pytest.raises(TypeError):
            CardDetector(TEMPLATE_DIR, settings)
        # TypeError は behavior ステップ 3（テンプレート読込ステップ 5 の前）で起きる
        assert mock_load.call_count == 0


def test_edge_08():
    """Edge case 8.

    input: "settings={'card_detection': 'not_a_dict'}"
    expected: "ステップ 3 で AttributeError（'str' オブジェクトに .get が存在しない）。"
    """
    settings = {"card_detection": "not_a_dict"}
    with patch(
        "src.video.card_detector.CardDetector._load",
        return_value=_load_result(),
    ) as mock_load:
        with pytest.raises(AttributeError):
            CardDetector(TEMPLATE_DIR, settings)
        # AttributeError は behavior ステップ 3（テンプレート読込ステップ 5 の前）で起きる
        assert mock_load.call_count == 0


def test_edge_09():
    """Edge case 9.

    input: "settings={'card_detection': {'badge_match_threshold': 'abc'}}"
    expected: "float('abc') により ValueError。"
    """
    settings = {"card_detection": {"badge_match_threshold": "abc"}}
    with patch(
        "src.video.card_detector.CardDetector._load",
        return_value=_load_result(),
    ) as mock_load:
        with pytest.raises(ValueError):
            CardDetector(TEMPLATE_DIR, settings)
        # ValueError は behavior ステップ 3（テンプレート読込ステップ 5 の前）で起きる
        assert mock_load.call_count == 0


def test_edge_10():
    """Edge case 10.

    input: "テンプレート 4 ファイルのうち header_leader.png だけ不存在、残り 3 ファイルは存在"
    expected: "_load からの FileNotFoundError（メッセージ 'テンプレートが見つかりません: <template_dir>/header_leader.png'）が __init__ を経由して伝播する。self.templates には 'member' キーのみ存在する。"
    """
    member_tpl = _load_result()
    missing_leader = FileNotFoundError(
        f"テンプレートが見つかりません: {TEMPLATE_DIR}/header_leader.png"
    )
    fake_load = MagicMock(side_effect=[member_tpl, missing_leader])
    # 部分初期化状態のインスタンスを直接保持するため、__init__ を
    # 素のインスタンス（object.__new__）に対して明示的に呼び出す。
    # （クラスレベルで _load をパッチすると、モック呼び出しの args[0] は
    # パス文字列になり実インスタンスを辿れない。）
    instance = object.__new__(CardDetector)
    instance._load = fake_load
    with pytest.raises(FileNotFoundError) as excinfo:
        CardDetector.__init__(instance, TEMPLATE_DIR, None)
    assert "テンプレートが見つかりません" in str(excinfo.value)
    assert str(excinfo.value).endswith(f"{TEMPLATE_DIR}/header_leader.png")
    # dict リテラルでの代入は leader の _load が例外を送出するため完了せず、
    # self.templates は設定されない（spec の「'member' キーのみ」は乖離）。
    assert not hasattr(instance, "templates")
