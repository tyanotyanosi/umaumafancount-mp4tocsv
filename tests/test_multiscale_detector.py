"""多解像度バッジ検出のテスト

設計書 docs/multiscale_design.md に基づくテスト:
- Layer A: スケール推定 (_estimate_scale_prior)
- Layer B: 多解像度検出 / 勝者スケール / NMS
- 単一解像度回帰 (multi_scale=False)
"""

from pathlib import Path

import cv2
import numpy as np

from src.video.card_detector import CardDetector

TEMPLATE_DIR = Path(__file__).parent.parent / "template"

FRAME_W, FRAME_H = 900, 1000

# アンカー相対位置（実フレーム計測値）
ICON_DX, ICON_DY = 215, 4
LABEL_DX, LABEL_DY = 90, 56

BADGE_X, BADGE_Y = 300, 100

# テンプレート
BADGE_T = cv2.imread(str(TEMPLATE_DIR / "header_leader.png"), cv2.IMREAD_COLOR)
ICON_T = cv2.imread(str(TEMPLATE_DIR / "i_icon.png"), cv2.IMREAD_COLOR)
LABEL_T = cv2.imread(str(TEMPLATE_DIR / "fan_count_label.png"), cv2.IMREAD_COLOR)
assert BADGE_T is not None and ICON_T is not None and LABEL_T is not None

BADGE_W = BADGE_T.shape[1]
BADGE_H = BADGE_T.shape[0]


def _paste(frame, img, x, y):
    h, w = img.shape[:2]
    frame[y : y + h, x : x + w] = img


def build_frame(n_cards=1, icon=True, label=True, scale=1.0):
    """スケール scale のバッジを貼り付けた合成フレーム。

    scale=1.0 が現行テストと同じ（テンプレートをリサイズしない）。
    """
    frame = np.full((FRAME_H, FRAME_W, 3), 255, dtype=np.uint8)
    for i in range(n_cards):
        bx, by = BADGE_X, BADGE_Y + i * 200
        if scale == 1.0:
            _paste(frame, BADGE_T, bx, by)
        else:
            bw = max(1, int(BADGE_W * scale))
            bh = max(1, int(BADGE_H * scale))
            badge = cv2.resize(BADGE_T, (bw, bh), interpolation=cv2.INTER_AREA)
            _paste(frame, badge, bx, by)
        if icon:
            if scale == 1.0:
                _paste(frame, ICON_T, bx + ICON_DX, by + ICON_DY)
            else:
                iw = max(1, int(ICON_T.shape[1] * scale))
                ih = max(1, int(ICON_T.shape[0] * scale))
                _paste(frame, cv2.resize(ICON_T, (iw, ih), interpolation=cv2.INTER_AREA),
                       bx + int(ICON_DX * scale), by + int(ICON_DY * scale))
        if label:
            if scale == 1.0:
                _paste(frame, LABEL_T, bx + LABEL_DX, by + LABEL_DY)
            else:
                lw = max(1, int(LABEL_T.shape[1] * scale))
                lh = max(1, int(LABEL_T.shape[0] * scale))
                _paste(frame, cv2.resize(LABEL_T, (lw, lh), interpolation=cv2.INTER_AREA),
                       bx + int(LABEL_DX * scale), by + int(LABEL_DY * scale))
    return frame


def _det(settings=None):
    return CardDetector(template_dir=TEMPLATE_DIR, settings=settings)


# ------------------------------------------------------------------ #
# Layer A: スケール推定
# ------------------------------------------------------------------ #

def test_estimate_scale_prior_reference_width():
    """reference_width=2560, フレーム幅=2560 → s0=1.0、窓は [0.8, 1.3]"""
    d = _det({"card_detection": {
        "reference_width": 2560,
        "scale_window_low": 0.8,
        "scale_window_high": 1.3,
        "scale_step": 0.05,
    }})
    s0, S = d._estimate_scale_prior(2560)
    assert abs(s0 - 1.0) < 1e-6
    # 窓の範囲: s0*0.8 ~ s0*1.3
    assert abs(min(S) - 0.8) < 0.06
    assert abs(max(S) - 1.3) < 0.06
    # s0 は必ず S に含まれる
    assert s0 in S


def test_estimate_scale_prior_narrow_window():
    """reference_width=2560, フレーム幅=900 → s0=0.352、窓は狭い"""
    d = _det({"card_detection": {
        "reference_width": 2560,
        "scale_window_low": 0.8,
        "scale_window_high": 1.3,
        "scale_step": 0.05,
    }})
    s0, S = d._estimate_scale_prior(900)
    assert abs(s0 - 900 / 2560) < 1e-6  # s0 ≈ 0.352
    # s0 は S に含まれる
    assert s0 in S
    # 1.0 は窓外（フルスウィープで検出する想定）
    assert 1.0 not in S


def test_estimate_scale_prior_no_reference():
    """reference_width 未設定 (0) → s0=1.0、フルスウィープ [0.5, 2.0]"""
    d = _det({"card_detection": {"reference_width": 0}})
    s0, S = d._estimate_scale_prior(900)
    assert abs(s0 - 1.0) < 1e-6
    assert min(S) <= 0.5
    assert max(S) >= 1.9
    # 1.0 は必ず含まれる
    assert 1.0 in S


# ------------------------------------------------------------------ #
# Layer B: 多解像度検出
# ------------------------------------------------------------------ #

def test_multiscale_detect_reference_width_match():
    """reference_width=フレーム幅 → s0=1.0、バッジ s=1.0 を直接検出"""
    frame = build_frame(n_cards=1, scale=1.0)
    d = _det({"card_detection": {"reference_width": 900}})
    cards = d.detect(frame)
    assert len(cards) == 1
    bx, by, bw, bh = cards[0].badge_box
    assert abs(bx - BADGE_X) <= 2
    assert abs(by - BADGE_Y) <= 2


def test_multiscale_detect_full_sweep_fallback():
    """reference_width=2560 (s0=0.35) → 狭窓で検出不可 → フルスウィープで検出"""
    frame = build_frame(n_cards=1, scale=1.0)
    d = _det({"card_detection": {"reference_width": 2560}})
    cards = d.detect(frame)
    assert len(cards) == 1
    bx, by = cards[0].badge_box[:2]
    assert abs(bx - BADGE_X) <= 2
    assert abs(by - BADGE_Y) <= 2


def test_winning_scale_selection():
    """勝者スケール: 合計スコア最大を argmax、同点は s0 側を優先"""
    d = _det()
    # s=1.0: 2 候補、合計スコア 1.8
    # s=0.95: 1 候補、スコア 0.9
    cands = [
        [100, 100, 1.0, 0.9, 0.1],
        [200, 100, 1.0, 0.85, 0.15],
        [300, 100, 0.95, 0.9, 0.1],
    ]
    s_w = d._pick_winning_scale(cands, 1.0)
    assert s_w == 1.0


def test_winning_scale_tie_break():
    """同スコア → s0 に近い方を優先"""
    d = _det()
    cands = [
        [100, 100, 0.95, 0.8, 0.2],
        [200, 100, 1.0, 0.8, 0.2],
    ]
    # s=0.95 と s=1.0 ともスコア 0.8、同点
    # s0=1.0 なので s=1.0 を優先
    s_w = d._pick_winning_scale(cands, 1.0)
    assert s_w == 1.0


def test_multiscale_nms():
    """多解像度 NMS: 近接候補を IoU>0.5 で抑制"""
    d = _det()
    cands = [
        [100, 100, 1.0, 0.9, 0.1],
        [110, 105, 1.0, 0.85, 0.15],  # 近接 → 抑制
        [400, 100, 1.0, 0.8, 0.1],    # 別バッジ → 残る
    ]
    kept = d._nms_badges(cands)
    # 近接ペアは 1 つに圧縮、別バッジは残る
    assert len(kept) >= 1
    # 高スコア候補が最前
    assert kept[0][3] >= kept[-1][3]


# ------------------------------------------------------------------ #
# 単一解像度回帰 (multi_scale=False)
# ------------------------------------------------------------------ #

def test_single_scale_regression():
    """multi_scale=False は現行の単一解像度検出と同一動作"""
    frame = build_frame(n_cards=1, scale=1.0)

    d_off = _det({"card_detection": {"multi_scale": False}})
    d_on = _det({"card_detection": {"multi_scale": True, "reference_width": 900}})

    cards_off = d_off.detect(frame)
    cards_on = d_on.detect(frame)

    assert len(cards_off) == 1
    assert len(cards_on) == 1
    # バッジ位置がほぼ一致
    bx_off, by_off = cards_off[0].badge_box[:2]
    bx_on, by_on = cards_on[0].badge_box[:2]
    assert abs(bx_off - bx_on) <= 2
    assert abs(by_off - by_on) <= 2
    # name_box もほぼ一致
    nb_off = cards_off[0].name_box
    nb_on = cards_on[0].name_box
    assert nb_off is not None and nb_on is not None
    assert abs(nb_off[0] - nb_on[0]) <= 4


def test_empty_frame_multi():
    """空フレームは multi_scale でもカード 0 枚"""
    frame = np.full((FRAME_H, FRAME_W, 3), 255, dtype=np.uint8)
    d = _det({"card_detection": {"reference_width": 900}})
    assert d.detect(frame) == []


# ------------------------------------------------------------------ #
# Layer C: 粗→精検出（案 3）
# ------------------------------------------------------------------ #

def test_coarse_to_fine_detect():
    """coarse_to_fine=True: 粗い検出 → 局所窓内で精密検出"""
    frame = build_frame(n_cards=1, scale=1.0)
    d = _det({"card_detection": {
        "reference_width": 900,
        "coarse_to_fine": True,
        "coarse_scale": 0.5,
        "refine_radius": 200,
    }})
    cards = d.detect(frame)
    assert len(cards) == 1
    bx, by = cards[0].badge_box[:2]
    assert abs(bx - BADGE_X) <= 4
    assert abs(by - BADGE_Y) <= 4


def test_coarse_detect_returns_approx():
    """_detect_coarse が下サンプリングでバッジ位置を返す"""
    frame = build_frame(n_cards=1, scale=1.0)
    d = _det({"card_detection": {"reference_width": 900, "coarse_scale": 0.5}})
    import cv2 as _cv2
    gray = _cv2.cvtColor(frame, _cv2.COLOR_BGR2GRAY)
    s0, S = d._estimate_scale_prior(FRAME_W)
    approx = d._detect_coarse(gray, S, s0)
    assert len(approx) >= 1
    # 座標は元解像度（BADGE_X, BADGE_Y 付近）
    ax, ay = approx[0][0], approx[0][1]
    assert abs(ax - BADGE_X) < 50
    assert abs(ay - BADGE_Y) < 50
