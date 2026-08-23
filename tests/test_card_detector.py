"""CardDetector 単体テスト（合成フレーム使用）"""

from pathlib import Path

import cv2
import numpy as np

from src.video.card_detector import CardDetector

TEMPLATE_DIR = Path(__file__).parent.parent / "template"

# 実フレームで計測したアンカー相対位置（scripts/anchor_probe.py 結果）
ICON_DX = 215  # i_icon.x - badge.x
ICON_DY = 4  # i_icon.y - badge.y
LABEL_DX = 90  # fan_count_label.x - badge.x
LABEL_DY = 56  # fan_count_label.y - badge.y

BADGE_X, BADGE_Y = 300, 100
FRAME_W, FRAME_H = 900, 1000

# テンプレート
BADGE_T = cv2.imread(str(TEMPLATE_DIR / "header_leader.png"), cv2.IMREAD_COLOR)
ICON_T = cv2.imread(str(TEMPLATE_DIR / "i_icon.png"), cv2.IMREAD_COLOR)
LABEL_T = cv2.imread(str(TEMPLATE_DIR / "fan_count_label.png"), cv2.IMREAD_COLOR)
assert BADGE_T is not None and ICON_T is not None and LABEL_T is not None

BADGE_W = BADGE_T.shape[1]
BADGE_X_RIGHT = BADGE_X + BADGE_W
LABEL_W = LABEL_T.shape[1]


def _paste(frame, img, x, y):
    h, w = img.shape[:2]
    frame[y : y + h, x : x + w] = img


def build_frame(n_cards=1, icon=True, label=True, edge=False):
    """テンプレートを貼り付けた白背景の合成フレームを生成する"""
    frame = np.full((FRAME_H, FRAME_W, 3), 255, dtype=np.uint8)
    for i in range(n_cards):
        bx = 5 if edge else BADGE_X
        by = BADGE_Y + i * 200
        _paste(frame, BADGE_T, bx, by)
        if icon:
            _paste(frame, ICON_T, bx + ICON_DX, by + ICON_DY)
        if label:
            _paste(frame, LABEL_T, bx + LABEL_DX, by + LABEL_DY)
    return frame


def _det():
    return CardDetector(template_dir=TEMPLATE_DIR)


def test_detect_full_card():
    """カード1枚（バッジ+i_icon+fan_label）を検出し、name_box/fan_box が正しい"""
    frame = build_frame(n_cards=1)
    cards = _det().detect(frame)

    assert len(cards) == 1
    card = cards[0]
    assert card.role == "leader"
    bx, by, bw, bh = card.badge_box
    assert abs(bx - BADGE_X) <= 2
    assert abs(by - BADGE_Y) <= 2

    # name_box: バッジ右端+8 → i_icon 左端、縦はバッジより上下 10px 拡張
    nb = card.name_box
    assert nb is not None
    assert abs(nb[0] - (BADGE_X_RIGHT + 8)) <= 2, f"name x 不一致: {nb}"
    assert abs(nb[2] - (BADGE_X + ICON_DX - (BADGE_X_RIGHT + 8))) <= 4, f"name w 不一致: {nb}"
    assert nb[1] == BADGE_Y - 10, f"name y 不一致: {nb}"
    assert nb[3] == BADGE_T.shape[0] + 20, f"name h 不一致: {nb}"

    # fan_box: fan_label 右端+8 起点、幅 280
    fb = card.fan_box
    assert fb is not None
    assert abs(fb[0] - (BADGE_X + LABEL_DX + LABEL_W + 8)) <= 2, f"fan x 不一致: {fb}"
    assert abs(fb[2] - 280) <= 1


def test_detect_three_cards_sorted():
    """3枚のカードを検出し y 座標でソートされる"""
    frame = build_frame(n_cards=3)
    cards = _det().detect(frame)

    assert len(cards) == 3
    ys = [c.badge_box[1] for c in cards]
    assert ys[0] < ys[1] < ys[2]
    for card in cards:
        assert card.name_box is not None
        assert card.fan_box is not None


def test_edge_badge_skipped():
    """画面端（edge_margin 以内）のバッジは除外される"""
    frame = build_frame(n_cards=1, edge=True)
    cards = _det().detect(frame)
    assert cards == []


def test_name_box_fallback_without_icon():
    """i_icon が無い場合、name_box は max_name_width 幅のフォールバックになる"""
    frame = build_frame(n_cards=1, icon=False)
    cards = _det().detect(frame)

    assert len(cards) == 1
    nb = cards[0].name_box
    assert nb is not None
    # 右端が badge.x + badge.w + max_name_width (=400) になる
    assert abs(nb[0] - (BADGE_X_RIGHT + 8)) <= 2
    assert nb[0] + nb[2] == BADGE_X + BADGE_W + 400
    assert cards[0].fan_box is not None  # fan_label は貼られている


def test_fan_box_none_without_label():
    """fan_count_label が無い場合、fan_box は None になる"""
    frame = build_frame(n_cards=1, label=False)
    cards = _det().detect(frame)

    assert len(cards) == 1
    assert cards[0].fan_box is None
    assert cards[0].name_box is not None


def test_empty_frame_no_cards():
    """空フレームはカード 0 枚"""
    frame = np.full((FRAME_H, FRAME_W, 3), 255, dtype=np.uint8)
    assert _det().detect(frame) == []
