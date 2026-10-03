"""scripts.anchor_probe.find_peak の仕様書駆動 pytest テスト。

spec: docs/00-Architecture/scripts_anchor_probe__find_peak.yaml
function: scripts.anchor_probe.find_peak

検証する行動:
  region を x0, y0, x1, y1 の 4 値に分解し、img[y0:y1, x0:x1] のスライスでサブ画像 sub を取得。
  cv2.matchTemplate(sub, tpl, cv2.TM_CCOEFF_NORMED) で応答マップ r を計算し、
  cv2.minMaxLoc(r) の最大応答位置 ml2 と最大値 mx を取り、
  (x0+ml2[0], y0+ml2[1], float(mx)) を返す。純粋関数（I/O・グローバル状態変更なし）。

モックした依存: なし（実 cv2 を使用）。
検証用画像は固定シードのランダムグレースケール（2 次元 uint8）で、
テンプレートが一意にマッチする位置を決定論的にする。

errors セクション（記録のみ・テストしない）:
  - region が 4 要素未満/超過 → 展開代入で ValueError。
  - region の要素が整数でない → numpy スライスで TypeError。
  - tpl が region スライスより大きい → cv2.matchTemplate がエラーを送出。
  - img と tpl のチャンネル数/dtype が適合しない → cv2.matchTemplate がエラーを送出。

spec 乖離（下方報告に集計）:
  - edge_01: 仕様 expected はスコア 1.0（推定）とするが、TM_CCOEFF_NORMED は同一画像でも
    浮動小数点精度により 1.0 に厳密に等しくならず（観測 0.9999995...）。pytest.approx で検証。
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

import scripts.anchor_probe as m


def _unique_img(h, w, seed=42):
    """固定シードのランダムグレースケール画像（2 次元 uint8）。テンプレートを一意にする。"""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w), dtype=np.uint8)


def test_edge_01():
    """input: tplがregionスライスと完全に一致する場合（サイズ・ピクセル値同一、ピクセルにばらつきあり）。例: region=(10,10,30,30) かつ tpl=subと同一
    expected: (10, 10, 1.0) が返る（TM_CCOEFF_NORMEDは同一画像の最大応答が1.0になる、と推定）
    """
    img = _unique_img(40, 40, seed=1)
    region = (10, 10, 30, 30)
    x0, y0, x1, y1 = region
    tpl = img[y0:y1, x0:x1].copy()
    result = m.find_peak(img, tpl, region)
    assert result[0] == 10
    assert result[1] == 10
    # 備考（spec 乖離）: 同一画像でも浮動小数点精度により厳密に 1.0 には等しくならない。
    assert result[2] == pytest.approx(1.0, abs=1e-4)


def test_edge_02():
    """input: region=(0,0,w,h) でimg全体をカバー（w, h はimgの幅・高さ）
    expected: 返される位置は応答マップの最大応答の生位置(ml2[0], ml2[1])（オフセット加算なし）で、第3要素は応答マップ最大値のfloat
    """
    img = _unique_img(40, 40, seed=2)
    # tpl は img のサブ領域 (rows 15:25, cols 8:18) のコピー。最大応答はその左上 (col=8, row=15)。
    tpl = img[15:25, 8:18].copy()
    region = (0, 0, 40, 40)
    result = m.find_peak(img, tpl, region)
    # オフセット (0,0) なので返る位置は生位置 ml2 = (8, 15)。
    assert result[0] == 8
    assert result[1] == 15
    assert result[2] == pytest.approx(1.0, abs=1e-4)


def test_edge_03():
    """input: region=(0,0,0,0)（空のregion）
    expected: subはshape (0,0)の空配列となり、cv2.matchTemplateがエラーを送出する（例外種別はこのファイルからは確認不可、missing参照）
    """
    img = _unique_img(40, 40, seed=3)
    tpl = img[15:25, 8:18].copy()
    with pytest.raises(cv2.error):
        m.find_peak(img, tpl, (0, 0, 0, 0))
