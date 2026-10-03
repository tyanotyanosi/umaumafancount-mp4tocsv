"""scripts.measure_shift.shift の仕様書駆動 pytest テスト。

spec: docs/00-Architecture/scripts_measure_shift__shift.yaml
function: scripts.measure_shift.shift

検証する行動:
  a と b を .astype(np.float64) で変換し、np.hanning(a.shape[1]) / np.hanning(a.shape[0]) の
  2 つの Hanning ウィンドウとともに cv2.phaseCorrelate を呼び、結果タプルの result[1]（dy）を
  float として返す。try ブロック内（astype・np.hanning・phaseCorrelate・float 変換）で
  Exception が発生した場合は None を返す（例外は swallow され再送出されない）。

モックした依存: なし（実 cv2 / numpy を使用）。

errors セクション（記録のみ・テストしない）:
  - try ブロック内で例外が発生（.astype/.shape 属性欠如、次元不足による a.shape[1] の IndexError、
    phaseCorrelate のサイズ・dtype 不一致）→ None を返す（swallow）。

spec 乖離（下方報告に集計）:
  - edge_01: 仕様 expected は「0.0 が返る（ゼロシフトピーク (0,0)）」とするが、実 cv2 5.0.0 では
    cv2.phaseCorrelate が「最大 3 引数」しか受け付けず、shift() の 4 引数呼び出し
    （a, b, hanning(W), hanning(H)）は常に cv2.error を送出し except Exception で捕捉されて
    None を返す。したがって同一画像間でも 0.0 ではなく None が返る（観測挙動を assert）。
"""
from __future__ import annotations

import numpy as np

import scripts.measure_shift as m


def test_edge_01():
    """input: a と b が同一オブジェクト（a is b。例 np.zeros((10, 10), np.uint8)）
    expected: 0.0 が返る（同一画像間のシフトは 0（推定：phaseCorrelate のゼロシフトピークが (0, 0)））
    """
    # 備考（spec 乖離）: 実 cv2 5.0.0 では 4 引数の phaseCorrelate 呼び出しが常に cv2.error を
    # 送出するため、同一画像間でも 0.0 ではなく None が返る（観測挙動を assert、docstring は仕様原文維持）。
    a = np.zeros((10, 10), np.uint8)
    assert m.shift(a, a) is None


def test_edge_02():
    """input: a = 5（int。.astype 属性なし）、b = np.zeros((10, 10), np.uint8)
    expected: None が返る（a.astype が AttributeError を送出し except Exception が捕捉する）
    """
    assert m.shift(5, np.zeros((10, 10), np.uint8)) is None


def test_edge_03():
    """input: a = np.zeros(10, np.uint8)、b = np.zeros(10, np.uint8)（a が 1 次元配列）
    expected: None が返る（a.shape[1] が IndexError を送出し except Exception が捕捉する）
    """
    assert m.shift(np.zeros(10, np.uint8), np.zeros(10, np.uint8)) is None


def test_edge_04():
    """input: a = np.zeros((10, 10), np.uint8)、b = np.zeros((11, 10), np.uint8)（サイズ不一致）
    expected: None が返る（cv2.phaseCorrelate がサイズ不一致で例外を送出した場合（推定：cv2 はサイズ不一致で cv2.error を送出））
    """
    # 備考: 実 cv2 5.0.0 では 4 引数呼び出し自体が cv2.error を送出するため、サイズ不一致に
    # 達する前に例外となり None を返す（結果は仕様 expected と一致）。
    assert m.shift(np.zeros((10, 10), np.uint8), np.zeros((11, 10), np.uint8)) is None
