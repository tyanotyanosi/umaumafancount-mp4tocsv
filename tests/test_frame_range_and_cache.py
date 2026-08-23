"""フレーム範囲指定 (--start/--end/--limit) と crop ハッシュ OCR キャッシュのユニットテスト。"""

import numpy as np
import pytest

from src.video.extractor import FrameExtractor, compute_frame_range
from src.ocr.cache import CachingOCR


# --------------------------------------------------------------------------- #
# compute_frame_range（純関数）
# --------------------------------------------------------------------------- #

class TestComputeFrameRange:
    def test_full_video_defaults(self):
        """デフォルト（start=0, end=0, limit=0）: 最初から最後まで。"""
        assert compute_frame_range(30.0, 900) == (0, 899)

    def test_start_only(self):
        """start のみ指定: そのフレームから最後まで。"""
        # fps=30, start=1.0s → フレーム 30 から最終フレーム 899 まで
        assert compute_frame_range(30.0, 900, start_sec=1.0) == (30, 899)

    def test_start_and_end(self):
        """start と end: 範囲内の両端を含む。"""
        # fps=30, start=1.0s, end=2.0s → フレーム 30..60
        assert compute_frame_range(30.0, 900, start_sec=1.0, end_sec=2.0) == (30, 60)

    def test_limit_caps_end(self):
        """limit は end より先に到達しても start+limit に切替する。"""
        # end=2.0s (フレーム 60) だが limit=0.5s (15 フレーム) → stop=30+15=45
        assert compute_frame_range(30.0, 900, start_sec=1.0, end_sec=2.0, limit_sec=0.5) == (30, 45)

    def test_limit_larger_than_end_keeps_end(self):
        """limit が end より大きい場合は end が支配する。"""
        # end=2.0s (フレーム 60)、limit=10s → stop=min(60, 30+300)=60
        assert compute_frame_range(30.0, 900, start_sec=1.0, end_sec=2.0, limit_sec=10.0) == (30, 60)

    def test_end_clamped_to_last_frame(self):
        """end が動画末尾を超える場合は最終フレームに丸める。"""
        # 動画 900 フレーム、end=100s (3000 フレーム) → 899 に丸める
        assert compute_frame_range(30.0, 900, end_sec=100.0) == (0, 899)

    def test_negative_start_clamped_to_zero(self):
        """負の start は 0 に丸める。"""
        assert compute_frame_range(30.0, 900, start_sec=-1.0) == (0, 899)

    def test_empty_video_raises(self):
        """フレーム総数が 0 以下: ValueError。"""
        with pytest.raises(ValueError):
            compute_frame_range(30.0, 0)

    def test_start_beyond_end_raises(self):
        """開始位置が動画長を超過: ValueError。"""
        # 動画 900 フレーム (30 秒)、start=999s は超過
        with pytest.raises(ValueError):
            compute_frame_range(30.0, 900, start_sec=999.0)

    def test_empty_range_raises(self):
        """start > stop（end が start より前）: ValueError。"""
        with pytest.raises(ValueError):
            compute_frame_range(30.0, 900, start_sec=2.0, end_sec=1.0)

    def test_nonpositive_fps_ok(self):
        """fps<=0 でも範囲自体は計算可能（0..total-1）。"""
        assert compute_frame_range(0.0, 10) == (0, 9)


# --------------------------------------------------------------------------- #
# FrameExtractor.iter_frames の max_frames
# --------------------------------------------------------------------------- #

class _FakeReader:
    """VideoReader を真似する最小ファイク（fps と連続フレームのみ）。"""

    def __init__(self, fps: float, n: int):
        self._fps = fps
        self._n = n
        self._counter = 0

    def get_fps(self):
        return self._fps

    def read_frame(self):
        # 0..n-1 のフレーム番号を ndarray として返す
        if self._counter < self._n:
            f = self._counter
            self._counter += 1
            return np.full((16, 16, 3), f % 255, dtype=np.uint8)
        return None


def _make_extractor(n_frames, interval=0.0):
    reader = _FakeReader(30.0, n_frames)
    ext = FrameExtractor(reader, interval_sec=interval)
    return ext


class TestIterFramesMaxFrames:
    def test_max_frames_limits_decoded(self):
        """max_frames はデコード枚数で切り取る（interval=0 で全フレーム採用）。"""
        ext = _make_extractor(100, interval=0.0)
        frames = list(ext.iter_frames(max_frames=10))
        assert len(frames) == 10

    def test_no_max_frames_reads_all(self):
        """max_frames=None は最後までデコードする。"""
        ext = _make_extractor(50, interval=0.0)
        frames = list(ext.iter_frames(max_frames=None))
        assert len(frames) == 50

    def test_interval_with_max_frames(self):
        """interval サンプリングと max_frames の併用: デコード上限内で間隔採用。"""
        # fps=30, interval=1.0s → 30 フレームに 1 枚採用。max_frames=90 → 3 枚
        ext = _make_extractor(200, interval=1.0)
        frames = list(ext.iter_frames(max_frames=90))
        assert len(frames) == 3

    def test_extract_default_args(self):
        """extract() の既定（max_frames=None）は全フレーム。"""
        ext = _make_extractor(40, interval=0.0)
        assert len(ext.extract()) == 40


# --------------------------------------------------------------------------- #
# CachingOCR（crop ハッシュキャッシュ）
# --------------------------------------------------------------------------- #

class _CountingOCR:
    """呼び出し回数を数える最小 OCR ファイク。"""

    def __init__(self):
        self.recognize_calls = 0
        self.rwc_calls = 0

    def recognize(self, image):
        self.recognize_calls += 1
        # 画像の内容に応じた偽結果（キャッシュの妥当性を検証するため）
        val = int(image.sum())
        return f"img{val}"

    def recognize_with_confidence(self, image):
        self.rwc_calls += 1
        val = int(image.sum())
        return [(f"img{val}", 0.9)]


class TestCachingOCR:
    def test_recognize_caches_identical_crop(self):
        """同一 crop の認識は 1 回のみ（2 回目はキャッシュ）。"""
        engine = _CountingOCR()
        wrapper = CachingOCR(engine, name="test")
        img = np.full((8, 8, 3), 10, dtype=np.uint8)

        r1 = wrapper.recognize(img)
        r2 = wrapper.recognize(img)
        assert r1 == r2
        assert engine.recognize_calls == 1  # 2 回目も再認識しない
        assert wrapper.stats["hits"] == 1
        assert wrapper.stats["misses"] == 1

    def test_different_crops_not_cached_together(self):
        """異なる crop は別キーで、それぞれ認識される。"""
        engine = _CountingOCR()
        wrapper = CachingOCR(engine, name="test")
        a = np.full((8, 8, 3), 10, dtype=np.uint8)
        b = np.full((8, 8, 3), 20, dtype=np.uint8)

        wrapper.recognize(a)
        wrapper.recognize(b)
        assert engine.recognize_calls == 2
        assert wrapper.stats["unique"] == 2

    def test_recognize_with_confidence_caches(self):
        """recognize_with_confidence もキャッシュ対象。"""
        engine = _CountingOCR()
        wrapper = CachingOCR(engine, name="test")
        img = np.full((8, 8, 3), 5, dtype=np.uint8)

        r1 = wrapper.recognize_with_confidence(img)
        r2 = wrapper.recognize_with_confidence(img)
        assert r1 == r2
        assert engine.rwc_calls == 1
        assert wrapper.stats["hits"] == 1

    def test_empty_result_cached(self):
        """空文字結果もキャッシュされる（再認識しない）。"""
        class _EmptyOCR:
            calls = 0
            def recognize(self, image):
                self.calls += 1
                return ""

        engine = _EmptyOCR()
        wrapper = CachingOCR(engine, name="test")
        img = np.zeros((8, 8, 3), dtype=np.uint8)

        assert wrapper.recognize(img) == ""
        assert wrapper.recognize(img) == ""
        assert engine.calls == 1
        assert wrapper.stats["hits"] == 1

    def test_delegates_other_attrs(self):
        """認識メソッド以外（例: エンジンの属性）は内部エンジンへ委譲する。"""
        engine = _CountingOCR()
        engine.some_flag = "x"
        wrapper = CachingOCR(engine, name="test")
        assert wrapper.some_flag == "x"

    def test_stats_shape(self):
        """stats に name/hits/misses/unique が揃う。"""
        wrapper = CachingOCR(_CountingOCR(), name="n")
        s = wrapper.stats
        assert s["name"] == "n"
        assert s["hits"] == 0 and s["misses"] == 0 and s["unique"] == 0
