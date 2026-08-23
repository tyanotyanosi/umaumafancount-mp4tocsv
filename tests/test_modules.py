import pytest
import json
from pathlib import Path


class TestResultParser:
    def test_parse_batch_basic(self):
        """カード入力の parse_batch 基本動作: 複数フレーム・複数カード → 正しく集計"""
        from src.parser.result_parser import ResultParser
        parser = ResultParser()
        frame_results = [
            {"cards": [
                {"role": "leader", "name_raw": "ろん", "fans_raw": "3,249,444,186人"},
                {"role": "member", "name_raw": "キュルス", "fans_raw": "2,823,905,018人"},
            ]},
            {"cards": [
                {"role": "leader", "name_raw": "ろん", "fans_raw": "3,249,444,186人"},
                {"role": "member", "name_raw": "キュルス", "fans_raw": "2,823,905,018人"},
            ]},
        ]
        merged = parser.parse_batch(frame_results)
        assert merged == {"ろん": 3249444186, "キュルス": 2823905018}

    def test_parse_batch_majority_vote(self):
        """多数決: 3フレーム中2回の一致値を採用、1回の誤認識値は上書きされる"""
        from src.parser.result_parser import ResultParser
        parser = ResultParser()
        frame_results = [
            {"cards": [{"name": "ろん", "fans": "3,249,444,186人"}]},
            {"cards": [{"name": "ろん", "fans": "3,249,444,186人"}]},
            {"cards": [{"name": "ろん", "fans": "3,249,444,1OO人"}]},
        ]
        merged = parser.parse_batch(frame_results)
        assert merged["ろん"] == 3249444186

    def test_parse_batch_tie_first_wins(self):
        """同票（桁数同じ）の場合は最初に出現した値を採用"""
        from src.parser.result_parser import ResultParser
        parser = ResultParser()
        frame_results = [
            {"cards": [{"name": "ろん", "fans": "3,249,444,186人"}]},
            {"cards": [{"name": "ろん", "fans": "3,249,444,187人"}]},
        ]
        merged = parser.parse_batch(frame_results)
        assert merged["ろん"] == 3249444186

    def test_parse_batch_tie_prefers_more_digits(self):
        """同票（桁数違い・桁脱落）の場合は桁数の多い方を採用"""
        from src.parser.result_parser import ResultParser
        parser = ResultParser()
        frame_results = [
            {"cards": [{"name": "モルガン", "fans": "1,87,201,99人"}]},
            {"cards": [{"name": "モルガン", "fans": "1,874201,199人"}]},
        ]
        merged = parser.parse_batch(frame_results)
        assert merged["モルガン"] == 1874201199

    def test_parse_batch_missing_fans_skipped(self):
        """欠損耐性: fans が None のカードは投票に参加しない"""
        from src.parser.result_parser import ResultParser
        parser = ResultParser()
        frame_results = [
            {"cards": [{"name": "ろん", "fans": None}]},
            {"cards": [{"name": "ろん", "fans": "3,249,444,186人"}]},
            {"cards": [{"name": "ろん", "fans": ""}]},
        ]
        merged = parser.parse_batch(frame_results)
        assert merged["ろん"] == 3249444186

    def test_parse_batch_empty_name_ignored(self):
        """欠損耐性: name が空のカードは無視される"""
        from src.parser.result_parser import ResultParser
        parser = ResultParser()
        frame_results = [
            {"cards": [
                {"name": "", "fans": "3,249,444,186人"},
                {"name": None, "fans": "3,249,444,186人"},
            ]},
        ]
        merged = parser.parse_batch(frame_results)
        assert merged == {}

    def test_parse_batch_zero_candidates_absent(self):
        """候補0件ユーザは結果にない"""
        from src.parser.result_parser import ResultParser
        parser = ResultParser()
        frame_results = [
            {"cards": [{"name": "ろん", "fans": "無効な文字列!!"}]},
        ]
        merged = parser.parse_batch(frame_results)
        assert "ろん" not in merged

    def test_correct_digit_confusion(self):
        """_correct_digit_confusion: O→0 等の混同補正"""
        from src.parser.result_parser import ResultParser
        parser = ResultParser()
        assert parser._correct_digit_confusion("2,715,661,O95") == "2,715,661,095"
        assert parser._correct_digit_confusion("3249444186") == "3249444186"
        assert parser._correct_digit_confusion("324944A186") == "3249444186"
        assert parser._correct_digit_confusion("2,715,661,A9G") == "2,715,661,495"

    def test_correct_digit_confusion_rejects_garbage(self):
        """置換後も数字以外の文字が残る場合は None"""
        from src.parser.result_parser import ResultParser
        parser = ResultParser()
        assert parser._correct_digit_confusion("3249444186x") is None
        assert parser._correct_digit_confusion("こんにちは") is None

    def test_parse_fan_text(self):
        """_parse_fan_text: 混同補正 + クリーン + 数値検証のパイプライン"""
        from src.parser.result_parser import ResultParser
        parser = ResultParser()
        assert parser._parse_fan_text("3,249,444,186人") == 3249444186
        assert parser._parse_fan_text("2,715,661,O95") == 2715661095
        assert parser._parse_fan_text("2,715,661,A9GJ") == 2715661495
        assert parser._parse_fan_text("無効") is None
        assert parser._parse_fan_text("") is None

    def test_clean_user_name_duplicate_lines(self):
        """同一行の繰り返し（OCR アーティファクト）は1行に統合される"""
        from src.parser.result_parser import ResultParser
        parser = ResultParser()
        assert parser._clean_user_name("キュルス\nキュルス") == "キュルス"
        assert parser._clean_user_name("ろん\nろん\nろん") == "ろん"
        # 2行で異なる名前は両方残る
        assert parser._clean_user_name("ザイ・\nツカコミヤ") == "ザイ・ツカコミヤ"

    def test_clean_user_name_with_parentheses(self):
        from src.parser.result_parser import ResultParser
        parser = ResultParser()
        name = "ろん(1)"
        cleaned = parser._clean_user_name(name)
        assert cleaned == "ろん"

    def test_clean_user_name_with_japanese_parentheses(self):
        from src.parser.result_parser import ResultParser
        parser = ResultParser()
        name = "キュルス（1）"
        cleaned = parser._clean_user_name(name)
        assert cleaned == "キュルス"

    def test_clean_user_name_with_spaces(self):
        from src.parser.result_parser import ResultParser
        parser = ResultParser()
        name = "  ろん   "
        cleaned = parser._clean_user_name(name)
        assert cleaned == "ろん"


class TestJSONWriter:
    def test_write_json(self, tmp_path):
        from src.output.json_writer import JSONWriter
        writer = JSONWriter(output_dir=str(tmp_path))
        data = {"ろん": 3249444186, "キュルス": 2823905018}
        result_path = writer.write(data)

        assert Path(result_path).exists()
        with open(result_path, 'r', encoding='utf-8') as f:
            loaded = json.load(f)
        assert loaded == data

    def test_write_json_auto_filename(self, tmp_path):
        from src.output.json_writer import JSONWriter
        writer = JSONWriter(output_dir=str(tmp_path))
        data = {"ろん": 3249444186}
        result_path = writer.write(data)

        assert "result_" in result_path
        assert result_path.endswith(".json")


class TestCSVWriter:
    def test_write_csv(self, tmp_path):
        from src.output.csv_writer import CSVWriter
        import csv
        writer = CSVWriter(output_dir=str(tmp_path))
        data = {"ろん": 3249444186, "キュルス": 2823905018}
        result_path = writer.write(data)

        assert Path(result_path).exists()
        with open(result_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.reader(f)
            rows = list(reader)
        assert rows[0] == ['ユーザ名', 'ファン数']
        assert len(rows) == 3  # header + 2 data rows


class TestDiffChecker:
    def test_first_frame_is_different(self):
        from src.video.diff_checker import DiffChecker
        import numpy as np
        checker = DiffChecker(threshold=0.1)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = checker.is_different(frame)
        assert result is True

    def test_same_frame_returns_false(self):
        from src.video.diff_checker import DiffChecker
        import numpy as np
        checker = DiffChecker(threshold=0.1)
        frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
        checker.update(frame1)
        frame2 = np.zeros((100, 100, 3), dtype=np.uint8)
        result = checker.is_different(frame2)
        assert not result

    def test_different_frame_returns_true(self):
        from src.video.diff_checker import DiffChecker
        import numpy as np
        checker = DiffChecker(threshold=0.1)
        frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
        checker.update(frame1)
        frame2 = np.ones((100, 100, 3), dtype=np.uint8) * 255
        result = checker.is_different(frame2)
        assert result


class _FakeReader:
    """VideoReader のインターフェースを模した最小リーダー（動画ファイル不要）"""

    def __init__(self, n: int, fps: float):
        self.n = n
        self.fps = fps
        self._idx = 0

    def get_fps(self):
        return self.fps

    def read_frame(self):
        if self._idx < self.n:
            self._idx += 1
            return __import__("numpy").zeros((10, 10, 3), dtype="uint8")
        return None


class TestFrameExtractor:
    def test_extract_samples_at_interval(self):
        """interval 毎のフレームのみを抽出（30fps・90フレーム・1秒間隔 → 0,30,60 の3本）"""
        from src.video.extractor import FrameExtractor
        reader = _FakeReader(n=90, fps=30.0)
        ext = FrameExtractor(reader, interval_sec=1.0)
        frames = ext.extract()
        assert len(frames) == 3

    def test_extract_05s_interval(self):
        """0.5秒間隔 → 約2本/秒（30fps・90フレーム → 0,15,30,... で6本）"""
        from src.video.extractor import FrameExtractor
        reader = _FakeReader(n=90, fps=30.0)
        ext = FrameExtractor(reader, interval_sec=0.5)
        frames = ext.extract()
        assert len(frames) == 6

    def test_extract_interval_larger_than_video(self):
        """間隔が動画全長より大きい場合は先頭1本のみの抽出"""
        from src.video.extractor import FrameExtractor
        reader = _FakeReader(n=20, fps=30.0)
        ext = FrameExtractor(reader, interval_sec=2.0)
        frames = ext.extract()
        assert len(frames) == 1

    def test_iter_frames_interval_0_yields_all(self):
        """interval_sec=0 → サンプリングせず全フレームを生成（差分判定のみモード）"""
        from src.video.extractor import FrameExtractor
        reader = _FakeReader(n=90, fps=30.0)
        ext = FrameExtractor(reader, interval_sec=0.0)
        frames = list(ext.iter_frames())
        assert len(frames) == 90

    def test_iter_frames_sampling_matches_extract(self):
        """iter_frames と extract は同一間隔で同じ本数を返す（ストリーミング整合性）"""
        from src.video.extractor import FrameExtractor
        # _FakeReader は1回読み切りの状態ありストリームなので、別 reader を使う
        ext_iter = FrameExtractor(_FakeReader(n=90, fps=30.0), interval_sec=1.0)
        n_iter = len(list(ext_iter.iter_frames()))
        ext_list = FrameExtractor(_FakeReader(n=90, fps=30.0), interval_sec=1.0)
        n_extract = len(ext_list.extract())
        assert n_iter == n_extract == 3


class TestImageProcessor:
    def test_to_grayscale(self):
        from src.utils.image_processor import ImageProcessor
        import cv2
        import numpy as np
        bgr_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        gray = ImageProcessor.to_grayscale(bgr_frame)
        assert gray.shape == (100, 100)
        assert len(gray.shape) == 2

    def test_crop_region(self):
        from src.utils.image_processor import ImageProcessor
        import numpy as np
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        cropped = ImageProcessor.crop_region(frame, region=(10, 10, 50, 50))
        assert cropped.shape == (50, 50, 3)
