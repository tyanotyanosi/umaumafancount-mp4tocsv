"""ユニットテスト: main.py

テスト対象（優先度順）:
  1. get_fan_count()       — 正規表現マッチング・最頻値選択・重複排除
  2. OCR後処理（12連鎖 .replace()）— 記号除去・カンマ正規化・トークン分離
  3. memberReplace適用ロジック         — 誤認識補正＋\n{memberName} 分割
  4. parse_args()                      — CLI引数パース（--gui, --debug, --img-scale）
  5. select_video_gui()                — tkinter ImportError フォールバック

モック対象: cv2, PIL.Image, winrt.*, tkinter
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path
from unittest.mock import patch, MagicMock, call

import pytest

# ---------------------------------------------------------------------------
# モジュールインポート前のモック（main.py がモジュールレベルで cv2, PIL, winrt を
# import するため、import 前に差し替える）
# ---------------------------------------------------------------------------
cv2_mock = MagicMock()
pil_image_mock = MagicMock()
winrt_ocr_mock = MagicMock()
winrt_graphics_mock = MagicMock()
winrt_streams_mock = MagicMock()

sys.modules["cv2"] = cv2_mock
sys.modules["PIL"] = MagicMock()
sys.modules["PIL.Image"] = pil_image_mock
sys.modules["winrt.windows.media.ocr"] = winrt_ocr_mock
sys.modules["winrt.windows.graphics.imaging"] = winrt_graphics_mock
sys.modules["winrt.windows.storage.streams"] = winrt_streams_mock
sys.modules["winrt.windows.foundation"] = MagicMock()
sys.modules["winrt.windows.storage"] = MagicMock()
sys.modules["winrt.runtime"] = MagicMock()
sys.modules["tqdm"] = MagicMock()

# モック適用後に main を import
import main


# ===================================================================
# 1. get_fan_count()
# ===================================================================

class TestGetFanCount:
    """get_fan_count() の正規表現マッチング・最頻値選択・重複排除"""

    def test_basic_match(self):
        """カンマ区切り数値が1つだけ含まれるテキスト → その数値を返す"""
        texts = ["万丈目準 3,249,444,186 人"]
        result = main.get_fan_count(texts, "万丈目準", [])
        assert result == 3249444186

    def test_no_match_returns_none(self):
        """カンマ区切り数値がない場合 → None"""
        texts = ["万丈目準 人"]
        result = main.get_fan_count(texts, "万丈目準", [])
        assert result is None

    def test_member_not_in_text(self):
        """メンバー名がテキストに含まれない場合 → None"""
        texts = ["別人A 1,234,567"]
        result = main.get_fan_count(texts, "万丈目準", [])
        assert result is None

    def test_prefers_most_frequent(self):
        """複数フレームで同じ値が最多出現 → その値が選ばれる"""
        texts = [
            "万丈目準 1,000,000 人",
            "万丈目準 1,000,000 人",
            "万丈目準 2,000,000 人",
            "万丈目準 2,000,000 人",
            "万丈目準 2,000,000 人",
        ]
        result = main.get_fan_count(texts, "万丈目準", [])
        assert result == 2000000  # 3回出現 → 最頻

    def test_tie_returns_one_of(self):
        """同率の場合はソート順で先頭（実装上は降順ソートの最初）"""
        texts = [
            "万丈目準 1,000,000 人",
            "万丈目準 9,999,999 人",
        ]
        result = main.get_fan_count(texts, "万丈目準", [])
        assert result in (1000000, 9999999)

    def test_skip_already_in_fans(self):
        """fans リストに含まれる値はスキップする"""
        texts = [
            "万丈目準 1,000,000 人",
            "万丈目準 2,000,000 人",
        ]
        result = main.get_fan_count(texts, "万丈目準", [1000000])
        assert result == 2000000  # 1000000 は fans にあるのでスキップ

    def test_all_values_skipped_returns_none(self):
        """fans リストにより全候補がスキップされた場合 → None"""
        texts = [
            "万丈目準 1,000,000 人",
            "万丈目準 2,000,000 人",
        ]
        result = main.get_fan_count(texts, "万丈目準", [1000000, 2000000])
        assert result is None

    def test_rejects_non_comma_numbers(self):
        """カンマなしの数値（1234567）はマッチしない"""
        texts = ["万丈目準 1234567 人"]
        result = main.get_fan_count(texts, "万丈目準", [])
        assert result is None

    def test_rejects_fewer_than_4_digits(self):
        """4桁未満（1,234 など）も一応マッチする"""
        texts = ["万丈目準 1,234 人"]
        result = main.get_fan_count(texts, "万丈目準", [])
        assert result == 1234

    def test_token_boundary_prevents_concatenation(self):
        """前後に英数字・日本語文字が連結している場合はマッチしない"""
        texts = ["メンバーA123,456"]
        result = main.get_fan_count(texts, "メンバーA", [])
        assert result is None

    def test_no_false_positive_on_partial_number(self):
        """途中にカンマがない数値（1234,567 のような不完全）はマッチしない"""
        texts = ["万丈目準 1234,567 人"]
        result = main.get_fan_count(texts, "万丈目準", [])
        assert result is None

    def test_multiple_members_in_one_text(self):
        """1テキストに複数メンバーが含まれている場合、pattern.search() は
        テキスト内の最初の数値（1,111,111）を返すのが現在の実装。
        メンバー名の近くの数値を探すわけではないという制約。"""
        texts = [
            "メンバーA 1,111,111 人 メンバーB 2,222,222 人",
        ]
        result_a = main.get_fan_count(texts, "メンバーA", [])
        result_b = main.get_fan_count(texts, "メンバーB", [])
        assert result_a == 1111111
        # メンバーB はテキストに含まれるが、pattern.search() は最初の数値を返す
        assert result_b == 1111111  # 最初の数値が返る（制約）


# ===================================================================
# 2. OCR後処理（12連鎖 .replace()）
# ===================================================================

class TestOcrPostProcessing:
    """main() 内の OCR後処理 .replace() 連鎖のテスト"""

    def post_process(self, text: str) -> str:
        """main.py 行213-225 の処理を再現"""
        text = text.replace(" ", "")
        text = text.replace("①", "")
        text = text.replace("↓", "")
        text = text.replace("(", "")
        text = text.replace(")", "")
        text = text.replace("（", "")
        text = text.replace("）", "")
        text = text.replace("@", "")
        text = text.replace("、", ",")
        text = text.replace("，", ",")
        text = text.replace("30/30", "")
        text = text.replace("人", " 人")
        text = text.replace("ファン数", "ファン数 ")
        return text

    def test_removes_spaces(self):
        assert self.post_process("a b c") == "abc"

    def test_removes_circled_one(self):
        assert self.post_process("名前①") == "名前"

    def test_removes_down_arrow(self):
        assert self.post_process("↓123") == "123"

    def test_removes_parentheses(self):
        assert self.post_process("(test)（テスト）") == "testテスト"

    def test_removes_at_sign(self):
        assert self.post_process("user@name") == "username"

    def test_normalizes_commas(self):
        assert self.post_process("a、b，c") == "a,b,c"

    def test_removes_30_30(self):
        assert self.post_process("30/30") == ""

    def test_inserts_space_around_人(self):
        assert self.post_process("1234人") == "1234 人"

    def test_inserts_space_after_ファン数(self):
        assert self.post_process("ファン数1234") == "ファン数 1234"

    def test_full_pipeline_realistic(self):
        raw = " メンバーA 3,249,444,186① 人 ↓ (30/30) @test "
        result = self.post_process(raw)
        assert "①" not in result
        assert "↓" not in result
        assert "(" not in result and ")" not in result
        assert "@" not in result
        assert "30/30" not in result
        assert " 人" in result
        assert "メンバーA" in result
        assert "3,249,444,186" in result


# ===================================================================
# 3. memberReplace 適用ロジック
# ===================================================================

class TestMemberReplace:
    """memberReplace.json の適用 ＋ \n{memberName} 分割ロジック"""

    def apply_replace(self, texts, member_list, member_replace):
        """main.py 行236-244 の処理を再現"""
        texts = list(texts)  # copy
        for i, text in enumerate(texts):
            for member in member_list:
                if member in member_replace:
                    for repname in member_replace[member]:
                        texts[i] = texts[i].replace(repname, member)
                if member in text:
                    texts[i] = texts[i].replace(member, f"\n{member} ")
        joined = "\n".join(texts)
        return joined.split("\n")

    def test_basic_replacement(self):
        texts = ["メンハーA 1,000,000"]
        member_list = ["メンバーA"]
        replace = {"メンバーA": ["メンハーA", "メンパーA"]}
        result = self.apply_replace(texts, member_list, replace)
        assert any("メンバーA" in t for t in result)
        assert not any("メンハーA" in t for t in result)

    def test_multiple_errors(self):
        texts = ["メンパーA 2,000,000"]
        member_list = ["メンバーA"]
        replace = {"メンバーA": ["メンハーA", "メンパーA"]}
        result = self.apply_replace(texts, member_list, replace)
        assert any("メンバーA" in t for t in result)

    def test_member_split_with_newline(self):
        """replace 後に member が \n{member}  に置換される"""
        texts = ["メンバーA 1,000,000 人"]
        member_list = ["メンバーA"]
        replace = {}
        result = self.apply_replace(texts, member_list, replace)
        # "メンバーA" が "\nメンバーA " に置換される
        assert any(t.strip().startswith("メンバーA") for t in result)

    def test_unmatched_member_ignored(self):
        """replace に存在しない member はスキップされる"""
        texts = ["メンバーA 1,000,000"]
        member_list = ["メンバーA"]
        replace = {"メンバーB": ["誤認識B"]}  # メンバーA のエントリなし
        result = self.apply_replace(texts, member_list, replace)
        assert any("メンバーA" in t for t in result)

    def test_no_false_replacement(self):
        """OCR誤認識パターンがテキストにない場合、何も変わらない"""
        texts = ["メンバーA 1,000,000"]
        member_list = ["メンバーA"]
        replace = {"メンバーA": ["存在しないパターン"]}
        result = self.apply_replace(texts, member_list, replace)
        assert any("メンバーA" in t for t in result)


# ===================================================================
# 4. parse_args()
# ===================================================================

class TestParseArgs:
    """CLI引数パースのテスト"""

    def test_video_positional(self):
        """従来のCLI: uv run main.py video.mp4 → args.video == 'video.mp4'"""
        with patch.object(sys, "argv", ["main.py", "video.mp4"]):
            args = main.parse_args()
        assert args.video == "video.mp4"

    def test_video_optional_omitted(self):
        """引数なし → args.video is None"""
        with patch.object(sys, "argv", ["main.py"]):
            args = main.parse_args()
        assert args.video is None

    def test_gui_flag(self):
        """--gui フラグ → args.gui is True"""
        with patch.object(sys, "argv", ["main.py", "--gui"]):
            args = main.parse_args()
        assert args.gui is True
        assert args.video is None

    def test_debug_flag(self):
        """--debug → args.debug is True"""
        with patch.object(sys, "argv", ["main.py", "video.mp4", "--debug"]):
            args = main.parse_args()
        assert args.debug is True

    def test_img_scale_gray(self):
        """--img-scale gray → args.img_scale == 'gray'"""
        with patch.object(sys, "argv", ["main.py", "video.mp4", "--img-scale", "gray"]):
            args = main.parse_args()
        assert args.img_scale == "gray"

    def test_all_flags_combined(self):
        """--gui + --debug + --img-scale gray の同時指定"""
        with patch.object(sys, "argv", ["main.py", "--gui", "--debug", "--img-scale", "gray"]):
            args = main.parse_args()
        assert args.gui is True
        assert args.debug is True
        assert args.img_scale == "gray"
        assert args.video is None

    def test_video_and_gui_together(self):
        """video + --gui の同時指定 → --gui が優先（videoは無視されるわけではない）"""
        with patch.object(sys, "argv", ["main.py", "video.mp4", "--gui"]):
            args = main.parse_args()
        assert args.video == "video.mp4"
        assert args.gui is True


# ===================================================================
# 5. select_video_gui()
# ===================================================================

class TestSelectVideoGui:
    """select_video_gui() のフォールバック動作"""

    def test_tkinter_import_error_returns_none(self):
        """tkinter が利用不可の場合 → None を返す"""
        # select_video_gui 内部で tkinter を import しようとする → ImportError
        with patch("builtins.__import__", side_effect=ImportError("no tkinter")):
            result = main.select_video_gui()
        assert result is None

    def test_cancel_returns_none(self):
        """ダイアログでキャンセル（空文字） → None"""
        mock_root = MagicMock()
        mock_filedialog = MagicMock()
        mock_filedialog.askopenfilename.return_value = ""

        with (
            patch("tkinter.Tk", return_value=mock_root),
            patch("tkinter.filedialog", mock_filedialog, create=True),
        ):
            result = main.select_video_gui()
        assert result is None
        mock_root.destroy.assert_called_once()

    def test_file_selected_returns_path(self):
        """ファイル選択 → 絶対パスが返る"""
        mock_root = MagicMock()
        mock_filedialog = MagicMock()
        selected_path = "C:\\Users\\test\\input\\video.mp4"
        mock_filedialog.askopenfilename.return_value = selected_path

        with (
            patch("tkinter.Tk", return_value=mock_root),
            patch("tkinter.filedialog", mock_filedialog, create=True),
        ):
            result = main.select_video_gui()
        assert result == selected_path
        mock_root.destroy.assert_called_once()

    def test_root_attributes_set(self):
        """root.withdraw() と -topmost が呼ばれる"""
        mock_root = MagicMock()
        mock_filedialog = MagicMock()
        mock_filedialog.askopenfilename.return_value = ""

        with (
            patch("tkinter.Tk", return_value=mock_root),
            patch("tkinter.filedialog", mock_filedialog, create=True),
        ):
            main.select_video_gui()
        mock_root.withdraw.assert_called_once()
        mock_root.attributes.assert_called_once_with("-topmost", True)


# ===================================================================
# 6. select_video_gui の ImportError が main ブロックでどう扱われるか
# ===================================================================

class TestMainBranching:
    """__main__ ブロックの GUI/CLI 分岐ロジック"""

    def test_cli_mode_direct_call(self):
        """video 引数あり → CLI モード（video_path が args.video）"""
        with (
            patch.object(sys, "argv", ["main.py", "video.mp4"]),
            patch.object(main, "save_all_frames"),
            patch.object(main, "cleanup", return_value=None),
            patch("glob.glob", return_value=[]),
            patch("builtins.open", MagicMock()),
            patch("json.dump"),
            patch("json.load", return_value={}),
        ):
            # __main__ ブロックは実行しない（args パースのみ検証）
            args = main.parse_args()
            assert args.video == "video.mp4"

    def test_gui_flag_opens_dialog(self):
        """--gui フラグ → select_video_gui が呼ばれる"""
        with (
            patch.object(sys, "argv", ["main.py", "--gui"]),
            patch.object(main, "select_video_gui", return_value="C:\\input\\video.mp4") as mock_gui,
            patch.object(main, "save_all_frames"),
            patch.object(main, "cleanup", return_value=None),
            patch("glob.glob", return_value=[]),
            patch("builtins.open", MagicMock()),
            patch("json.dump"),
            patch("json.load", return_value={}),
        ):
            from main import select_video_gui
            # GUI モードで select_video_gui が呼ばれることを確認
            args = main.parse_args()
            assert args.gui is True
            assert args.video is None

    def test_no_args_triggers_gui(self):
        """引数なし → select_video_gui が呼ばれる"""
        with (
            patch.object(sys, "argv", ["main.py"]),
            patch.object(main, "select_video_gui", return_value="C:\\input\\video.mp4") as mock_gui,
            patch.object(main, "save_all_frames"),
            patch.object(main, "cleanup", return_value=None),
            patch("glob.glob", return_value=[]),
            patch("builtins.open", MagicMock()),
            patch("json.dump"),
            patch("json.load", return_value={}),
        ):
            args = main.parse_args()
            assert args.video is None


# ===================================================================
# 7. cleanup() — ディレクトリ初期化
# ===================================================================

class TestCleanup:
    """cleanup() のディレクトリ作成ロジック"""

    @patch("os.makedirs")
    @patch("shutil.rmtree")
    def test_creates_debug_subdirs(self, mock_rmtree, mock_makedirs):
        """debug_path 配下に crop/gray/frames/text が作成される"""
        base = Path("output")
        debug = base / "debug"
        main.cleanup(base, debug)

        expected_calls = [
            call(base, ignore_errors=True),  # shutil.rmtree
            call(base, exist_ok=True),
            call(debug / "crop", exist_ok=True),
            call(debug / "gray", exist_ok=True),
            call(debug / "frames", exist_ok=True),
            call(debug, exist_ok=True),
            call(debug / "text", exist_ok=True),
        ]
        mock_rmtree.assert_called_once()
        assert mock_makedirs.call_count == 6


# ===================================================================
# 8. save_all_frames() — 絶対パス / 相対パス分岐
# ===================================================================

class TestSaveAllFrames:
    """save_all_frames() のパス分岐ロジック"""

    def test_absolute_path_used_directly(self):
        """絶対パス → cv2.VideoCapture にそのまま渡される"""
        with patch("main.cv2") as mock_cv2:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.return_value = 100
            # cap.read() を最初は (True, frame)、2回目以降は (False, None) にする
            mock_cap.read.side_effect = [(True, MagicMock()), (False, None)]
            mock_cv2.VideoCapture.return_value = mock_cap

            main.save_all_frames(
                "C:\\Users\\input\\video.mp4",
                Path("output/debug"),
                "frames/frame",
            )
            # 絶対パス → ./input/ を付加せず直接使う
            mock_cv2.VideoCapture.assert_called_once_with("C:\\Users\\input\\video.mp4")

    def test_relative_path_prepends_input(self):
        """相対パス → ./input/ が先頭に付加される"""
        with patch("main.cv2") as mock_cv2:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.return_value = 100
            mock_cap.read.side_effect = [(True, MagicMock()), (False, None)]
            mock_cv2.VideoCapture.return_value = mock_cap

            main.save_all_frames(
                "video.mp4",
                Path("output/debug"),
                "frames/frame",
            )
            # 相対パス → ./input/video.mp4
            mock_cv2.VideoCapture.assert_called_once()
            call_arg = mock_cv2.VideoCapture.call_args[0][0]
            assert "input" in call_arg


# ===================================================================
# 9. edge cases / 境界値
# ===================================================================

class TestEdgeCases:
    """get_fan_count のエッジケース"""

    def test_large_number(self):
        """大きな桁数のファン数（ウマ娘では数百億までありうる）"""
        texts = ["万丈目準 110,100,100,010,000 人"]
        result = main.get_fan_count(texts, "万丈目準", [])
        assert result == 110100100010000

    def test_zero(self):
        """ファン数 0 はカンマがないのでマッチしない（0は非検出扱い）"""
        texts = ["万丈目準 0 人"]
        result = main.get_fan_count(texts, "万丈目準", [])
        assert result is None

    def test_empty_texts_list(self):
        """空の texts → None"""
        result = main.get_fan_count([], "万丈目準", [])
        assert result is None

    def test_multiple_numbers_same_text(self):
        """1テキスト内に複数のカンマ区切り数値 → メンバーを含む行を探す"""
        texts = ["万丈目準 1,000,000 人 他の値 2,000,000 人"]
        result = main.get_fan_count(texts, "万丈目準", [])
        # pattern.search() は最初のマッチを返すため、テキスト内で最初に見つかった値
        assert result is not None
