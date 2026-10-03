"""Tests for ``cli.main.process_video``.

Specification: docs/00-Architecture/cli_main__process_video.yaml

``process_video`` is the main video-processing logic: frame extraction,
diff-based adoption, card detection + OCR, name mapping, result
aggregation, and JSON/CSV output in one flow:

- Step 1: create an ``ErrorHandler`` instance
- Step 2: in a ``try`` block: if quiet is False, print
  「動画読み込み中: {video_path}」 and open ``VideoReader(video_path)``
  as a context manager
- Step 3: get fps / frames_count via ``reader.get_fps()`` /
  ``reader.get_frame_count()``; if quiet is False, print both
- Step 4: compute ``(start_frame, stop_frame)`` via
  ``compute_frame_range(fps, frames_count, start_sec, end_sec, limit_sec)``
  and let ``max_frames = stop_frame - start_frame + 1``
- Step 5: if ``start_frame > 0``, call ``reader.seek(start_frame)``
  (if quiet is False, print the seeked count); if quiet is False, print
  「範囲: フレーム {start_frame}..{stop_frame}（{max_frames} 枚）」
- Step 6: create ``FrameExtractor(reader, interval_sec=interval)``
- Step 7: if use_diff is True: if settings is None, call
  ``load_settings()``; create ``DiffChecker(threshold=...)`` with
  ``float(settings.get("video", {}).get("diff_threshold", 0.03))``; if
  use_diff is False, ``diff_checker = None`` (in either case, if quiet is
  False, print progress)
- Step 8: iterate ``extractor.iter_frames(max_frames=max_frames)`` and
  count total; if diff_checker is not None, adopt only frames for which
  ``is_different(frame)`` is True (calling ``update(frame)``); if None,
  adopt all frames
- Step 9: if quiet is False, print
  「計 {total} フレームを処理し、{len(frames)} フレームを採用」
- Step 10: if ocr_engine is not in ("meiki", "gemma4"), call
  ``error_handler.handle(ErrorLevel.USER_ERROR, "不明なOCRエンジン: {ocr_engine}")``
  and return ``{}``
- Step 11: if settings is None, call ``load_settings()``; generate
  name_ocr / name_ocr_low / fan_ocr via
  ``_build_ocr_engines(ocr_engine, settings, quiet=quiet)``
- Step 12: if detector is None (if quiet is False, print
  「カード検出器初期化中...」), generate it via ``build_detector(settings)``
- Step 13: get all_results via
  ``_process_frames(frames, detector, name_ocr, name_ocr_low, fan_ocr,
  output_dir, debug, quiet=quiet)``
- Step 14: for each of the 3 OCR engines, if it is a ``CachingOCR``, read
  hits / misses / unique from ``eng.stats``, compute the hit rate, and
  print 「OCRキャッシュ ({eng.name}): ...」 (independent of quiet)
- Step 15: print
  「処理完了: 計 {sum(len(r["cards"]) for r in all_results)} 件のカード」
  (independent of quiet)
- Step 16: generate mapper via
  ``build_name_mapper(settings, name_mapping_file, no_name_mapping)``; if
  mapper is not None, print the definition file path (name_mapping_file
  or settings' name_mapping.file, default "config/name_mapping.json")
- Step 17: get merged via ``ResultParser().parse_batch(all_results,
  mapper=mapper)`` and print 「ユーザ情報を抽出: {len(merged)} 件」
- Step 18: if mapper is not None, print mapper.warnings (approximate
  match warnings) one by one; if mapper.unmapped_names is non-empty,
  print the sorted(set(...)) unmapped names as
  「未マッピングの検知: {n} 件（unmapped_action=...）」
- Step 19: call ``_write_outputs(merged, output_dir, fmt)`` and return
  merged
- Step 20: as ``except Exception``, call
  ``error_handler.handle(ErrorLevel.SYSTEM_ERROR, "動画処理中にエラーが発生: {e}", e)``
  and re-raise the original exception

Mocked / stand-in dependencies (per the test-generation rules):
``cli.main.VideoReader`` / ``cli.main.FrameExtractor`` are mocks (the
mocked reader reports fps=30.0 / frame_count=10 and the mocked extractor
yields 3 frames); ``cli.main.load_settings`` /
``cli.main._build_ocr_engines`` / ``cli.main.build_detector`` /
``cli.main._process_frames`` (returns ``[]``) /
``cli.main.build_name_mapper`` (returns None) / ``cli.main.ResultParser``
(``parse_batch`` returns ``{}``) / ``cli.main._write_outputs`` /
``cli.main.DiffChecker`` / ``cli.main.ErrorHandler`` are mocks.
``compute_frame_range`` runs real (pure function; observed to return
(0, 9) for fps=30.0, frame_count=10, start_sec=0.0, end_sec=0.0,
limit_sec=0.0).

``errors`` section of the spec (documented here only, NOT tested, per the
test-generation rules):

- condition: "with VideoReader ブロック内の処理で例外が発生（動画の読み込み失敗、デコードエラー、設定の型不正、OCR・検出の例外、出力書き出し失敗等）"
  behavior: "error_handler.handle(ErrorLevel.SYSTEM_ERROR, "動画処理中にエラーが発生: {e}", e) が呼ばれた後、元の例外が呼び出し側に再送出される"
"""

import contextlib
import re
from unittest import mock

import pytest

from cli.main import process_video
from src.ocr.cache import CachingOCR
from src.utils.error_handler import ErrorLevel
from src.video.card_detector import CardDetector

# フレーム数（モック extractor が yield する枚数）
_FRAMES = 3


@contextlib.contextmanager
def _mocks(settings=None, load_settings_value=None, quiet=False,
           ocr_engine="meiki", detector=None, use_diff=True,
           diff_is_different=None, engines=None):
    """Patch the dependencies of ``process_video`` and yield a dict of
    the mocks (plus the mocked reader / extractor).

    Defaults: ``load_settings_value`` is ``{}`` (load_settings returns an
    empty dict when the settings file is missing/empty/invalid YAML);
    ``engines`` is ``(None, None, None)`` because
    ``_build_ocr_engines``'s return value is unpacked into 3 engines.
    """
    if load_settings_value is None:
        load_settings_value = {}
    if engines is None:
        engines = (None, None, None)
    frames = [b"frame_%d" % i for i in range(_FRAMES)]
    with mock.patch("cli.main.VideoReader") as mock_vr, \
            mock.patch("cli.main.FrameExtractor") as mock_fe, \
            mock.patch("cli.main.load_settings",
                       return_value=load_settings_value) as mock_ls, \
            mock.patch("cli.main._build_ocr_engines",
                       return_value=engines) as mock_boe, \
            mock.patch("cli.main.build_detector") as mock_bd, \
            mock.patch("cli.main._process_frames",
                       return_value=[]) as mock_pf, \
            mock.patch("cli.main.build_name_mapper",
                       return_value=None) as mock_bnm, \
            mock.patch("cli.main.ResultParser") as mock_rp, \
            mock.patch("cli.main._write_outputs") as mock_wo, \
            mock.patch("cli.main.DiffChecker") as mock_dc, \
            mock.patch("cli.main.ErrorHandler") as mock_eh:
        reader = mock.MagicMock(name="reader")
        reader.get_fps.return_value = 30.0
        reader.get_frame_count.return_value = 10
        mock_vr.return_value.__enter__.return_value = reader
        mock_vr.return_value.__exit__.return_value = False
        mock_fe.return_value.iter_frames.return_value = iter(frames)
        mock_rp.return_value.parse_batch.return_value = {}
        if diff_is_different is not None:
            mock_dc.return_value.is_different.return_value = diff_is_different
        yield {
            "reader": reader,
            "mock_vr": mock_vr,
            "mock_fe": mock_fe,
            "mock_ls": mock_ls,
            "mock_boe": mock_boe,
            "mock_bd": mock_bd,
            "mock_pf": mock_pf,
            "mock_bnm": mock_bnm,
            "mock_rp": mock_rp,
            "mock_wo": mock_wo,
            "mock_dc": mock_dc,
            "mock_eh": mock_eh,
        }


def test_edge_01(capsys):
    """
    input: ocr_engine="tesseract"（meiki/gemma4 以外）、他引数は既定値、読み込み可能な動画
    expected: フレーム抽出が完了した上で error_handler.handle(ErrorLevel.USER_ERROR, "不明なOCRエンジン: tesseract") が呼ばれ {} が返る; _build_ocr_engines / _process_frames / _write_outputs は呼ばれない
    """
    with _mocks(ocr_engine="tesseract") as m:
        ret = process_video(video_path="v.mp4", ocr_engine="tesseract")
        assert ret == {}
        m["mock_eh"].return_value.handle.assert_called_once_with(
            ErrorLevel.USER_ERROR, "不明なOCRエンジン: tesseract")
        m["mock_boe"].assert_not_called()
        m["mock_pf"].assert_not_called()
        m["mock_wo"].assert_not_called()


def test_edge_02(capsys):
    """
    input: use_diff=False
    expected: DiffChecker は生成されず、extractor.iter_frames からの全フレームが採用される（採用枚数 == total）
    """
    with _mocks(use_diff=False) as m:
        ret = process_video(video_path="v.mp4", use_diff=False)
        assert ret == {}
        m["mock_dc"].assert_not_called()
        out = capsys.readouterr().out
        match = re.search(
            r"計 (\d+) フレームを処理し、(\d+) フレームを採用", out)
        assert match is not None
        total, adopted = int(match.group(1)), int(match.group(2))
        assert adopted == total
        # 全フレームが _process_frames に渡される
        assert len(m["mock_pf"].call_args.args[0]) == adopted


def test_edge_03(capsys):
    """
    input: use_diff=True、settings=None
    expected: load_settings() が呼ばれ、settings["video"]["diff_threshold"]（無ければ 0.03）が float 変換されて DiffChecker(threshold=...) が生成される
    """
    with _mocks(settings=None, load_settings_value={}) as m:
        ret = process_video(video_path="v.mp4", use_diff=True,
                            settings=None)
        assert ret == {}
        m["mock_ls"].assert_called_once()
        m["mock_dc"].assert_called_once()
        call = m["mock_dc"].call_args
        assert call.kwargs.get("threshold") == 0.03 or (
            call.args and call.args[0] == 0.03)


def test_edge_04(capsys):
    """
    input: settings={"video": {"diff_threshold": 0.1}}、use_diff=True
    expected: DiffChecker(threshold=0.1) が生成される
    """
    settings = {"video": {"diff_threshold": 0.1}}
    with _mocks(settings=settings) as m:
        ret = process_video(video_path="v.mp4", use_diff=True,
                            settings=settings)
        assert ret == {}
        m["mock_ls"].assert_not_called()
        m["mock_dc"].assert_called_once()
        call = m["mock_dc"].call_args
        assert call.kwargs.get("threshold") == 0.1 or (
            call.args and call.args[0] == 0.1)
        # settings が _build_ocr_engines に渡される
        assert settings in m["mock_boe"].call_args.args


def test_edge_05(capsys):
    """
    input: detector に CardDetector インスタンスを渡入
    expected: build_detector は呼ばれず、渡入されたインスタンスが _process_frames に使用される
    """
    detector = mock.Mock(spec=CardDetector)
    with _mocks(detector=detector) as m:
        ret = process_video(video_path="v.mp4", detector=detector)
        assert ret == {}
        m["mock_bd"].assert_not_called()
        assert m["mock_pf"].call_args.args[1] is detector


def test_edge_06(capsys):
    """
    input: start_sec=0.0（start_frame が 0 になる場合）
    expected: reader.seek は呼ばれない
    """
    with _mocks() as m:
        ret = process_video(video_path="v.mp4", start_sec=0.0)
        assert ret == {}
        m["reader"].seek.assert_not_called()
        # compute_frame_range(30.0, 10, 0.0, 0.0, 0.0) = (0, 9)
        # → max_frames = 9 - 0 + 1 = 10
        m["mock_fe"].return_value.iter_frames.assert_called_once_with(
            max_frames=10)


def test_edge_07(capsys):
    """
    input: 採用フレームが0枚（差分判定で全フレームが除外される等）
    expected: _process_frames は [] を返し「処理完了: 計 0 件のカード」が出力され、ResultParser に空リストが渡され、_write_outputs は空の結果で書き出し、関数はその merged が返る
    """
    with _mocks(diff_is_different=False) as m:
        ret = process_video(video_path="v.mp4")
        out = capsys.readouterr().out
        assert "処理完了: 計 0 件のカード" in out
        # 空のフレームリストが _process_frames に渡される
        assert m["mock_pf"].call_args.args[0] == []
        # ResultParser に空リストが渡される
        m["mock_rp"].return_value.parse_batch.assert_called_once()
        assert m["mock_rp"].return_value.parse_batch.call_args.args[0] == []
        # _write_outputs は空の結果で書き出し
        m["mock_wo"].assert_called_once()
        assert m["mock_wo"].call_args.args[0] == {}
        assert ret == {}


def test_edge_08(capsys):
    """
    input: quiet=True
    expected: 「動画読み込み中」・FPS/フレーム数・範囲・採用枚数の進行表示はされないが、「処理完了」「ユーザ情報を抽出」・OCRキャッシュ統計・出力パス等の表示はされる（これらは quiet 判定の外にある）
    """
    engines = []
    for name in ("name_ocr", "name_ocr_low", "fan_ocr"):
        eng = mock.Mock(spec=CachingOCR)
        eng.name = name
        eng.stats = {"hits": 5, "misses": 3, "unique": 8}
        engines.append(eng)
    with _mocks(quiet=True, engines=engines) as m:
        ret = process_video(video_path="v.mp4", quiet=True)
        assert ret == {}
        out = capsys.readouterr().out
        # quiet=True で抑制される進行表示
        assert "動画読み込み中" not in out
        assert "範囲: フレーム" not in out
        assert "フレームを採用" not in out
        # quiet 判定の外の表示はされる
        assert "処理完了: 計 0 件のカード" in out
        assert "ユーザ情報を抽出: 0 件" in out
        assert "OCRキャッシュ (name_ocr)" in out


def test_edge_09(capsys):
    """
    input: settings={"video": {"diff_threshold": "abc"}}、use_diff=True
    expected: float() 変換で ValueError が送出され、error_handler.handle(ErrorLevel.SYSTEM_ERROR, ...) 呼び出し後に例外が再送出される
    """
    settings = {"video": {"diff_threshold": "abc"}}
    with _mocks(settings=settings) as m:
        with pytest.raises(ValueError):
            process_video(video_path="v.mp4", use_diff=True,
                          settings=settings)
        m["mock_eh"].return_value.handle.assert_called_once()
        args = m["mock_eh"].return_value.handle.call_args.args
        assert args[0] is ErrorLevel.SYSTEM_ERROR
        assert "動画処理中にエラーが発生" in args[1]
        assert isinstance(args[2], ValueError)
