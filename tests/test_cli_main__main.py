"""Tests for ``cli.main.main``.

Specification: docs/00-Architecture/cli_main__main.yaml

``main`` is the CLI entry point: it parses the command-line arguments,
loads the settings, and then calls ``process_video`` to run the video
processing:

- Step 1: ``settings = load_settings()``
- Step 2: ``default_interval = float(settings.get("video", {}).get("frame_interval", 1.0))``
- Step 3: create ``argparse.ArgumentParser`` (description
  「ウマ娘の動画から総獲得ファン数を抽出」)
- Step 4: register arguments: --video/-v (required), --output/-o
  (default "output"), --format/-f (choices json/csv/all, default all),
  --ocr/-O (choices meiki/gemma4, default meiki), --interval/-i (float,
  default default_interval), --start (float, default 0.0), --end (float,
  default 0.0), --limit (float, default 0.0), --quiet/-q (store_true),
  --diff/-d (store_true, default True), --no-diff (store_true),
  --debug/-D (store_true), --name-mapping-file (default None),
  --no-name-mapping (store_true)
- Step 5: ``args = parser.parse_args()``
- Step 6: if ``args.no_diff`` is True, set ``args.diff = False``
  (--no-diff takes precedence)
- Step 7: call ``process_video`` with
  video_path=args.video, ocr_engine=args.ocr, interval=args.interval,
  use_diff=args.diff, output_dir=args.output, fmt=args.format,
  debug=args.debug, settings=settings,
  name_mapping_file=args.name_mapping_file,
  no_name_mapping=args.no_name_mapping, start_sec=args.start,
  end_sec=args.end, limit_sec=args.limit, quiet=args.quiet (detector is
  not passed and stays the default None)

The function returns ``None`` (no return statement); the return value of
``process_video`` (merged dict) is discarded.

Mocked / stand-in dependencies (per the test-generation rules):
``sys.argv`` is patched per test; ``cli.main.load_settings`` is patched to
return a settings dict per test; ``cli.main.process_video`` is a mock
whose call arguments are asserted. ``main`` itself runs real.

``errors`` section of the spec (documented here only, NOT tested, per the
test-generation rules):

- condition: 'process_video 内部で例外が発生'
  behavior: 'main には try/except がなく、例外はそのまま呼び出し側（スクリプト）へ伝播する'
"""

from unittest import mock

import pytest

from cli.main import main


def test_edge_01():
    """
    input: コマンドラインに --video が含まれない
    expected: argparse がエラーを出力し SystemExit（終了コード2）が送出される
    """
    with mock.patch("sys.argv", ["prog"]), \
            mock.patch("cli.main.load_settings", return_value={}):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 2


def test_edge_02():
    """
    input: --format xml（choices にない値）
    expected: SystemExit（終了コード2）が送出される
    """
    with mock.patch("sys.argv", ["prog", "--video", "v.mp4",
                                "--format", "xml"]), \
            mock.patch("cli.main.load_settings", return_value={}):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 2


def test_edge_03():
    """
    input: --ocr tesseract（choices にない値）
    expected: SystemExit（終了コード2）が送出される
    """
    with mock.patch("sys.argv", ["prog", "--video", "v.mp4",
                                "--ocr", "tesseract"]), \
            mock.patch("cli.main.load_settings", return_value={}):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 2


def test_edge_04():
    """
    input: --interval abc（float に変換不能）
    expected: SystemExit（終了コード2）が送出される
    """
    with mock.patch("sys.argv", ["prog", "--video", "v.mp4",
                                "--interval", "abc"]), \
            mock.patch("cli.main.load_settings", return_value={}):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 2


def test_edge_05():
    """
    input: --diff と --no-diff が同時に指定
    expected: use_diff=False として process_video が呼ばれる（--no-diff が優先）
    """
    with mock.patch("sys.argv", ["prog", "--video", "v.mp4",
                                "--diff", "--no-diff"]), \
            mock.patch("cli.main.load_settings", return_value={}), \
            mock.patch("cli.main.process_video") as mock_pv:
        ret = main()
        assert ret is None
        mock_pv.assert_called_once()
        kwargs = mock_pv.call_args.kwargs
        assert kwargs["use_diff"] is False


def test_edge_06():
    """
    input: --output 未指定
    expected: output_dir="output" として process_video が呼ばれる
    """
    with mock.patch("sys.argv", ["prog", "--video", "v.mp4"]), \
            mock.patch("cli.main.load_settings", return_value={}), \
            mock.patch("cli.main.process_video") as mock_pv:
        ret = main()
        assert ret is None
        mock_pv.assert_called_once()
        kwargs = mock_pv.call_args.kwargs
        assert kwargs["output_dir"] == "output"


def test_edge_07():
    """
    input: --interval 未指定、config/settings.yaml に video: {frame_interval: 2.5}
    expected: interval=2.5 として process_video が呼ばれる
    """
    with mock.patch("sys.argv", ["prog", "--video", "v.mp4"]), \
            mock.patch("cli.main.load_settings",
                       return_value={"video": {"frame_interval": 2.5}}), \
            mock.patch("cli.main.process_video") as mock_pv:
        ret = main()
        assert ret is None
        mock_pv.assert_called_once()
        kwargs = mock_pv.call_args.kwargs
        assert kwargs["interval"] == 2.5


def test_edge_08():
    """
    input: config/settings.yaml が video: 5（非dict）を含む
    expected: default_interval 計算時に AttributeError: "int" object has no attribute "get" が送出される
    """
    with mock.patch("sys.argv", ["prog", "--video", "v.mp4"]), \
            mock.patch("cli.main.load_settings", return_value={"video": 5}):
        with pytest.raises(AttributeError):
            main()


def test_edge_09():
    """
    input: --name-mapping-file 未指定、--no-name-mapping 未指定
    expected: name_mapping_file=None、no_name_mapping=False として process_video が呼ばれる
    """
    with mock.patch("sys.argv", ["prog", "--video", "v.mp4"]), \
            mock.patch("cli.main.load_settings", return_value={}), \
            mock.patch("cli.main.process_video") as mock_pv:
        ret = main()
        assert ret is None
        mock_pv.assert_called_once()
        kwargs = mock_pv.call_args.kwargs
        assert kwargs["name_mapping_file"] is None
        assert kwargs["no_name_mapping"] is False
