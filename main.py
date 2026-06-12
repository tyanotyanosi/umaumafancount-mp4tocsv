import sys
import argparse
import time
from pathlib import Path

# Ensure the 'src' directory is in the module search path
import sys as sys_module
sys_module.path.append(str(Path(__file__).parent))

from src.domain.models import PipelineConfig
from src.app.gui.main_window import AppWindow
from src.core.pipeline import PipelineRunner

def _run_batch_mode(config: PipelineConfig):
    """GUIを開かず、直接パイプラインを実行（バッチモード用）"""
    runner = PipelineRunner()
    result = runner.run(
        config,
        lambda msg, pct=None: print(f"[{time.strftime('%H:%M:%S')}] {msg}")
    )

    if result.error:
        print(f"エラーが発生しました: {result.error}")
        sys.exit(1)

    print("=== 処理完了 ===")
    import json
    print(json.dumps(result.fan_counts, ensure_ascii=False, indent=4))

def _build_parser() -> argparse.ArgumentParser:
    """テスト容易性のためパーサー構築を分離"""
    parser = argparse.ArgumentParser(
        description="Umamusume Pretty Derby の動画からファン数を抽出するツール(WinRt OCR使用)"
    )
    parser.add_argument(
        "video", nargs="?", type=str, default=None,
        help="入力動画のファイル名。省略時はGUIが起動します。"
    )
    parser.add_argument("--gui", action="store_true", help="GUIを起動する")
    parser.add_argument("--debug", action="store_true", help="中間ファイルを削除せずに残す")
    parser.add_argument(
        "--img-scale", type=str, default=None,
        help="グレースケールで文字認識する場合は'gray'を指定"
    )
    return parser

def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """CLI引数をパースする（テスト用に独立）"""
    return _build_parser().parse_args(args)

def main():
    args = parse_args()

    # バッチモード：動画パスがCLIで指定された場合、GUIを開かずに直接実行
    if args.video is not None:
        config = PipelineConfig(
            video_path=Path(args.video),
            debug=args.debug,
            img_scale=args.img_scale
        )
        _run_batch_mode(config)

    # GUI モード：動画パスなし、または --gui フラグ指定時
    else:
        app = AppWindow(debug=args.debug, img_scale=args.img_scale)
        app.run()

if __name__ == "__main__":
    main()
