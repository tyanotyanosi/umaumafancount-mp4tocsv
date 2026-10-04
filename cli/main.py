import argparse
import logging
import sys
from pathlib import Path
import cv2
import yaml

from src.video.reader import VideoReader
from src.video.extractor import FrameExtractor, compute_frame_range
from src.video.diff_checker import DiffChecker
from src.video.card_detector import CardDetector
from src.ocr.meiki_ocr import MeikiOCRWrapper
from src.ocr.gemma4 import Gemma4Wrapper
from src.ocr.cache import CachingOCR
from src.parser.result_parser import ResultParser
from src.parser.name_mapper import NameMapper, NameMapperLoader, ensure_name_mapping_file
from src.output.json_writer import JSONWriter
from src.output.csv_writer import CSVWriter
from src.utils.error_handler import ErrorHandler, ErrorLevel
from src.utils.app_paths import data_path


def load_settings(settings_path: str | None = None) -> dict:
    """設定ファイルを読み込み。

    ファイルが存在しない / 空 / 不正な YAML の場合は空 dict を返す
    （全設定はデフォルト値で動くため、クラッシュせずに続行する）。

    ``settings_path`` 未指定時は exe 対応パス解決（app_paths.data_path）で
    ``config/settings.yaml`` を探す（exe 同置 → _internal 同梱の順）。
    """
    p = Path(settings_path) if settings_path else data_path("config/settings.yaml")
    if not p.exists():
        logging.getLogger(__name__).warning(
            "設定ファイルが見つかりません: %s（デフォルト値を使用）", p)
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError) as e:
        logging.getLogger(__name__).warning(
            "設定ファイルの読み込みに失敗しました: %s（デフォルト値を使用）: %s", p, e)
        return {}


def build_detector(settings: dict) -> CardDetector:
    """settings から CardDetector を生成"""
    cd = settings.get("card_detection", {}) if settings else {}
    template_dir = cd.get("template_dir", "template")
    # exe 同置 / exe 内同梱の順でテンプレートを解決（絶対パス指定も尊重）
    return CardDetector(template_dir=str(data_path(template_dir)), settings=settings)


def build_name_mapper(settings: dict, name_mapping_file: str = None,
                      no_name_mapping: bool = False):
    """settings の ``name_mapping`` セクションから NameMapper を構築。

    - ``no_name_mapping=True`` または ``enable`` が false の場合は ``None``
      を返し、現状維持（検知のまま集計）となる。
    - ``name_mapping_file`` が指定された場合は設定の ``file`` を上書きする。
    """
    nm = (settings or {}).get("name_mapping", {})
    if no_name_mapping or not nm.get("enable", False):
        return None
    path = name_mapping_file or nm.get("file", "config/name_mapping.json")
    # exe 同置 / exe 内同梱の順で解決（絶対パス指定も尊重）
    path = str(data_path(path))
    # 初回実行時にマッピング定義ファイルを作成（ユーザー別データ）
    ensure_name_mapping_file(path)
    mapping = NameMapperLoader.load_mapping(path)
    return NameMapper(
        mapping=mapping,
        edit_distance_threshold=int(nm.get("edit_distance_threshold", 2)),
        unmapped_action=nm.get("unmapped_action", "suggest"),
        warn_on_approx=bool(nm.get("warn_on_approx", True)),
    )


def _extract_crop(frame, box, min_height: int = 60):
    """box の crop を抽出。高さが min_height 未満の場合は 2x アップスケールして返す。"""
    x, y, w, h = box
    crop = frame[y:y + h, x:x + w]
    if crop.size == 0:
        return crop
    if h < min_height:
        crop = cv2.resize(crop, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    return crop


def _build_ocr_engines(ocr_engine: str, settings: dict, quiet: bool = False):
    """OCRエンジンを構築し、(name_ocr, name_ocr_low, fan_ocr) を返す。

    ``ocr.cache``（デフォルト true）が有効な場合、各エンジンは ``CachingOCR``
    でラップされ、crop ハッシュをキーに結果がキャッシュされる。
    ``quiet=True`` の場合は初期化の進行表示を抑制する。
    """
    cache_enabled = bool((settings or {}).get("ocr", {}).get("cache", True))

    def _wrap(eng, name):
        return CachingOCR(eng, name=name) if cache_enabled else eng

    if ocr_engine == "meiki":
        if not quiet:
            print("meikiOCR初期化中...")
        meiki_config = settings.get("ocr", {}).get("meiki", {})
        name_det_threshold = meiki_config.get("name_det_threshold", 0.3)
        name_rec_threshold = meiki_config.get("name_rec_threshold", 0.2)
        fan_det_threshold = meiki_config.get("fan_det_threshold", 0.3)
        fan_rec_threshold = meiki_config.get("fan_rec_threshold", 0.05)
        if not quiet:
            print(f"  meikiOCR name_det_threshold={name_det_threshold}, name_rec_threshold={name_rec_threshold}")
            print(f"  meikiOCR fan_det_threshold={fan_det_threshold}, fan_rec_threshold={fan_rec_threshold}")
        name_det_low = meiki_config.get("name_det_threshold_low", 0.2)
        name_rec_low = meiki_config.get("name_rec_threshold_low", 0.1)
        name_ocr = MeikiOCRWrapper(det_threshold=name_det_threshold, rec_threshold=name_rec_threshold)
        name_ocr_low = MeikiOCRWrapper(det_threshold=name_det_low, rec_threshold=name_rec_low)
        fan_ocr = MeikiOCRWrapper(det_threshold=fan_det_threshold, rec_threshold=fan_rec_threshold)
        return _wrap(name_ocr, "name"), _wrap(name_ocr_low, "name_low"), _wrap(fan_ocr, "fan")

    if not quiet:
        print("Gemma4初期化中...")
    ocr = Gemma4Wrapper()
    return _wrap(ocr, "name"), None, _wrap(ocr, "fan")


def _process_frames(frames, detector: CardDetector, name_ocr, name_ocr_low, fan_ocr,
                    output_dir: str, debug: bool = False, quiet: bool = False) -> list:
    """フレームを順に処理（カード検出 + OCR）。フレームごとの結果リストを返す。

    ``quiet=True`` の場合はフレーム毎・カード毎の進行表示を抑制する
    （エラーとサマリーは残す）。
    """
    all_results = []
    debug_dir = Path(output_dir) / "debug"
    if debug:
        debug_dir.mkdir(parents=True, exist_ok=True)
        print(f"デバッグモード: {debug_dir} に画像を保存")

    for i, frame in enumerate(frames):
        if not quiet:
            print(f"フレーム {i+1}/{len(frames)} 処理中...")

        if debug:
            cv2.imwrite(str(debug_dir / f"frame_{i:04d}.png"), frame)

        cards = detector.detect(frame)
        frame_result = {"cards": []}

        for c, card in enumerate(cards):
            name_raw = None
            if card.name_box:
                crop = _extract_crop(frame, card.name_box)
                if debug:
                    cv2.imwrite(str(debug_dir / f"frame_{i:04d}_card{c}_name.png"), crop)
                name_raw = name_ocr.recognize(crop)
                if not name_raw and name_ocr_low is not None:
                    name_raw = name_ocr_low.recognize(crop)
                if debug:
                    with open(debug_dir / f"frame_{i:04d}_card{c}_name_ocr.txt", "w", encoding="utf-8") as f:
                        f.write(name_raw if name_raw else "")
            if card.fan_box:
                crop = _extract_crop(frame, card.fan_box)
                fans_raw = None
                if debug:
                    cv2.imwrite(str(debug_dir / f"frame_{i:04d}_card{c}_fan.png"), crop)
                fans_raw = fan_ocr.recognize(crop)
                if debug:
                    with open(debug_dir / f"frame_{i:04d}_card{c}_fan_ocr.txt", "w", encoding="utf-8") as f:
                        f.write(fans_raw if fans_raw else "")
            else:
                fans_raw = None
            frame_result["cards"].append({
                "role": card.role,
                "name_raw": name_raw,
                "fans_raw": fans_raw,
                "name_box": card.name_box,
                "fan_box": card.fan_box,
            })
            if not quiet and i < 5:
                print(f"  カード{c} ({card.role}): name={repr(name_raw)}, fans={repr(fans_raw)}")

        all_results.append(frame_result)

    return all_results


def _write_outputs(merged: dict, output_dir: str, fmt: str = "all") -> tuple:
    """JSON/CSV を書き出し、パスを返す。

    ``fmt`` は "all" / "json" / "csv"。"json" / "csv" を渡した場合、
    該当フォーマットのみを書き出し、書かない方のパスは ``None`` を返す。
    """
    json_path = csv_path = None
    if fmt in ("all", "json"):
        json_writer = JSONWriter(Path(output_dir) / "json")
        json_path = json_writer.write(merged)
        print(f"JSON出力: {json_path}")
    if fmt in ("all", "csv"):
        csv_writer = CSVWriter(Path(output_dir) / "csv")
        csv_path = csv_writer.write(merged)
        print(f"CSV出力: {csv_path}")
    return json_path, csv_path


def process_video(video_path: str, ocr_engine: str = "meiki",
                  interval: float = 1.0, use_diff: bool = True,
                  output_dir: str = "output", fmt: str = "all",
                  debug: bool = False,
                  detector: CardDetector = None,
                  settings: dict = None,
                  name_mapping_file: str = None,
                  no_name_mapping: bool = False,
                  start_sec: float = 0.0, end_sec: float = 0.0, limit_sec: float = 0.0,
                  quiet: bool = False):
    """動画処理のメインロジック

    - ``start_sec`` / ``end_sec`` / ``limit_sec``: 処理範囲（秒）。
      start は動画開始からの秒、end は絶対秒（0 = 最後まで）、
      limit は start からの最大処理時間（0 = 無制限）。
    - ``quiet``: フレーム毎・カード毎の進行表示を抑制する。
    """
    error_handler = ErrorHandler()

    try:
        if not quiet:
            print(f"動画読み込み中: {video_path}")
        with VideoReader(video_path) as reader:
            fps = reader.get_fps()
            frames_count = reader.get_frame_count()
            if not quiet:
                print(f"  FPS: {fps}, フレーム数: {frames_count}")

            # フレーム範囲の決定（seek + デコード上限）
            start_frame, stop_frame = compute_frame_range(
                fps, frames_count, start_sec, end_sec, limit_sec
            )
            max_frames = stop_frame - start_frame + 1
            if start_frame > 0:
                reader.seek(start_frame)
                if not quiet:
                    print(f"  範囲: 開始 {start_sec}s → フレーム {start_frame}（シーク済み）")
            if not quiet:
                print(f"  範囲: フレーム {start_frame}..{stop_frame}（{max_frames} 枚）")

            extractor = FrameExtractor(reader, interval_sec=interval)

            if use_diff:
                if settings is None:
                    settings = load_settings()
                diff_threshold = float(settings.get("video", {}).get("diff_threshold", 0.03))
                diff_checker = DiffChecker(threshold=diff_threshold)
                if not quiet:
                    print(f"フレームをストリーミングし差分判定中（間隔: {interval}秒、閾値: {diff_threshold}）...")
            else:
                diff_checker = None
                if not quiet:
                    print(f"フレーム抽出中（間隔: {interval}秒、差分判定なし）...")

            frames = []
            total = 0
            for frame in extractor.iter_frames(max_frames=max_frames):
                total += 1
                if diff_checker is not None:
                    if diff_checker.is_different(frame):
                        frames.append(frame)
                        diff_checker.update(frame)
                else:
                    frames.append(frame)
            if not quiet:
                print(f"  計 {total} フレームを処理し、{len(frames)} フレームを採用")

            if ocr_engine not in ("meiki", "gemma4"):
                error_handler.handle(ErrorLevel.USER_ERROR, f"不明なOCRエンジン: {ocr_engine}")
                return {}
            if settings is None:
                settings = load_settings()
            name_ocr, name_ocr_low, fan_ocr = _build_ocr_engines(ocr_engine, settings, quiet=quiet)

            if detector is None:
                if not quiet:
                    print("カード検出器初期化中...")
                detector = build_detector(settings)

            all_results = _process_frames(frames, detector, name_ocr, name_ocr_low,
                                           fan_ocr, output_dir, debug, quiet=quiet)

            # OCR キャッシュ統計（有効な場合）
            for eng in (name_ocr, name_ocr_low, fan_ocr):
                if isinstance(eng, CachingOCR):
                    s = eng.stats
                    rate = (s["hits"] / (s["hits"] + s["misses"]) * 100) if (s["hits"] + s["misses"]) else 0.0
                    print(f"OCRキャッシュ ({eng.name}): ヒット {s['hits']} / ミス {s['misses']}（キャッシュ種別 {s['unique']}、ヒット率 {rate:.0f}%）")

            print(f"\n処理完了: 計 {sum(len(r['cards']) for r in all_results)} 件のカード")

            mapper = build_name_mapper(settings, name_mapping_file, no_name_mapping)
            if mapper is not None:
                file_path = name_mapping_file or settings.get("name_mapping", {}).get("file", "config/name_mapping.json")
                print(f"名前マッピングを適用中（定義ファイル: {file_path}）")
            merged = ResultParser().parse_batch(all_results, mapper=mapper)
            print(f"ユーザ情報を抽出: {len(merged)} 件")

            if mapper is not None:
                for warning in mapper.warnings:
                    print(f"  警告（近似一致）: {warning}")
                if mapper.unmapped_names:
                    unmapped = sorted(set(mapper.unmapped_names))
                    print(f"未マッピングの検知: {len(unmapped)} 件（unmapped_action='{mapper.unmapped_action}'、集計には含めず）")
                    for name in unmapped:
                        print(f"  - {name}")

            _write_outputs(merged, output_dir, fmt)

            return merged

    except Exception as e:
        error_handler.handle(ErrorLevel.SYSTEM_ERROR, f"動画処理中にエラーが発生: {e}", e)
        raise


def main():
    # 日本語 help/ログを、リダイレクト先（CI のパイプ = cp1252 等）でも
    # UnicodeEncodeError にならないよう UTF-8 に固定する。
    # exe エントリ(build/entry_cli.py)・console script・python -m cli.main を
    # 全てカバーするため、エントリ側ではなくここに置く。
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    settings = load_settings()
    default_interval = float(settings.get("video", {}).get("frame_interval", 1.0))

    parser = argparse.ArgumentParser(description="ウマ娘の動画から総獲得ファン数を抽出")
    parser.add_argument("--video", "-v", required=True, help="入力動画パス")
    parser.add_argument("--output", "-o", default="output", help="出力ディレクトリ")
    parser.add_argument("--format", "-f", choices=["json", "csv", "all"], default="all")
    parser.add_argument("--ocr", "-O", choices=["meiki", "gemma4"], default="meiki")
    parser.add_argument("--interval", "-i", type=float, default=default_interval, help="抽出間隔（秒）")
    parser.add_argument("--start", type=float, default=0.0, metavar="SEC",
                        help="処理開始位置（動画開始からの秒、デフォルト 0）")
    parser.add_argument("--end", type=float, default=0.0, metavar="SEC",
                        help="処理終了位置（動画開始からの絶対秒、0 未満 = 最後まで）")
    parser.add_argument("--limit", type=float, default=0.0, metavar="SEC",
                        help="処理の最大時間（start からの秒、0 未満 = 無制限）")
    parser.add_argument("--quiet", "-q", action="store_true", default=False,
                        help="フレーム毎・カード毎の進行表示を抑制（エラーとサマリーは表示）")
    parser.add_argument("--diff", "-d", action="store_true", default=True, help="差分判定有効")
    parser.add_argument("--no-diff", action="store_true", help="差分判定無効")
    parser.add_argument("--debug", "-D", action="store_true", default=False, help="デバッグモード（フレームとcrop画像を保存）")
    parser.add_argument("--name-mapping-file", default=None,
                        help="名前マッピング定義ファイルのパス（設定の name_mapping.file を上書き）")
    parser.add_argument("--no-name-mapping", action="store_true",
                        help="名前マッピングを無効化（設定の name_mapping.enable を無視）")

    args = parser.parse_args()

    if args.no_diff:
        args.diff = False

    process_video(
        video_path=args.video,
        ocr_engine=args.ocr,
        interval=args.interval,
        use_diff=args.diff,
        output_dir=args.output,
        fmt=args.format,
        debug=args.debug,
        settings=settings,
        name_mapping_file=args.name_mapping_file,
        no_name_mapping=args.no_name_mapping,
        start_sec=args.start,
        end_sec=args.end,
        limit_sec=args.limit,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()
