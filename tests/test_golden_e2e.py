"""ゴールデン回帰テスト (E2E): フル動画で本番パイプラインを走らせ、ファン数の値集合が安定していることを保証する。

- 名前表記は OCR により揺れるため、**ファンカウントの値集合（unique）** のみを比較する。
- 実行時間がかかる（数十秒〜数分）ため `slow` マーカー付き。
- デフォルトではスキップ（オプトイン: 環境変数 ``MOV_E2E=1`` を付けて実行）。
  動画ファイルが無い、または meikiocr が無い環境でも自動的にスキップする。

実行例::

    MOV_E2E=1 pytest -m slow tests/test_golden_e2e.py -v
"""

import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
GOLDEN = ROOT / "tests" / "golden"
EXPECTED_FANS = GOLDEN / "e2e_fans_expected.json"
VIDEOS_DIR = ROOT / "inputmov"

# 本番と同一の条件
SETTINGS_PATH = ROOT / "config" / "settings.yaml"


def _video():
    vids = sorted(VIDEOS_DIR.glob("*.mp4"))
    assert vids, f"inputmov に動画が見つかりません: {VIDEOS_DIR}"
    return vids[0]


@pytest.fixture
def golden_fans():
    return json.load(open(EXPECTED_FANS, encoding="utf-8"))["unique_counts"]


def test_e2e_fan_counts_match_golden(golden_fans, tmp_path):
    # オプトイン: デフォルトでは実行しない
    if os.environ.get("MOV_E2E") != "1":
        pytest.skip("E2E は MOV_E2E=1 でオプトインして実行します（数十秒〜数分）")

    video = _video()
    if not video.exists():
        pytest.skip(f"動画が不存在: {video}")

    pytest.importorskip("meikiocr", reason="meikiocr がインストールされていません")

    from cli.main import process_video, load_settings

    settings = load_settings(str(SETTINGS_PATH))
    out_dir = str(tmp_path / "out")

    # 本番と同一: meiki / 全フレーム + 差分判定 / 名前マッピング無効
    merged = process_video(
        video_path=str(video),
        ocr_engine="meiki",
        interval=0.0,
        use_diff=True,
        output_dir=out_dir,
        settings=settings,
        no_name_mapping=True,
    )

    got = sorted(set(merged.values()))
    expected = sorted(golden_fans)
    assert got == expected, (
        "ファンカウントの値集合がゴールデンと異なる:\n"
        f"  期待 ({len(expected)}): {expected}\n"
        f"  実際 ({len(got)}): {got}\n"
        f"  不足: {sorted(set(expected) - set(got))}\n"
        f"  余分: {sorted(set(got) - set(expected))}"
    )
