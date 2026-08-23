# mov-to-fan-count

ウマ娘の動画（.mp4）から、表示された各ユーザの**ユーザ名**と**総獲得ファン数**を
テンプレートマッチング＋OCR で自動抽出し、JSON / CSV で出力するツール。

- 解像度 2560x1072（21:9）の動画を想定
- 1 フレーム最大 3 ヶードの検出（リーダー / メンバーバッジ）
- カード内でユーザ名とファン数を個別 OCR し、(ユーザ名, ファン数) を対で確定
- フレーム間の同一ユーザ候補に**多数決**で誤読を補正

## 動作要件

- Python **3.11** 以上
- Windows（検証済み）/ Linux・macOS は未検証

依存パッケージ（`pip` で自動インストール）:
`opencv-python` / `meikiocr` / `customtkinter` / `Pillow` / `pyyaml`

## インストール

```bash
# 仮想環境を作成し、開発モードでインストール
python -m venv .venv
.venv\Scripts\activate            # Windows（Linux/macOS は .venv/bin/activate）
pip install -e .

# テスト用ツール（任意）
pip install -e ".[dev]"
```

インストールするとコマンドラインツール `mov-to-fan-count` と
GUI ツール `mov-to-fan-count-gui` が使えるようになる。

## クイックスタート

```bash
mov-to-fan-count -v "path/to/video.mp4" -o output
```

出力:
- `output/json/result_<timestamp>.json` — `{ "ユーザ名": ファン数, ... }`
- `output/csv/result_<timestamp>.csv` — `ユーザ名,ファン数` の行

## CLI オプション

| オプション | 説明 |
|---|---|
| `--video, -v` | 入力動画パス（必須） |
| `--output, -o` | 出力ディレクトリ（既定 `output`） |
| `--format, -f` | `json` / `csv` / `all`（既定 `all`） |
| `--ocr, -O` | OCR エンジン: `meiki` / `gemma4`（既定 `meiki`） |
| `--interval, -i` | 抽出間隔（秒、既定は設定ファイルの `frame_interval`） |
| `--start` | 処理開始位置（動画開始からの秒、既定 0） |
| `--end` | 処理終了位置（動画開始からの絶対秒、0 未満 = 最後まで） |
| `--limit` | 処理の最大時間（start からの秒、0 未満 = 無制限） |
| `--quiet, -q` | フレーム毎・カード毎の進行表示を抑制（エラーとサマリーは表示） |
| `--diff, -d` | フレーム差分判定を有効（既定） |
| `--no-diff` | フレーム差分判定を無効 |
| `--debug, -D` | デバッグモード（フレームと crop 画像を `output/debug` に保存） |
| `--name-mapping-file` | 名前マッピング定義ファイルのパス（設定を上書き） |
| `--no-name-mapping` | 名前マッピングを無効化 |

### 範囲指定の例

```bash
# 動画の先頭 30 秒だけ処理
mov-to-fan-count -v video.mp4 --limit 30

# 60 秒目から 90 秒目まで（30 秒分）
mov-to-fan-count -v video.mp4 --start 60 --limit 30

# 60 秒目から 120 秒目まで
mov-to-fan-count -v video.mp4 --start 60 --end 120

# 静かに（サマリーのみ表示）
mov-to-fan-count -v video.mp4 --quiet -o output
```

## 名前マッピング

OCR の誤読（例: 「ろん」→「たけし」）を、ユーザ定義のマッピング定義で実ユーザ名に
正しくする機能。

- 定義ファイル: `config/name_mapping.json`（ユーザが編集）
- 有効化・閾値: `config/settings.yaml` の `name_mapping` セクション
  - `enable: true`、`edit_distance_threshold: 2`（近似一致のレーベンシュタイン距離）
  - `unmapped_action: suggest` / `keep` / `drop`
- 近似一致は警告を出力、未マッピング検知は `suggest` では集計に含まれない

## 設定（config/settings.yaml）

```yaml
video:
  frame_interval: 0        # 0 = 差分判定のみで採用フレームを決定
  enable_diff_check: true
  diff_threshold: 0.03     # 静止区間ノイズ床より上、スクロールステップより下

card_detection:
  template_dir: "template"
  badge_match_threshold: 0.6
  label_match_threshold: 0.7
  icon_match_threshold: 0.7
  max_cards: 3
  reference_width: 2560    # テンプレート撮影時のフレーム幅（スケール先行推定）
  multi_scale: true
  scale_cache: true

ocr:
  engine: meiki
  cache: true              # crop ハッシュ OCR キャッシュ（同一 crop は再認識しない）
  meiki:
    name_det_threshold: 0.3
    name_rec_threshold: 0.2
    name_det_threshold_low: 0.2   # 読み取り失敗時のフォールバック
    name_rec_threshold_low: 0.1
    fan_det_threshold: 0.3
    fan_rec_threshold: 0.05

name_mapping:
  file: "config/name_mapping.json"
  enable: true
  edit_distance_threshold: 2
  unmapped_action: "suggest"
```

## 処理パイプライン

1. **VideoReader** — 動画のデコード（`src/video/reader.py`）
2. **FrameExtractor** — 間隔サンプリング／範囲指定（`--start/--end/--limit`）（`src/video/extractor.py`）
3. **DiffChecker** — フレーム全体の差分で冗長フレームを除外（`src/video/diff_checker.py`）
4. **CardDetector** — 多スケールテンプレートマッチングでカードを検出（`src/video/card_detector.py`）
5. **OCR** — meiki でカード内のユーザ名・ファン数を認識（`src/ocr/meiki_ocr.py`）。
   `ocr.cache` 有効時は crop ハッシュで同一 crop の再認識を省略（`src/ocr/cache.py`）
6. **ResultParser** — カード結果の正規化・多数決・桁混同補正（`src/parser/result_parser.py`）
7. **出力** — JSON / CSV（`cli/main.py`）

## テスト

```bash
# 全テスト（E2E slow は MOV_E2E=1 でオプトイン）
pytest -q

# フル動画 E2E（数分〜十数分かかる）
$env:MOV_E2E="1"; pytest -m slow
```

- `tests/test_golden_detection.py` — ゴールデン検出回帰（9 サンプルフレーム）
- `tests/test_golden_e2e.py` — フル動画 E2E（`slow` マーカー、`MOV_E2E=1` で実行）
- `tests/golden/` — ゴールデン資産（サンプルフレーム・期待結果 JSON）
- `scripts/make_golden.py` — 検出ゴールデンの再生成（検出ロジック変更時の差分確認用）

## 出力例

```json
{
    "たけし": 3249444186
}
```

## ディレクトリ構成

```
cli/main.py          # CLI エントリポイント
src/video/           # 動画読み込み・フレーム抽出・差分判定・カード検出
src/ocr/             # OCR ラッパー（meiki / gemma4 / キャッシュ）
src/parser/          # 結果パース（多数決・桁混同補正）
config/              # 設定（settings.yaml / name_mapping.json）
template/            # カード検出テンプレート（4 画像）
tests/               # テスト + ゴールデン資産
scripts/             # 補助スクリプト（make_golden など）
docs/                # 設計・開発ログ・改善提案
```
