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

> OCR エンジン `gemma4` は任意（後述）。

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

> JSON と CSV は**既定で両方書き出される**（`--format all`）。
> `--format` で `json` / `csv` のいずれか一方のみに限定できます。

## Windows exe 版（Python 不要 / オフライン動作）

Python をインストールしない PC で使う場合、[GitHub Releases](releases) の
`mov-to-fan-count-v1.0.0-win64.zip` をダウンロードして使う。

- exe 版は **v1.0.0 以降** のリリースから公開される
  （v0.0.x のリリースは旧アーキテクチャ時代のソース配布で、exe は含まれない）

1. zip を任意のフォルダに解凍（移動していても問題ない）
2. `mov-to-fan-count-gui.exe` をダブルクリック（初回起動は 10〜30 秒）
3. 動画を選択し「動画処理」→ 結果を JSON / CSV で保存

- OCR に使う meiki モデル（約 46 MB）は exe に同梱されているため、
  **ネットワーク接続は不要**（オフラインで初回から動作）
- `config\settings.yaml` / `template\*.png` は exe の隣に同梱されており、
  テキストエディタで編集できる（exe 内部同梱版より優先される）
- `mov-to-fan-count.exe` が CLI 版（コンソール）。オプションは後述と同じ
- 詳細・トラブルシュート: `user-guide/usage.txt`
  （zip 内には「使い方.txt」として同梱）

## CLI オプション

| オプション | 説明 |
|---|---|
| `--video, -v` | 入力動画パス（必須） |
| `--output, -o` | 出力ディレクトリ（既定 `output`） |
| `--format, -f` | `json` / `csv` / `all`（既定 `all`）— 書き出すフォーマットを選択 |
| `--ocr, -O` | OCR エンジン: `meiki` / `gemma4`（既定 `meiki`） |
| `--interval, -i` | 抽出間隔（秒、既定は設定ファイルの `frame_interval`、0 = サンプリングせず差分判定のみで採用を決定） |
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

> CLI の間隔は `--interval` と `video.frame_interval` で決まる
> （`0` = サンプリングせず差分判定のみ）。設定ファイルの `video.diff_only` は
> **GUI 専用**で CLI には影響しない。

### OCR エンジン

- `meiki`（既定）— `meikiocr` パッケージ。ユーザ名用・ファン数用の別閾値の
  ラッパーを使い、ユーザ名の読み取りに失敗した場合は低閾値
  （`name_det_threshold_low` / `name_rec_threshold_low`）で再認識する。
- `gemma4`（オプション）— `pip install -e ".[gemma4]"`（`torch` / `transformers`）
  が必要。**現在は使用不可**（認識結果は空文字を返す）。

## 名前マッピング

OCR の誤読（例: 「のびた」→「のび太」）を、ユーザ定義のマッピング定義で実ユーザ名に
正しくする機能。

- 定義ファイル: `config/name_mapping.json`（ユーザが編集、JSON 形式）
- 有効化・閾値: `config/settings.yaml` の `name_mapping` セクション
  - `enable: true`、`edit_distance_threshold: 2`（近似一致のレーベンシュタイン距離）
  - `warn_on_approx: true`（近似一致時に警告を出力）
  - `unmapped_action: suggest` / `keep` / `drop`
- 近似一致は警告を出力、`suggest` では未マッピング検知は集計に含まれない
  （一覧として末尾に表示）、`keep` は検知名のまま集計、`drop` は除外

定義ファイルの例（`config/name_mapping.json`）:

```json
{
  "user_names": {
    "のび太": { "aliases": ["のび太", "のびた", "のび", "Nobita"] }
  },
  "raw_to_user": {
    "固定の検知名A": "ジャイアン"
  }
}
```

- `user_names` — 実ユーザ名 → OCR 検知名候補（`aliases`）。完全一致を優先し、
  一致なしの場合は編集距離 ≤ `edit_distance_threshold` で近似一致を試みる
- `raw_to_user` — 検知名 → 実ユーザ名の直接対応表（完全一致）

## 設定（config/settings.yaml）

```yaml
video:
  frame_interval: 0        # 0 = サンプリングしない（差分判定のみで採用を決定）
  diff_threshold: 0.03     # 静止区間ノイズ床より上、スクロールステップより下

card_detection:
  template_dir: "template"
  badge_match_threshold: 0.6
  label_match_threshold: 0.7
  icon_match_threshold: 0.7
  max_cards: 3
  reference_width: 2560    # テンプレート撮影時のフレーム幅（スケール先行推定）
  multi_scale: true        # マルチスケール検出の総スイッチ
  scale_cache: true        # 勝者スケールをフレーム間でキャッシュ

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
  warn_on_approx: true
  unmapped_action: "suggest"
```

> 上記は抜粋。実際のファイルには `card_detection` の余白・ウィンドウ
> （`edge_margin` / `name_margin` / `scale_window_low` / `coarse_to_fine` 等）、
> `ocr.gemma4`、`output` 等の追加項目がある。

## GUI

`mov-to-fan-count-gui`（customtkinter 製）。動画を選択し「動画処理」を実行すると、
meiki エンジン＋差分判定でユーザ名とファン数を抽出し、テーブルに表示する。
JSON / CSV へのエクスポートも可能。

```bash
mov-to-fan-count-gui
```

サンプリング間隔は `config/settings.yaml` で決まる:
`video.diff_only: true` → 間隔サンプリングなし（interval 0）で全フレームを
差分判定に任せる。そうでない場合は `video.frame_interval` のサンプリング＋
差分判定を行う。

## 出力例

```json
{
    "のび太": 3249444186
}
```
