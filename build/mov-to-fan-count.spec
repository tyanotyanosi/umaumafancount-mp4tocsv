# -*- coding: utf-8 -*-
"""PyInstaller spec: GUI / CLI の 2 EXE を 1 つの onedir にまとめる。

ビルド（プロジェクトルートから）:
    uv run python scripts/download_models.py models   # 前提: models/ に ONNX 3 件
    uv run pyinstaller build/mov-to-fan-count.spec --noconfirm

注意:
- パスは ``SPECPATH``（このファイルのディレクトリ）基準で解決する
  （PyInstaller 6 は spec 内の相対パスを spec 側で解決するため、
  プロジェクトルート基準の相対パスだと壊れる）。
- ``a.scripts`` は**プログラムのリストではなく TOC**（rthook エントリ +
  プログラムスクリプトエントリ）である。C ブートローダは CArchive の TOC に
  並ぶ**全 PYSOURCE を TOC 順に ``__main__`` として実行**するため（= 最後
  の PYSOURCE が実効的なメインスクリプト）、各 EXE には「rthook 全部 +
  該当プログラムスクリプト1本（末尾）」の TOC を渡す（_script_toc 参照）。
  めぐり: PyInstaller 6.22 ``bootloader/src/pyi_launch.c``
  ``pyi_launch_run_scripts``（2026-07 時点の master で確認）。
- EXE の割り当てが逆になると GUI/CLI が入れ替わるので、
  ビルド後必ず「--help が CLI 側 / GUI 側はウィンドウ起動」で確認すること。
- gemma4（torch/transformers）は**意図的に除外**（exe には含めない）。

設計: ``docs/04-Design/exe-build-pipeline-design.md`` §5.7
"""
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

SPEC_DIR = Path(SPECPATH)          # build/（この spec のディレクトリ）
ROOT = SPEC_DIR.parent             # プロジェクトルート

app_name = "mov-to-fan-count"

# 同梱モデルが無いと exe がオフライン動作しなくなるため、ビルド前に強制チェック
for _required in ("config", "template", "models"):
    if not (ROOT / _required).is_dir():
        sys.exit(
            f"エラー: '{_required}/' がプロジェクトルートに見つかりません。\n"
            "  - config/ / template/ はリポジトリに同梱されています\n"
            "  - models/ は `uv run python scripts/download_models.py models` で取得してください"
        )

a = Analysis(
    [str(SPEC_DIR / "entry_gui.py"), str(SPEC_DIR / "entry_cli.py")],
    pathex=[str(ROOT)],
    datas=[
        (str(ROOT / "config" / "settings.yaml"), "config"),  # 同梱デフォルト設定（name_mapping.json はユーザー別なので非同梱）
        (str(ROOT / "template"), "template"),  # テンプレ画像 4 件
        (str(ROOT / "models"), "models"),      # meiki ONNX 3 件（約 46 MB）
    ] + collect_data_files("customtkinter"),  # テーマ JSON 等のデータファイル
    hiddenimports=[
        "meikiocr",
        "onnxruntime",
        "yaml",
        *collect_submodules("customtkinter"),
    ],
)
pyz = PYZ(a.pure)

# a.scripts = [rthook エントリ..., entry_gui, entry_cli]（TOC）
# エントリ形式は (dest_name, src_path, type) で、rthook の dest_name は
# 'pyi_*'（例: 'pyi_rth__tkinter'）、プログラムスクリプトの dest_name は
# 拡張子無しのファイル名（例: 'entry_cli'）である（6.22 で実測）。
def _script_toc(script_stem):
    """指定のプログラムスクリプトを「最後尾」にした TOC（rthook + 1 スクリプト）。"""
    rthooks = [e for e in a.scripts if e[0].startswith("pyi_")]
    progs = [e for e in a.scripts if e[0] == script_stem]
    if len(progs) != 1:
        sys.exit(f"エラー: a.scripts に {script_stem} が恰好1件見つかりません: {a.scripts}")
    return rthooks + progs

# onedir: EXE は WORKPATH に作り、COLLECT が dist フォルダへコピーする。
# exclude_binaries 無し（デフォルト False）だと dist/ 直下にも exe が
# 直接出力され、_internal/ の無い「動かない exe」が混入する（実機で確認済み）。
exe_gui = EXE(
    pyz, _script_toc("entry_gui"),
    name=f"{app_name}-gui",
    debug=False,
    strip=False,
    upx=False,
    exclude_binaries=True,
    console=False,          # GUI: コンソール非表示
    # icon="build/icon.ico",  # 任意: アイコンを用意したら指定
)
exe_cli = EXE(
    pyz, _script_toc("entry_cli"),
    name=app_name,
    debug=False,
    strip=False,
    upx=False,
    exclude_binaries=True,
    console=True,           # CLI: コンソール表示
)
coll = COLLECT(
    exe_gui, exe_cli, a.binaries, a.datas,
    name=app_name,
)