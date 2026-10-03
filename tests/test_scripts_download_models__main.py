"""scripts.download_models.main の仕様書駆動 pytest テスト。

spec: docs/00-Architecture/scripts_download_models__main.yaml
function: scripts.download_models.main

検証する行動:
  Path(out_dir) を out.mkdir(parents=True, exist_ok=True) で確保し、MODELS（(repo_id, filename) の 3 組）を
  順番に走査。各組について hf_hub_download(repo_id=..., filename=..., local_dir=str(out)) でローカル
  ファイルパス p を得て、p のサイズを MB 換算（st_size/1024/1024）し "downloaded: {p} ({size_mb:.1f} MB)"
  を出力。ループ終了後に要約行 "done: {len(MODELS)} files -> {out}/" を出力し None を返す。

モックした依存:
  - hf_hub_download を mock で置換（ネットワーク I/O を回避）。side_effect で local_dir 直下に
    実ファイルを作成しパスを返すことで、p.stat() が実ファイルサイズを返すようにする。

errors セクション（記録のみ・テストしない）:
  - out_dir のパスに通常ファイルが存在 → Path.mkdir が FileExistsError を送出。
  - out_dir が権限により作成できない → PermissionError。
  - ネットワーク障害・リポジトリ/ファイル不在 → hf_hub_download が例外を送出。
"""
from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

import scripts.download_models as m


def _hf_side_effect(repo_id, filename, local_dir):
    """hf_hub_download の置き換え。local_dir 直下に 2MB の実ファイルを作成してパスを返す。"""
    p = Path(local_dir) / filename
    p.write_bytes(b'x' * (2 * 1024 * 1024))
    return str(p)


def test_edge_01(tmp_path, capsys):
    """input: out_dir が既存ディレクトリ（例: "models" が既に存在）
    expected: エラーは発生せず（exist_ok=True）、そのディレクトリにファイルがダウンロードされ、要約行が出力される
    """
    out_dir = tmp_path / "models"
    out_dir.mkdir()  # 既存ディレクトリ
    with mock.patch('scripts.download_models.hf_hub_download', side_effect=_hf_side_effect):
        m.main(out_dir=str(out_dir))
    out = capsys.readouterr().out
    assert out.count("downloaded: ") == 3
    assert f"done: 3 files -> {out_dir}/" in out


def test_edge_02(tmp_path, capsys):
    """input: out_dir が存在しない階層パス（例: "a/b/c"）
    expected: 親ディレクトリが再帰的に作成され（parents=True）、"a/b/c/" 配下にファイルが配置され、要約行が出力される
    """
    out_dir = tmp_path / "a" / "b" / "c"  # 存在しない階層パス
    with mock.patch('scripts.download_models.hf_hub_download', side_effect=_hf_side_effect):
        m.main(out_dir=str(out_dir))
    out = capsys.readouterr().out
    assert out.count("downloaded: ") == 3
    assert f"done: 3 files -> {out_dir}/" in out
    assert out_dir.is_dir()
    for repo_id, filename in m.MODELS:
        assert (out_dir / filename).is_file()


def test_edge_03(tmp_path):
    """input: out_dir がパスとして存在する通常ファイル（ディレクトリではない）
    expected: out.mkdir がFileExistsErrorを送出する（exist_okは既存ディレクトリに対してのみ有効）
    """
    file_path = tmp_path / "afile"
    file_path.write_text("data")  # 通常ファイル
    with mock.patch('scripts.download_models.hf_hub_download') as mock_hf:
        with pytest.raises(FileExistsError):
            m.main(out_dir=str(file_path))
    mock_hf.assert_not_called()
