"""Tests for ``gui.main_window.MainWindow._output_csv``.

Specification: docs/00-Architecture/gui_main_window__MainWindow__output_csv.yaml

``MainWindow._output_csv`` is an instance method (per the spec's ``purpose``
field) that, when held result data exists, shows a save-destination dialog
and saves the data as a CSV file via CSVWriter:

- Step 1: if ``self.result_data`` is falsy (None, empty dict, etc.), set
  the status_label text to "出力データがありません" and return immediately
- Step 2: otherwise, show the save dialog via
  ``ctk.filedialog.asksaveasfilename`` (title=CSV保存先,
  defaultextension=.csv, フィルタ=*.csv)
- Step 3: if the dialog's return value is falsy (None or '', i.e. cancel),
  do nothing and return
- Step 4: otherwise, target = Path(戻り値), and generate CSVWriter with
  str(target.parent) as output_dir
- Step 5: call path = writer.write(self.result_data, target.name), and set
  the status_label text to "CSV出力完了: {path}"

Mock targets: ``ctk.filedialog.asksaveasfilename`` (the modal save dialog,
which requires user interaction) is replaced with ``unittest.mock``
(``ctk.filedialog`` is the same module object as ``tkinter.filedialog``).
The lazily-imported ``CSVWriter`` is the real implementation, and it writes
the actual file under ``tmp_path``.

``errors`` section of the spec (documented here only, NOT tested, per the
test-generation rules):

- condition: 'self.result_data がtruthyで、src.output.csv_writer のインポート、CSVWriterの生成、または write の呼び出し中に例外が発生（モジュール欠落、不正なディレクトリ等）'
  behavior: '関数内にtry/exceptはなく、例外は呼び出し元へそのまま伝播する。status_labelのテキストは更新されない'
"""

import time
import tkinter
from unittest.mock import patch

from gui.main_window import MainWindow


def _build_window(result_data):
    """Build a MainWindow instance per the spec's inputs/preconditions.

    ``__init__`` is bypassed (``object.__new__``): the spec requires only an
    instance with ``__init__`` completed, ``_setup_ui`` executed, and
    ``self.result_data`` and ``self.status_label`` present. Unrelated
    dependencies (window/UI creation) are not exercised. ``status_label``
    is a real tkinter Label hosted on a minimal withdrawn root window,
    which the caller destroys at test end.

    ``tkinter.Tk()`` is retried up to 5 times: the Tcl library of the
    uv-managed interpreter intermittently fails to read its own .tcl files
    (environment issue observed as TclError "couldn't read file ..."), so a
    transient load failure is retried before the test body runs.
    """
    root = None
    for attempt in range(5):
        try:
            root = tkinter.Tk()
            break
        except tkinter.TclError:
            if attempt == 4:
                raise
            time.sleep(0.25)
    root.withdraw()
    label = tkinter.Label(root, text="initial status")
    window = object.__new__(MainWindow)
    window.result_data = result_data
    window.status_label = label
    return root, window


def test_edge_01():
    """
    input: self.result_data が None（__init__ 19行目の初期値）
    expected: status_labelのテキストが「出力データがありません」に設定され、ダイアログは表示されず、関数はNoneを返して終了する
    """
    root, window = _build_window(None)
    try:
        with patch("tkinter.filedialog.asksaveasfilename") as dialog:
            result = window._output_csv()
        assert result is None
        assert window.status_label.cget("text") == "出力データがありません"
        dialog.assert_not_called()
    finally:
        root.destroy()


def test_edge_02():
    """
    input: self.result_data が {}（空dict、falsy）
    expected: Noneの場合と同一。status_labelのテキストが「出力データがありません」になり、ダイアログ表示・ファイル書き込みは発生しない
    """
    root, window = _build_window({})
    try:
        with patch("tkinter.filedialog.asksaveasfilename") as dialog:
            result = window._output_csv()
        assert result is None
        assert window.status_label.cget("text") == "出力データがありません"
        dialog.assert_not_called()
    finally:
        root.destroy()


def test_edge_03():
    """
    input: self.result_data が空でないdict、ユーザーがダイアログでキャンセルし戻り値が ''（またはNone）
    expected: ファイル書き込みもCSVWriterの生成も発生せず、status_labelのテキストは呼び出し前のまま変更されない。関数はNoneを返す
    """
    root, window = _build_window({"count": 42})
    try:
        with patch("tkinter.filedialog.asksaveasfilename", return_value=""), \
                patch("src.output.csv_writer.CSVWriter") as writer_cls:
            result = window._output_csv()
        assert result is None
        assert window.status_label.cget("text") == "initial status"
        writer_cls.assert_not_called()
    finally:
        root.destroy()


def test_edge_04(tmp_path):
    """
    input: self.result_data が空でないdict、ユーザーがダイアログで D/out/result.csv を選択
    expected: CSVWriter(output_dir=str(Path('D/out/result.csv').parent)) が生成され、.write(self.result_data, 'result.csv') が呼び出される。status_labelのテキストが「CSV出力完了: {writeの返り値}」になる
    注: 仕様書 expected 原文 'CSVWriter(output_dir=str(Path('D/out/result.csv').parent)) が生成され、.write(self.result_data, 'result.csv') が呼び出される。status_labelのテキストが「CSV出力完了: {writeの返り値}」になる'。実測挙動を assert（選択先を tmp_path にマッピングし、実ファイルが tmp_path に書き込まれ、返り値が status テキストに埋め込まれることを観測）。
    """
    result_data = {"count": 42}
    root, window = _build_window(result_data)
    chosen = str(tmp_path / "result.csv")
    try:
        with patch("tkinter.filedialog.asksaveasfilename", return_value=chosen) as dialog:
            result = window._output_csv()
        assert result is None
        dialog.assert_called_once()
        output_file = tmp_path / "result.csv"
        assert output_file.is_file()
        content = output_file.read_text(encoding="utf-8")
        assert "count" in content
        status = window.status_label.cget("text")
        assert status.startswith("CSV出力完了: ")
        assert "result.csv" in status
    finally:
        root.destroy()
