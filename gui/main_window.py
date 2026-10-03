import tkinter as tk
import customtkinter as ctk
from pathlib import Path

from gui.video_player import VideoPreviewFrame
from gui.result_view import ResultTableView
from gui.settings_dialog import SettingsDialog
from src.utils.app_paths import data_path, work_dir


class MainWindow(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("mov-to-fan-count")
        self.geometry("1200x800")

        self.video_path = None
        self.result_data = None

        self.settings = self._load_settings()

        self._setup_ui()

    def _load_settings(self):
        # frozen（exe）時は exe 同置 / _internal 同梱の順で解決
        settings_file = data_path("config/settings.yaml")
        if settings_file.exists():
            import yaml
            with open(settings_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        return {}

    def _setup_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        self._create_toolbar()
        self._create_main_content()
        self._create_status_bar()

        # ウィンドウクローズ時は動画キャプチャを解放（再生中のクローズでも安全に）
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_close(self):
        self.video_preview.close()
        self.destroy()

    def _create_toolbar(self):
        toolbar = ctk.CTkFrame(self, height=50)
        toolbar.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        toolbar.grid_columnconfigure((0, 1, 2, 3), weight=1)

        self.btn_settings = ctk.CTkButton(
            toolbar, text="設定", command=self._open_settings
        )
        self.btn_settings.grid(row=0, column=0, padx=5, pady=10)

        self.btn_process = ctk.CTkButton(
            toolbar, text="動画処理", command=self._process_video
        )
        self.btn_process.grid(row=0, column=1, padx=5, pady=10)

        self.btn_output_json = ctk.CTkButton(
            toolbar, text="JSON出力", command=self._output_json
        )
        self.btn_output_json.grid(row=0, column=2, padx=5, pady=10)

        self.btn_output_csv = ctk.CTkButton(
            toolbar, text="CSV出力", command=self._output_csv
        )
        self.btn_output_csv.grid(row=0, column=3, padx=5, pady=10)

    def _create_main_content(self):
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(1, weight=1)

        self.video_preview = VideoPreviewFrame(self.main_frame)
        self.video_preview.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.video_preview.on_process = self._on_video_process

        self.result_table = ResultTableView(self.main_frame)
        self.result_table.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

    def _create_status_bar(self):
        self.status_bar = ctk.CTkFrame(self, height=30)
        self.status_bar.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 10))

        self.status_label = ctk.CTkLabel(
            self.status_bar, text="準備完了", anchor="w"
        )
        self.status_label.pack(side="left", padx=10, pady=5)

    def _open_settings(self):
        dialog = SettingsDialog(self)
        dialog.wait_window()
        self.settings = self._load_settings()

    def _process_video(self):
        video_path = ctk.filedialog.askopenfilename(
            title="動画を選択",
            filetypes=[("MP4ファイル", "*.mp4"), ("すべてのファイル", "*.*")]
        )

        if not video_path:
            return

        self.video_path = video_path
        self.status_label.configure(text=f"動画選択: {video_path}")
        self.video_preview.load_video(video_path)

    def _on_video_process(self, video_path: str):
        self.status_label.configure(text="処理中...")
        self.video_preview.btn_process.configure(state="disabled")

        # 差分判定オンリー（設定）→ interval 0（全フレームを差分判定に任せる）
        # それ以外 → 設定の frame_interval（既定 1.0）のサンプリング＋差分判定
        diff_only = bool(self.settings.get("video", {}).get("diff_only", False))
        if diff_only:
            interval = 0.0
        else:
            interval = float(self.settings.get("video", {}).get("frame_interval", 1.0))

        # OCR エンジンは設定（設定ダイアログで保存される ocr.engine）に従う
        ocr_engine = str(self.settings.get("ocr", {}).get("engine", "meiki"))

        import threading

        def _safe_after(func):
            """ワーカースレッドから after を安全に実行（ウィンドウクローズ後は無視）。"""
            try:
                self.after(0, func)
            except tk.TclError:
                pass  # ウィンドウは既に破棄済み → UI 更新を諦める

        def _still_exists():
            try:
                return self.winfo_exists()
            except tk.TclError:
                return False

        def process_thread():
            try:
                from cli.main import process_video
                result = process_video(
                    video_path=video_path,
                    ocr_engine=ocr_engine,
                    interval=interval,
                    use_diff=True,
                    # frozen（exe）時は exe 同置の output/ に出力
                    output_dir=str(work_dir() / "output"),
                    settings=self.settings,
                )

                def _apply():
                    if _still_exists():
                        self.set_result_data(result)
                _safe_after(_apply)
            except Exception as e:
                def _err():
                    if _still_exists():
                        self.status_label.configure(text=f"エラー: {e}")
                _safe_after(_err)
            finally:
                def _reenable():
                    if _still_exists():
                        self.video_preview.btn_process.configure(state="normal")
                _safe_after(_reenable)

        threading.Thread(target=process_thread, daemon=True).start()

    def _output_json(self):
        if not self.result_data:
            self.status_label.configure(text="出力データがありません")
            return

        output_dir = ctk.filedialog.asksaveasfilename(
            title="JSON保存先",
            defaultextension=".json",
            filetypes=[("JSONファイル", "*.json")]
        )

        if output_dir:
            from src.output.json_writer import JSONWriter
            target = Path(output_dir)
            # ユーザが選択した保存先ディレクトリを尊重する
            # （writer の既定ディレクトリにファイル名だけを書くと選択先が失われていた）
            writer = JSONWriter(output_dir=str(target.parent))
            path = writer.write(self.result_data, target.name)
            self.status_label.configure(text=f"JSON出力完了: {path}")

    def _output_csv(self):
        if not self.result_data:
            self.status_label.configure(text="出力データがありません")
            return

        output_dir = ctk.filedialog.asksaveasfilename(
            title="CSV保存先",
            defaultextension=".csv",
            filetypes=[("CSVファイル", "*.csv")]
        )

        if output_dir:
            from src.output.csv_writer import CSVWriter
            target = Path(output_dir)
            # ユーザが選択した保存先ディレクトリを尊重する
            # （writer の既定ディレクトリにファイル名だけを書くと選択先が失われていた）
            writer = CSVWriter(output_dir=str(target.parent))
            path = writer.write(self.result_data, target.name)
            self.status_label.configure(text=f"CSV出力完了: {path}")

    def set_result_data(self, data: dict):
        self.result_data = data
        self.result_table.update_data(data)
        self.status_label.configure(text=f"処理完了: {len(data)}件のユーザ情報を抽出")

    def update_status(self, message: str):
        self.status_label.configure(text=message)
