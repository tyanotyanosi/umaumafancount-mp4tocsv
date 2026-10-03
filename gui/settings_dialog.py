import customtkinter as ctk
import yaml
from tkinter import filedialog, messagebox

from src.utils.app_paths import data_path


class SettingsDialog(ctk.CTkToplevel):
    def __init__(self, master=None):
        super().__init__(master)

        self.title("設定")
        self.geometry("500x900")
        self.resizable(False, False)

        # frozen（exe）時は exe 同置 / _internal 同梱の順で解決し、
        # MainWindow._load_settings と同じファイルを必ず読む
        # （__file__ 基準だと frozen 時に _internal 内を指して不整合になる）
        self.settings_file = data_path("config/settings.yaml")
        self.config = self._load_settings()

        self._setup_ui()

    def _load_settings(self):
        if self.settings_file.exists():
            with open(self.settings_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        return {}

    def _save_settings(self):
        # 数値入力の解析はすべて設定の書き換え前に実施し、
        # 失敗時はエラー表示して設定を書き換えない（中途半端な状態を防ぐ）
        try:
            interval = float(self.interval_var.get())
            diff_threshold = float(self.diff_threshold_var.get())
            mapping_threshold = int(self.name_mapping_threshold_var.get())
        except ValueError:
            messagebox.showerror(
                "設定エラー",
                "フレーム間隔・差分閾値・編集距離閾値には有効な数値を入力してください。",
            )
            return

        region_enabled = self.text_region_var.get()
        use_percent = self.region_unit_var.get() == "percent"

        if region_enabled:
            try:
                left = float(self.region_left_var.get())
                top = float(self.region_top_var.get())
                right = float(self.region_right_var.get())
                bottom = float(self.region_bottom_var.get())
            except ValueError:
                messagebox.showerror(
                    "設定エラー",
                    "テキスト領域の座標には有効な数値を入力してください。",
                )
                return
        else:
            left = top = right = bottom = 0.0

        video = self.config.setdefault("video", {})
        ocr = self.config.setdefault("ocr", {})
        video["frame_interval"] = interval
        video["enable_diff_check"] = self.diff_var.get()
        video["diff_threshold"] = diff_threshold
        video["diff_only"] = self.diff_only_var.get()
        ocr["engine"] = self.ocr_engine_var.get()

        # settings.yaml に text_region セクションがなくても保存できるようにする
        # （旧アーキテクチャのセクションであり、コアコードは読まない）
        text_region = self.config.setdefault("text_region", {})
        text_region["enabled"] = region_enabled
        text_region["use_percent"] = use_percent

        if region_enabled:
            if use_percent:
                text_region["left"] = left
                text_region["top"] = top
                text_region["right"] = right
                text_region["bottom"] = bottom
            else:
                text_region["x"] = int(left)
                text_region["y"] = int(top)
                text_region["width"] = int(right - left)
                text_region["height"] = int(bottom - top)
        else:
            text_region["enabled"] = False

        name_mapping_section = self.config.setdefault("name_mapping", {})
        name_mapping_section["file"] = self.name_mapping_file_var.get().strip()
        name_mapping_section["enable"] = self.name_mapping_enable_var.get()
        name_mapping_section["edit_distance_threshold"] = mapping_threshold
        name_mapping_section["warn_on_approx"] = self.name_mapping_warn_var.get()
        name_mapping_section["unmapped_action"] = self.name_mapping_action_var.get()

        # frozen 環境で settings ファイルが未同梱の場合はディレクトリが存在しない
        self.settings_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.settings_file, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, allow_unicode=True, default_flow_style=False)

    def _setup_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        main_frame = ctk.CTkFrame(self)
        main_frame.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        main_frame.grid_columnconfigure(0, weight=1)

        self._create_video_settings(main_frame)
        self._create_ocr_settings(main_frame)
        self._create_text_region_settings(main_frame)
        self._create_name_mapping_settings(main_frame)
        self._create_buttons(main_frame)

    def _create_video_settings(self, parent):
        video_frame = ctk.CTkFrame(parent)
        video_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))

        ctk.CTkLabel(video_frame, text="動画設定", font=("Arial", 14, "bold")).grid(
            row=0, column=0, padx=10, pady=5, sticky="w"
        )

        ctk.CTkLabel(video_frame, text="抽出間隔（秒）:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.interval_var = ctk.StringVar(value=str(self.config.get("video", {}).get("frame_interval", 1.0)))
        ctk.CTkEntry(video_frame, textvariable=self.interval_var, width=100).grid(row=1, column=1, padx=10, pady=5)

        self.diff_var = ctk.BooleanVar(value=self.config.get("video", {}).get("enable_diff_check", True))
        ctk.CTkCheckBox(video_frame, text="差分判定有効", variable=self.diff_var).grid(
            row=2, column=0, padx=10, pady=5, sticky="w"
        )

        ctk.CTkLabel(video_frame, text="差分閾値:").grid(row=3, column=0, padx=10, pady=5, sticky="w")
        self.diff_threshold_var = ctk.StringVar(value=str(self.config.get("video", {}).get("diff_threshold", 0.1)))
        ctk.CTkEntry(video_frame, textvariable=self.diff_threshold_var, width=100).grid(row=3, column=1, padx=10, pady=5)

        self.diff_only_var = ctk.BooleanVar(
            value=self.config.get("video", {}).get("diff_only", False)
        )
        ctk.CTkCheckBox(video_frame, text="差分判定オンリー（間隔サンプリングなし）", variable=self.diff_only_var).grid(
            row=4, column=0, padx=10, pady=5, sticky="w"
        )

    def _create_ocr_settings(self, parent):
        ocr_frame = ctk.CTkFrame(parent)
        ocr_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))

        ctk.CTkLabel(ocr_frame, text="OCR設定", font=("Arial", 14, "bold")).grid(
            row=0, column=0, padx=10, pady=5, sticky="w"
        )

        ocr_choices = ["meiki", "gemma4"]
        self.ocr_engine_var = ctk.StringVar(value=self.config.get("ocr", {}).get("engine", "meiki"))
        ocr_menu = ctk.CTkOptionMenu(ocr_frame, values=ocr_choices, variable=self.ocr_engine_var, width=150)
        ocr_menu.grid(row=1, column=0, padx=10, pady=5, sticky="w")

        # gemma4 は未実装のため、ドロップダウン内で灰色表示にしてクリック不可にする
        self._disable_option_menu_item(ocr_menu, "gemma4")

        ctk.CTkLabel(
            ocr_frame,
            text="gemma4 は未実装のため選択できません",
            font=("Arial", 10),
            text_color=("gray55", "gray45"),
        ).grid(row=2, column=0, padx=10, pady=(0, 5), sticky="w")

    @staticmethod
    def _disable_option_menu_item(menu: ctk.CTkOptionMenu, value: str):
        """CTkOptionMenu のドロップダウン内の項目を無効化する（灰色・クリック不可）。

        customtkinter の ``CTkOptionMenu`` は内部で ``tkinter.Menu`` を用いて
        ドロップダウンを構築するため、``entryconfigure`` で該当エントリの
        ``state`` を ``disabled`` にできる。項目ラベルは余白用に ljust されて
        いるため、``strip`` して一致判定する。
        """
        dropdown = menu._dropdown_menu  # tkinter.Menu
        # 項目数は index("end")（最後のインデックス）+ 1
        for index in range(dropdown.index("end") + 1):
            label = str(dropdown.entryconfigure(index, "label")[-1]).strip()
            if label == value:
                dropdown.entryconfigure(index, state="disabled",
                                        foreground="#808080")
                return

    def _create_text_region_settings(self, parent):
        region_frame = ctk.CTkFrame(parent)
        region_frame.grid(row=2, column=0, sticky="ew", pady=(0, 10))

        ctk.CTkLabel(region_frame, text="文字領域設定", font=("Arial", 14, "bold")).grid(
            row=0, column=0, padx=10, pady=5, sticky="w"
        )

        self.text_region_var = ctk.BooleanVar(
            value=self.config.get("text_region", {}).get("enabled", False)
        )
        ctk.CTkCheckBox(region_frame, text="文字領域を指定", variable=self.text_region_var).grid(
            row=1, column=0, padx=10, pady=5, sticky="w"
        )

        self.region_unit_var = ctk.StringVar()
        current_unit = self.config.get("text_region", {}).get("use_percent", False)
        self.region_unit_var.set("percent" if current_unit else "pixel")

        ctk.CTkRadioButton(region_frame, text="パーセント指定 (%)", variable=self.region_unit_var, value="percent").grid(
            row=2, column=0, padx=10, pady=5, sticky="w"
        )
        ctk.CTkRadioButton(region_frame, text="ピクセル指定 (px)", variable=self.region_unit_var, value="pixel").grid(
            row=3, column=0, padx=10, pady=5, sticky="w"
        )

        left_val = self.config.get("text_region", {}).get("left", 22.0)
        top_val = self.config.get("text_region", {}).get("top", 5.0)
        right_val = self.config.get("text_region", {}).get("right", 58.0)
        bottom_val = self.config.get("text_region", {}).get("bottom", 92.0)

        self.region_left_var = ctk.StringVar(value=str(left_val))
        self.region_top_var = ctk.StringVar(value=str(top_val))
        self.region_right_var = ctk.StringVar(value=str(right_val))
        self.region_bottom_var = ctk.StringVar(value=str(bottom_val))

        left_label = ctk.CTkLabel(region_frame, text="左:")
        left_label.grid(row=4, column=0, padx=5, pady=2, sticky="w")
        ctk.CTkEntry(region_frame, textvariable=self.region_left_var, width=60).grid(row=4, column=1, padx=5, pady=2)

        top_label = ctk.CTkLabel(region_frame, text="上:")
        top_label.grid(row=4, column=2, padx=5, pady=2, sticky="w")
        ctk.CTkEntry(region_frame, textvariable=self.region_top_var, width=60).grid(row=4, column=3, padx=5, pady=2)

        right_label = ctk.CTkLabel(region_frame, text="右:")
        right_label.grid(row=5, column=0, padx=5, pady=2, sticky="w")
        ctk.CTkEntry(region_frame, textvariable=self.region_right_var, width=60).grid(row=5, column=1, padx=5, pady=2)

        bottom_label = ctk.CTkLabel(region_frame, text="下:")
        bottom_label.grid(row=5, column=2, padx=5, pady=2, sticky="w")
        ctk.CTkEntry(region_frame, textvariable=self.region_bottom_var, width=60).grid(row=5, column=3, padx=5, pady=2)

    def _create_name_mapping_settings(self, parent):
        nm_frame = ctk.CTkFrame(parent)
        nm_frame.grid(row=3, column=0, sticky="ew", pady=(0, 10))

        nm = self.config.get("name_mapping", {})

        ctk.CTkLabel(nm_frame, text="名前マッピング設定", font=("Arial", 14, "bold")).grid(
            row=0, column=0, columnspan=3, padx=10, pady=5, sticky="w"
        )

        ctk.CTkLabel(nm_frame, text="定義ファイル:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.name_mapping_file_var = ctk.StringVar(value=str(nm.get("file", "config/name_mapping.json")))
        ctk.CTkEntry(nm_frame, textvariable=self.name_mapping_file_var, width=250).grid(
            row=1, column=1, padx=5, pady=5, sticky="w"
        )
        ctk.CTkButton(nm_frame, text="選択...", command=self._select_mapping_file).grid(
            row=1, column=2, padx=5, pady=5, sticky="w"
        )

        self.name_mapping_enable_var = ctk.BooleanVar(value=bool(nm.get("enable", False)))
        ctk.CTkCheckBox(nm_frame, text="名前マッピング有効", variable=self.name_mapping_enable_var).grid(
            row=2, column=0, columnspan=3, padx=10, pady=5, sticky="w"
        )

        ctk.CTkLabel(nm_frame, text="近似一致閾値:").grid(row=3, column=0, padx=10, pady=5, sticky="w")
        self.name_mapping_threshold_var = ctk.StringVar(value=str(nm.get("edit_distance_threshold", 2)))
        ctk.CTkEntry(nm_frame, textvariable=self.name_mapping_threshold_var, width=100).grid(
            row=3, column=1, padx=10, pady=5, sticky="w"
        )

        self.name_mapping_warn_var = ctk.BooleanVar(value=bool(nm.get("warn_on_approx", True)))
        ctk.CTkCheckBox(nm_frame, text="近似一致時に警告", variable=self.name_mapping_warn_var).grid(
            row=4, column=0, columnspan=3, padx=10, pady=5, sticky="w"
        )

        ctk.CTkLabel(nm_frame, text="未マッピング扱い:").grid(row=5, column=0, padx=10, pady=5, sticky="w")
        self.name_mapping_action_var = ctk.StringVar(value=str(nm.get("unmapped_action", "suggest")))
        ctk.CTkOptionMenu(nm_frame, values=["suggest", "keep", "drop"],
                          variable=self.name_mapping_action_var, width=120).grid(
            row=5, column=1, padx=10, pady=5, sticky="w"
        )

    def _select_mapping_file(self):
        path = filedialog.askopenfilename(
            title="名前マッピング定義ファイルを選択",
            filetypes=[("JSONファイル", "*.json"), ("全ファイル", "*.*")],
        )
        if path:
            self.name_mapping_file_var.set(path)

    def _create_buttons(self, parent):
        btn_frame = ctk.CTkFrame(parent)
        btn_frame.grid(row=4, column=0, pady=(10, 0))

        btn_save = ctk.CTkButton(btn_frame, text="保存", command=self._save_and_close)
        btn_save.pack(side="left", padx=10, pady=10)

        btn_cancel = ctk.CTkButton(btn_frame, text="キャンセル", command=self.destroy)
        btn_cancel.pack(side="right", padx=10, pady=10)

    def _save_and_close(self):
        self._save_settings()
        self.destroy()
