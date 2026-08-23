import customtkinter as ctk
from typing import Optional


class ResultTableView(ctk.CTkFrame):
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)

        self.data = {}
        self.entries = []

        self._setup_ui()

    def _setup_ui(self):
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        header_frame = ctk.CTkFrame(self)
        header_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=(5, 0))
        header_frame.grid_columnconfigure(0, weight=1)
        header_frame.grid_columnconfigure(1, weight=2)

        label_user = ctk.CTkLabel(header_frame, text="ユーザ名", anchor="w")
        label_user.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        label_count = ctk.CTkLabel(header_frame, text="ファン数", anchor="e")
        label_count.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        self.scroll_frame = ctk.CTkScrollableFrame(self, label_text="")
        self.scroll_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        self.scroll_frame.grid_columnconfigure(0, weight=1)
        self.scroll_frame.grid_columnconfigure(1, weight=2)

    def update_data(self, data: dict):
        for entry in self.entries:
            entry.destroy()
        self.entries.clear()
        self.data = data

        for user_name, fan_count in data.items():
            row_frame = ctk.CTkFrame(self.scroll_frame)
            row_frame.grid(row=len(self.entries), column=0, columnspan=2, sticky="ew", padx=2, pady=2)

            label_name = ctk.CTkLabel(row_frame, text=user_name, anchor="w")
            label_name.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

            label_count = ctk.CTkLabel(row_frame, text=str(fan_count), anchor="e")
            label_count.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

            self.entries.append(row_frame)

    def clear(self):
        for entry in self.entries:
            entry.destroy()
        self.entries.clear()
        self.data = {}

    def get_selected(self) -> Optional[tuple]:
        return None
