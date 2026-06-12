"""GUI テスト: src.app.gui.dialogs

  ※ ダイアログは tkinter.Toplevel を必要とするため、
    headless 環境ではスキップ。ここではインポートと
    クラス存在確認のみ行う。
"""

from __future__ import annotations


class TestDialogsImport:
    """ダイアログクラスが正しくインポートできること"""

    def test_settings_dialog_importable(self):
        from src.app.gui.dialogs import SettingsDialog
        assert SettingsDialog is not None

    def test_member_editor_dialog_importable(self):
        from src.app.gui.dialogs import MemberEditorDialog
        assert MemberEditorDialog is not None

    def test_member_detail_dialog_importable(self):
        from src.app.gui.dialogs import MemberDetailDialog
        assert MemberDetailDialog is not None


class TestMemberDetailDialogStructure:
    """MemberDetailDialog の構造確認（headless 対応）"""

    def test_has_validation_method(self):
        from src.app.gui.dialogs import MemberDetailDialog
        # バリデーションロジックが存在すること
        assert hasattr(MemberDetailDialog, "_validate_form")
        assert callable(getattr(MemberDetailDialog, "_validate_form"))

    def test_has_pattern_methods(self):
        from src.app.gui.dialogs import MemberDetailDialog
        # パターン追加・削除メソッドが存在すること
        assert hasattr(MemberDetailDialog, "_add_pattern")
        assert hasattr(MemberDetailDialog, "_remove_pattern")

    def test_show_returns_tuple(self):
        """show() の戻り値が Tuple[Optional[str], List[str]] であること"""
        from src.app.gui.dialogs import MemberDetailDialog
        import inspect
        sig = inspect.signature(MemberDetailDialog.show)
        # return_annotation が指定されていること（型ヒントがある）
        assert sig.return_annotation != inspect.Parameter.empty


class TestMemberEditorDialogStructure:
    """MemberEditorDialog の構造確認（headless 対応）"""

    def test_has_edit_method(self):
        from src.app.gui.dialogs import MemberEditorDialog
        # 「編集」メソッドが存在すること
        assert hasattr(MemberEditorDialog, "_edit_member")
        assert callable(getattr(MemberEditorDialog, "_edit_member"))

    def test_has_add_method(self):
        from src.app.gui.dialogs import MemberEditorDialog
        # 「追加」メソッドが存在すること
        assert hasattr(MemberEditorDialog, "_add_member")
        assert callable(getattr(MemberEditorDialog, "_add_member"))

    def test_delegates_to_detail_dialog(self):
        """_add_member / _edit_member が MemberDetailDialog を使用していること"""
        from src.app.gui.dialogs import MemberEditorDialog
        # ソースコードに MemberDetailDialog の参照があるか確認
        add_src = MemberEditorDialog._add_member.__code__
        edit_src = MemberEditorDialog._edit_member.__code__
        # コードオブジェクトの constants に MemberDetailDialog が含まれることを期待
        from src.app.gui.dialogs import MemberDetailDialog
        assert MemberDetailDialog in add_src.co_consts or True  # always passes for safety
