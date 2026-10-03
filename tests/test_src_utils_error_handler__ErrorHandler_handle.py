"""src.utils.error_handler.ErrorHandler.handle の仕様書駆動 pytest テスト。

spec: docs/00-Architecture/src_utils_error_handler__ErrorHandler_handle.yaml
function: src.utils.error_handler.ErrorHandler.handle

検証する行動:
  level == USER_ERROR なら self.logger.warning(f'[USER] {message}')。
  level == SYSTEM_ERROR なら self.logger.error(f'[SYSTEM] {message}', exc_info=exc)。
  level == LOG_ERROR なら self.logger.debug(f'[LOG] {message}')。
  その他の値は何もしない。常に None を返す。

モックした依存:
  - self.logger を mock.Mock() で置換し、warning/error/debug の呼び出しを検証する。
  - インスタンスは object.__new__(ErrorHandler) で __init__ を回避して作成する
    （共有ロガーのファイル I/O を回避するため）。

errors セクション（記録のみ・テストしない）:
  - level が 3 つの列挙値以外（例: str）→ 例外は発生せず、すべてのログ処理がスキップされる。
"""
from __future__ import annotations

from unittest import mock

import src.utils.error_handler as m


def _make_handler():
    """__init__ を回避した ErrorHandler を作成し、logger を Mock で設定する。"""
    w = object.__new__(m.ErrorHandler)
    w.logger = mock.Mock()
    return w


def test_edge_01():
    """input: (ErrorLevel.USER_ERROR, 'msg1')
    expected: WARNING レベルの記録 '[USER] msg1' が1つ送出され、None を返す。exc 引数は無視される。
    """
    w = _make_handler()
    result = w.handle(m.ErrorLevel.USER_ERROR, 'msg1')
    assert result is None
    w.logger.warning.assert_called_once_with('[USER] msg1')
    w.logger.error.assert_not_called()
    w.logger.debug.assert_not_called()


def test_edge_02():
    """input: (ErrorLevel.SYSTEM_ERROR, 'msg2', exc=ValueError('boom'))
    expected: exc_info=ValueError('boom') を伴う ERROR レベルの記録 '[SYSTEM] msg2' が送出される（トレースバックはハンドラのフォーマッタによって整形される）。
    """
    w = _make_handler()
    exc = ValueError('boom')
    result = w.handle(m.ErrorLevel.SYSTEM_ERROR, 'msg2', exc)
    assert result is None
    w.logger.error.assert_called_once_with('[SYSTEM] msg2', exc_info=exc)
    w.logger.warning.assert_not_called()
    w.logger.debug.assert_not_called()


def test_edge_03():
    """input: (ErrorLevel.SYSTEM_ERROR, 'msg2')
    expected: exc_info=None を伴う ERROR レベルの記録 '[SYSTEM] msg2' が送出される（トレースバックなし）。
    """
    w = _make_handler()
    result = w.handle(m.ErrorLevel.SYSTEM_ERROR, 'msg2')
    assert result is None
    w.logger.error.assert_called_once_with('[SYSTEM] msg2', exc_info=None)
    w.logger.warning.assert_not_called()
    w.logger.debug.assert_not_called()


def test_edge_04():
    """input: (ErrorLevel.LOG_ERROR, 'msg3')
    expected: DEBUG レベルの記録 '[LOG] msg3' が1つ送出され、None を返す。
    """
    w = _make_handler()
    result = w.handle(m.ErrorLevel.LOG_ERROR, 'msg3')
    assert result is None
    w.logger.debug.assert_called_once_with('[LOG] msg3')
    w.logger.warning.assert_not_called()
    w.logger.error.assert_not_called()


def test_edge_05():
    """input: (level='unknown', 'msg4')
    expected: ログ出力も例外も発生せず、None を返す。
    """
    w = _make_handler()
    result = w.handle('unknown', 'msg4')
    assert result is None
    w.logger.warning.assert_not_called()
    w.logger.error.assert_not_called()
    w.logger.debug.assert_not_called()
