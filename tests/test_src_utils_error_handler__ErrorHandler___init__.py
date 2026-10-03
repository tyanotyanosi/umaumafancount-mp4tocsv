"""src.utils.error_handler.ErrorHandler.__init__ の仕様書駆動 pytest テスト。

spec: docs/00-Architecture/src_utils_error_handler__ErrorHandler___init__.yaml
function: src.utils.error_handler.ErrorHandler.__init__

検証する行動:
  logging.getLogger('mov-to-fan-count') で共有ロガーを取得しレベルを DEBUG に設定。
  self.logger.handlers が空でなければハンドラ追加をスキップ。
  空の場合: FileHandler(log_file, encoding='utf-8')（レベル DEBUG）と StreamHandler（レベル INFO）を
  作成し、共通フォーマッタ '%(asctime)s - %(levelname)s - %(message)s' を両方に設定して、
  ファイルハンドラ→コンソールハンドラの順に追加する。

モックした依存:
  - logging.FileHandler を mock で置換し、ファイル I/O を回避しながら
    生成引数・ハンドラ追加を検証する。
  - 共有ロガー 'mov-to-fan-count' の handlers/level を fixture で保存・復元する。

errors セクション（記録のみ・テストしない）:
  - ハンドラ未登録で log_file が開けない → logging.FileHandler のファイル系エラーが伝播し __init__ が中断する。
"""
from __future__ import annotations

import logging
from unittest import mock

import pytest

import src.utils.error_handler as m

LOGGER_NAME = 'mov-to-fan-count'


@pytest.fixture
def shared_logger():
    """共有ロガーの handlers/level を保存し、テスト中は空にリセットする。"""
    logger = logging.getLogger(LOGGER_NAME)
    saved_handlers = list(logger.handlers)
    saved_level = logger.level
    logger.handlers = []
    yield logger
    logger.handlers = saved_handlers
    logger.setLevel(saved_level)


def test_edge_01(shared_logger):
    """input: 初回構築（共有ロガーにハンドラなし）、log_file はデフォルト 'error.log'
    expected: 共有ロガーのレベルが DEBUG になり、FileHandler('error.log', encoding='utf-8') と StreamHandler がこの順に追加され、合計2つのハンドラを持つ。
    """
    with mock.patch('src.utils.error_handler.logging.FileHandler') as mock_fh:
        m.ErrorHandler()
    assert shared_logger.level == logging.DEBUG
    assert len(shared_logger.handlers) == 2
    mock_fh.assert_called_once_with('error.log', encoding='utf-8')
    # 1 番目は（モックされた）FileHandler、2 番目は実 StreamHandler
    assert isinstance(shared_logger.handlers[1], logging.StreamHandler)
    assert shared_logger.handlers[1].level == logging.INFO


def test_edge_02(shared_logger):
    """input: 2回目以降の構築（共有ロガーにハンドラ既登録）、log_file='other.log'
    expected: ハンドラは追加されず 'other.log' は開かれない。ロガーのレベルだけが DEBUG に設定される。
    """
    dummy = logging.StreamHandler()
    shared_logger.addHandler(dummy)
    with mock.patch('src.utils.error_handler.logging.FileHandler') as mock_fh:
        m.ErrorHandler(log_file='other.log')
    assert len(shared_logger.handlers) == 1
    assert shared_logger.handlers[0] is dummy
    mock_fh.assert_not_called()
    assert shared_logger.level == logging.DEBUG


def test_edge_03(shared_logger):
    """input: 初回構築、log_file='logs/sub/error.log'（logs/ ディレクトリが存在しない）
    expected: FileHandler の生成時にファイル系エラー（例: FileNotFoundError / PermissionError）が発生し、構築が失敗する。
    """
    with mock.patch('src.utils.error_handler.logging.FileHandler', side_effect=FileNotFoundError):
        with pytest.raises(FileNotFoundError):
            m.ErrorHandler(log_file='logs/sub/error.log')
    assert len(shared_logger.handlers) == 0
