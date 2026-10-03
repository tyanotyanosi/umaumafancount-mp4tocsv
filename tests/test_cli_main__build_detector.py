"""Tests for ``cli.main.build_detector``.

Specification: docs/00-Architecture/cli_main__build_detector.yaml

``build_detector`` resolves the template directory from the settings
``card_detection`` section and creates / returns a ``CardDetector``:

- ``cd = settings.get('card_detection', {}) if settings else {}``
- ``template_dir = cd.get('template_dir', 'template')``
- create and return
  ``CardDetector(template_dir=str(data_path(template_dir)),
  settings=settings)``

Mocked / stand-in dependencies (per the test-generation rules):
``cli.main.data_path`` (returns a stand-in resolved path) and
``cli.main.CardDetector`` (constructor mock) are mocked; their
implementations are outside the read scope.

``errors`` section of the spec (documented here only, NOT tested, per the
test-generation rules):

- condition: "CardDetectorのコンストラクタが例外を送出する（例: テンプレート関連の問題等。実装依存）"
  behavior: "本関数は捕捉せず、例外は呼び出し元へそのまま送出される"
"""

from unittest import mock

from cli.main import build_detector

_RESOLVED_PATH = "D:/fake/resolved/template"


def _mocks():
    """Patch data_path / CardDetector and yield
    ``(mock_dp, mock_cd)``."""
    mock_dp = mock.patch("cli.main.data_path",
                         return_value=_RESOLVED_PATH).start()
    mock_cd = mock.patch("cli.main.CardDetector").start()
    return mock_dp, mock_cd


def test_edge_01():
    """
    input: settings={}
    expected: CardDetector(template_dir=str(data_path('template')), settings={}) が返る。
    """
    mock_dp, mock_cd = _mocks()
    try:
        ret = build_detector({})
        mock_dp.assert_called_once_with("template")
        mock_cd.assert_called_once_with(
            template_dir=str(_RESOLVED_PATH), settings={})
        assert ret is mock_cd.return_value
    finally:
        mock_dp.stop()
        mock_cd.stop()


def test_edge_02():
    """
    input: settings=None
    expected: cdは{}となりtemplate_dirは'template'だが、CardDetectorにはsettings=Noneがそのまま渡される。
    """
    mock_dp, mock_cd = _mocks()
    try:
        ret = build_detector(None)
        mock_dp.assert_called_once_with("template")
        mock_cd.assert_called_once_with(
            template_dir=str(_RESOLVED_PATH), settings=None)
        assert ret is mock_cd.return_value
    finally:
        mock_dp.stop()
        mock_cd.stop()


def test_edge_03():
    """
    input: settings={'card_detection': {}}
    expected: template_dirは既定の'template'で、CardDetectorが生成される。
    """
    mock_dp, mock_cd = _mocks()
    try:
        settings = {"card_detection": {}}
        ret = build_detector(settings)
        mock_dp.assert_called_once_with("template")
        mock_cd.assert_called_once_with(
            template_dir=str(_RESOLVED_PATH), settings=settings)
        assert ret is mock_cd.return_value
    finally:
        mock_dp.stop()
        mock_cd.stop()


def test_edge_04():
    """
    input: settings={'card_detection': {'template_dir': 'custom_dir'}}
    expected: CardDetector(template_dir=str(data_path('custom_dir')), settings=settings) が返る。
    """
    mock_dp, mock_cd = _mocks()
    try:
        settings = {"card_detection": {"template_dir": "custom_dir"}}
        ret = build_detector(settings)
        mock_dp.assert_called_once_with("custom_dir")
        mock_cd.assert_called_once_with(
            template_dir=str(_RESOLVED_PATH), settings=settings)
        assert ret is mock_cd.return_value
    finally:
        mock_dp.stop()
        mock_cd.stop()
