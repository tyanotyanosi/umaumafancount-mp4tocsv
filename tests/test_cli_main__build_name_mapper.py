"""Tests for ``cli.main.build_name_mapper``.

Specification: docs/00-Architecture/cli_main__build_name_mapper.yaml

``build_name_mapper`` builds a ``NameMapper`` from the settings
``name_mapping`` section and returns it; when the mapping is disabled
it returns ``None`` (maintaining the current behavior of counting while
detected):

- ``nm = (settings or {}).get('name_mapping', {})``
- if ``no_name_mapping`` is True, or ``nm.get('enable', False)`` is
  falsy, return ``None``
- ``path = name_mapping_file`` if truthy, otherwise the ``'file'`` value
  of nm (default ``'config/name_mapping.json'``)
- ``path = str(data_path(path))``
- ``mapping = NameMapperLoader.load_mapping(path)``
- return ``NameMapper(mapping=mapping,
  edit_distance_threshold=int(nm.get('edit_distance_threshold', 2)),
  unmapped_action=nm.get('unmapped_action', 'suggest'),
  warn_on_approx=bool(nm.get('warn_on_approx', True)))``

Mocked / stand-in dependencies (per the test-generation rules):
``cli.main.data_path`` (returns a stand-in resolved path),
``cli.main.NameMapperLoader`` (``load_mapping`` mock) and
``cli.main.NameMapper`` (constructor mock) are all mocked; their
implementations are outside the read scope.

``errors`` section of the spec (documented here only, NOT tested, per the
test-generation rules):

- condition: "NameMapperLoader.load_mappingが例外を送出する（例: マッピングファイル不存在・パースエラー等。実装依存）"
  behavior: "本関数は捕捉せず、例外は呼び出し元へそのまま送出される"
- condition: "nm['edit_distance_threshold']がintに変換できない値（例: 数値でない文字列）"
  behavior: "int()でValueErrorが送出される"
"""

from unittest import mock

from cli.main import build_name_mapper

_RESOLVED_PATH = "D:/fake/resolved/name_mapping.json"


def _mocks():
    """Patch data_path / NameMapperLoader / NameMapper / ensure_name_mapping_file
    and yield ``(mock_dp, mock_loader, mock_mapper, mock_ensure)``."""
    mock_dp = mock.patch("cli.main.data_path",
                         return_value=_RESOLVED_PATH)
    mock_loader = mock.patch("cli.main.NameMapperLoader")
    mock_mapper = mock.patch("cli.main.NameMapper")
    mock_ensure = mock.patch("cli.main.ensure_name_mapping_file")
    started = [p.start() for p in (mock_dp, mock_loader, mock_mapper, mock_ensure)]
    return started


def _stop(*patches):
    for p in patches:
        p.stop()


def test_edge_01():
    """
    input: settings={}, name_mapping_file=None, no_name_mapping=False
    expected: name_mapping.enableが既定FalseのためNoneを返す。マッピングファイルは読み込まれない。
    """
    patches = _mocks()
    try:
        mock_dp, mock_loader, mock_mapper, mock_ensure = patches
        ret = build_name_mapper({}, None, False)
        assert ret is None
        mock_loader.load_mapping.assert_not_called()
        mock_dp.assert_not_called()
        mock_mapper.assert_not_called()
    finally:
        _stop(*patches)


def test_edge_02():
    """
    input: settings=None
    expected: settings={}の場合と同様にNoneを返す。
    """
    patches = _mocks()
    try:
        mock_dp, mock_loader, mock_mapper, mock_ensure = patches
        ret = build_name_mapper(None)
        assert ret is None
        mock_loader.load_mapping.assert_not_called()
        mock_dp.assert_not_called()
        mock_mapper.assert_not_called()
    finally:
        _stop(*patches)


def test_edge_03():
    """
    input: no_name_mapping=True, settings={'name_mapping': {'enable': True}}
    expected: Noneを返す。マッピングファイルは読み込まれない。
    """
    patches = _mocks()
    try:
        mock_dp, mock_loader, mock_mapper, mock_ensure = patches
        ret = build_name_mapper({"name_mapping": {"enable": True}},
                                no_name_mapping=True)
        assert ret is None
        mock_loader.load_mapping.assert_not_called()
        mock_dp.assert_not_called()
        mock_mapper.assert_not_called()
    finally:
        _stop(*patches)


def test_edge_04():
    """
    input: settings={'name_mapping': {'enable': True}}
    expected: str(data_path('config/name_mapping.json'))をNameMapperLoader.load_mappingで読み、edit_distance_threshold=2, unmapped_action='suggest', warn_on_approx=TrueのNameMapperを返す。
    """
    patches = _mocks()
    try:
        mock_dp, mock_loader, mock_mapper, mock_ensure = patches
        mapping_value = {"a": "b"}
        mock_loader.load_mapping.return_value = mapping_value
        ret = build_name_mapper({"name_mapping": {"enable": True}})
        mock_dp.assert_called_once_with("config/name_mapping.json")
        mock_loader.load_mapping.assert_called_once_with(_RESOLVED_PATH)
        mock_mapper.assert_called_once_with(
            mapping=mapping_value,
            edit_distance_threshold=2,
            unmapped_action="suggest",
            warn_on_approx=True)
        assert ret is mock_mapper.return_value
    finally:
        _stop(*patches)


def test_edge_05():
    """
    input: settings={'name_mapping': {'enable': True, 'file': 'm.json'}}, name_mapping_file='override.json'
    expected: name_mapping_fileが優先され、str(data_path('override.json'))を読み込んでNameMapperを返す。
    """
    patches = _mocks()
    try:
        mock_dp, mock_loader, mock_mapper, mock_ensure = patches
        mapping_value = {"a": "b"}
        mock_loader.load_mapping.return_value = mapping_value
        ret = build_name_mapper(
            {"name_mapping": {"enable": True, "file": "m.json"}},
            name_mapping_file="override.json")
        mock_dp.assert_called_once_with("override.json")
        mock_loader.load_mapping.assert_called_once_with(_RESOLVED_PATH)
        mock_mapper.assert_called_once_with(
            mapping=mapping_value,
            edit_distance_threshold=2,
            unmapped_action="suggest",
            warn_on_approx=True)
        assert ret is mock_mapper.return_value
    finally:
        _stop(*patches)


def test_edge_06():
    """
    input: settings={'name_mapping': {'enable': True, 'edit_distance_threshold': 5, 'unmapped_action': 'skip', 'warn_on_approx': False}}
    expected: edit_distance_threshold=5, unmapped_action='skip', warn_on_approx=FalseのNameMapperを返す。
    """
    patches = _mocks()
    try:
        mock_dp, mock_loader, mock_mapper, mock_ensure = patches
        mapping_value = {"a": "b"}
        mock_loader.load_mapping.return_value = mapping_value
        ret = build_name_mapper(
            {"name_mapping": {"enable": True,
                              "edit_distance_threshold": 5,
                              "unmapped_action": "skip",
                              "warn_on_approx": False}})
        mock_dp.assert_called_once_with("config/name_mapping.json")
        mock_loader.load_mapping.assert_called_once_with(_RESOLVED_PATH)
        mock_mapper.assert_called_once_with(
            mapping=mapping_value,
            edit_distance_threshold=5,
            unmapped_action="skip",
            warn_on_approx=False)
        assert ret is mock_mapper.return_value
    finally:
        _stop(*patches)
