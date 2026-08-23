"""Project-local pytest configuration.

This file exists to make the ``tmp_path`` fixture work under the DSH
sandbox. Two issues:

1. On Windows, pytest's ``rm_rf`` converts paths to the ``\\?\\``
   extended-length form via ``ensure_extended_length_path``. Under this
   sandbox those ``\\?\\`` operations are denied (WinError 5 / 123), while
   ordinary absolute paths work normally (verified: scandir / rmtree on
   plain workspace paths succeed). So we neutralize the conversion.

2. The default ``tmp_path`` base lives in the system TEMP dir, which holds
   a stale ``pytest-of-<user>`` directory that is ACL-locked (scandir
   denied). We point the base at a project-local folder that is freely
   accessible instead.

3. Under this sandbox, ``os.mkdir(path, 0o700)`` creates a directory whose
   ACL denies access even to the creating process (verified: icacls and
   scandir both return WinError 5 on such a directory), while
   ``os.mkdir(path, 0o777)`` / ``os.mkdir(path)`` create a directory with
   the normal inherited ACL. pytest's tmpdir machinery creates every
   tmp_path/basetemp directory with an explicit ``mode=0o700`` (see
   ``_pytest/tmpdir.py``), so we drop the mode argument from ``os.mkdir``
   on win32. ``pathlib.Path.mkdir`` resolves ``os.mkdir`` at call time, so
   this covers all of pytest's directory creation.
"""
import os
import sys

if sys.platform.startswith("win32"):
    import _pytest.pathlib as _pathlib

    def _ensure_extended_length_path_noop(path):
        # Return the path unchanged: an ordinary absolute path with no
        # ``\\?\\`` prefix, which this sandbox allows.
        return path

    _pathlib.ensure_extended_length_path = _ensure_extended_length_path_noop

    _os_mkdir_orig = os.mkdir

    def _os_mkdir_ignore_mode(path, *args, **kwargs):
        # See module docstring issue 3: an explicit restrictive mode breaks
        # the resulting directory's ACL in this sandbox.
        return _os_mkdir_orig(path)

    os.mkdir = _os_mkdir_ignore_mode


def pytest_configure(config):
    """Point the tmp_path base at a project-local folder if not set."""
    if not config.option.basetemp:
        project = os.path.dirname(os.path.abspath(__file__))
        config.option.basetemp = os.path.join(project, ".pytest_tmp")
