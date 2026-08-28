"""The decoder closure zip builds from an installed package, deterministically.

`czip.decoder_artifact` resolves its sources relative to the package's parent
directory, so it works the same from a source checkout and from site-packages.
These tests run wherever czip is importable and need no graph-tool.
"""

from __future__ import annotations

import zipfile

from czip.decoder_artifact import (DECODE_MODULES, PINNED_REQUIREMENTS,
                                   README_NAME, build_zip)


def test_build_zip_ships_exactly_the_declared_modules(tmp_path):
    out = build_zip(tmp_path / "czip_decoder.zip")
    assert out.exists() and out.stat().st_size > 0
    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
    assert tuple(names) == DECODE_MODULES + (README_NAME,)


def test_build_zip_members_are_the_real_sources(tmp_path):
    out = build_zip(tmp_path / "czip_decoder.zip")
    with zipfile.ZipFile(out) as zf:
        head = zf.read("czip/czip.py").decode("utf-8")[:200]
        readme = zf.read(README_NAME).decode("utf-8")
    assert "czip" in head
    for pin in PINNED_REQUIREMENTS:
        assert pin in readme


def test_build_zip_is_byte_identical_across_builds(tmp_path):
    a = build_zip(tmp_path / "a.zip").read_bytes()
    b = build_zip(tmp_path / "b.zip").read_bytes()
    assert a == b
