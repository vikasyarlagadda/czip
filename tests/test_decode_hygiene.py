"""Decode-closure hygiene: what a czip decode actually has to import.

The portable size of the decoder is the package modules plus the third-party
wheels a fresh interpreter must load to turn a `.cz` back into a graph. That
number is only honest if the closure is *pinned*, so this module measures it
the only way that cannot drift: a subprocess with the encode-side heavyweights
poisoned in ``sys.modules``, decoding a real container, then reporting every
module that got loaded.

Poisoning (rather than uninstalling) is what makes the test runnable inside a
full development environment: a poisoned name raises ``ImportError`` on
import, exactly as it would where the package is absent.

The containers come from the ``sample_containers`` fixture, so every branch is
measured against a container this codebase just wrote.
"""

from __future__ import annotations

import functools
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests.conftest import BRANCHES

# Encode-side heavyweights. `graph_tool` is reachable only through
# `czip.czip_autofit`, which `czip.czip` imports inside the `--model auto`
# encode branch; `czip.sbm` has a module-level graph-tool import and is
# reached only from there. None of it may be on a decode path.
POISON = ("graph_tool", "torch", "pandas", "polars", "czip.sbm",
          "czip.czip_autofit")

# --- the pinned closure -------------------------------------------------
#
# REQUIRED is the closure the portable size is charged for: three wheels, no
# more. Anything new appearing here is a real cost increase and must be argued
# for, not absorbed.
REQUIRED_THIRD_PARTY = frozenset({"constriction", "numpy", "scipy"})

# ALLOWED_INCIDENTAL are names that may or may not appear depending on what
# else is installed beside the closure, and are NOT charged:
#   `cython_runtime` is a pseudo-module Cython's compiled extensions inject
#     into sys.modules. It has no file and ships in no wheel — scipy's own
#     bytes already pay for it.
#   `charset_normalizer` is an OPTIONAL soft dependency of
#     `numpy.f2py.crackfortran` (imported under try/except ImportError, used
#     only to sniff a Fortran source file's encoding), pulled in transitively
#     because `scipy.sparse` imports `numpy.f2py`. It is absent from the
#     pinned environment, and
#     `test_decode_survives_without_the_optional_soft_deps` proves decode does
#     not need it, so charging the closure for it would overstate the tool.
ALLOWED_INCIDENTAL = frozenset({"charset_normalizer", "cython_runtime"})

# The package half of the closure — the 7 modules `czip.decoder_artifact`
# ships.
DECODE_SRC_MODULES = (
    "czip",
    "czip.coder",
    "czip.czip",
    "czip.nested_coder",
    "czip.sbm_coder",
    "czip.weights",
    "czip.weights_coder",
)

_PROBE = '''
import json, os, sys, tempfile
for _name in {poison!r}:
    sys.modules[_name] = None
from czip import czip
_out = os.path.join(tempfile.mkdtemp(), "decoded.npz")
_rc = czip.main(["decode", sys.argv[1], "-o", _out])
_std = sys.stdlib_module_names
_live = [m for m in sys.modules if sys.modules[m] is not None]
_third = sorted({{m.split(".")[0] for m in _live
                 if not m.startswith(("_", "czip"))
                 and m.split(".")[0] not in _std}})
_pkg = sorted(m for m in _live if m == "czip" or m.startswith("czip."))
print("@@" + json.dumps({{"rc": _rc, "exists": os.path.exists(_out),
                          "size": os.path.getsize(_out) if os.path.exists(_out) else 0,
                          "third_party": _third, "package": _pkg}}))
'''


@functools.lru_cache(maxsize=None)
def _probe(container: Path, poison: tuple[str, ...] = POISON) -> dict:
    """Decode `container` in a fresh interpreter; return the probe's report.

    Cached: a probe is a pure function of (container, poison) and each one
    costs a full interpreter start plus a real decode, so the two tests that
    read the same probe share one subprocess.
    """
    r = subprocess.run(
        [sys.executable, "-c", _PROBE.format(poison=poison), str(container)],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    line = [ln for ln in r.stdout.splitlines() if ln.startswith("@@")]
    assert len(line) == 1, f"probe emitted no report:\n{r.stdout}"
    report = json.loads(line[0][2:])
    report["stdout"] = r.stdout
    return report


@pytest.mark.parametrize("branch", BRANCHES)
def test_decode_needs_no_graph_tool(sample_containers, branch):
    """A decode must complete with every encode-side heavyweight unimportable."""
    rep = _probe(sample_containers[branch])
    assert rep["rc"] == 0
    assert rep["exists"] and rep["size"] > 0
    # the CLI's decode stamp: a decode checks the container against ITSELF,
    # which is what this line is careful to say
    assert "(params + source digests verified)" in rep["stdout"]


@pytest.mark.parametrize("branch", BRANCHES)
def test_decoder_closure_module_list(sample_containers, branch):
    """The loaded closure is exactly the pinned one (plus uncharged incidentals)."""
    rep = _probe(sample_containers[branch])
    third = set(rep["third_party"])
    assert REQUIRED_THIRD_PARTY <= third, (
        f"pinned dependency missing: {sorted(REQUIRED_THIRD_PARTY - third)}")
    unpinned = third - REQUIRED_THIRD_PARTY - ALLOWED_INCIDENTAL
    assert not unpinned, (
        f"decode pulled in unpinned third-party modules {sorted(unpinned)}; "
        "each one is a real addition to the portable size of the decoder — "
        "justify it and add it to REQUIRED_THIRD_PARTY, or remove the import")
    assert tuple(rep["package"]) == DECODE_SRC_MODULES


def test_decode_survives_without_the_optional_soft_deps(sample_containers):
    """The uncharged incidentals really are optional: poison them and decode."""
    rep = _probe(sample_containers["nested-weighted"],
                 POISON + ("charset_normalizer",))
    assert rep["rc"] == 0 and rep["exists"]
    assert "charset_normalizer" not in rep["third_party"]


def test_artifact_module_list_matches_hygiene_list():
    """`czip.decoder_artifact` ships exactly the modules the probe observes."""
    from czip.decoder_artifact import DECODE_MODULES

    as_modules = tuple(
        p[:-len("/__init__.py")].replace("/", ".") if p.endswith("/__init__.py")
        else p[:-len(".py")].replace("/", ".")
        for p in DECODE_MODULES)
    assert tuple(sorted(as_modules)) == DECODE_SRC_MODULES
