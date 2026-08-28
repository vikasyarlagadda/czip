"""The hermetic czip decoder closure, as one reproducible zip.

The portable size of the decoder is *this* artifact plus the
three pinned wheels — everything a fresh interpreter needs to turn a saved
`.cz` container back into a graph, and nothing else. Encode-side machinery
(the SBM fitters, graph-tool) is not in here: `tests/test_decode_hygiene.py`
holds that line by decoding a container with those names poisoned.

The zip is NOT committed. It is rebuilt byte-identically from the working tree
by::

    python -m czip.decoder_artifact --out czip_decoder.zip

which is what lets a size be quoted for it without a binary
blob entering git. Determinism comes from a fixed timestamp, fixed permissions
and creator, a fixed member order, and a generated README with no variable
content — so two builds of the same source tree are the same bytes.
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# The decode closure, as paths relative to the package's parent directory.
# This is the single source of truth for what "the decoder" is;
# `tests/test_decode_hygiene.py` asserts it equals the `czip.*` modules a real
# decode actually loads.
DECODE_MODULES: tuple[str, ...] = (
    "czip/__init__.py",
    "czip/czip.py",
    "czip/sbm_coder.py",
    "czip/nested_coder.py",
    "czip/coder.py",
    "czip/weights_coder.py",
    "czip/weights.py",
)

# The pinned versions the closure is measured against. The wheels
# themselves are not shipped in here.
PINNED_REQUIREMENTS: tuple[str, ...] = (
    "numpy==2.5.2",
    "scipy==1.18.0",
    "constriction==0.5.0",
)

# Fixed zip metadata. 1980-01-01 00:00:00 is the earliest timestamp the zip
# format can express; 0o644 regular-file permissions in the high 16 bits is
# what `external_attr` wants; create_system 3 is "unix".
_ZIP_DATE = (1980, 1, 1, 0, 0, 0)
_ZIP_ATTR = (0o100644 << 16)
_ZIP_CREATE_SYSTEM = 3

README_NAME = "README.txt"

README_TEXT = """\
czip decoder closure
====================

Everything needed to decode a `.cz` container produced by the czip coder, and
nothing needed to produce one. Unzip beside a working directory so that
`czip/` is importable, install the pinned requirements below, then:

    czip decode <file.cz> -o <out.npz>

`<out.npz>` is the adjacency matrix in scipy CSR form (`scipy.sparse.load_npz`);
add `--labels-out <labels.npy>` for the block partition. A decode checks the
container against itself: both header digests (parameters and source) must
match or it fails.

Pinned requirements (nothing else is imported on a decode path):

    numpy==2.5.2
    scipy==1.18.0
    constriction==0.5.0

This archive is generated, not committed. Rebuild it byte-identically from the
source tree with:

    python -m czip.decoder_artifact --out czip_decoder.zip
"""


def build_zip(dest: Path) -> Path:
    """Write the decoder closure to `dest`; return `dest`.

    Byte-identical across runs and machines: every ZipInfo carries the same
    timestamp, permissions and creator, members are written in DECODE_MODULES
    order, and the trailing README is a fixed string.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=9) as zf:
        for rel in DECODE_MODULES:
            zf.writestr(_info(rel), (REPO / rel).read_bytes())
        zf.writestr(_info(README_NAME), README_TEXT.encode("utf-8"))
    return dest


def _info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(filename=name, date_time=_ZIP_DATE)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = _ZIP_ATTR
    info.create_system = _ZIP_CREATE_SYSTEM
    return info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="path to write the zip to")
    args = ap.parse_args(argv)
    out = build_zip(Path(args.out))
    print(f"[decoder_artifact] wrote {out} ({out.stat().st_size} bytes, "
          f"{len(DECODE_MODULES)} modules + {README_NAME})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
