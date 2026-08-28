"""Shared fixtures: small containers built on the fly, one per model branch.

The suite needs real `.cz` files to decode — the decode-closure probe runs a
fresh interpreter against one, and the header-compatibility tests read one
back. They are generated here rather than committed, so the tests carry no
data and stay honest about what they cover: every container is encoded by the
code under test, from a seeded synthetic graph, with no fitting step and
therefore no graph-tool.

Both models are covered on both payloads:

    flat-topology     dcsbm                 explicit partition, weights dropped
    nested-topology   nested-dcsbm          explicit hierarchy, weights dropped
    flat-weighted     dcsbm+weights         explicit partition
    nested-weighted   nested-dcsbm+weights  explicit hierarchy

Session-scoped: encoding four containers costs a second or so, and the probe
subprocesses are cached on the container path, so they must be stable for the
whole run.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from czip import czip

N_NODES = 120
N_BLOCKS = 4
N_EDGES = 600
#: level-1 labels over the 4 base blocks (2 superblocks), then a single top
TOP_SPLIT = (0, 1, 1, 0)

#: the four (model, payload) branches, by fixture key
BRANCHES = ("flat-topology", "nested-topology",
            "flat-weighted", "nested-weighted")


def make_graph(seed: int = 0, weighted: bool = False):
    """(CSR adjacency, block labels): seeded, simple, directed, no self-loops."""
    rng = np.random.default_rng(seed)
    labels = np.sort(rng.integers(0, N_BLOCKS, size=N_NODES))
    labels[:N_BLOCKS] = np.arange(N_BLOCKS)
    labels = np.sort(labels).astype(np.int64)

    pairs: set[tuple[int, int]] = set()
    while len(pairs) < N_EDGES:
        s, d = rng.integers(0, N_NODES, size=2)
        if s != d:
            pairs.add((int(s), int(d)))
    src, dst = (np.asarray(a, dtype=np.int64) for a in zip(*sorted(pairs)))

    if weighted:
        data = rng.geometric(0.4, size=src.size).astype(np.int64)  # >= 1
    else:
        data = np.ones(src.size, dtype=np.int64)
    adj = sp.csr_matrix((data, (src, dst)), shape=(N_NODES, N_NODES),
                        dtype=np.int64)
    return adj, labels


def make_hierarchy(labels):
    """Canonical 3-level hierarchy over a base partition with B0 = N_BLOCKS."""
    mid = np.asarray(TOP_SPLIT, dtype=np.int64)
    assert mid.shape[0] == int(labels.max()) + 1
    return [np.asarray(labels, dtype=np.int64), mid,
            np.zeros(int(mid.max()) + 1, dtype=np.int64)]


@pytest.fixture(scope="session")
def sample_containers(tmp_path_factory) -> dict:
    """{branch name: path to a freshly encoded .cz} for all four branches."""
    out = tmp_path_factory.mktemp("containers")
    blobs = {}

    adj, labels = make_graph(seed=1)
    blobs["flat-topology"] = czip.encode_topology(adj, labels)
    blobs["nested-topology"] = czip.encode_topology(
        adj, hierarchy=make_hierarchy(labels))

    wadj, wlabels = make_graph(seed=2, weighted=True)
    blobs["flat-weighted"] = czip.encode_weighted(wadj, wlabels, wmin=1)
    blobs["nested-weighted"] = czip.encode_weighted(
        wadj, hierarchy=make_hierarchy(wlabels), wmin=1)

    paths = {}
    for name, blob in blobs.items():
        path = out / f"{name.replace('-', '_')}.cz"
        path.write_bytes(blob)
        paths[name] = path
    return paths


@pytest.fixture(scope="session")
def legacy_container() -> bytes:
    """A container in the shape written before per-container stream counts.

    Encoded normally, then reopened and stripped of ``n_word_streams`` /
    ``n_byte_streams``. That is exactly what an older writer produced, and it
    is what makes the bits gate exercise its fallback window instead of the
    counts the encoder now records.
    """
    adj, labels = make_graph(seed=3, weighted=True)
    header, streams = czip.unpack(czip.encode_weighted(adj, labels, wmin=1))
    for key in ("n_word_streams", "n_byte_streams"):
        header["report"]["topology"].pop(key)
    return czip.pack(header, streams)
