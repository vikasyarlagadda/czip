"""Generate a small synthetic connectome to try czip on.

The graph is a directed, self-loop-free, block-structured multigraph-free
network: 300 neurons in 6 blocks, ~2500 connections, each carrying an integer
synapse count of 1 or more. Blocks connect to themselves more densely than to
each other, which is the structure a DC-SBM model is able to exploit — so the
coded container is meaningfully smaller than a naive edge list.

Nothing here is real data. The generator is seeded, so two runs with the same
``--seed`` produce byte-identical files.

Writes three files into ``--out-dir``:

    example_graph.npz      scipy CSR adjacency, int64 synapse counts
    example_partition.npy  block label per node, 0..5, sorted
    example_edges.csv      the same graph as (src,dst,weight) rows

Usage::

    python examples/make_example.py --out-dir .
    czip encode example_graph.npz --partition example_partition.npy \\
        --wmin 1 -o example.cz
    czip info example.cz
    czip decode example.cz -o example_decoded.npz
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import scipy.sparse as sp

N_NODES = 300
N_BLOCKS = 6
N_EDGES = 2500
#: how much likelier a within-block pair is to be connected than a cross-block one
WITHIN_BLOCK_ODDS = 12.0
#: success probability of the geometric synapse-count draw (mean ~1/p)
WEIGHT_P = 0.45


def make_graph(seed: int = 0):
    """(CSR adjacency, block labels) for one seeded synthetic connectome."""
    rng = np.random.default_rng(seed)

    labels = np.sort(rng.integers(0, N_BLOCKS, size=N_NODES))
    labels[:N_BLOCKS] = np.arange(N_BLOCKS)  # no empty block
    labels = np.sort(labels).astype(np.int64)

    # Rejection-sample distinct ordered pairs, favouring within-block ones.
    pairs: set[tuple[int, int]] = set()
    while len(pairs) < N_EDGES:
        s, d = rng.integers(0, N_NODES, size=2)
        if s == d:
            continue
        if labels[s] != labels[d] and rng.random() > 1.0 / WITHIN_BLOCK_ODDS:
            continue
        pairs.add((int(s), int(d)))

    src, dst = (np.asarray(a, dtype=np.int64) for a in zip(*sorted(pairs)))
    w = rng.geometric(WEIGHT_P, size=src.size).astype(np.int64)  # >= 1
    adj = sp.csr_matrix((w, (src, dst)), shape=(N_NODES, N_NODES),
                        dtype=np.int64)
    return adj, labels


def write_edge_list(adj: sp.csr_matrix, path: Path) -> None:
    """Write the adjacency as a (src,dst,weight) CSV with a header row.

    Node ids are zero-padded to a fixed width. The edge-list loader maps ids
    to matrix indices in sorted TEXT order, so the padding is what makes the
    CSV produce the same node numbering as the .npz — and therefore lets the
    same ``example_partition.npy`` describe both.
    """
    width = len(str(adj.shape[0] - 1))
    coo = adj.tocoo()
    order = np.lexsort((coo.col, coo.row))
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["src", "dst", "weight"])
        for i in order:
            writer.writerow([f"{int(coo.row[i]):0{width}d}",
                             f"{int(coo.col[i]):0{width}d}",
                             int(coo.data[i])])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=".", help="directory to write into")
    ap.add_argument("--seed", type=int, default=0, help="generator seed")
    args = ap.parse_args(argv)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    adj, labels = make_graph(args.seed)

    graph_path = out / "example_graph.npz"
    partition_path = out / "example_partition.npy"
    edges_path = out / "example_edges.csv"

    sp.save_npz(graph_path, adj)
    np.save(partition_path, labels)
    write_edge_list(adj, edges_path)

    print(f"{graph_path}: {adj.shape[0]} nodes, {adj.nnz} edges, "
          f"weights {int(adj.data.min())}..{int(adj.data.max())}")
    print(f"{partition_path}: {int(labels.max()) + 1} blocks")
    print(f"{edges_path}: {adj.nnz} rows + header")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
