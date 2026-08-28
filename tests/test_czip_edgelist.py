"""Tests for the czip edge-list (CSV/TSV) loader — the
"arbitrary connectome" input path. An edge list is (src, dst[, weight]) rows;
node ids are arbitrary tokens mapped to 0..n-1 in sorted order, duplicate
(src, dst) pairs sum their weights (ConnectsTo aggregation convention)."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from czip.czip import load_edgelist

REPO = Path(__file__).resolve().parents[1]


def test_csv_with_header_and_weights(tmp_path):
    p = tmp_path / "g.csv"
    p.write_text("pre,post,weight\n10,20,3\n20,30,5\n30,10,1\n")
    adj, ids = load_edgelist(p)
    assert list(ids) == ["10", "20", "30"]
    assert adj.shape == (3, 3)
    assert adj[0, 1] == 3 and adj[1, 2] == 5 and adj[2, 0] == 1
    assert adj.nnz == 3


def test_tsv_without_header_unweighted(tmp_path):
    p = tmp_path / "g.tsv"
    p.write_text("a\tb\nb\tc\n")
    adj, ids = load_edgelist(p)
    assert list(ids) == ["a", "b", "c"]
    assert adj[0, 1] == 1 and adj[1, 2] == 1
    assert adj.nnz == 2


def test_duplicate_pairs_sum(tmp_path):
    p = tmp_path / "g.csv"
    p.write_text("1,2,3\n1,2,4\n")
    adj, ids = load_edgelist(p)
    assert adj.nnz == 1
    assert adj[0, 1] == 7


def test_self_loops_kept(tmp_path):
    p = tmp_path / "g.csv"
    p.write_text("1,1,2\n1,2,1\n")
    adj, _ = load_edgelist(p)
    assert adj[0, 0] == 2


def test_cli_encode_decode_roundtrip_from_csv(tmp_path):
    rng = np.random.default_rng(0)
    n = 30
    rows, cols = np.nonzero(rng.random((n, n)) < 0.2)
    w = rng.integers(1, 9, size=rows.size)
    csv = tmp_path / "toy.csv"
    csv.write_text("pre,post,weight\n" + "\n".join(
        f"n{r:03d},n{c:03d},{wi}" for r, c, wi in zip(rows, cols, w)) + "\n")
    part = tmp_path / "labels.npy"
    np.save(part, np.zeros(n, dtype=np.int64))
    cz = tmp_path / "toy.cz"

    r = subprocess.run(
        [sys.executable, "-m", "czip.czip", "encode", str(csv),
         "--partition", str(part), "--wmin", "1", "-o", str(cz)],
        cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    ids_file = tmp_path / "toy.cz.node_ids.npy"
    assert ids_file.exists()
    assert sorted(np.load(ids_file)) == sorted(f"n{i:03d}" for i in range(n))

    out = tmp_path / "dec.npz"
    r = subprocess.run(
        [sys.executable, "-m", "czip.czip", "decode", str(cz),
         "-o", str(out)], cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    dec = sp.load_npz(out)
    ref = sp.csr_matrix((w, (rows, cols)), shape=(n, n), dtype=np.int64)
    assert (ref != dec).nnz == 0
