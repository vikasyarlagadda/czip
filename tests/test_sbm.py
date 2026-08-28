"""Tests for czip/sbm.py — SBM fitting + bit accounting.

Conventions under test:
- bits = nats / ln 2
- L(theta) = partition_dl + degree_dl + edges_dl; L(G|theta) = adjacency term
- additivity: L_total = L(theta) + L(G|theta), asserted to float precision
- non-degree-corrected states have degree_dl == 0
- partitions saved to .npy reload into a state with identical entropy
  (recomputed from disk instead of re-fitted)
"""

import math

import numpy as np
import pytest
import scipy.sparse as sp

gt = pytest.importorskip("graph_tool.all")

from czip.sbm import (
    csr_to_graph_tool,
    nats_to_bits,
    fit_flat,
    fit_nested,
    fixed_partition_state,
    save_partition,
    load_partition_state,
    save_hierarchy,
    load_nested_state,
)


def _planted_graph(seed=7, n=300, B=4):
    """Small directed planted-partition graph with clear block structure."""
    np.random.seed(seed)
    gt.seed_rng(seed)
    g, bm = gt.random_graph(
        n,
        lambda: (np.random.poisson(6), np.random.poisson(6)),
        model="blockmodel",
        block_membership=lambda: np.random.randint(B),
        edge_probs=lambda a, b: 0.9 if a == b else 0.03,
    )
    return g


def _tiny_csr():
    # 4 nodes, directed edges: 0->1, 0->2, 1->2, 2->0, 3->1
    rows = np.array([0, 0, 1, 2, 3])
    cols = np.array([1, 2, 2, 0, 1])
    data = np.ones(5, dtype=np.int64)
    return sp.csr_matrix((data, (rows, cols)), shape=(4, 4))


class TestCsrToGraphTool:
    def test_counts_and_directedness(self):
        g = csr_to_graph_tool(_tiny_csr())
        assert g.is_directed()
        assert g.num_vertices() == 4
        assert g.num_edges() == 5

    def test_edge_set_matches(self):
        g = csr_to_graph_tool(_tiny_csr())
        edges = {(int(e.source()), int(e.target())) for e in g.edges()}
        assert edges == {(0, 1), (0, 2), (1, 2), (2, 0), (3, 1)}

    def test_isolated_nodes_kept(self):
        # node 3 has no edges; n must stay 4 (the n-convention)
        m = sp.csr_matrix(
            (np.ones(2), (np.array([0, 1]), np.array([1, 2]))), shape=(4, 4)
        )
        g = csr_to_graph_tool(m)
        assert g.num_vertices() == 4
        assert g.num_edges() == 2


class TestNatsToBits:
    def test_conversion(self):
        assert nats_to_bits(math.log(2)) == pytest.approx(1.0)
        assert nats_to_bits(0.0) == 0.0
        assert nats_to_bits(10.0) == pytest.approx(10.0 / math.log(2))


class TestFlatItemization:
    def test_additive_and_nonnegative(self):
        g = _planted_graph()
        state, bits = fit_flat(g, seed=42)
        assert bits["L_total_bits"] == pytest.approx(
            bits["L_theta_bits"] + bits["L_adjacency_bits"], rel=1e-9
        )
        assert bits["L_theta_bits"] == pytest.approx(
            bits["L_partition_bits"] + bits["L_degree_bits"] + bits["L_edges_bits"],
            rel=1e-9,
        )
        for k, v in bits.items():
            if k.startswith("L_"):
                assert v >= 0.0, f"{k} negative: {v}"
        assert bits["B"] >= 1
        assert bits["deg_corrected"] is True

    def test_total_matches_entropy_in_bits(self):
        g = _planted_graph()
        state, bits = fit_flat(g, seed=42)
        assert bits["L_total_bits"] == pytest.approx(
            state.entropy() / math.log(2), rel=1e-12
        )

    def test_non_dc_degree_term_zero(self):
        g = _planted_graph()
        state, bits = fit_flat(g, seed=42, deg_corr=False)
        assert bits["L_degree_bits"] == 0.0
        assert bits["deg_corrected"] is False

    def test_seed_determinism(self):
        # Seeded fits are only deterministic single-threaded: graph-tool's
        # parallel MCMC diverges under OpenMP scheduling,
        # so pin to 1 thread for the determinism contract being tested.
        import graph_tool.all as gt
        n_threads = gt.openmp_get_num_threads()
        gt.openmp_set_num_threads(1)
        try:
            g = _planted_graph()
            _, bits1 = fit_flat(g, seed=123)
            _, bits2 = fit_flat(g, seed=123)
        finally:
            gt.openmp_set_num_threads(n_threads)
        assert bits1["L_total_bits"] == bits2["L_total_bits"]
        assert bits1["B"] == bits2["B"]


class TestNestedItemization:
    def test_additive(self):
        g = _planted_graph()
        state, bits = fit_nested(g, seed=42)
        assert bits["L_total_bits"] == pytest.approx(
            bits["L_theta_bits"] + bits["L_adjacency_bits"], rel=1e-9
        )
        assert bits["L_total_bits"] == pytest.approx(
            state.entropy() / math.log(2), rel=1e-12
        )
        assert bits["levels_B"][0] >= 1


class TestFixedPartition:
    def test_entropy_only_no_fitting(self):
        g = _planted_graph()
        labels = np.random.RandomState(0).randint(0, 5, g.num_vertices())
        state, bits = fixed_partition_state(g, labels)
        # blocks must be exactly the given labels, untouched by any fitting
        assert np.array_equal(np.asarray(state.get_blocks().a), labels)
        assert bits["L_total_bits"] == pytest.approx(
            bits["L_theta_bits"] + bits["L_adjacency_bits"], rel=1e-9
        )

    def test_worse_than_fitted(self):
        # a random partition must not beat the fitted one on a planted graph
        g = _planted_graph()
        _, fitted = fit_flat(g, seed=42)
        labels = np.random.RandomState(0).randint(0, 5, g.num_vertices())
        _, fixed = fixed_partition_state(g, labels)
        assert fixed["L_total_bits"] > fitted["L_total_bits"]


class TestPartitionPersistence:
    def test_roundtrip_entropy_identical(self, tmp_path):
        g = _planted_graph()
        state, bits = fit_flat(g, seed=42)
        path = tmp_path / "partition.npy"
        save_partition(state, path)
        state2, bits2 = load_partition_state(g, path)
        assert bits2["L_total_bits"] == pytest.approx(
            bits["L_total_bits"], rel=1e-12
        )
        assert np.array_equal(
            np.asarray(state.get_blocks().a), np.asarray(state2.get_blocks().a)
        )


class TestTypePartition:
    def test_factorizes_with_untyped_shared_block(self):
        pl = pytest.importorskip("polars")
        from czip.sbm import partition_from_types

        s = pl.Series("cell_type", ["A", "B", None, "A", None, "C"])
        labels = partition_from_types(s)
        assert labels.shape == (6,)
        # same type -> same block
        assert labels[0] == labels[3]
        # distinct types -> distinct blocks
        assert len({labels[0], labels[1], labels[5]}) == 3
        # all untyped share ONE extra block, distinct from every typed block
        assert labels[2] == labels[4]
        assert labels[2] not in {labels[0], labels[1], labels[5]}
        # labels are a dense 0..B-1 range
        assert set(labels) == set(range(4))

    def test_no_untyped_no_extra_block(self):
        pl = pytest.importorskip("polars")
        from czip.sbm import partition_from_types

        s = pl.Series("cell_type", ["A", "B", "A"])
        labels = partition_from_types(s)
        assert set(labels) == {0, 1}


class TestHierarchyPersistence:
    """Nested runs must be fully recomputable from disk.

    save_partition only keeps the base level; save_hierarchy keeps every
    level so load_nested_state reproduces the complete nested DL without
    re-fitting.
    """

    def test_roundtrip_entropy_identical(self, tmp_path):
        g = _planted_graph()
        state, bits = fit_nested(g, seed=11)
        path = tmp_path / "hierarchy.npz"
        save_hierarchy(state, path)
        state2, bits2 = load_nested_state(g, path)
        for k in ("L_theta_bits", "L_adjacency_bits", "L_total_bits"):
            assert bits2[k] == pytest.approx(bits[k], rel=1e-12), k
        assert bits2["levels_B"] == bits["levels_B"]

    def test_base_level_matches_save_partition(self, tmp_path):
        g = _planted_graph()
        state, _ = fit_nested(g, seed=12)
        save_hierarchy(state, tmp_path / "h.npz")
        save_partition(state.get_levels()[0], tmp_path / "p.npy")
        with np.load(tmp_path / "h.npz") as z:
            base = z["level_0"]
        assert np.array_equal(base, np.load(tmp_path / "p.npy"))

    def test_deg_corr_propagates_to_base_state(self, tmp_path):
        # graph-tool 3.6 silently ignores a wrong kwarg name; guard that
        # deg_corr actually reaches the rebuilt base state
        g = _planted_graph()
        state, _ = fit_nested(g, seed=13)
        save_hierarchy(state, tmp_path / "h.npz")
        s_dc, _ = load_nested_state(g, tmp_path / "h.npz", deg_corr=True)
        s_nd, _ = load_nested_state(g, tmp_path / "h.npz", deg_corr=False)
        assert bool(s_dc.get_levels()[0].deg_corr) is True
        assert bool(s_nd.get_levels()[0].deg_corr) is False
