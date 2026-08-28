"""Parity tests pinning the nested DC-SBM decomposition to graph-tool.

The flat coder's rule applies unchanged to
the hierarchy: graph-tool is the sole formula authority, so every per-level
term in czip.nested_coder must match `NestedBlockState.levels[l].entropy(...)`
and the whole message must match `state.entropy(degree_dl_kind='uniform')`.
Synthetic tiny graphs only.
"""

import numpy as np
import pytest

gt = pytest.importorskip("graph_tool.all")

from czip import nested_coder, sbm_coder
from czip.sbm import _term, nats_to_bits


def _graph(src, dst, n):
    g = gt.Graph(directed=True)
    g.add_vertex(n)
    g.add_edge_list(np.column_stack([src, dst]))
    return g


def _nested_case(seed, n=36, e=180, sizes=(6, 3, 2)):
    """Random tiny directed multigraph (self-loops + parallel edges possible)
    with a random dense hierarchy, as a deg-corr NestedBlockState.

    `sizes` is (B0, B1, ...); the returned hierarchy is already canonical and
    terminates in the single top block graph-tool expects.
    """
    rng = np.random.default_rng(seed)
    src = rng.integers(0, n, e)
    dst = rng.integers(0, n, e)
    g = _graph(src, dst, n)
    hierarchy = []
    items = n
    for B in sizes:
        lab = rng.integers(0, B, items)
        lab[:B] = np.arange(B)              # every block non-empty
        _, lab = np.unique(lab, return_inverse=True)
        hierarchy.append(lab.reshape(-1).astype(np.int64))
        items = int(lab.max()) + 1
    hierarchy.append(np.zeros(items, dtype=np.int64))
    state = gt.NestedBlockState(g, bs=hierarchy,
                                base_state_args=dict(deg_corr=True))
    return state, g, hierarchy, src, dst


def _fitted_case(seed=3, n=60, e=1200, K=6, p_in=0.85):
    """A planted-block graph fitted with minimize_nested_blockmodel_dl — the
    only way to get graph-tool's PADDED, non-dense get_bs() arrays."""
    rng = np.random.default_rng(seed)
    grp = np.repeat(np.arange(K), n // K)
    members = [np.where(grp == r)[0] for r in range(K)]
    src, dst = [], []
    for _ in range(e):
        r = int(rng.integers(0, K))
        s = r if rng.random() < p_in else int(rng.integers(0, K))
        src.append(int(rng.choice(members[r])))
        dst.append(int(rng.choice(members[s])))
    src = np.array(src)
    dst = np.array(dst)
    g = _graph(src, dst, n)
    np.random.seed(seed)
    gt.seed_rng(seed)
    state = gt.minimize_nested_blockmodel_dl(g, state_args=dict(deg_corr=True))
    return state, g, src, dst


def _degenerate_case(seed=0, n=25, e=80):
    """B0 = 1: the hierarchy is a single level and there is no expansion."""
    rng = np.random.default_rng(seed)
    src = rng.integers(0, n, e)
    dst = rng.integers(0, n, e)
    g = _graph(src, dst, n)
    hierarchy = [np.zeros(n, dtype=np.int64)]
    state = gt.NestedBlockState(g, bs=hierarchy,
                                base_state_args=dict(deg_corr=True))
    return state, g, hierarchy, src, dst


# ---------------------------------------------------------------------------
# Hierarchy normal form — densification, truncation, validation
# ---------------------------------------------------------------------------

class TestCanonicalHierarchy:
    def test_densifies_padded_get_bs(self):
        state, _, _, _ = _fitted_case()
        bs = [np.asarray(b, dtype=np.int64) for b in state.get_bs()]
        # precondition: graph-tool really did hand back padded, sparse labels
        assert len(bs[0]) == len(bs[1])                  # level 1 padded to n
        assert int(bs[0].max()) + 1 > len(np.unique(bs[0]))

        h = nested_coder.canonical_hierarchy(bs)
        B = [int(lab.max()) + 1 for lab in h]
        assert B == [s.get_B() for s in state.levels][:len(h)]
        assert [len(lab) for lab in h] == [len(bs[0])] + B[:-1]
        for lab in h:
            counts = np.bincount(lab)
            assert counts.min() >= 1                      # no empty blocks
            assert set(lab.tolist()) == set(range(int(lab.max()) + 1))
        # a relabelling of the fitted partition, not a repartition: the raw
        # label is constant on each dense block and the map is injective
        assert len(set(zip(h[0].tolist(), bs[0].tolist()))) == B[0]

    def test_blocks_are_numbered_by_first_appearance(self):
        state, _, _, _ = _fitted_case()
        h = nested_coder.canonical_hierarchy(state.get_bs())
        for lab in h:
            first = np.unique(lab, return_index=True)[1]
            assert np.array_equal(first, np.sort(first))
            assert lab[0] == 0

    def test_truncates_at_first_single_block(self):
        state, _, _, _ = _fitted_case()
        h = nested_coder.canonical_hierarchy(state.get_bs())
        assert int(h[-1].max()) + 1 == 1
        assert all(int(lab.max()) + 1 > 1 for lab in h[:-1])
        assert len(h) < len(state.get_bs())               # padding dropped

    def test_trailing_levels_cost_nothing(self):
        state, _, _, _ = _fitted_case()
        h = nested_coder.canonical_hierarchy(state.get_bs())
        for l in range(len(h), len(state.levels)):
            assert state.level_entropy(l) == 0.0

    def test_already_canonical_is_a_fixed_point(self):
        _, _, hierarchy, _, _ = _nested_case(1)
        h = nested_coder.canonical_hierarchy(hierarchy)
        assert len(h) == len(hierarchy)
        for got, want in zip(h, hierarchy):
            assert np.array_equal(got, want)

    def test_canonical_hierarchy_reproduces_entropy(self):
        state, g, _, _ = _fitted_case()
        h = nested_coder.canonical_hierarchy(state.get_bs())
        rebuilt = gt.NestedBlockState(g, bs=h,
                                      base_state_args=dict(deg_corr=True))
        assert rebuilt.entropy() == pytest.approx(state.entropy(), rel=1e-9)
        assert rebuilt.entropy(degree_dl_kind="uniform") == pytest.approx(
            state.entropy(degree_dl_kind="uniform"), rel=1e-9)

    def test_degenerate_single_level(self):
        _, _, hierarchy, _, _ = _degenerate_case()
        h = nested_coder.canonical_hierarchy(hierarchy)
        assert len(h) == 1
        assert np.array_equal(h[0], np.zeros(len(hierarchy[0])))

    def test_unterminated_hierarchy_raises(self):
        _, _, hierarchy, _, _ = _nested_case(2)
        with pytest.raises(ValueError, match="terminate"):
            nested_coder.canonical_hierarchy(hierarchy[:-1])

    def test_short_parent_level_raises(self):
        bad = [np.array([0, 1, 2, 2]), np.array([0, 0])]
        with pytest.raises(ValueError, match="too short"):
            nested_coder.canonical_hierarchy(bad)

    def test_negative_labels_raise(self):
        with pytest.raises(ValueError, match="negative"):
            nested_coder.canonical_hierarchy([np.array([0, -1, 0])])

    def test_empty_base_raises(self):
        with pytest.raises(ValueError, match="non-empty base"):
            nested_coder.canonical_hierarchy([np.array([], dtype=np.int64)])


class TestCanonicalValidation:
    """The public ideal-bits entry points only accept the normal form."""

    def test_empty_block_in_a_level_raises(self):
        _, _, hierarchy, src, dst = _nested_case(3)
        bad = list(hierarchy)
        bad[1] = bad[1].copy()
        bad[1][bad[1] == 0] = 2                    # block 0 of level 1 emptied
        with pytest.raises(ValueError, match="no empty blocks"):
            nested_coder.nested_total_ideal_bits(src, dst, bad)

    def test_length_mismatch_between_levels_raises(self):
        _, _, hierarchy, src, dst = _nested_case(3)
        bad = list(hierarchy)
        bad[1] = np.concatenate([bad[1], bad[1][:1]])
        with pytest.raises(ValueError, match="items but level"):
            nested_coder.nested_total_ideal_bits(src, dst, bad)

    def test_single_block_before_the_top_raises(self):
        _, _, hierarchy, src, dst = _nested_case(3)
        bad = [hierarchy[0], np.zeros(len(hierarchy[1]), dtype=np.int64),
               np.zeros(1, dtype=np.int64)]
        with pytest.raises(ValueError, match="truncate"):
            nested_coder.nested_total_ideal_bits(src, dst, bad)

    def test_unterminated_hierarchy_is_refused_by_the_validator(self):
        # `canonical_hierarchy` refuses this too, but the ideal-bits path does
        # not go through it: `_check_canonical` carries its OWN copy of the
        # termination check, and an unterminated hierarchy is a different
        # model (with no level above it graph-tool charges the top level's
        # edges_dl), so neither entry point may price one.
        _, _, hierarchy, src, dst = _nested_case(3)
        with pytest.raises(ValueError, match="terminate"):
            nested_coder.nested_total_ideal_bits(src, dst, hierarchy[:-1])
        with pytest.raises(ValueError, match="terminate"):
            nested_coder.ers_levels((src, dst), hierarchy[:-1])

    def test_expand_ideal_bits_rejects_non_dense_parent(self):
        E = np.array([[3, 1], [0, 2]], dtype=np.int64)
        with pytest.raises(ValueError, match="no empty blocks"):
            nested_coder.expand_ideal_bits(E, np.array([0, 2]))


# ---------------------------------------------------------------------------
# Per-level e_rs aggregation
# ---------------------------------------------------------------------------

class TestErsLevels:
    def test_matches_direct_aggregation_and_conserves_edges(self):
        _, _, hierarchy, src, dst = _nested_case(4)
        E = nested_coder.ers_levels((src, dst), hierarchy)
        assert len(E) == len(hierarchy)
        assert E[-1].shape == (1, 1) and int(E[-1][0, 0]) == len(src)
        for l in range(1, len(E)):
            par = hierarchy[l]
            Bp = int(par.max()) + 1
            want = np.zeros((Bp, Bp), dtype=np.int64)
            for r in range(E[l - 1].shape[0]):
                for s in range(E[l - 1].shape[1]):
                    want[par[r], par[s]] += E[l - 1][r, s]
            assert np.array_equal(E[l], want)
            assert E[l].sum() == len(src)

    def test_accepts_a_prebuilt_base_matrix_dense_or_sparse(self):
        import scipy.sparse as sp
        _, _, hierarchy, src, dst = _nested_case(5)
        E = nested_coder.ers_levels((src, dst), hierarchy)
        from_dense = nested_coder.ers_levels(E[0], hierarchy)
        from_sparse = nested_coder.ers_levels(sp.csr_matrix(E[0]), hierarchy)
        for a, b, c in zip(E, from_dense, from_sparse):
            assert np.array_equal(a, b) and np.array_equal(a, c)

    def test_base_shape_mismatch_raises(self):
        _, _, hierarchy, src, dst = _nested_case(5)
        with pytest.raises(ValueError, match="shape"):
            nested_coder.ers_levels(np.zeros((2, 2), dtype=np.int64),
                                    hierarchy)


# ---------------------------------------------------------------------------
# Per-level parity vs graph-tool
# ---------------------------------------------------------------------------

class TestLevelParity:
    @pytest.mark.parametrize("seed,sizes", [
        (0, (6, 3, 2)), (1, (5, 2)), (2, (8, 4, 2)), (3, (4,)),
    ])
    def test_partition_dl_per_level(self, seed, sizes):
        state, _, hierarchy, _, _ = _nested_case(seed, sizes=sizes)
        for l, lab in enumerate(hierarchy):
            expected = nats_to_bits(
                _term(state.levels[l], partition_dl=True, propagate=False))
            assert sbm_coder.partition_ideal_bits(lab) == pytest.approx(
                expected, rel=1e-9)

    @pytest.mark.parametrize("seed,sizes", [
        (0, (6, 3, 2)), (1, (5, 2)), (2, (8, 4, 2)), (3, (4,)),
    ])
    def test_expansion_per_level(self, seed, sizes):
        state, _, hierarchy, src, dst = _nested_case(seed, sizes=sizes)
        E = nested_coder.ers_levels((src, dst), hierarchy)
        for l in range(1, len(hierarchy)):
            expected = nats_to_bits(state.levels[l].entropy(
                adjacency=True, dl=False, propagate=False))
            assert nested_coder.expand_ideal_bits(
                E[l - 1], hierarchy[l]) == pytest.approx(expected, rel=1e-9)

    @pytest.mark.parametrize("seed,sizes", [(0, (6, 3, 2)), (2, (8, 4, 2))])
    def test_level_entropy_splits_as_documented(self, seed, sizes):
        """level 0 = adjacency + partition + degree (NO edges_dl);
        level l>=1 = expansion + partition, no degree term."""
        state, _, hierarchy, _, _ = _nested_case(seed, sizes=sizes)
        for l, s in enumerate(state.levels):
            adj = s.entropy(adjacency=True, dl=False, propagate=False)
            part = _term(s, partition_dl=True, propagate=False)
            deg = _term(s, degree_dl=True, propagate=False)
            assert state.level_entropy(l) == pytest.approx(
                adj + part + deg, rel=1e-9)
            if l:
                assert deg == 0.0
        assert state.entropy() == pytest.approx(
            sum(state.level_entropy(l) for l in range(len(state.levels))),
            rel=1e-12)

    @pytest.mark.parametrize("seed,sizes", [(0, (6, 3, 2)), (2, (8, 4, 2))])
    def test_base_edges_dl_is_excluded_from_level_zero(self, seed, sizes):
        state, _, _, _, _ = _nested_case(seed, sizes=sizes)
        base = state.levels[0]
        edges = _term(base, edges_dl=True, propagate=False)
        assert edges > 0.0                                # not vacuously true
        flat_style = (base.entropy(adjacency=True, dl=False, propagate=False)
                      + _term(base, partition_dl=True, propagate=False)
                      + _term(base, degree_dl=True, propagate=False) + edges)
        assert state.level_entropy(0) == pytest.approx(
            flat_style - edges, rel=1e-12)

    @pytest.mark.parametrize("seed,sizes", [(0, (6, 3, 2)), (1, (5, 2))])
    def test_top_expansion_equals_flat_edges_dl_below(self, seed, sizes):
        """At the B=1 top, multiset(B_{L-2}^2, e) IS the flat edges_dl of the
        level below — how the hierarchy pays for what level 0 dropped."""
        state, _, hierarchy, src, dst = _nested_case(seed, sizes=sizes)
        E = nested_coder.ers_levels((src, dst), hierarchy)
        top = len(hierarchy) - 1
        expansion = nested_coder.expand_ideal_bits(E[top - 1], hierarchy[top])
        below = nats_to_bits(
            _term(state.levels[top - 1], edges_dl=True, propagate=False))
        assert expansion == pytest.approx(below, rel=1e-9)
        assert expansion == pytest.approx(
            sbm_coder.edges_ideal_bits(E[top - 1]), rel=1e-12)


# ---------------------------------------------------------------------------
# Whole-message parity vs state.entropy(degree_dl_kind='uniform')
# ---------------------------------------------------------------------------

class TestNestedTotal:
    @pytest.mark.parametrize("seed,sizes", [
        (0, (6, 3, 2)), (1, (5, 2)), (2, (8, 4, 2)), (3, (4,)), (4, (7, 2)),
    ])
    def test_total_matches_graph_tool_uniform_kind(self, seed, sizes):
        state, _, hierarchy, src, dst = _nested_case(seed, sizes=sizes)
        expected = nats_to_bits(state.entropy(degree_dl_kind="uniform"))
        assert nested_coder.nested_total_ideal_bits(
            src, dst, hierarchy) == pytest.approx(expected, rel=1e-9)

    def test_total_on_padded_fitted_hierarchy(self):
        state, _, src, dst = _fitted_case()
        h = nested_coder.canonical_hierarchy(state.get_bs())
        expected = nats_to_bits(state.entropy(degree_dl_kind="uniform"))
        assert nested_coder.nested_total_ideal_bits(
            src, dst, h) == pytest.approx(expected, rel=1e-9)

    def test_total_on_degenerate_single_level(self):
        state, _, hierarchy, src, dst = _degenerate_case()
        expected = nats_to_bits(state.entropy(degree_dl_kind="uniform"))
        rep = nested_coder.nested_ideal_bits(src, dst, hierarchy)
        assert rep["ideal_bits"] == pytest.approx(expected, rel=1e-9)
        assert rep["L_expansion_bits"] == 0.0
        assert rep["levels_B"] == [1]

    def test_report_is_itemized_and_additive(self):
        _, _, hierarchy, src, dst = _nested_case(6)
        rep = nested_coder.nested_ideal_bits(src, dst, hierarchy)
        itemized = (rep["L_partition_bits"] + rep["L_expansion_bits"]
                    + rep["L_degree_bits"] + rep["L_adjacency_bits"])
        assert rep["ideal_bits"] == pytest.approx(itemized, rel=1e-12)
        assert rep["L_theta_bits"] == pytest.approx(
            rep["ideal_bits"] - rep["L_adjacency_bits"], rel=1e-12)
        assert rep["expansion_bits_per_level"][0] == 0.0
        assert len(rep["partition_bits_per_level"]) == len(hierarchy)
        assert rep["levels_B"] == [int(l.max()) + 1 for l in hierarchy]

    def test_itemization_matches_graph_tool_level_by_level(self):
        state, _, hierarchy, src, dst = _nested_case(7, sizes=(6, 3, 2))
        rep = nested_coder.nested_ideal_bits(src, dst, hierarchy)
        for l in range(len(hierarchy)):
            level_bits = (rep["partition_bits_per_level"][l]
                          + rep["expansion_bits_per_level"][l])
            if l == 0:
                level_bits += rep["L_degree_bits"] + rep["L_adjacency_bits"]
                expected = nats_to_bits(
                    state.level_entropy(0)
                    - _term(state.levels[0], degree_dl=True, propagate=False)
                    + _term(state.levels[0], degree_dl=True,
                            degree_dl_kind="uniform", propagate=False))
            else:
                expected = nats_to_bits(state.level_entropy(l))
            assert level_bits == pytest.approx(expected, rel=1e-9)

    def test_hierarchy_beats_flat_edges_dl_accounting(self):
        """Sanity on the swap: nested total = flat-uniform total
        - flat edges_dl + (all partitions above + all expansions)."""
        state, g, hierarchy, src, dst = _nested_case(8, sizes=(6, 3, 2))
        rep = nested_coder.nested_ideal_bits(src, dst, hierarchy)
        base = state.levels[0]
        flat_uniform = nats_to_bits(
            base.entropy(adjacency=True, dl=False, propagate=False)
            + _term(base, partition_dl=True, propagate=False)
            + _term(base, edges_dl=True, propagate=False)
            + _term(base, degree_dl=True, degree_dl_kind="uniform",
                    propagate=False))
        above = (sum(rep["partition_bits_per_level"][1:])
                 + rep["L_expansion_bits"])
        edges_flat = nats_to_bits(_term(base, edges_dl=True, propagate=False))
        assert rep["ideal_bits"] == pytest.approx(
            flat_uniform - edges_flat + above, rel=1e-9)


# ---------------------------------------------------------------------------
# Expansion codec: the actual entropy code for E^{(l)} -> E^{(l-1)}
# ---------------------------------------------------------------------------

def _expand_case(total, groups=(3, 2, 3), seed=0):
    """(child e_rs, parent labels) with a prescribed edge total.

    Parent block R holds groups[R] consecutive child blocks, so the layout is
    already the sorted-by-parent one the codec permutes into.
    """
    rng = np.random.default_rng(seed)
    parent = np.repeat(np.arange(len(groups)), groups).astype(np.int64)
    B0 = int(parent.size)
    flat = rng.multinomial(total, np.ones(B0 * B0) / (B0 * B0))
    return flat.reshape(B0, B0).astype(np.int64), parent


def _scrambled_case(total, seed=0):
    """Same, but with parent labels that do NOT arrive in sorted order — the
    child-cell convention must follow the labels, not the array layout."""
    rng = np.random.default_rng(seed)
    parent = np.array([2, 0, 1, 0, 2, 1, 0], dtype=np.int64)
    B0 = int(parent.size)
    flat = rng.multinomial(total, np.ones(B0 * B0) / (B0 * B0))
    return flat.reshape(B0, B0).astype(np.int64), parent


def _acceptance_slack(report, n_streams=1):
    """Acceptance criterion: realized - ideal <= 32*n_streams + 3e-4*n_symbols
    (+ the rank branch's byte padding)."""
    return 32 * n_streams + 8 + 3e-4 * report["n_symbols"]


RANK_MAX = sbm_coder.COMPOSITION_RANK_MAX_TOTAL


class TestExpandCodec:
    """encode_expand / decode_expand — one level's expansion, both branches."""

    @pytest.mark.parametrize("seed,sizes", [
        (0, (6, 3, 2)), (1, (5, 2)), (2, (8, 4, 2)),
    ])
    def test_roundtrip_every_level_rank_branch(self, seed, sizes):
        _, _, hierarchy, src, dst = _nested_case(seed, sizes=sizes)
        E = nested_coder.ers_levels((src, dst), hierarchy)
        assert int(E[0].sum()) <= RANK_MAX          # precondition: rank branch
        for l in range(1, len(hierarchy)):
            payload, rep = nested_coder.encode_expand(E[l - 1], hierarchy[l])
            assert isinstance(payload, (bytes, bytearray))
            got = nested_coder.decode_expand(payload, E[l], hierarchy[l])
            assert np.array_equal(got, E[l - 1])
            assert rep["lossless"] is True

    def test_roundtrip_every_level_walk_branch(self):
        _, _, hierarchy, src, dst = _nested_case(9, n=40, e=6000,
                                                 sizes=(6, 3, 2))
        E = nested_coder.ers_levels((src, dst), hierarchy)
        assert int(E[0].sum()) > RANK_MAX           # precondition: walk branch
        for l in range(1, len(hierarchy)):
            payload, rep = nested_coder.encode_expand(E[l - 1], hierarchy[l])
            assert isinstance(payload, np.ndarray)
            got = nested_coder.decode_expand(payload, E[l], hierarchy[l])
            assert np.array_equal(got, E[l - 1])
            assert rep["lossless"] is True
            assert rep["n_symbols"] > 0

    def test_roundtrip_on_a_fitted_padded_hierarchy(self):
        state, _, src, dst = _fitted_case()
        h = nested_coder.canonical_hierarchy(state.get_bs())
        E = nested_coder.ers_levels((src, dst), h)
        for l in range(1, len(h)):
            payload, _ = nested_coder.encode_expand(E[l - 1], h[l])
            got = nested_coder.decode_expand(payload, E[l], h[l])
            assert np.array_equal(got, E[l - 1])

    @pytest.mark.parametrize("total", [RANK_MAX, RANK_MAX + 1])
    def test_branch_selection_is_wholesale_on_the_level_total(self, total):
        child, parent = _expand_case(total, seed=1)
        assert int(child.sum()) == total
        payload, rep = nested_coder.encode_expand(child, parent)
        if total <= RANK_MAX:
            assert isinstance(payload, (bytes, bytearray))
            assert rep["n_symbols"] == 0
        else:
            assert isinstance(payload, np.ndarray)
            assert payload.dtype == np.uint32
        parent_ers = nested_coder._aggregate(child, parent)
        got = nested_coder.decode_expand(payload, parent_ers, parent)
        assert np.array_equal(got, child)

    @pytest.mark.parametrize("total", [37, RANK_MAX + 500])
    def test_scrambled_parent_labels_roundtrip(self, total):
        child, parent = _scrambled_case(total, seed=2)
        payload, rep = nested_coder.encode_expand(child, parent)
        parent_ers = nested_coder._aggregate(child, parent)
        got = nested_coder.decode_expand(payload, parent_ers, parent)
        assert np.array_equal(got, child)
        assert rep["ideal_bits"] == pytest.approx(
            nested_coder.expand_ideal_bits(child, parent), rel=1e-12)

    @pytest.mark.parametrize("total", [0, 1, 37, RANK_MAX, RANK_MAX + 1, 9000])
    def test_ideal_equals_expand_ideal_bits(self, total):
        child, parent = _expand_case(total, seed=3)
        _, rep = nested_coder.encode_expand(child, parent)
        assert rep["ideal_bits"] == pytest.approx(
            nested_coder.expand_ideal_bits(child, parent), rel=1e-12)

    @pytest.mark.parametrize("total", [37, RANK_MAX + 1, 20000])
    def test_realized_within_acceptance(self, total):
        child, parent = _expand_case(total, groups=(5, 4, 4, 3), seed=4)
        _, rep = nested_coder.encode_expand(child, parent)
        assert (rep["realized_bits"] - rep["ideal_bits"]
                <= _acceptance_slack(rep))
        assert rep["overhead_bits"] == pytest.approx(
            rep["realized_bits"] - rep["ideal_bits"], rel=1e-12)

    @pytest.mark.parametrize("total", [37, RANK_MAX + 1])
    def test_encode_is_deterministic(self, total):
        child, parent = _expand_case(total, seed=5)
        a, rep_a = nested_coder.encode_expand(child, parent)
        b, rep_b = nested_coder.encode_expand(child, parent)
        assert bytes(np.asarray(a).tobytes() if isinstance(a, np.ndarray)
                     else a) == bytes(
            np.asarray(b).tobytes() if isinstance(b, np.ndarray) else b)
        assert rep_a == rep_b

    @pytest.mark.parametrize("total", [37, RANK_MAX + 1])
    def test_top_level_expansion_is_byte_identical_to_the_flat_edges_code(
            self, total):
        """At B=1 the expansion IS the flat edges_dl of the level below, and
        the emitted code must be the same bitstream, not merely the same
        length — same weak composition over the same row-major cells."""
        child, _ = _expand_case(total, seed=6)
        parent = np.zeros(child.shape[0], dtype=np.int64)
        payload, rep = nested_coder.encode_expand(child, parent)
        flat_payload, flat_rep = sbm_coder.encode_edges(child)
        if isinstance(payload, np.ndarray):
            assert np.array_equal(payload, flat_payload)
        else:
            assert payload == flat_payload
        assert rep["ideal_bits"] == pytest.approx(flat_rep["ideal_bits"],
                                                  rel=1e-12)

    def test_walk_chunking_does_not_change_the_bitstream(self):
        """The walk stream is fed to the range coder in bounded chunks; the
        chunk size is a memory knob, never part of the format."""
        child, parent = _expand_case(RANK_MAX + 2000, seed=7)
        big, _ = nested_coder.encode_expand(child, parent)
        saved = nested_coder._WALK_CHUNK
        try:
            nested_coder._WALK_CHUNK = 7
            small, _ = nested_coder.encode_expand(child, parent)
        finally:
            nested_coder._WALK_CHUNK = saved
        assert np.array_equal(big, small)

    def test_zero_edges_costs_nothing(self):
        child = np.zeros((6, 6), dtype=np.int64)
        parent = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
        payload, rep = nested_coder.encode_expand(child, parent)
        assert rep["ideal_bits"] == 0.0
        assert rep["realized_bits"] == 0.0
        got = nested_coder.decode_expand(
            payload, np.zeros((3, 3), dtype=np.int64), parent)
        assert np.array_equal(got, child)

    @pytest.mark.parametrize("total", [40, RANK_MAX + 40])
    def test_single_child_block_is_free(self, total):
        child = np.array([[total]], dtype=np.int64)
        parent = np.zeros(1, dtype=np.int64)
        payload, rep = nested_coder.encode_expand(child, parent)
        assert rep["ideal_bits"] == 0.0
        got = nested_coder.decode_expand(payload, child, parent)
        assert np.array_equal(got, child)

    def test_report_keys_mirror_the_flat_stage_reports(self):
        child, parent = _expand_case(500, seed=8)
        _, rep = nested_coder.encode_expand(child, parent)
        _, flat_rep = sbm_coder.encode_edges(child)
        assert set(rep) == set(flat_rep)

    def test_verify_false_skips_the_self_decode(self):
        child, parent = _expand_case(500, seed=9)
        payload, rep = nested_coder.encode_expand(child, parent, verify=False)
        assert rep["lossless"] is None
        parent_ers = nested_coder._aggregate(child, parent)
        assert np.array_equal(
            nested_coder.decode_expand(payload, parent_ers, parent), child)

    def test_child_shape_mismatch_raises(self):
        child, parent = _expand_case(50, seed=10)
        with pytest.raises(ValueError, match="shape"):
            nested_coder.encode_expand(child[:-1, :-1], parent)

    def test_non_dense_parent_raises(self):
        child, _ = _expand_case(50, groups=(3, 2, 3), seed=11)
        bad = np.repeat(np.array([0, 1, 3]), (3, 2, 3)).astype(np.int64)
        with pytest.raises(ValueError, match="no empty blocks"):
            nested_coder.encode_expand(child, bad)

    def test_parent_ers_shape_mismatch_raises_on_decode(self):
        child, parent = _expand_case(50, seed=12)
        payload, _ = nested_coder.encode_expand(child, parent)
        with pytest.raises(ValueError, match="shape"):
            nested_coder.decode_expand(payload, np.zeros((2, 2), np.int64),
                                       parent)


# ---------------------------------------------------------------------------
# Whole-message assembly: encode_nested_dcsbm / decode_nested_dcsbm
# ---------------------------------------------------------------------------

def _streams(message):
    """The payload streams of a nested message (everything but the report)."""
    return {k: v for k, v in message.items() if k != "report"}


def _raw_bytes(stream):
    return (bytes(stream) if isinstance(stream, (bytes, bytearray))
            else np.asarray(stream).tobytes())


def _expected_stream_names(L):
    names = ["header", "levels_header"]
    for l in range(L):
        names += [f"level_{l}_partition_rank", f"level_{l}_partition_words"]
    names += [f"level_{l}_expand_payload" for l in range(L - 1, 0, -1)]
    return names + ["degrees_payload", "adjacency_words"]


def _message_slack(message):
    """Acceptance criterion at whole-message scale: 32 bits per stream (which also
    absorbs each byte-padded payload's <8 padding bits) + 3e-4 per symbol,
    plus the explicit header bits, which carry no ideal-bits counterpart."""
    rep = message["report"]
    return (rep["header_bits"] + 32 * len(_streams(message))
            + 3e-4 * rep["n_symbols"])


def _medium_case():
    """The flat suite's medium scale (n=1500) with a four-level hierarchy:
    e > COMPOSITION_RANK_MAX_TOTAL, so every composition stage is on the
    slot-walk branch."""
    return _nested_case(21, n=1500, e=15000, sizes=(40, 10, 3))


class TestNestedFullMessage:
    @pytest.mark.parametrize("seed,sizes", [
        (0, (6, 3, 2)), (1, (5, 2)), (2, (8, 4, 2)),
    ])
    def test_roundtrip_recovers_graph_and_hierarchy(self, seed, sizes):
        _, _, hierarchy, src, dst = _nested_case(seed, sizes=sizes)
        message = nested_coder.encode_nested_dcsbm(src, dst, hierarchy)
        dec_h, dec_src, dec_dst = nested_coder.decode_nested_dcsbm(
            message, len(hierarchy[0]))
        assert len(dec_h) == len(hierarchy)
        for got, want in zip(dec_h, hierarchy):
            assert np.array_equal(got, want)
        assert sorted(zip(dec_src.tolist(), dec_dst.tolist())) == \
            sorted(zip(src.tolist(), dst.tolist()))
        assert message["report"]["lossless"] is True

    def test_roundtrip_on_the_medium_fixture(self):
        state, _, hierarchy, src, dst = _medium_case()
        message = nested_coder.encode_nested_dcsbm(src, dst, hierarchy)
        rep = message["report"]
        assert rep["lossless"] is True
        assert rep["ideal_bits"] == pytest.approx(
            nats_to_bits(state.entropy(degree_dl_kind="uniform")), rel=1e-9)
        dec_h, dec_src, dec_dst = nested_coder.decode_nested_dcsbm(
            message, len(hierarchy[0]))
        for got, want in zip(dec_h, hierarchy):
            assert np.array_equal(got, want)
        assert sorted(zip(dec_src.tolist(), dec_dst.tolist())) == \
            sorted(zip(src.tolist(), dst.tolist()))
        assert rep["realized_bits"] - rep["ideal_bits"] <= _message_slack(
            message)

    def test_raw_padded_hierarchy_is_canonicalized_in_the_message(self):
        """`hierarchy` is accepted raw; the CANONICAL form is what travels."""
        state, _, src, dst = _fitted_case()
        bs = [np.asarray(b, dtype=np.int64) for b in state.get_bs()]
        canon = nested_coder.canonical_hierarchy(bs)
        message = nested_coder.encode_nested_dcsbm(src, dst, bs)
        assert list(_streams(message)) == _expected_stream_names(len(canon))
        dec_h, dec_src, dec_dst = nested_coder.decode_nested_dcsbm(
            message, len(bs[0]))
        for got, want in zip(dec_h, canon):
            assert np.array_equal(got, want)
        assert sorted(zip(dec_src.tolist(), dec_dst.tolist())) == \
            sorted(zip(src.tolist(), dst.tolist()))

    def test_raw_and_canonical_input_give_the_same_message(self):
        """Canonicalization happens before any bit is emitted, so the padded
        get_bs() form and its normal form are the same transmission."""
        state, _, src, dst = _fitted_case()
        bs = [np.asarray(b, dtype=np.int64) for b in state.get_bs()]
        canon = nested_coder.canonical_hierarchy(bs)
        raw_msg = nested_coder.encode_nested_dcsbm(src, dst, bs)
        canon_msg = nested_coder.encode_nested_dcsbm(src, dst, canon)
        assert list(_streams(raw_msg)) == list(_streams(canon_msg))
        for name, stream in _streams(raw_msg).items():
            assert _raw_bytes(stream) == _raw_bytes(canon_msg[name])
        assert raw_msg["report"] == canon_msg["report"]

    @pytest.mark.parametrize("seed,sizes", [
        (0, (6, 3, 2)), (1, (5, 2)), (3, (4,)), (4, (7, 2)),
    ])
    def test_ideal_total_matches_graph_tool_uniform_kind(self, seed, sizes):
        state, _, hierarchy, src, dst = _nested_case(seed, sizes=sizes)
        message = nested_coder.encode_nested_dcsbm(src, dst, hierarchy)
        assert message["report"]["ideal_bits"] == pytest.approx(
            nats_to_bits(state.entropy(degree_dl_kind="uniform")), rel=1e-9)

    def test_report_reproduces_nested_ideal_bits_itemization(self):
        _, _, hierarchy, src, dst = _nested_case(6)
        message = nested_coder.encode_nested_dcsbm(src, dst, hierarchy)
        rep = message["report"]
        ideal = nested_coder.nested_ideal_bits(src, dst, hierarchy)
        assert rep["ideal_bits"] == pytest.approx(ideal["ideal_bits"],
                                                  rel=1e-12)
        for key, want in ideal.items():
            if isinstance(want, float):
                assert rep[key] == pytest.approx(want, rel=1e-12)
            elif isinstance(want, list) and want and isinstance(want[0],
                                                                float):
                assert rep[key] == pytest.approx(want, rel=1e-12)
            else:
                assert rep[key] == want

    def test_report_is_itemized_and_additive(self):
        _, _, hierarchy, src, dst = _nested_case(7, sizes=(6, 3, 2))
        message = nested_coder.encode_nested_dcsbm(src, dst, hierarchy)
        rep = message["report"]
        itemized = (rep["L_partition_bits"] + rep["L_expansion_bits"]
                    + rep["L_degree_bits"] + rep["L_adjacency_bits"])
        assert rep["ideal_bits"] == pytest.approx(itemized, rel=1e-12)
        # per-stage realized/ideal reconcile with the whole-message totals
        stages = rep["stages"]
        assert set(stages) == (
            {f"level_{l}_partition" for l in range(len(hierarchy))}
            | {f"level_{l}_expand" for l in range(1, len(hierarchy))}
            | {"degrees", "adjacency"})
        assert sum(s["ideal_bits"] for s in stages.values()) == pytest.approx(
            rep["ideal_bits"], rel=1e-12)
        assert sum(
            s["realized_bits"] for s in stages.values()) == pytest.approx(
            rep["realized_bits"] - rep["header_realized_bits"], rel=1e-12)
        assert rep["header_bits"] == pytest.approx(
            rep["header_e_bits"] + rep["header_levels_bits"], rel=1e-12)
        assert rep["header_levels_bits"] == float(
            sbm_coder.elias_gamma_bits(len(hierarchy)))
        assert rep["n_levels"] == len(hierarchy)
        # the two headers carry exactly e and L, nothing else
        assert sbm_coder.elias_gamma_decode(message["header"]) == len(src)
        assert sbm_coder.elias_gamma_decode(
            message["levels_header"]) == len(hierarchy)

    def test_realized_matches_the_streams_it_names(self):
        _, _, hierarchy, src, dst = _nested_case(8, sizes=(6, 3, 2))
        message = nested_coder.encode_nested_dcsbm(src, dst, hierarchy)
        counted = sum(sbm_coder._payload_bits(v)
                      for v in _streams(message).values())
        assert message["report"]["realized_bits"] == pytest.approx(
            counted, rel=1e-12)

    def test_stream_kinds_are_stable(self):
        """Container work: bytes payloads vs uint32 word arrays."""
        _, _, hierarchy, src, dst = _nested_case(9, n=40, e=6000,
                                                 sizes=(6, 3, 2))
        message = nested_coder.encode_nested_dcsbm(src, dst, hierarchy)
        streams = _streams(message)
        assert isinstance(streams["header"], (bytes, bytearray))
        assert isinstance(streams["levels_header"], (bytes, bytearray))
        for l in range(len(hierarchy)):
            assert isinstance(streams[f"level_{l}_partition_rank"],
                              (bytes, bytearray))
            assert streams[f"level_{l}_partition_words"].dtype == np.uint32
        # e > COMPOSITION_RANK_MAX_TOTAL: expansions and degrees are walks
        for l in range(1, len(hierarchy)):
            assert streams[f"level_{l}_expand_payload"].dtype == np.uint32
        assert streams["degrees_payload"].dtype == np.uint32
        assert streams["adjacency_words"].dtype == np.uint32

    def test_composition_stream_kinds_follow_the_rank_branch(self):
        """Below the rank threshold the composition stages are byte payloads,
        exactly as in the flat message — the container must carry both kinds
        for the same stream name."""
        _, _, hierarchy, src, dst = _nested_case(9, sizes=(6, 3, 2))
        assert len(src) <= RANK_MAX                 # precondition
        streams = _streams(nested_coder.encode_nested_dcsbm(src, dst,
                                                            hierarchy))
        for l in range(1, len(hierarchy)):
            assert isinstance(streams[f"level_{l}_expand_payload"],
                              (bytes, bytearray))
        assert isinstance(streams["degrees_payload"], (bytes, bytearray))
        assert streams["adjacency_words"].dtype == np.uint32

    def test_encode_is_deterministic(self):
        _, _, hierarchy, src, dst = _nested_case(10, sizes=(5, 2))
        a = nested_coder.encode_nested_dcsbm(src, dst, hierarchy)
        b = nested_coder.encode_nested_dcsbm(src, dst, hierarchy)
        assert list(_streams(a)) == list(_streams(b))
        for name, stream in _streams(a).items():
            assert _raw_bytes(stream) == _raw_bytes(b[name])
        assert a["report"] == b["report"]

    def test_degenerate_single_level_roundtrip(self):
        state, _, hierarchy, src, dst = _degenerate_case()
        message = nested_coder.encode_nested_dcsbm(src, dst, hierarchy)
        rep = message["report"]
        assert list(_streams(message)) == _expected_stream_names(1)
        assert rep["L_expansion_bits"] == 0.0
        assert rep["levels_B"] == [1]
        assert rep["ideal_bits"] == pytest.approx(
            nats_to_bits(state.entropy(degree_dl_kind="uniform")), rel=1e-9)
        dec_h, dec_src, dec_dst = nested_coder.decode_nested_dcsbm(
            message, len(hierarchy[0]))
        assert len(dec_h) == 1 and np.array_equal(dec_h[0], hierarchy[0])
        assert sorted(zip(dec_src.tolist(), dec_dst.tolist())) == \
            sorted(zip(src.tolist(), dst.tolist()))

    def test_verify_false_leaves_lossless_none(self):
        _, _, hierarchy, src, dst = _nested_case(11, sizes=(6, 3, 2))
        eager = nested_coder.encode_nested_dcsbm(src, dst, hierarchy)
        lazy = nested_coder.encode_nested_dcsbm(src, dst, hierarchy,
                                                verify=False)
        assert eager["report"]["lossless"] is True
        assert lazy["report"]["lossless"] is None
        # verify is an encode-time check, never part of the format
        for name, stream in _streams(eager).items():
            assert _raw_bytes(stream) == _raw_bytes(lazy[name])
        dec_h, dec_src, dec_dst = nested_coder.decode_nested_dcsbm(
            lazy, len(hierarchy[0]))
        for got, want in zip(dec_h, hierarchy):
            assert np.array_equal(got, want)
        assert sorted(zip(dec_src.tolist(), dec_dst.tolist())) == \
            sorted(zip(src.tolist(), dst.tolist()))

    def test_realized_within_acceptance_tiny(self):
        _, _, hierarchy, src, dst = _nested_case(12, sizes=(8, 4, 2))
        message = nested_coder.encode_nested_dcsbm(src, dst, hierarchy)
        rep = message["report"]
        assert rep["realized_bits"] - rep["ideal_bits"] <= _message_slack(
            message)
        assert rep["overhead_bits"] == pytest.approx(
            rep["realized_bits"] - rep["ideal_bits"], rel=1e-12)
