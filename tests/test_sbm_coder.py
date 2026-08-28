"""Parity-test-driven coder for graph-tool's microcanonical DL terms.

Graph-tool is the
sole formula authority — every sub-code's ideal bits must match
`czip.sbm._term` on synthetic graphs; unexplained additive constants are
banned. These tests cover the P1/P2 primitives and per-term codecs on tiny
synthetic graphs only.
"""

import math

import numpy as np
import pytest

import constriction
gt = pytest.importorskip("graph_tool.all")

from czip import sbm_coder
from czip.sbm import _term, nats_to_bits


def _tiny_state(seed: int, n_lo=10, n_hi=40, m_lo=20, m_hi=120, b_hi=6):
    """Random tiny directed multigraph (self-loops + parallel edges possible)
    with a random dense partition, as a deg-corr BlockState."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(n_lo, n_hi))
    m = int(rng.integers(m_lo, m_hi))
    src = rng.integers(0, n, m)
    dst = rng.integers(0, n, m)
    g = gt.Graph(directed=True)
    g.add_vertex(n)
    g.add_edge_list(np.column_stack([src, dst]))
    labels = rng.integers(0, int(rng.integers(2, b_hi)), n)
    _, labels = np.unique(labels, return_inverse=True)
    b = g.new_vertex_property("int", vals=labels)
    state = gt.BlockState(g, b=b, deg_corr=True)
    return state, g, labels, src, dst


# ---------------------------------------------------------------------------
# P1 — exact big-int rank stream (mixed radix)
# ---------------------------------------------------------------------------

class TestRankStream:
    def test_roundtrip_preserves_ranks(self):
        enc = sbm_coder.RankEncoder()
        factors = [(3, 7), (0, 1), (12, 100), (5, 6), (0, 2)]
        for rank, n in factors:
            enc.append(rank, n)
        payload = enc.to_bytes()
        dec = sbm_coder.RankDecoder(payload)
        for rank, n in factors:
            assert dec.pop(n) == rank

    def test_ideal_bits_is_log2_product_of_choices(self):
        enc = sbm_coder.RankEncoder()
        for rank, n in [(1, 5), (2, 3), (999, 10**6)]:
            enc.append(rank, n)
        expected = math.log2(5) + math.log2(3) + math.log2(10**6)
        assert enc.ideal_bits == pytest.approx(expected, rel=1e-12)

    def test_realized_bits_within_one_byte_of_ideal(self):
        enc = sbm_coder.RankEncoder()
        enc.append(4, 5)
        enc.append(100, 101)
        payload = enc.to_bytes()
        assert 8 * len(payload) >= enc.ideal_bits
        assert 8 * len(payload) - enc.ideal_bits < 8 + 1  # byte pad + <1 bit

    def test_rank_out_of_range_raises(self):
        enc = sbm_coder.RankEncoder()
        with pytest.raises(ValueError):
            enc.append(5, 5)
        with pytest.raises(ValueError):
            enc.append(-1, 5)

    def test_empty_stream_is_zero_bits(self):
        enc = sbm_coder.RankEncoder()
        assert enc.ideal_bits == 0.0
        assert enc.to_bytes() == b""

    def test_huge_factor_uses_exact_bigint(self):
        # N = 500! is far beyond float range; ranks must stay exact.
        n_choices = math.factorial(500)
        rank = n_choices - 1
        enc = sbm_coder.RankEncoder()
        enc.append(rank, n_choices)
        dec = sbm_coder.RankDecoder(enc.to_bytes())
        assert dec.pop(n_choices) == rank


# ---------------------------------------------------------------------------
# Combinatorial rank/unrank bijections
# ---------------------------------------------------------------------------

class TestWeakCompositionRank:
    """Weak compositions of total t into k nonneg parts <-> [0, C(t+k-1, t))."""

    def test_bijection_exhaustive_small(self):
        t, k = 5, 3
        n = math.comb(t + k - 1, t)
        seen = set()
        for r in range(n):
            comp = sbm_coder.weak_composition_unrank(r, t, k)
            assert len(comp) == k and sum(comp) == t and min(comp) >= 0
            assert sbm_coder.weak_composition_rank(comp, t) == r
            seen.add(tuple(comp))
        assert len(seen) == n

    def test_rank_range(self):
        comp = [7, 0, 3, 2]
        r = sbm_coder.weak_composition_rank(comp, 12)
        assert 0 <= r < math.comb(12 + 4 - 1, 12)

    def test_single_part(self):
        assert sbm_coder.weak_composition_rank([9], 9) == 0
        assert sbm_coder.weak_composition_unrank(0, 9, 1) == [9]


class TestUrnCode:
    """P2 — decrementing-urn categorical: uniform over arrangements of a
    known multiset; ideal bits = log2( n! / prod n_r! )."""

    def test_roundtrip_is_lossless(self):
        rng = np.random.default_rng(0)
        counts = np.array([5, 3, 9, 1])
        labels = rng.permutation(np.repeat(np.arange(4), counts))
        words, report = sbm_coder.urn_encode(labels, counts)
        decoded = sbm_coder.urn_decode(words, counts)
        assert np.array_equal(decoded, labels)

    def test_ideal_bits_is_log2_multinomial(self):
        counts = np.array([4, 2, 6])
        labels = np.repeat(np.arange(3), counts)
        _, report = sbm_coder.urn_encode(labels, counts)
        expected = math.log2(
            math.factorial(12)
            // (math.factorial(4) * math.factorial(2) * math.factorial(6))
        )
        assert report["ideal_bits"] == pytest.approx(expected, rel=1e-12)

    def test_realized_bits_close_to_ideal(self):
        rng = np.random.default_rng(1)
        counts = np.array([300, 200, 500])
        labels = rng.permutation(np.repeat(np.arange(3), counts))
        _, report = sbm_coder.urn_encode(labels, counts)
        # design acceptance: <= 32 * n_streams + 3e-4 * n_symbols
        assert report["realized_bits"] - report["ideal_bits"] <= 32 + 3e-4 * 1000

    def test_medium_scale_roundtrip(self):
        # ge1-shaped stress (chunked vectorized encode path): n=60k, k=400
        rng = np.random.default_rng(5)
        k = 400
        counts = rng.multinomial(60_000, np.ones(k) / k)
        counts = np.maximum(counts, 1)
        labels = rng.permutation(np.repeat(np.arange(k), counts))
        words, report = sbm_coder.urn_encode(labels, counts)
        decoded = sbm_coder.urn_decode(words, counts)
        assert np.array_equal(decoded, labels)
        assert (report["realized_bits"] - report["ideal_bits"]
                <= 32 + 3e-4 * report["n_symbols"])

    def test_counts_mismatch_raises(self):
        with pytest.raises(ValueError):
            sbm_coder.urn_encode(np.array([0, 0, 1]), np.array([1, 1]))

    def test_empty_sequence(self):
        words, report = sbm_coder.urn_encode(
            np.array([], dtype=np.int64), np.array([], dtype=np.int64))
        assert report["ideal_bits"] == 0.0
        assert np.array_equal(
            sbm_coder.urn_decode(words, np.array([], dtype=np.int64)),
            np.array([], dtype=np.int64))


class TestCompositionWalk:
    """Scalable composition codec: stars-and-bars slot walk coded as
    sequential Bernoulli draws P(bar) = bars_left / slots_left. The product
    telescopes to 1 / C(total + k - 1, k - 1), so ideal bits equal the same
    multiset count the exact rank code realizes — used above
    COMPOSITION_RANK_MAX_TOTAL where big-int ranking is infeasible."""

    def test_walk_bits_match_multiset_count(self):
        rng = np.random.default_rng(0)
        for _ in range(5):
            k = int(rng.integers(2, 40))
            comp = rng.multinomial(int(rng.integers(0, 3000)), np.ones(k) / k)
            total = int(comp.sum())
            _, report = sbm_coder.composition_walk_encode(comp)
            expected = math.log2(math.comb(total + k - 1, total)) if total else 0.0
            assert report["walk_bits"] == pytest.approx(expected, rel=1e-9, abs=1e-9)
            assert report["ideal_bits"] == pytest.approx(expected, rel=1e-9, abs=1e-9)

    def test_roundtrip_random(self):
        rng = np.random.default_rng(1)
        for _ in range(5):
            k = int(rng.integers(2, 300))
            comp = rng.multinomial(int(rng.integers(1, 5000)), np.ones(k) / k)
            words, _ = sbm_coder.composition_walk_encode(comp)
            decoded = sbm_coder.composition_walk_decode(
                words, int(comp.sum()), k)
            assert np.array_equal(decoded, comp)

    def test_edge_cases(self):
        for comp in ([7], [0, 0, 0], [0, 5, 0, 0, 2]):
            comp = np.array(comp, dtype=np.int64)
            words, report = sbm_coder.composition_walk_encode(comp)
            decoded = sbm_coder.composition_walk_decode(
                words, int(comp.sum()), len(comp))
            assert np.array_equal(decoded, comp)

    def test_realized_within_acceptance(self):
        rng = np.random.default_rng(2)
        k = 256
        comp = rng.multinomial(20000, np.ones(k) / k)
        _, report = sbm_coder.composition_walk_encode(comp)
        assert (report["realized_bits"] - report["ideal_bits"]
                <= 32 + 3e-4 * report["n_symbols"])


class TestPositiveCompositionRank:
    """Compositions of n into B positive parts <-> [0, C(n-1, B-1))."""

    def test_bijection_exhaustive_small(self):
        n, b = 7, 3
        count = math.comb(n - 1, b - 1)
        seen = set()
        for r in range(count):
            comp = sbm_coder.positive_composition_unrank(r, n, b)
            assert len(comp) == b and sum(comp) == n and min(comp) >= 1
            assert sbm_coder.positive_composition_rank(comp) == r
            seen.add(tuple(comp))
        assert len(seen) == count


# ---------------------------------------------------------------------------
# Term codec: partition_dl parity (graph-tool is the formula authority)
# ---------------------------------------------------------------------------

class TestPartitionCodec:
    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_ideal_bits_matches_graph_tool_partition_dl(self, seed):
        state, g, labels, _, _ = _tiny_state(seed)
        expected = nats_to_bits(_term(state, partition_dl=True))
        assert sbm_coder.partition_ideal_bits(labels) == pytest.approx(
            expected, rel=1e-10)

    def test_roundtrip_recovers_labels(self):
        _, _, labels, _, _ = _tiny_state(5)
        rank_payload, words, report = sbm_coder.encode_partition(labels)
        decoded = sbm_coder.decode_partition(rank_payload, words, len(labels))
        assert np.array_equal(decoded, labels)
        assert report["lossless"] is True

    def test_realized_within_acceptance(self):
        # design criterion: realized - ideal <= 32*n_streams + 3e-4*n_symbols
        _, _, labels, _, _ = _tiny_state(6)
        _, _, report = sbm_coder.encode_partition(labels)
        slack = 32 * 2 + 8 + 3e-4 * len(labels)  # 2 streams + byte pad
        assert report["realized_bits"] - report["ideal_bits"] <= slack

    def test_single_block_partition(self):
        labels = np.zeros(12, dtype=np.int64)
        rank_payload, words, report = sbm_coder.encode_partition(labels)
        decoded = sbm_coder.decode_partition(rank_payload, words, 12)
        assert np.array_equal(decoded, labels)


# ---------------------------------------------------------------------------
# Term codec: edges_dl parity (e_rs matrix as a weak composition over B^2)
# ---------------------------------------------------------------------------

class TestEdgesCodec:
    @staticmethod
    def _ers(labels, src, dst):
        B = int(labels.max()) + 1
        ers = np.zeros((B, B), dtype=np.int64)
        np.add.at(ers, (labels[src], labels[dst]), 1)
        return ers

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_ideal_bits_matches_graph_tool_edges_dl(self, seed):
        state, _, labels, src, dst = _tiny_state(seed)
        ers = self._ers(labels, src, dst)
        expected = nats_to_bits(_term(state, edges_dl=True))
        assert sbm_coder.edges_ideal_bits(ers) == pytest.approx(
            expected, rel=1e-10)

    def test_roundtrip_recovers_ers(self):
        _, _, labels, src, dst = _tiny_state(7)
        ers = self._ers(labels, src, dst)
        payload, report = sbm_coder.encode_edges(ers)
        decoded = sbm_coder.decode_edges(payload, ers.shape[0],
                                         int(ers.sum()))
        assert np.array_equal(decoded, ers)
        assert report["lossless"] is True

    def test_realized_within_byte_of_ideal(self):
        _, _, labels, src, dst = _tiny_state(8)
        ers = self._ers(labels, src, dst)
        payload, report = sbm_coder.encode_edges(ers)
        assert report["realized_bits"] - report["ideal_bits"] < 9

    def test_walk_path_roundtrip_medium(self):
        # e above COMPOSITION_RANK_MAX_TOTAL: slot-walk stream takes over
        rng = np.random.default_rng(3)
        B, e = 24, 50_000
        assert e > sbm_coder.COMPOSITION_RANK_MAX_TOTAL
        ers = rng.multinomial(e, np.ones(B * B) / (B * B)).reshape(B, B)
        payload, report = sbm_coder.encode_edges(ers)
        assert report["lossless"] is True
        decoded = sbm_coder.decode_edges(payload, B, e)
        assert np.array_equal(decoded, ers)
        assert report["ideal_bits"] == pytest.approx(
            sbm_coder.edges_ideal_bits(ers), rel=1e-12)
        assert (report["realized_bits"] - report["ideal_bits"]
                <= 32 + 3e-4 * report["n_symbols"])


# ---------------------------------------------------------------------------
# Term codec: degree_dl (uniform kind) parity — per-block stub compositions
# ---------------------------------------------------------------------------

class TestDegreeUniformCodec:
    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_ideal_bits_matches_graph_tool_degree_dl_uniform(self, seed):
        state, _, labels, src, dst = _tiny_state(seed)
        n = len(labels)
        kout = np.bincount(src, minlength=n)
        kin = np.bincount(dst, minlength=n)
        expected = nats_to_bits(
            _term(state, degree_dl=True, degree_dl_kind="uniform"))
        assert sbm_coder.degrees_uniform_ideal_bits(
            kout, kin, labels) == pytest.approx(expected, rel=1e-10)

    def test_roundtrip_recovers_degrees(self):
        _, _, labels, src, dst = _tiny_state(9)
        n = len(labels)
        kout = np.bincount(src, minlength=n)
        kin = np.bincount(dst, minlength=n)
        ers = TestEdgesCodec._ers(labels, src, dst)
        payload, report = sbm_coder.encode_degrees_uniform(kout, kin, labels)
        dec_out, dec_in = sbm_coder.decode_degrees_uniform(
            payload, labels, ers)
        assert np.array_equal(dec_out, kout)
        assert np.array_equal(dec_in, kin)
        assert report["lossless"] is True

    def test_walk_path_roundtrip_medium(self):
        # total stubs above COMPOSITION_RANK_MAX_TOTAL: slot-walk stream
        rng = np.random.default_rng(4)
        n, B, e = 1200, 6, 30_000
        assert e > sbm_coder.COMPOSITION_RANK_MAX_TOTAL
        labels = np.sort(rng.integers(0, B, size=n))
        labels[:B] = np.arange(B)
        labels = np.sort(labels)
        src = rng.integers(0, n, size=e)
        dst = rng.integers(0, n, size=e)
        kout = np.bincount(src, minlength=n)
        kin = np.bincount(dst, minlength=n)
        ers = TestEdgesCodec._ers(labels, src, dst)
        payload, report = sbm_coder.encode_degrees_uniform(kout, kin, labels)
        assert report["lossless"] is True
        dec_out, dec_in = sbm_coder.decode_degrees_uniform(
            payload, labels, ers)
        assert np.array_equal(dec_out, kout)
        assert np.array_equal(dec_in, kin)
        assert report["ideal_bits"] == pytest.approx(
            sbm_coder.degrees_uniform_ideal_bits(kout, kin, labels),
            rel=1e-12)
        assert (report["realized_bits"] - report["ideal_bits"]
                <= 32 + 3e-4 * report["n_symbols"])


# ---------------------------------------------------------------------------
# Term codec: adjacency (deg-corr, directed) — sequential hypergeometric
# stub-composition code; parity vs graph-tool's adjacency entropy
# ---------------------------------------------------------------------------

class TestAdjacencyCodec:
    @staticmethod
    def _arrays(labels, src, dst):
        n = len(labels)
        kout = np.bincount(src, minlength=n)
        kin = np.bincount(dst, minlength=n)
        B = int(labels.max()) + 1
        ers = np.zeros((B, B), dtype=np.int64)
        np.add.at(ers, (labels[src], labels[dst]), 1)
        return kout, kin, ers

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_ideal_bits_matches_graph_tool_adjacency(self, seed):
        state, _, labels, src, dst = _tiny_state(seed)
        kout, kin, ers = self._arrays(labels, src, dst)
        expected = nats_to_bits(state.entropy(adjacency=True, dl=False))
        assert sbm_coder.adjacency_ideal_bits(
            src, dst, labels, kout, kin, ers) == pytest.approx(
            expected, rel=1e-9)

    @pytest.mark.parametrize("seed", [5, 6, 7])
    def test_roundtrip_recovers_edge_multiset(self, seed):
        _, _, labels, src, dst = _tiny_state(seed)
        kout, kin, ers = self._arrays(labels, src, dst)
        words, report = sbm_coder.encode_adjacency(
            src, dst, labels, kout, kin, ers)
        dec_src, dec_dst = sbm_coder.decode_adjacency(
            words, labels, kout, kin, ers)
        orig = sorted(zip(src.tolist(), dst.tolist()))
        got = sorted(zip(dec_src.tolist(), dec_dst.tolist()))
        assert got == orig
        assert report["lossless"] is True

    def test_realized_within_acceptance(self):
        _, _, labels, src, dst = _tiny_state(8)
        kout, kin, ers = self._arrays(labels, src, dst)
        words, report = sbm_coder.encode_adjacency(
            src, dst, labels, kout, kin, ers)
        slack = 32 + 8 + 3e-4 * report["n_symbols"] + 1
        assert report["realized_bits"] - report["ideal_bits"] <= slack

    def test_medium_scale_parity_and_roundtrip(self):
        # scaling path: grouped per-block-pair iteration must stay exact
        rng = np.random.default_rng(13)
        n, B, e = 1500, 8, 12_000
        labels = np.sort(rng.integers(0, B, size=n))
        labels[:B] = np.arange(B)
        labels = np.sort(labels)
        src = rng.integers(0, n, size=e)
        dst = rng.integers(0, n, size=e)
        kout, kin, ers = self._arrays(labels, src, dst)
        g = gt.Graph(directed=True)
        g.add_vertex(n)
        g.add_edge_list(np.column_stack([src, dst]))
        state = gt.BlockState(
            g, b=g.new_vertex_property("int", vals=labels), deg_corr=True)
        expected = nats_to_bits(state.entropy(adjacency=True, dl=False))
        assert sbm_coder.adjacency_ideal_bits(
            src, dst, labels, kout, kin, ers) == pytest.approx(
            expected, rel=1e-9)
        words, report = sbm_coder.encode_adjacency(
            src, dst, labels, kout, kin, ers)
        assert report["lossless"] is True
        assert (report["realized_bits"] - report["ideal_bits"]
                <= 32 + 3e-4 * report["n_symbols"] + 1)

    def test_parallel_edges_and_self_loops_covered(self):
        # deterministic multigraph with explicit parallel edges + self-loops
        labels = np.array([0, 0, 1, 1, 1], dtype=np.int64)
        src = np.array([0, 0, 0, 2, 2, 4, 1, 3, 3], dtype=np.int64)
        dst = np.array([1, 1, 0, 2, 3, 4, 4, 3, 0], dtype=np.int64)
        kout, kin, ers = self._arrays(labels, src, dst)
        g = gt.Graph(directed=True)
        g.add_vertex(5)
        g.add_edge_list(np.column_stack([src, dst]))
        state = gt.BlockState(
            g, b=g.new_vertex_property("int", vals=labels), deg_corr=True)
        expected = nats_to_bits(state.entropy(adjacency=True, dl=False))
        assert sbm_coder.adjacency_ideal_bits(
            src, dst, labels, kout, kin, ers) == pytest.approx(
            expected, rel=1e-9)
        words, _ = sbm_coder.encode_adjacency(
            src, dst, labels, kout, kin, ers)
        dec_src, dec_dst = sbm_coder.decode_adjacency(
            words, labels, kout, kin, ers)
        assert sorted(zip(dec_src.tolist(), dec_dst.tolist())) == \
            sorted(zip(src.tolist(), dst.tolist()))

    def test_quantization_gap_zero_on_typical_data(self):
        # model-typical draws never hit the ~2^-23 categorical floor
        _, _, labels, src, dst = _tiny_state(9)
        kout, kin, ers = self._arrays(labels, src, dst)
        _, report = sbm_coder.encode_adjacency(
            src, dst, labels, kout, kin, ers)
        assert report["n_floored_draws"] == 0
        assert report["tail_gap_bits"] == 0.0

    def test_quantization_gap_accumulates_below_floor(self, monkeypatch):
        # lowering the floor makes ordinary draws trip the diagnostic; the
        # ideal itemization must be unaffected (REPORT-only)
        _, _, labels, src, dst = _tiny_state(9)
        kout, kin, ers = self._arrays(labels, src, dst)
        ideal_before = sbm_coder.adjacency_ideal_bits(
            src, dst, labels, kout, kin, ers)
        monkeypatch.setattr(sbm_coder, "QUANT_FLOOR_BITS", 0.5)
        _, report = sbm_coder.encode_adjacency(
            src, dst, labels, kout, kin, ers)
        assert report["n_floored_draws"] > 0
        assert report["tail_gap_bits"] > 0.0
        assert report["ideal_bits"] == pytest.approx(ideal_before, rel=1e-12)

    def test_underflow_tail_draw_finite_bits(self):
        # A draw >~1074 bits below the row mode underflows np.exp to an exact
        # float 0.0 in _hyper_row_norm (hit for real on dcsbm_ge1, 2026-08-13):
        # the ideal accumulation must fall back to log-space bits, not crash,
        # and constriction must still round-trip the symbol (its quantizer
        # floors every symbol to nonzero weight).
        k, K, m = 2000, 4000, 2000
        lo, hi, W = 0, 2000, 2001
        value = 2000  # all m draws marked: -log2 P = log2 C(4000, 2000)
        row = sbm_coder._hyper_row_norm(k, K, m, lo, hi, W)
        assert row[value] == 0.0  # the underflow precondition
        exact = (math.lgamma(K + 1) - 2 * math.lgamma(k + 1)) / math.log(2)
        bits = sbm_coder._hyper_bits_exact(k, K, m, lo, hi, value)
        assert math.isfinite(bits)
        assert bits == pytest.approx(exact, rel=1e-9)
        enc = constriction.stream.queue.RangeEncoder()
        enc.encode(np.array([value], dtype=np.int32),
                   sbm_coder._CATEGORICAL_FAMILY, row[None, :])
        dec = constriction.stream.queue.RangeDecoder(enc.get_compressed())
        assert int(dec.decode(sbm_coder._CATEGORICAL_FAMILY,
                              row[None, :])[0]) == value


# ---------------------------------------------------------------------------
# Full-message assembly: flat DC-SBM round-trip on a whole tiny graph
# (degree term coded under degree_dl_kind='uniform'; the distributed-kind
#  delta is reported, not coded, at real-data time)
# ---------------------------------------------------------------------------

class TestFullMessage:
    @staticmethod
    def _uniform_kind_total_bits(state):
        return nats_to_bits(
            state.entropy(adjacency=True, dl=False)
            + _term(state, partition_dl=True)
            + _term(state, edges_dl=True)
            + _term(state, degree_dl=True, degree_dl_kind="uniform"))

    @pytest.mark.parametrize("seed", [0, 3, 7])
    def test_roundtrip_recovers_graph_and_partition(self, seed):
        state, _, labels, src, dst = _tiny_state(seed)
        message = sbm_coder.encode_dcsbm(src, dst, labels)
        dec_labels, dec_src, dec_dst = sbm_coder.decode_dcsbm(
            message, len(labels))
        assert np.array_equal(dec_labels, labels)
        assert sorted(zip(dec_src.tolist(), dec_dst.tolist())) == \
            sorted(zip(src.tolist(), dst.tolist()))

    @pytest.mark.parametrize("seed", [0, 3, 7])
    def test_ideal_total_matches_graph_tool_uniform_kind(self, seed):
        state, _, labels, src, dst = _tiny_state(seed)
        message = sbm_coder.encode_dcsbm(src, dst, labels)
        expected = self._uniform_kind_total_bits(state)
        assert message["report"]["ideal_bits"] == pytest.approx(
            expected, rel=1e-9)

    def test_realized_total_within_acceptance(self):
        # design criterion: realized - ideal <= 32*n_streams
        # + 3e-4*n_symbols, plus the explicit header and byte padding
        state, _, labels, src, dst = _tiny_state(11)
        message = sbm_coder.encode_dcsbm(src, dst, labels)
        rep = message["report"]
        slack = (rep["header_bits"] + 32 * 2 + 8 * 3 + 1
                 + 3e-4 * rep["n_symbols"])
        assert rep["realized_bits"] - rep["ideal_bits"] <= slack

    def test_medium_scale_total_parity_and_roundtrip(self):
        rng = np.random.default_rng(14)
        n, B, e = 1500, 8, 12_000
        labels = np.sort(rng.integers(0, B, size=n))
        labels[:B] = np.arange(B)
        labels = np.sort(labels)
        src = rng.integers(0, n, size=e)
        dst = rng.integers(0, n, size=e)
        g = gt.Graph(directed=True)
        g.add_vertex(n)
        g.add_edge_list(np.column_stack([src, dst]))
        state = gt.BlockState(
            g, b=g.new_vertex_property("int", vals=labels), deg_corr=True)
        message = sbm_coder.encode_dcsbm(src, dst, labels)
        rep = message["report"]
        assert rep["lossless"] is True
        assert rep["ideal_bits"] == pytest.approx(
            self._uniform_kind_total_bits(state), rel=1e-9)
        dec_labels, dec_src, dec_dst = sbm_coder.decode_dcsbm(message, n)
        assert np.array_equal(dec_labels, labels)
        assert sorted(zip(dec_src.tolist(), dec_dst.tolist())) == \
            sorted(zip(src.tolist(), dst.tolist()))

    def test_report_is_itemized_and_additive(self):
        _, _, labels, src, dst = _tiny_state(12)
        message = sbm_coder.encode_dcsbm(src, dst, labels)
        rep = message["report"]
        itemized = (rep["L_partition_bits"] + rep["L_degree_bits"]
                    + rep["L_edges_bits"] + rep["L_adjacency_bits"])
        assert rep["ideal_bits"] == pytest.approx(itemized, rel=1e-12)
        # the quantization-gap diagnostic surfaces in the full-message report
        assert rep["n_floored_draws"] == 0
        assert rep["tail_gap_bits"] == 0.0
