"""Tests for czip.czip — the .cz v1 container + CLI.

The container is a dispatcher around the coders (czip.sbm_coder);
losslessness here means decode(encode(G)) reproduces the exact topology
(and, for a weighted container, the exact weights). A topology-only
container carries the DC-SBM payload alone, explicit in the header.
"""

import json

import numpy as np
import pytest
import scipy.sparse as sp

from czip import czip


def _tiny_graph(seed=0, n=40, B=3, e=220):
    """Simple directed graph (no self-loops, no multi-edges) + partition."""
    rng = np.random.default_rng(seed)
    labels = np.sort(rng.integers(0, B, size=n))
    labels[:B] = np.arange(B)
    labels = np.sort(labels)
    pairs = set()
    while len(pairs) < e:
        s, d = rng.integers(0, n, size=2)
        if s != d:
            pairs.add((int(s), int(d)))
    src, dst = map(np.asarray, zip(*sorted(pairs)))
    adj = sp.csr_matrix((np.ones(e, dtype=np.int64), (src, dst)),
                        shape=(n, n))
    return adj, labels.astype(np.int64)


class TestContainer:
    def test_pack_unpack_roundtrip(self):
        header = {"model_id": "dcsbm", "n_nodes": 7}
        streams = {"a": b"\x01\x02\xff",
                   "b": np.array([3, 1 << 31, 0], dtype=np.uint32),
                   "c": b""}
        blob = czip.pack(header, streams)
        hdr, out = czip.unpack(blob)
        assert hdr["model_id"] == "dcsbm" and hdr["n_nodes"] == 7
        assert out["a"] == streams["a"]
        assert out["b"].dtype == np.uint32
        assert np.array_equal(out["b"], streams["b"])
        assert out["c"] == b""

    def test_magic_prefix(self):
        assert czip.pack({}, {}).startswith(b"CZIP")

    def test_bad_magic_rejected(self):
        blob = b"NOPE" + czip.pack({}, {})[4:]
        with pytest.raises(ValueError, match="magic"):
            czip.unpack(blob)

    def test_future_version_rejected(self):
        blob = bytearray(czip.pack({}, {}))
        blob[4:6] = (czip.FORMAT_VERSION + 1).to_bytes(2, "little")
        with pytest.raises(ValueError, match="version"):
            czip.unpack(bytes(blob))

    def test_truncated_blob_rejected(self):
        blob = czip.pack({"x": 1}, {"s": b"payload-bytes"})
        with pytest.raises(ValueError, match="truncated"):
            czip.unpack(blob[:-4])


class TestEncodeDecode:
    def test_topology_roundtrip_is_lossless(self):
        adj, labels = _tiny_graph()
        blob = czip.encode_topology(adj, labels)
        dec_labels, dec_adj = czip.decode(blob)
        assert np.array_equal(dec_labels, labels)
        assert (dec_adj != adj).nnz == 0

    def test_encode_is_deterministic(self):
        adj, labels = _tiny_graph(seed=5)
        assert czip.encode_topology(adj, labels) == \
            czip.encode_topology(adj, labels)

    def test_header_carries_itemization_and_payload_kind(self):
        adj, labels = _tiny_graph(seed=2)
        blob = czip.encode_topology(adj, labels)
        hdr, _ = czip.unpack(blob)
        assert hdr["format_version"] == czip.FORMAT_VERSION
        assert hdr["model_id"] == "dcsbm"
        assert hdr["payload"] == "topology"
        assert hdr["n_nodes"] == adj.shape[0]
        assert hdr["n_edges"] == adj.nnz
        rep = hdr["report"]
        for key in ("ideal_bits", "realized_bits", "L_partition_bits",
                    "L_edges_bits", "L_degree_bits", "L_adjacency_bits"):
            assert key in rep
        # container overhead itemized and accounts for the full blob size
        assert hdr["container_overhead_bits"] > 0
        assert 8 * len(blob) == pytest.approx(
            rep["realized_bits"] + hdr["container_overhead_bits"])

    def test_decode_verifies_source_digest(self):
        adj, labels = _tiny_graph(seed=3)
        blob = czip.encode_topology(adj, labels)
        hdr, streams = czip.unpack(blob)
        hdr["source_digest"] = "0" * 64
        with pytest.raises(ValueError, match="digest"):
            czip.decode(czip.pack(hdr, streams))

    def test_non_dense_labels_are_densified(self):
        # graph-tool fits store raw non-contiguous block labels; czip must
        # densify them (an order-preserving mapping to 0..B-1)
        adj, labels = _tiny_graph(seed=8)
        sparse_labels = labels * 10 + 3  # gaps, non-zero-based
        blob = czip.encode_topology(adj, sparse_labels)
        dec_labels, dec_adj = czip.decode(blob)
        assert np.array_equal(dec_labels, labels)
        assert (dec_adj != adj).nnz == 0

    def test_weighted_input_rejected_without_topology_only(self):
        adj, labels = _tiny_graph(seed=4)
        adj = adj.copy()
        adj.data[:] = 7  # synapse counts > 1
        with pytest.raises(ValueError, match="topology"):
            czip.encode_topology(adj, labels)
        blob = czip.encode_topology(adj, labels, allow_weight_drop=True)
        _, dec_adj = czip.decode(blob)
        assert (dec_adj != (adj > 0).astype(np.int64)).nnz == 0


def _tiny_weighted_graph(seed=0, wmin=1, **kw):
    adj, labels = _tiny_graph(seed=seed, **kw)
    rng = np.random.default_rng(seed + 1000)
    adj = adj.copy()
    adj.data = wmin + rng.geometric(0.4, size=adj.nnz).astype(np.int64) - 1
    return adj, labels


class TestEncodeWeighted:
    def test_weighted_roundtrip_is_lossless(self):
        adj, labels = _tiny_weighted_graph()
        blob = czip.encode_weighted(adj, labels, wmin=1)
        dec_labels, dec_adj = czip.decode(blob)
        assert np.array_equal(dec_labels, labels)
        assert (dec_adj != adj).nnz == 0
        assert dec_adj.data.dtype == adj.data.dtype

    def test_weighted_roundtrip_wmin5(self):
        adj, labels = _tiny_weighted_graph(seed=9, wmin=5)
        blob = czip.encode_weighted(adj, labels, wmin=5)
        _, dec_adj = czip.decode(blob)
        assert (dec_adj != adj).nnz == 0

    def test_weighted_header(self):
        adj, labels = _tiny_weighted_graph(seed=2)
        blob = czip.encode_weighted(adj, labels, wmin=1)
        hdr, _ = czip.unpack(blob)
        assert hdr["model_id"] == "dcsbm+weights"
        assert hdr["payload"] == "graph"
        assert hdr["wmin"] == 1
        rep = hdr["report"]
        assert "weights" in rep and "topology" in rep
        assert rep["weights"]["L_total_bits"] > 0

    def test_weighted_encode_is_deterministic(self):
        adj, labels = _tiny_weighted_graph(seed=3)
        assert czip.encode_weighted(adj, labels, wmin=1) == \
            czip.encode_weighted(adj, labels, wmin=1)

    def test_weight_below_wmin_rejected(self):
        adj, labels = _tiny_weighted_graph(seed=4, wmin=1)
        with pytest.raises(ValueError, match="wmin"):
            czip.encode_weighted(adj, labels, wmin=5)

    def test_digest_covers_weights(self):
        # two graphs with same topology but different weights must have
        # different source digests (losslessness self-check sees weights)
        adj, labels = _tiny_weighted_graph(seed=5)
        adj2 = adj.copy()
        adj2.data = adj2.data + 1
        h1, _ = czip.unpack(czip.encode_weighted(adj, labels, wmin=1))
        h2, _ = czip.unpack(czip.encode_weighted(adj2, labels, wmin=1))
        assert h1["source_digest"] != h2["source_digest"]


class TestCli:
    def test_cli_encode_decode_info(self, tmp_path, capsys):
        adj, labels = _tiny_graph(seed=6)
        npz, part = tmp_path / "g.npz", tmp_path / "p.npy"
        sp.save_npz(npz, adj)
        np.save(part, labels)
        cz, out = tmp_path / "g.cz", tmp_path / "dec.npz"

        assert czip.main(["encode", str(npz), "--partition", str(part),
                          "-o", str(cz)]) == 0
        assert czip.main(["info", str(cz)]) == 0
        info = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
        assert info["model_id"] == "dcsbm"
        assert czip.main(["decode", str(cz), "-o", str(out)]) == 0
        dec = sp.load_npz(out)
        assert (dec != adj).nnz == 0

    def test_cli_weighted_roundtrip(self, tmp_path):
        adj, labels = _tiny_weighted_graph(seed=7, wmin=5)
        npz, part = tmp_path / "g.npz", tmp_path / "p.npy"
        sp.save_npz(npz, adj)
        np.save(part, labels)
        cz, out = tmp_path / "g.cz", tmp_path / "dec.npz"
        assert czip.main(["encode", str(npz), "--partition", str(part),
                          "--wmin", "5", "-o", str(cz)]) == 0
        assert czip.main(["decode", str(cz), "-o", str(out)]) == 0
        assert (sp.load_npz(out) != adj).nnz == 0


# ---------------------------------------------------------------------------
# Nested DC-SBM payload: the hierarchy replaces the flat
# partition as the model, czip.nested_coder replaces czip.sbm_coder as the coder,
# and the container dispatches on model_id exactly as before.
# ---------------------------------------------------------------------------

def _tiny_hierarchy(labels, top_split=(0, 1, 0)):
    """Canonical 3-level hierarchy over a base partition with B0 = 3."""
    mid = np.asarray(top_split, dtype=np.int64)
    assert mid.shape[0] == int(labels.max()) + 1
    return [np.asarray(labels, dtype=np.int64), mid,
            np.zeros(int(mid.max()) + 1, dtype=np.int64)]


def _tiny_nested(seed=0, top_split=(0, 1, 0), **kw):
    adj, labels = _tiny_graph(seed=seed, **kw)
    return adj, _tiny_hierarchy(labels, top_split)


def _tiny_nested_weighted(seed=0, wmin=1, top_split=(0, 1, 0), **kw):
    adj, labels = _tiny_weighted_graph(seed=seed, wmin=wmin, **kw)
    return adj, _tiny_hierarchy(labels, top_split)


def _padded(hierarchy, n):
    """graph-tool's get_bs() shape: every level padded to n with junk."""
    out = [np.asarray(hierarchy[0], dtype=np.int64)]
    for lab in hierarchy[1:]:
        pad = np.full(n, 999, dtype=np.int64)
        pad[:lab.shape[0]] = lab
        out.append(pad)
    return out


def _deep_nested(seed=3, n=60, e=300):
    """A four-level hierarchy: 15 message segments, so segment padding alone
    outgrows the flat model's six-segment allowance."""
    adj, _ = _tiny_graph(seed=seed, n=n, B=8, e=e)
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 8, n)
    labels[:8] = np.arange(8)
    labels = np.unique(labels, return_inverse=True)[1].astype(np.int64)
    return adj, [labels,
                 np.array([0, 1, 2, 0, 1, 2, 3, 3], dtype=np.int64),
                 np.array([0, 0, 1, 1], dtype=np.int64),
                 np.zeros(2, dtype=np.int64)]


def _nested_stream_names(L):
    names = ["header", "levels_header"]
    for l in range(L):
        names += [f"level_{l}_partition_rank", f"level_{l}_partition_words"]
    names += [f"level_{l}_expand_payload" for l in range(L - 1, 0, -1)]
    return names + ["degrees_payload", "adjacency_words"]


class TestNestedTopology:
    def test_nested_roundtrip_is_lossless(self):
        adj, hier = _tiny_nested()
        blob = czip.encode_topology(adj, hierarchy=hier)
        dec_labels, dec_adj = czip.decode(blob)
        assert np.array_equal(dec_labels, hier[0])
        assert (dec_adj != adj).nnz == 0

    def test_nested_header(self):
        adj, hier = _tiny_nested(seed=2)
        hdr, _ = czip.unpack(czip.encode_topology(adj, hierarchy=hier))
        assert hdr["format_version"] == czip.FORMAT_VERSION
        assert hdr["model_id"] == "nested-dcsbm"
        assert hdr["payload"] == "topology"
        assert hdr["n_levels"] == 3
        rep = hdr["report"]
        for key in ("ideal_bits", "realized_bits", "L_partition_bits",
                    "L_expansion_bits", "L_degree_bits", "L_adjacency_bits"):
            assert key in rep
        assert 8 * len(czip.encode_topology(adj, hierarchy=hier)) == \
            pytest.approx(rep["realized_bits"]
                          + hdr["container_overhead_bits"])

    def test_nested_stream_table_geometry(self):
        adj, hier = _tiny_nested(seed=3)
        hdr, streams = czip.unpack(czip.encode_topology(adj, hierarchy=hier))
        assert [e["name"] for e in hdr["stream_table"]] == \
            _nested_stream_names(3)
        assert list(streams) == _nested_stream_names(3)

    def test_nested_encode_is_deterministic(self):
        adj, hier = _tiny_nested(seed=4)
        assert czip.encode_topology(adj, hierarchy=hier) == \
            czip.encode_topology(adj, hierarchy=hier)

    def test_raw_padded_hierarchy_gives_the_canonical_container(self):
        adj, hier = _tiny_nested(seed=5)
        raw = _padded(hier, adj.shape[0])
        assert czip.encode_topology(adj, hierarchy=raw) == \
            czip.encode_topology(adj, hierarchy=hier)

    def test_labels_and_hierarchy_are_mutually_exclusive(self):
        adj, hier = _tiny_nested(seed=6)
        with pytest.raises(ValueError, match="hierarchy"):
            czip.encode_topology(adj, hier[0], hierarchy=hier)

    def test_neither_labels_nor_hierarchy_rejected(self):
        adj, _ = _tiny_nested(seed=6)
        with pytest.raises(ValueError, match="hierarchy"):
            czip.encode_topology(adj)

    def test_hostile_n_levels_is_rejected(self):
        # n_levels is a third-party count like n_nodes/n_edges: it
        # drives the stream-name loop, so it is bounded before it allocates.
        # Both ends of 1 <= L <= n, and the message must name the bound —
        # "missing stream(s)" further in is the unbounded path, not this one.
        adj, hier = _tiny_nested(seed=22)
        hdr, streams = czip.unpack(czip.encode_topology(adj, hierarchy=hier))
        for bad in (0, adj.shape[0] + 1):
            hdr["n_levels"] = bad
            with pytest.raises(ValueError, match="at most n levels"):
                czip.decode(czip.pack(hdr, streams))

    def test_weighted_input_rejected_without_topology_only(self):
        adj, hier = _tiny_nested(seed=7)
        adj = adj.copy()
        adj.data[:] = 7
        with pytest.raises(ValueError, match="topology"):
            czip.encode_topology(adj, hierarchy=hier)


class TestNestedParamsDigest:
    def test_upper_level_split_changes_the_params_digest(self):
        # the flat digest sees only the base labels; the nested one must
        # cover every level, or a swapped hierarchy would decode "clean"
        adj, hier = _tiny_nested(seed=8, top_split=(0, 1, 0))
        _, other = _tiny_nested(seed=8, top_split=(0, 1, 1))
        h1, _ = czip.unpack(czip.encode_topology(adj, hierarchy=hier))
        h2, _ = czip.unpack(czip.encode_topology(adj, hierarchy=other))
        assert np.array_equal(hier[0], other[0])
        assert h1["params_digest"] != h2["params_digest"]

    def test_the_level_count_is_part_of_the_digest(self):
        # depth is model, not layout: these two hierarchies serialize to the
        # SAME concatenated level bytes, so a digest over the arrays alone
        # would collide across a change of depth
        deep = [np.array([0, 1, 0, 1], dtype=np.int64),
                np.array([0, 1], dtype=np.int64),
                np.array([0, 0], dtype=np.int64)]
        shallow = [np.array([0, 1, 0, 1, 0, 1], dtype=np.int64),
                   np.array([0, 0], dtype=np.int64)]
        assert (b"".join(lab.astype("<i8").tobytes() for lab in deep)
                == b"".join(lab.astype("<i8").tobytes() for lab in shallow))
        assert czip._params_digest_nested(deep) != \
            czip._params_digest_nested(shallow)

    def test_flipped_hierarchy_digest_fails_nested_decode(self):
        adj, hier = _tiny_nested(seed=9, top_split=(0, 1, 0))
        _, other = _tiny_nested(seed=9, top_split=(0, 1, 1))
        hdr, streams = czip.unpack(czip.encode_topology(adj, hierarchy=hier))
        flipped, _ = czip.unpack(czip.encode_topology(adj, hierarchy=other))
        hdr["params_digest"] = flipped["params_digest"]
        with pytest.raises(ValueError, match="params_digest"):
            czip.decode(czip.pack(hdr, streams))

    def test_missing_params_digest_fails_nested_decode(self):
        adj, hier = _tiny_nested(seed=10)
        hdr, streams = czip.unpack(czip.encode_topology(adj, hierarchy=hier))
        hdr.pop("params_digest")
        with pytest.raises(ValueError, match="params_digest"):
            czip.decode(czip.pack(hdr, streams))


class TestNestedWeighted:
    def test_nested_weighted_roundtrip_is_lossless(self):
        adj, hier = _tiny_nested_weighted()
        blob = czip.encode_weighted(adj, wmin=1, hierarchy=hier)
        dec_labels, dec_adj = czip.decode(blob)
        assert np.array_equal(dec_labels, hier[0])
        assert (dec_adj != adj).nnz == 0

    def test_nested_weighted_header(self):
        adj, hier = _tiny_nested_weighted(seed=11)
        hdr, _ = czip.unpack(czip.encode_weighted(adj, wmin=1,
                                                  hierarchy=hier))
        assert hdr["model_id"] == "nested-dcsbm+weights"
        assert hdr["payload"] == "graph"
        assert hdr["n_levels"] == 3
        assert hdr["wmin"] == 1
        rep = hdr["report"]
        assert "topology" in rep and "weights" in rep
        assert rep["topology"]["L_expansion_bits"] > 0

    def test_nested_weighted_roundtrip_wmin5(self):
        adj, hier = _tiny_nested_weighted(seed=12, wmin=5)
        blob = czip.encode_weighted(adj, wmin=5, hierarchy=hier)
        _, dec_adj = czip.decode(blob)
        assert (dec_adj != adj).nnz == 0

    def test_nested_weighted_is_deterministic(self):
        adj, hier = _tiny_nested_weighted(seed=13)
        assert czip.encode_weighted(adj, wmin=1, hierarchy=hier) == \
            czip.encode_weighted(adj, wmin=1, hierarchy=hier)

    def test_nested_weighted_verify_stamps_lossless(self):
        adj, hier = _tiny_nested_weighted(seed=14)
        blob = czip.encode_weighted(adj, wmin=1, hierarchy=hier)
        hdr, _ = czip.unpack(czip.verify_lossless(blob, adj=adj,
                                                  labels=hier[0]))
        assert hdr["report"]["topology"]["lossless"] is True
        assert hdr["report"]["weights"]["lossless"] is True


class TestStreamCountsDriveTheGate:
    """The gate's padding slack is per-container, derived from the actual
    stream table, not from the flat model's hardcoded 2 word / 4 byte
    streams; old containers that predate the counts keep the constants."""

    def _counts(self, streams):
        words = sum(1 for v in streams.values()
                    if not isinstance(v, (bytes, bytearray)))
        return words, len(streams) - words

    def test_flat_topology_counts_match_the_stream_table(self):
        adj, labels = _tiny_graph(seed=15)
        hdr, streams = czip.unpack(czip.encode_topology(adj, labels))
        rep = hdr["report"]
        assert (rep["n_word_streams"], rep["n_byte_streams"]) == \
            self._counts(streams)
        # the tiny fixture is on the rank branch, so this is still the old
        # (2, 4) shape and the gate window is unchanged
        assert (rep["n_word_streams"], rep["n_byte_streams"]) == (2, 4)

    def test_nested_counts_match_the_stream_table(self):
        adj, hier = _tiny_nested(seed=16)
        hdr, streams = czip.unpack(czip.encode_topology(adj, hierarchy=hier))
        rep = hdr["report"]
        assert (rep["n_word_streams"], rep["n_byte_streams"]) == \
            self._counts(streams)

    def test_nested_topology_gate_is_green(self):
        adj, hier = _tiny_nested(seed=17)
        hdr, _ = czip.unpack(czip.encode_topology(adj, hierarchy=hier))
        gate = hdr["report"]["bits_gate"]
        assert gate["within_slack"] is True
        assert gate["lower_bits"] <= gate["realized_bits"] <= \
            gate["upper_bits"]
        assert czip.bits_gate(hdr)["within_slack"] is True

    def test_nested_weighted_gate_is_green(self):
        adj, hier = _tiny_nested_weighted(seed=18)
        hdr, _ = czip.unpack(czip.encode_weighted(adj, wmin=1,
                                                  hierarchy=hier))
        assert hdr["report"]["topology"]["bits_gate"]["within_slack"] is True
        assert hdr["report"]["weights"]["bits_gate"]["within_slack"] is True

    def test_weights_streams_do_not_widen_the_topology_slack(self):
        # the topology counts must cover the topology segments only
        adj, hier = _tiny_nested_weighted(seed=19)
        hdr, _ = czip.unpack(czip.encode_weighted(adj, wmin=1,
                                                  hierarchy=hier))
        topo = hdr["report"]["topology"]
        n_topo = len(_nested_stream_names(3))
        assert topo["n_word_streams"] + topo["n_byte_streams"] == n_topo

    def test_a_deep_hierarchy_outgrows_the_flat_constant_window(self):
        # why the constants had to go: a four-level message pads 15 segments,
        # and that padding alone exceeds the flat model's 160-bit allowance,
        # so _require_gate would refuse a perfectly calibrated container
        adj, hier = _deep_nested()
        hdr, _ = czip.unpack(czip.encode_topology(adj, hierarchy=hier))
        rep = hdr["report"]
        assert (rep["n_word_streams"], rep["n_byte_streams"]) == (5, 10)
        assert rep["bits_gate"]["within_slack"] is True
        stripped = {k: v for k, v in rep.items()
                    if k not in ("n_word_streams", "n_byte_streams")}
        assert czip.bits_gate({"report": stripped})["within_slack"] is False

    def test_present_counts_size_the_window_for_this_container(self):
        # the companion to the fallback test below: when the counts ARE
        # present the window is the same formula over THEM, so the deep
        # container's window is its own (5, 10) one and not the flat (2, 4)
        adj, hier = _deep_nested()
        hdr, _ = czip.unpack(czip.encode_topology(adj, hierarchy=hier))
        rep = hdr["report"]
        assert (rep["n_word_streams"], rep["n_byte_streams"]) != \
            (czip._TOPOLOGY_WORD_STREAMS, czip._TOPOLOGY_BYTE_STREAMS)
        assert czip.bits_gate(hdr)["topology"]["slack_budget_bits"] == \
            pytest.approx(32.0 * 2 * rep["n_word_streams"]
                          + 8.0 * rep["n_byte_streams"]
                          + czip.GATE_DRIFT_BITS_PER_SYMBOL
                          * rep["n_symbols"])

    def test_absent_counts_fall_back_to_the_flat_constants(self):
        adj, labels = _tiny_graph(seed=20)
        hdr, _ = czip.unpack(czip.encode_topology(adj, labels))
        with_counts = czip.bits_gate(hdr)["topology"]["slack_budget_bits"]
        hdr["report"].pop("n_word_streams")
        hdr["report"].pop("n_byte_streams")
        fallback = czip.bits_gate(hdr)["topology"]["slack_budget_bits"]
        assert fallback == with_counts
        assert fallback == pytest.approx(
            32.0 * 2 * czip._TOPOLOGY_WORD_STREAMS
            + 8.0 * czip._TOPOLOGY_BYTE_STREAMS
            + czip.GATE_DRIFT_BITS_PER_SYMBOL
            * hdr["report"]["n_symbols"])


class TestNestedCli:
    def _save_hierarchy(self, path, hierarchy):
        np.savez(path, **{f"level_{i}": np.asarray(lab, dtype=np.int64)
                          for i, lab in enumerate(hierarchy)})

    def test_cli_hierarchy_encode_decode_info(self, tmp_path, capsys):
        adj, hier = _tiny_nested(seed=21)
        npz, hpath = tmp_path / "g.npz", tmp_path / "h.npz"
        sp.save_npz(npz, adj)
        self._save_hierarchy(hpath, _padded(hier, adj.shape[0]))
        cz, out = tmp_path / "g.cz", tmp_path / "dec.npz"
        lab = tmp_path / "labels.npy"

        assert czip.main(["encode", str(npz), "--hierarchy", str(hpath),
                          "-o", str(cz)]) == 0
        assert czip.main(["info", str(cz)]) == 0
        info = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
        assert info["model_id"] == "nested-dcsbm"
        assert info["n_levels"] == 3
        assert info["report"]["lossless"] is True
        assert czip.main(["decode", str(cz), "-o", str(out),
                          "--labels-out", str(lab)]) == 0
        assert (sp.load_npz(out) != adj).nnz == 0
        assert np.array_equal(np.load(lab), hier[0])

    def test_cli_hierarchy_weighted_roundtrip(self, tmp_path):
        adj, hier = _tiny_nested_weighted(seed=22, wmin=5)
        npz, hpath = tmp_path / "g.npz", tmp_path / "h.npz"
        sp.save_npz(npz, adj)
        self._save_hierarchy(hpath, hier)
        cz, out = tmp_path / "g.cz", tmp_path / "dec.npz"
        assert czip.main(["encode", str(npz), "--hierarchy", str(hpath),
                          "--wmin", "5", "-o", str(cz)]) == 0
        hdr, _ = czip.unpack(cz.read_bytes())
        assert hdr["model_id"] == "nested-dcsbm+weights"
        assert hdr["report"]["topology"]["lossless"] is True
        assert hdr["report"]["weights"]["lossless"] is True
        assert czip.main(["decode", str(cz), "-o", str(out)]) == 0
        assert (sp.load_npz(out) != adj).nnz == 0

    def test_cli_rejects_partition_and_hierarchy_together(self, tmp_path):
        adj, hier = _tiny_nested(seed=23)
        npz, hpath = tmp_path / "g.npz", tmp_path / "h.npz"
        part = tmp_path / "p.npy"
        sp.save_npz(npz, adj)
        self._save_hierarchy(hpath, hier)
        np.save(part, hier[0])
        with pytest.raises(SystemExit):
            czip.main(["encode", str(npz), "--partition", str(part),
                       "--hierarchy", str(hpath), "-o",
                       str(tmp_path / "g.cz")])
