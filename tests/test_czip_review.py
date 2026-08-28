"""Guard tests for the ways a container could claim more than it proves.

One test class per hazard: the container's input validation (weights,
labels, node/edge counts), the CLI's explicit decode proof, the
stream-table geometry checks, and the realized-vs-ideal bits gate ported
from ``weights_coder.weights_roundtrip``.

The last three classes are negative controls for false-losslessness paths:
a fractional CSV weight rounded before the integral guard saw it,
``verify_lossless`` truncating the source array it was meant to compare
against, and a ``params_digest`` that is written but never read.
"""

from __future__ import annotations

import importlib.util
import json
import struct

import numpy as np
import pytest
import scipy.sparse as sp

from czip import czip
from tests.test_czip import _tiny_graph, _tiny_weighted_graph

#: graph-tool is an encode-side-only, conda-forge-only dependency
needs_graph_tool = pytest.mark.skipif(
    importlib.util.find_spec("graph_tool") is None,
    reason="graph-tool is not installed")


# ------------------------------------------------ weight integrality
class TestWeightIntegrality:
    def test_non_integral_float_weights_rejected(self):
        adj, labels = _tiny_graph(seed=11, n=6, e=3)
        adj = adj.astype(np.float64)
        adj.data[:] = np.array([2.7, 1.4, 3.0])
        with pytest.raises(ValueError, match="integer"):
            czip.encode_weighted(adj, labels, wmin=1)

    def test_integral_float_weights_accepted_and_lossless(self):
        adj, labels = _tiny_weighted_graph(seed=12)
        fadj = adj.astype(np.float64)
        blob = czip.encode_weighted(fadj, labels, wmin=1)
        _, dec = czip.decode(blob)
        assert (dec != adj).nnz == 0

    def test_digest_computed_on_untruncated_weights(self):
        # 2.7 must never be silently coded (and digested) as 2
        adj, labels = _tiny_graph(seed=13, n=6, e=3)
        adj = adj.astype(np.float64)
        adj.data[:] = np.array([2.7, 1.4, 3.0])
        with pytest.raises(ValueError):
            czip.encode_weighted(adj, labels, wmin=1)


# ------------------------------------------------ graph preconditions
class TestLabelLength:
    def test_topology_rejects_short_labels(self):
        adj, labels = _tiny_graph(seed=14)
        with pytest.raises(ValueError, match="labels"):
            czip.encode_topology(adj, labels[:-1])

    def test_weighted_rejects_long_labels(self):
        adj, labels = _tiny_weighted_graph(seed=15)
        with pytest.raises(ValueError, match="labels"):
            czip.encode_weighted(adj, np.r_[labels, labels[:1]], wmin=1)


# ------------------------------------------------ partition length
class TestDegenerateGraphs:
    def test_zero_edge_graph_rejected(self):
        adj = sp.csr_matrix((4, 4), dtype=np.int64)
        labels = np.zeros(4, dtype=np.int64)
        with pytest.raises(ValueError, match="at least one edge"):
            czip.encode_topology(adj, labels)

    def test_zero_node_graph_rejected(self):
        adj = sp.csr_matrix((0, 0), dtype=np.int64)
        with pytest.raises(ValueError, match="at least one node"):
            czip.encode_topology(adj, np.zeros(0, dtype=np.int64))

    def test_zero_edge_graph_rejected_weighted(self):
        adj = sp.csr_matrix((4, 4), dtype=np.int64)
        labels = np.zeros(4, dtype=np.int64)
        with pytest.raises(ValueError, match="at least one edge"):
            czip.encode_weighted(adj, labels, wmin=1)


# ------------------------------------------------ stream-table geometry
class TestStreamTableGeometry:
    def _tampered(self, mutate):
        """Rebuild a container by hand with a tampered stream table."""
        blob = czip.pack({"model_id": "dcsbm"},
                         {"a": b"0123", "b": b"456789"})
        hdr, _ = czip.unpack(blob)
        hdr.pop("container_overhead_bits", None)
        mutate(hdr["stream_table"])
        hdr_bytes = json.dumps(hdr, sort_keys=True,
                               separators=(",", ":")).encode("utf-8")
        return (czip.MAGIC
                + struct.pack("<HI", czip.FORMAT_VERSION, len(hdr_bytes))
                + hdr_bytes + b"0123456789")

    def test_negative_offset_rejected(self):
        def mutate(table):
            table[1]["offset"] = -4
        with pytest.raises(ValueError, match="negative"):
            czip.unpack(self._tampered(mutate))

    def test_overlapping_streams_rejected(self):
        def mutate(table):
            table[1]["offset"] = 2
        with pytest.raises(ValueError, match="tile"):
            czip.unpack(self._tampered(mutate))

    def test_gap_between_streams_rejected(self):
        def mutate(table):
            table[1]["offset"] = 5
            table[1]["length"] = 5
        with pytest.raises(ValueError, match="tile"):
            czip.unpack(self._tampered(mutate))

    def test_trailing_bytes_rejected(self):
        def mutate(table):
            table[1]["length"] = 4
        with pytest.raises(ValueError, match="tile"):
            czip.unpack(self._tampered(mutate))

    def test_overhead_bits_measured_from_blob_extent(self):
        blob = czip.pack({"model_id": "dcsbm"}, {"a": b"0123"})
        hdr, _ = czip.unpack(blob)
        assert hdr["container_overhead_bits"] == 8 * (len(blob) - 4)


# ------------------------------------------------ declared-count bounds
class TestDecodeHeaderBounds:
    def test_absurd_n_nodes_rejected_before_allocation(self):
        adj, labels = _tiny_graph(seed=16)
        hdr, streams = czip.unpack(czip.encode_topology(adj, labels))
        hdr["n_nodes"] = 2 ** 60
        with pytest.raises(ValueError, match="n_nodes"):
            czip.decode(czip.pack(hdr, streams))

    def test_absurd_n_edges_rejected(self):
        adj, labels = _tiny_graph(seed=17)
        hdr, streams = czip.unpack(czip.encode_topology(adj, labels))
        hdr["n_edges"] = 2 ** 60
        with pytest.raises(ValueError, match="n_edges"):
            czip.decode(czip.pack(hdr, streams))

    def test_nonpositive_n_nodes_rejected(self):
        adj, labels = _tiny_graph(seed=18)
        hdr, streams = czip.unpack(czip.encode_topology(adj, labels))
        hdr["n_nodes"] = 0
        with pytest.raises(ValueError, match="n_nodes"):
            czip.decode(czip.pack(hdr, streams))


# ------------------------------------------------ the losslessness stamp
class TestCliVerify:
    def _encode(self, tmp_path, extra=(), weighted=False):
        if weighted:
            adj, labels = _tiny_weighted_graph(seed=19, wmin=1)
        else:
            adj, labels = _tiny_graph(seed=19)
        npz, part = tmp_path / "g.npz", tmp_path / "p.npy"
        sp.save_npz(npz, adj)
        np.save(part, labels)
        cz = tmp_path / "g.cz"
        argv = ["encode", str(npz), "--partition", str(part), "-o", str(cz)]
        if weighted:
            argv += ["--wmin", "1"]
        assert czip.main(argv + list(extra)) == 0
        return adj, czip.unpack(cz.read_bytes())[0]

    def test_encode_stamps_lossless_by_default(self, tmp_path):
        _, hdr = self._encode(tmp_path)
        assert hdr["report"]["lossless"] is True

    def test_encode_weighted_stamps_lossless(self, tmp_path):
        _, hdr = self._encode(tmp_path, weighted=True)
        assert hdr["report"]["topology"]["lossless"] is True
        assert hdr["report"]["weights"]["lossless"] is True

    def test_no_verify_leaves_the_claim_unproven(self, tmp_path):
        _, hdr = self._encode(tmp_path, extra=["--no-verify"])
        assert hdr["report"]["lossless"] is None

    def test_verify_lossless_catches_a_corrupted_stream(self):
        adj, labels = _tiny_graph(seed=20)
        blob = czip.encode_topology(adj, labels)
        hdr, streams = czip.unpack(blob)
        words = np.array(streams["adjacency_words"])
        words[0] ^= np.uint32(1 << 5)
        streams["adjacency_words"] = words
        with pytest.raises(ValueError):
            czip.verify_lossless(czip.pack(hdr, streams))

    def test_verify_lossless_compares_against_the_source(self):
        adj, labels = _tiny_graph(seed=21)
        other, _ = _tiny_graph(seed=22)
        blob = czip.encode_topology(adj, labels)
        with pytest.raises(ValueError, match="verify"):
            czip.verify_lossless(blob, adj=other, labels=labels)


# ------------------------------------------------ wmin is an origin shift
class TestWminIsAnOriginShift:
    def test_cli_checks_wmin_before_fitting(self, tmp_path, monkeypatch):
        adj, labels = _tiny_weighted_graph(seed=23, wmin=1)
        npz = tmp_path / "g.npz"
        sp.save_npz(npz, adj)

        import czip.czip_autofit as autofit

        def _boom(*a, **kw):  # pragma: no cover - must never run
            raise AssertionError("fit started before --wmin was checked")

        monkeypatch.setattr(autofit, "fit_auto", _boom)
        with pytest.raises(SystemExit):
            czip.main(["encode", str(npz), "--model", "auto",
                       "--wmin", "99", "-o", str(tmp_path / "g.cz")])

    def test_cli_wmin_error_names_the_offending_weight(self, tmp_path,
                                                       capsys):
        adj, labels = _tiny_weighted_graph(seed=24, wmin=1)
        npz, part = tmp_path / "g.npz", tmp_path / "p.npy"
        sp.save_npz(npz, adj)
        np.save(part, labels)
        with pytest.raises(SystemExit):
            czip.main(["encode", str(npz), "--partition", str(part),
                       "--wmin", "99", "-o", str(tmp_path / "g.cz")])
        assert "wmin" in capsys.readouterr().err


# ------------------------------------------------ fit metadata
@needs_graph_tool
class TestFitMetaRecordsThreads:
    def test_fit_auto_records_omp_threads(self):
        from tests.test_czip_autofit import planted_two_block
        from czip.czip_autofit import fit_auto

        meta = fit_auto(planted_two_block(), seed0=100,
                        restarts=1)["fit_meta"]
        assert isinstance(meta["omp_threads"], int)
        assert meta["omp_threads"] >= 1


# ------------------------------------------------ the bits gate
class TestBitsGate:
    def test_topology_encode_stamps_a_passing_gate(self):
        adj, labels = _tiny_graph(seed=25)
        hdr, _ = czip.unpack(czip.encode_topology(adj, labels))
        gate = hdr["report"]["bits_gate"]
        assert gate["within_slack"] is True
        assert gate["realized_bits"] >= gate["lower_bits"]
        assert gate["realized_bits"] <= gate["upper_bits"]

    def test_weighted_encode_stamps_both_gates(self):
        adj, labels = _tiny_weighted_graph(seed=26)
        hdr, _ = czip.unpack(czip.encode_weighted(adj, labels, wmin=1))
        assert hdr["report"]["topology"]["bits_gate"]["within_slack"] is True
        assert hdr["report"]["weights"]["bits_gate"]["within_slack"] is True

    def test_gate_reads_a_header_read_only(self):
        adj, labels = _tiny_weighted_graph(seed=27)
        hdr, _ = czip.unpack(czip.encode_weighted(adj, labels, wmin=1))
        gate = czip.bits_gate(hdr)
        assert gate["within_slack"] is True
        assert gate["topology"]["within_slack"] is True
        assert gate["weights"]["within_slack"] is True

    def test_gate_fails_when_realized_exceeds_expected_plus_slack(self):
        adj, labels = _tiny_graph(seed=28)
        hdr, _ = czip.unpack(czip.encode_topology(adj, labels))
        hdr["report"]["realized_bits"] += 10 * hdr["report"]["ideal_bits"]
        gate = czip.bits_gate(hdr)
        assert gate["within_slack"] is False
        assert gate["topology"]["within_slack"] is False

    def test_gate_fails_below_the_floored_symbol_deficit_bound(self):
        adj, labels = _tiny_graph(seed=29)
        hdr, _ = czip.unpack(czip.encode_topology(adj, labels))
        rep = hdr["report"]
        rep["realized_bits"] = (rep["ideal_bits"] - rep["tail_gap_bits"]
                                - 10.0 * rep["n_symbols"] - 1e6)
        assert czip.bits_gate(hdr)["within_slack"] is False

    def test_encode_raises_when_the_gate_fails(self, monkeypatch):
        adj, labels = _tiny_graph(seed=30)
        failed = {"within_slack": False, "weights": None,
                  "topology": {"within_slack": False}}
        monkeypatch.setattr(czip, "bits_gate", lambda header: failed)
        with pytest.raises(ValueError, match="bits gate"):
            czip.encode_topology(adj, labels)


# ------------------------------------------------ weights-header round-trip
class TestWeightsHeaderRoundTrip:
    def test_params_desync_is_caught_at_encode(self, monkeypatch):
        # a header codec that hands the decoder different params than were
        # fitted must be caught at encode, not by a corrupted artifact
        adj, labels = _tiny_weighted_graph(seed=31)
        real = czip.read_header
        state = {"n": 0}

        def _skewed(br):
            fam, delta, par = real(br)
            state["n"] += 1
            if state["n"] == 1:
                par = {k: v * 0.5 + 1.0 for k, v in par.items()}
            return fam, delta, par

        monkeypatch.setattr(czip, "read_header", _skewed)
        with pytest.raises(AssertionError, match="param"):
            czip.encode_weighted(adj, labels, wmin=1)


# ------------------------------------------------ fractional CSV weights
class TestEdgeListWeightIntegrality:
    """The CSV/TSV loader must not round a weight past the integral guard."""

    def _write(self, tmp_path, text, name="edges.csv"):
        p = tmp_path / name
        p.write_text(text)
        return p

    def test_fractional_csv_weight_rejected(self, tmp_path):
        path = self._write(tmp_path, "a,b,2.7\nb,c,4\n")
        with pytest.raises(ValueError, match="integer"):
            czip.load_edgelist(path)

    def test_fractional_csv_error_names_the_row(self, tmp_path):
        path = self._write(tmp_path, "a,b,4\nb,c,2.7\n")
        with pytest.raises(ValueError, match=r":2:"):
            czip.load_edgelist(path)

    def test_integral_float_csv_weight_accepted(self, tmp_path):
        path = self._write(tmp_path, "a,b,4.0\nb,c,2\n")
        adj, ids = czip.load_edgelist(path)
        assert adj.dtype == np.int64
        assert sorted(adj.tocoo().data.tolist()) == [2, 4]
        assert list(ids) == ["a", "b", "c"]

    def test_tsv_fractional_weight_rejected(self, tmp_path):
        path = self._write(tmp_path, "a\tb\t2.5\n", name="edges.tsv")
        with pytest.raises(ValueError, match="integer"):
            czip.load_edgelist(path)

    def test_header_row_still_skipped(self, tmp_path):
        path = self._write(tmp_path, "src,dst,weight\na,b,2\nb,c,3\n")
        adj, _ = czip.load_edgelist(path)
        assert adj.nnz == 2

    def test_cli_refuses_a_fractional_source_and_writes_nothing(self,
                                                                tmp_path):
        path = self._write(tmp_path, "a,b,2.7\nb,c,4\n")
        out = tmp_path / "g.cz"
        part = tmp_path / "p.npy"
        np.save(part, np.zeros(3, dtype=np.int64))
        with pytest.raises(ValueError, match="integer"):
            czip.main(["encode", str(path), "--partition", str(part),
                       "--wmin", "1", "-o", str(out)])
        assert not out.exists()


# ------------------------------------------------ the source comparison
class TestVerifySourceIntegrality:
    """``verify_lossless`` may not cast the source it is comparing to."""

    def _fractional_source(self, seed=41):
        adj, labels = _tiny_weighted_graph(seed=seed, wmin=1)
        adj = adj.tocsr()
        adj.data[0] = 2  # the value the decode will produce
        blob = czip.encode_weighted(adj, labels, wmin=1)
        src = adj.astype(np.float64).tocsr()
        src.data[0] = 2.7  # ... a source differing only in a fractional part
        return blob, src, labels

    def test_fractional_source_raises_instead_of_stamping(self):
        blob, src, labels = self._fractional_source()
        with pytest.raises(ValueError, match="integer"):
            czip.verify_lossless(blob, adj=src, labels=labels)

    def test_fractional_source_leaves_the_blob_unstamped(self):
        blob, src, labels = self._fractional_source(seed=42)
        with pytest.raises(ValueError):
            czip.verify_lossless(blob, adj=src, labels=labels)
        hdr, _ = czip.unpack(blob)
        assert hdr["report"]["weights"].get("lossless") is None
        assert hdr["report"]["topology"].get("lossless") is None

    def test_integral_float_source_still_verifies(self):
        adj, labels = _tiny_weighted_graph(seed=43, wmin=1)
        blob = czip.encode_weighted(adj, labels, wmin=1)
        stamped = czip.verify_lossless(blob, adj=adj.astype(np.float64),
                                       labels=labels)
        hdr, _ = czip.unpack(stamped)
        assert hdr["report"]["weights"]["lossless"] is True


# ------------------------------------------------ stamping needs a source
class TestStampRequiresSourceArrays:
    """A losslessness stamp requires source arrays; a digest-only check is
    recorded distinctly."""

    def test_no_arrays_does_not_stamp_lossless(self):
        adj, labels = _tiny_weighted_graph(seed=44, wmin=1)
        blob = czip.encode_weighted(adj, labels, wmin=1)
        hdr, _ = czip.unpack(czip.verify_lossless(blob))
        assert hdr["report"]["weights"]["lossless"] is None
        assert hdr["report"]["topology"]["lossless"] is None

    def test_no_arrays_records_digest_verified(self):
        adj, labels = _tiny_weighted_graph(seed=45, wmin=1)
        blob = czip.encode_weighted(adj, labels, wmin=1)
        hdr, _ = czip.unpack(czip.verify_lossless(blob))
        assert hdr["report"]["weights"]["digest_verified"] is True
        assert hdr["report"]["topology"]["digest_verified"] is True

    def test_adj_without_labels_does_not_stamp_lossless(self):
        adj, labels = _tiny_graph(seed=46)
        blob = czip.encode_topology(adj, labels)
        hdr, _ = czip.unpack(czip.verify_lossless(blob, adj=adj))
        assert hdr["report"]["lossless"] is None
        assert hdr["report"]["digest_verified"] is True

    def test_both_arrays_stamp_lossless(self):
        adj, labels = _tiny_graph(seed=47)
        blob = czip.encode_topology(adj, labels)
        hdr, _ = czip.unpack(czip.verify_lossless(blob, adj=adj,
                                                  labels=labels))
        assert hdr["report"]["lossless"] is True
        assert hdr["report"]["digest_verified"] is True


# ------------------------------------------------ the params digest
class TestParamsDigestIsChecked:
    def test_tampered_params_digest_fails_decode(self):
        adj, labels = _tiny_graph(seed=48)
        hdr, streams = czip.unpack(czip.encode_topology(adj, labels))
        hdr["params_digest"] = "0" * 64
        with pytest.raises(ValueError, match="params_digest"):
            czip.decode(czip.pack(hdr, streams))

    def test_tampered_params_digest_fails_weighted_decode(self):
        adj, labels = _tiny_weighted_graph(seed=49, wmin=1)
        hdr, streams = czip.unpack(czip.encode_weighted(adj, labels, wmin=1))
        hdr["params_digest"] = "0" * 64
        with pytest.raises(ValueError, match="params_digest"):
            czip.decode(czip.pack(hdr, streams))

    def test_missing_params_digest_fails_decode(self):
        adj, labels = _tiny_graph(seed=50)
        hdr, streams = czip.unpack(czip.encode_topology(adj, labels))
        hdr.pop("params_digest")
        with pytest.raises(ValueError, match="params_digest"):
            czip.decode(czip.pack(hdr, streams))

    def test_tampered_params_digest_is_never_stamped(self):
        adj, labels = _tiny_graph(seed=51)
        hdr, streams = czip.unpack(czip.encode_topology(adj, labels))
        hdr["params_digest"] = "0" * 64
        with pytest.raises(ValueError, match="params_digest"):
            czip.verify_lossless(czip.pack(hdr, streams), adj=adj,
                                 labels=labels)

    def test_intact_params_digest_decodes(self):
        adj, labels = _tiny_weighted_graph(seed=52, wmin=1)
        blob = czip.encode_weighted(adj, labels, wmin=1)
        _, dec = czip.decode(blob)
        assert (dec != adj).nnz == 0
