"""czip — versioned container + CLI around the coders.

The `.cz` v1 container is a dispatcher, not a coder:
model bits come verbatim from the coder message segments (czip.sbm_coder
for DC-SBM topology, czip.weights_coder for weights), and the
container adds no padding of its own; its own cost — the magic bytes, the
version and header-length prefix, and the header JSON — is measured as
`container_overhead_bits`. That cost is itemized, not excluded: it is
reported on its own line AND included in every whole-container
bits-per-edge figure.

Layout (little-endian): magic ``CZIP`` | format_version u16 | header_len u32
| header (canonical JSON, sorted keys, UTF-8) | concatenated stream bytes.
The header's ``stream_table`` records (name, kind, offset, length) per
stream; ``kind`` says how to read a stream's bytes and nothing more — "u32"
is a constriction word stream (little-endian uint32 words), "bytes" is an
opaque byte payload taken as-is (the message header, the edge and degree
payloads, the weights header, and rank payloads are all "bytes"). Which
kind a segment takes can depend on the branch its coder chose for this
graph, so the table is always read, never assumed.

A topology-only container carries the DC-SBM payload alone — explicit in
the header as ``payload: "topology"``; weighted inputs are refused unless
the caller opts into dropping weights.

`encode` decodes the blob it wrote and compares it against the source before
returning (an explicit decode pass), stamping
``report.lossless: true``; ``--no-verify`` opts out and leaves the claim
unproven. ``--wmin W`` is the weight ORIGIN (the code transmits ``w - W``),
never a filter: a weight below W is an error, checked before any fit.

Models: ``dcsbm`` / ``dcsbm+weights`` take a flat partition; ``nested-dcsbm``
/ ``nested-dcsbm+weights`` take a whole hierarchy and code it with
czip.nested_coder. Both share FORMAT_VERSION 1 — the model_id dispatches, so
containers written before the nested models joined keep decoding.

Usage:
    czip encode g.npz --partition p.npy -o g.cz
    czip encode g.npz --hierarchy h.npz -o g.cz
      # h.npz is a czip.sbm.save_hierarchy dump (level_0..level_{k-1})
    czip encode g.csv --model auto --wmin 1 -o g.cz
      # edge list (src,dst[,weight]) CSV/TSV; node id map saved alongside
    czip decode g.cz -o g_decoded.npz [--labels-out p.npy]
    czip info g.cz
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import struct
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from czip import nested_coder
from czip import sbm_coder
from czip import weights as W
from czip.weights_coder import (BitReader, BitWriter, _shift_base,
                               decode_weights_stream, encode_weights_stream,
                               read_header, write_header)

MAGIC = b"CZIP"
FORMAT_VERSION = 1

# header keys that describe the container itself; rebuilt on every pack
_COMPUTED_KEYS = ("stream_table", "container_overhead_bits", "format_version")


def pack(header: dict, streams: dict) -> bytes:
    """Serialize header + named streams into a .cz blob."""
    table = []
    chunks = []
    offset = 0
    for name, payload in streams.items():
        if isinstance(payload, (bytes, bytearray)):
            kind, raw = "bytes", bytes(payload)
        else:
            arr = np.asarray(payload)
            if arr.dtype != np.uint32:
                raise ValueError(f"stream {name!r}: expected uint32 words or "
                                 f"bytes, got dtype {arr.dtype}")
            kind, raw = "u32", arr.astype("<u4").tobytes()
        table.append({"name": name, "kind": kind,
                      "offset": offset, "length": len(raw)})
        chunks.append(raw)
        offset += len(raw)
    hdr = {k: v for k, v in header.items() if k not in _COMPUTED_KEYS}
    hdr["format_version"] = FORMAT_VERSION
    hdr["stream_table"] = table
    hdr_bytes = json.dumps(hdr, sort_keys=True,
                           separators=(",", ":")).encode("utf-8")
    return (MAGIC + struct.pack("<HI", FORMAT_VERSION, len(hdr_bytes))
            + hdr_bytes + b"".join(chunks))


def unpack(blob: bytes):
    """Parse a .cz blob -> (header, streams). Injects the computed
    ``container_overhead_bits`` (container bytes beyond the streams).

    The stream table is validated as a geometry, not trusted: every
    offset/length is non-negative and the entries must tile the stream
    region ``[0, len(blob) - base)`` exactly — monotonic, no gaps, no
    overlaps, no trailing bytes. ``container_overhead_bits`` is then the
    measured blob extent beyond that tiling (magic + version + header_len +
    header JSON), never a sum of self-declared lengths.
    """
    if blob[:4] != MAGIC:
        raise ValueError(f"bad magic {blob[:4]!r}, expected {MAGIC!r}")
    if len(blob) < 10:
        raise ValueError("truncated container: no header")
    version, hdr_len = struct.unpack("<HI", blob[4:10])
    if version != FORMAT_VERSION:
        raise ValueError(f"unsupported format version {version} "
                         f"(this reader handles {FORMAT_VERSION})")
    if 10 + hdr_len > len(blob):
        raise ValueError("truncated container: header extends past blob")
    header = json.loads(blob[10:10 + hdr_len].decode("utf-8"))
    base = 10 + hdr_len
    table = header.get("stream_table", []) or []
    streams = {}
    covered = 0
    for entry in table:
        name, offset, length = entry["name"], entry["offset"], entry["length"]
        if offset < 0 or length < 0:
            raise ValueError(f"bad stream table: stream {name!r} has a "
                             f"negative offset/length ({offset}, {length})")
        if base + offset + length > len(blob):
            raise ValueError(f"truncated container: stream "
                             f"{name!r} extends past blob")
        if offset != covered:
            raise ValueError(f"bad stream table: stream {name!r} starts at "
                             f"{offset} but the streams must tile the "
                             f"container (expected offset {covered})")
        lo = base + offset
        raw = blob[lo:lo + length]
        if entry["kind"] == "u32":
            streams[name] = np.frombuffer(raw, dtype="<u4")
        else:
            streams[name] = raw
        covered = offset + length
    if covered != len(blob) - base:
        raise ValueError(f"bad stream table: streams tile {covered} bytes of "
                         f"the {len(blob) - base}-byte stream region")
    header["container_overhead_bits"] = 8 * (len(blob) - covered)
    return header, streams


# --------------------------------------------------------------- bits gate

# Quantization-drift allowance per coded symbol, same budget the verified
# weights round-trip uses (weights_coder.weights_roundtrip).
GATE_DRIFT_BITS_PER_SYMBOL = 3e-4
# Fallback segment counts for containers written before the encoders recorded
# their own (``report.n_word_streams`` / ``report.n_byte_streams``): the flat
# DC-SBM message on the rank branch is 2 constriction word streams
# (partition_words, adjacency_words) padded to whole 32-bit words and 4
# byte-aligned payloads (header, partition_rank, edges_payload,
# degrees_payload). The nested model has one pair of partition streams per
# level plus the expansions, so the window can no longer be a constant.
_TOPOLOGY_WORD_STREAMS = 2
_TOPOLOGY_BYTE_STREAMS = 4


def _stream_counts(streams: dict) -> tuple[int, int]:
    """(word streams, byte streams) of a message's stream dict.

    The gate's padding allowance is per segment, and which segments are words
    and which are bytes is a property of the message the encoder just built
    (both the flat edges/degrees payloads and the nested expansion payloads
    switch branch on the edge total), so it is measured here rather than
    assumed.
    """
    words = sum(1 for v in streams.values()
                if not isinstance(v, (bytes, bytearray)))
    return words, len(streams) - words


def _gate(realized: float, expected: float, deficit: float,
          slack: float) -> dict:
    lower = expected - deficit - slack
    upper = expected + slack
    return {
        "realized_bits": float(realized),
        "expected_bits": float(expected),
        "deficit_bound_bits": float(deficit),
        "slack_budget_bits": float(slack),
        "lower_bits": float(lower),
        "upper_bits": float(upper),
        "margin_low_bits": float(realized - lower),
        "margin_high_bits": float(upper - realized),
        "within_slack": bool(lower <= realized <= upper),
    }


def bits_gate(header: dict) -> dict:
    """Realized-vs-ideal reconciliation gate for a .cz header.

    Ports the ``within_slack`` window of the verified weights round-trip
    (weights_coder.weights_roundtrip) to the container: each entropy-coded
    segment's realized bits must land inside
    ``[ideal − deficit_bound − slack, ideal + slack]``. The deficit bound is
    the lower side because constriction's probability floor makes deep-tail
    symbols cost *less* than −log₂P — that is the (legitimate) negative
    ``overhead_bits`` every real header carries — and it is exactly the
    coder's own floored-symbol accounting (``tail_gap_bits`` for topology,
    ``deep_tail_deficit_bound_bits`` for weights). ``slack`` budgets segment
    padding plus float-quantization drift; the padding term counts the
    container's OWN topology segments (``report.n_word_streams`` /
    ``n_byte_streams``, written by the encoders), falling back to the flat
    model's constants for containers written before those were recorded.

    Honest limitation: the drift term is ``3e-4 * n_symbols``, linear in
    message length, not O(1) — this is a tight engineering reconciliation
    window, not an asymptotic bound. It is the same per-symbol budget the
    verified ``weights_coder.weights_roundtrip`` uses; at hemibrain-ge5
    scale the whole slack is 0.06% of ideal bits, so a coder miscalibrated
    by even 0.01 bits/symbol fails the gate.

    Header-only and read-only: nothing is re-encoded, so the gate can be run
    against a shipped artifact's header. Returns
    ``{"topology": {...}, "weights": {...} | None, "within_slack": bool}``.
    """
    rep = header.get("report", {})
    topo = rep.get("topology", rep)
    n_words = int(topo.get("n_word_streams", _TOPOLOGY_WORD_STREAMS))
    n_bytes = int(topo.get("n_byte_streams", _TOPOLOGY_BYTE_STREAMS))
    topo_slack = (32.0 * 2 * n_words + 8.0 * n_bytes
                  + GATE_DRIFT_BITS_PER_SYMBOL * float(topo["n_symbols"]))
    out = {"topology": _gate(topo["realized_bits"], topo["ideal_bits"],
                             topo["tail_gap_bits"], topo_slack),
           "weights": None}
    weights = rep.get("weights")
    if weights is not None:
        n_levels = int(header.get("n_weight_levels", 0))
        w_slack = (32.0 * 2 * n_levels
                   + GATE_DRIFT_BITS_PER_SYMBOL * float(weights["n_symbols"]))
        out["weights"] = _gate(weights["realized_data_bits"],
                               weights["ideal_data_bits"],
                               weights["deep_tail_deficit_bound_bits"],
                               w_slack)
    out["within_slack"] = bool(
        out["topology"]["within_slack"]
        and (out["weights"] is None or out["weights"]["within_slack"]))
    return out


def _require_gate(header: dict) -> dict:
    """Run bits_gate on a freshly built header; fail the encode loudly."""
    gate = bits_gate(header)
    if not gate["within_slack"]:
        raise ValueError(
            "czip bits gate failed: realized bits fall outside the expected "
            f"± slack window — {json.dumps(_jsonable(gate), sort_keys=True)}")
    return gate


def _canonical_edges(src, dst):
    order = np.lexsort((dst, src))
    return (np.asarray(src, dtype=np.int64)[order],
            np.asarray(dst, dtype=np.int64)[order],
            order)


def _source_digest(labels, src, dst, w=None) -> str:
    src, dst, order = _canonical_edges(src, dst)
    h = hashlib.sha256()
    h.update(np.asarray(labels, dtype="<i8").tobytes())
    h.update(src.astype("<i8").tobytes())
    h.update(dst.astype("<i8").tobytes())
    if w is not None:
        h.update(np.asarray(w, dtype="<i8")[order].tobytes())
    return h.hexdigest()


def _params_digest(labels) -> str:
    """SHA256 of the model params (the densified partition), the exact byte
    content both encoders write into ``params_digest`` and decode re-derives
    from the decoded partition."""
    return hashlib.sha256(
        np.asarray(labels, dtype=np.int64).astype("<i8").tobytes()).hexdigest()


def _params_digest_nested(levels) -> str:
    """SHA256 of a nested model's params: the level count followed by every
    canonical level array, each serialized ``<i8``.

    The flat ``_params_digest`` would see only the base labels, so two
    containers whose hierarchies differ only above level 0 would carry the
    same params digest — and decode's digest check would pass on a swapped
    model. Every level is part of the model, so every level is digested.
    """
    h = hashlib.sha256()
    h.update(np.asarray([len(levels)], dtype="<i8").tobytes())
    for lab in levels:
        h.update(np.asarray(lab, dtype=np.int64).astype("<i8").tobytes())
    return h.hexdigest()


def _validate_graph(adj: sp.spmatrix, labels: np.ndarray) -> None:
    """Preconditions shared by both encoders.

    czip needs n >= 1 and e >= 1: the DC-SBM message header codes e with
    Elias-gamma, which has no codeword for 0, and the partition/degree
    segments assume at least one node. A partition must give exactly one
    block label per node — a mismatch used to surface only at decode time,
    as a math.comb error inside the partition decoder.
    """
    if adj.shape[0] != adj.shape[1]:
        raise ValueError(f"adjacency must be square, got {adj.shape}")
    n = int(adj.shape[0])
    if n < 1:
        raise ValueError("czip needs a graph with at least one node")
    if int(adj.nnz) < 1:
        raise ValueError(
            "czip needs a graph with at least one edge (e >= 1: the message "
            "header codes e with Elias-gamma, undefined for 0)")
    if labels.shape[0] != n:
        raise ValueError(
            f"labels length {labels.shape[0]} != n_nodes {n}: the partition "
            "must carry exactly one block label per node")


def _integral_weights(data) -> np.ndarray:
    """int64 edge weights, refusing anything non-integral.

    Weights are synapse counts. A float input that is not integral used to
    be truncated by ``astype(int64)`` *before* the source digest was taken,
    so the container's losslessness certificate covered the truncated graph
    rather than the input.
    """
    arr = np.asarray(data)
    if arr.dtype.kind == "f":
        if not np.all(np.isfinite(arr)) or not np.all(arr == np.rint(arr)):
            bad = arr[~np.isfinite(arr) | (arr != np.rint(arr))][:3]
            raise ValueError(
                "weights must be integer synapse counts; got non-integral "
                f"values (e.g. {bad.tolist()}) — czip refuses to round them")
    elif arr.dtype.kind not in ("i", "u", "b"):
        raise ValueError(f"weights must be integers, got dtype {arr.dtype}")
    return arr.astype(np.int64)


def _integral_weight(value: float, where: str) -> int:
    """One edge-list weight as an int, refusing non-integral input.

    Row-level counterpart of ``_integral_weights``: the CSV/TSV loader used
    to do ``int(round(float(...)))``, so a ``2.7`` in the source file became
    a ``3`` in the container *before* the array-level guard or the source
    digest could see it — and the container then stamped that changed graph
    lossless. ``where`` names the offending row.
    """
    if not math.isfinite(value) or value != math.floor(value):
        raise ValueError(
            f"{where}: weight {value!r} is not an integer synapse count — "
            "czip refuses to round it")
    return int(value)


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def _resolve_model(labels, hierarchy):
    """(base labels, canonical levels or None) for one encode call.

    Exactly one of ``labels`` / ``hierarchy`` describes the model. A raw
    (padded, ``get_bs()``-style) hierarchy is canonicalized here, and its
    dense level 0 IS the base partition every downstream stage keys on — the
    weights layer included, since ``_group_layout`` reads base labels only.
    """
    if (labels is None) == (hierarchy is None):
        raise ValueError("pass exactly one of labels / hierarchy: a flat "
                         "DC-SBM takes a partition, a nested one takes the "
                         "whole hierarchy")
    if hierarchy is not None:
        levels = nested_coder.canonical_hierarchy(hierarchy)
        return levels[0], levels
    # graph-tool fits store raw non-contiguous block labels; densify to
    # 0..B-1 (order-preserving)
    return np.unique(np.asarray(labels),
                     return_inverse=True)[1].astype(np.int64), None


def _topology_message(src, dst, labels, levels):
    """The coded topology message + the header fields naming its model."""
    if levels is None:
        message = sbm_coder.encode_dcsbm(src, dst, labels, verify=False)
        meta = {"model_id": "dcsbm", "params_digest": _params_digest(labels)}
        streams = {
            "header": message["header"],
            "partition_rank": message["partition_rank"],
            "partition_words": np.asarray(message["partition_words"],
                                          dtype=np.uint32),
            "edges_payload": message["edges_payload"],
            "degrees_payload": message["degrees_payload"],
            "adjacency_words": np.asarray(message["adjacency_words"],
                                          dtype=np.uint32),
        }
    else:
        message = nested_coder.encode_nested_dcsbm(src, dst, levels,
                                                   verify=False)
        meta = {"model_id": "nested-dcsbm",
                "n_levels": len(levels),
                "params_digest": _params_digest_nested(levels)}
        # transmission order IS the message's dict order (nested_coder builds
        # it segment by segment); the report is not a stream
        streams = {k: (v if isinstance(v, (bytes, bytearray))
                       else np.asarray(v, dtype=np.uint32))
                   for k, v in message.items() if k != "report"}
    report = _jsonable(message["report"])
    n_words, n_bytes = _stream_counts(streams)
    report["n_word_streams"] = n_words
    report["n_byte_streams"] = n_bytes
    return streams, report, meta


def encode_topology(adj: sp.csr_matrix, labels: np.ndarray | None = None,
                    allow_weight_drop: bool = False,
                    fit_meta: dict | None = None,
                    hierarchy=None) -> bytes:
    """Encode (model, topology) of a directed graph as a .cz blob.

    The payload is the verified DC-SBM message: flat (sbm_coder.encode_dcsbm,
    model ``dcsbm``) from ``labels``, or nested
    (nested_coder.encode_nested_dcsbm, model ``nested-dcsbm``) from
    ``hierarchy`` — a raw padded ``get_bs()``/``save_hierarchy`` level list is
    accepted and canonicalized. Either way the encode-side self-decode is
    skipped and losslessness is proven by an explicit
    decode(). Weighted inputs are refused unless allow_weight_drop=True,
    because this payload transmits topology only.
    """
    adj = adj.tocsr()
    labels, levels = _resolve_model(labels, hierarchy)
    _validate_graph(adj, np.asarray(labels))
    if adj.data.size and not np.all(adj.data == 1):
        if not allow_weight_drop:
            raise ValueError(
                "input has non-binary weights but the payload is topology "
                "only; pass allow_weight_drop=True to encode the binarized "
                "graph")
        adj = (adj > 0).astype(np.int64).tocsr()
    coo = adj.tocoo()
    src, dst, _ = _canonical_edges(coo.row, coo.col)
    streams, report, meta = _topology_message(src, dst, labels, levels)
    header = {
        "tool": "czip",
        **meta,
        "payload": "topology",
        "n_nodes": int(adj.shape[0]),
        "n_edges": int(src.shape[0]),
        "source_digest": _source_digest(labels, src, dst),
        "report": report,
    }
    header["report"]["bits_gate"] = _require_gate(header)["topology"]
    if fit_meta is not None:
        header["fit_meta"] = _jsonable(fit_meta)
    return pack(header, streams)


def _group_layout(labels, src, dst):
    """Ordered-block-pair group ids per canonical edge + per-group slicing.

    Returns (group_ids, order, gids, starts, ends) with `order` the stable
    group sort of the canonical edge sequence — identical on both sides of
    the round-trip because it derives from transmitted topology only.
    """
    B = int(labels.max()) + 1
    group_ids = labels[src] * B + labels[dst]
    order = np.argsort(group_ids, kind="stable")
    gs = group_ids[order]
    starts = np.flatnonzero(np.r_[True, gs[1:] != gs[:-1]])
    ends = np.r_[starts[1:], gs.size]
    return group_ids, order, gs[starts], starts, ends


def _check_header_roundtrip(weights_header: bytes, flags, gids, groups,
                            pooled) -> None:
    """Re-read the written weights header and assert it reproduces the fit.

    Same decoder-side cross-check ``weights_coder.weights_roundtrip`` runs:
    the params the decoder will see must equal the fitted
    ``params_q`` exactly, so a header-codec desync can never reach a shipped
    artifact (where it would surface as a silent weight corruption caught
    only by the source digest).
    """
    br = BitReader(weights_header)
    flags_dec = [br.read(1) for _ in flags]
    if flags_dec != list(flags):
        raise AssertionError("weights header: own/pooled flag decode mismatch")
    pooled_hdr = read_header(br) if pooled is not None else None
    for g, f in zip(gids, flags):
        if f:
            fam, _delta, par = read_header(br)
            ref_fam, ref = groups[str(g)]["family"], groups[str(g)]["params_q"]
        else:
            fam, _delta, par = pooled_hdr
            ref_fam, ref = pooled["family"], pooled["params_q"]
        if fam != ref_fam:
            raise AssertionError(f"weights header: family mismatch group {g}: "
                                 f"{fam} vs {ref_fam}")
        for name, val in ref.items():
            if not math.isclose(par[name], val, rel_tol=0, abs_tol=1e-12):
                raise AssertionError(
                    f"weights header: param mismatch group {g} {name}: "
                    f"{par[name]} vs {val}")


def encode_weighted(adj: sp.csr_matrix, labels: np.ndarray | None = None,
                    *, wmin: int, fit_meta: dict | None = None,
                    hierarchy=None) -> bytes:
    """Encode (model, topology, weights) as a .cz blob.

    Topology = the verified DC-SBM message, flat from ``labels`` or nested
    from ``hierarchy`` (models ``dcsbm+weights`` / ``nested-dcsbm+weights``);
    weights = the verified per-group OWN/POOLED code
    (weights.weight_code_batched fit + weights_coder streams), conditioned on
    the transmitted topology exactly as the theoretical accounting does. The
    weights layer keys on the base partition alone (``_group_layout`` groups
    edges by ordered block pair), so a nested model changes nothing above it.
    Losslessness is proven by decode() + the source digest (which covers the
    weight values).
    """
    adj = adj.tocsr()
    labels, levels = _resolve_model(labels, hierarchy)
    _validate_graph(adj, np.asarray(labels))
    coo = adj.tocoo()
    src, dst, order = _canonical_edges(coo.row, coo.col)
    w = _integral_weights(coo.data)[order]
    topo_streams, topo_report, meta = _topology_message(src, dst, labels,
                                                        levels)

    group_ids, gorder, gids, starts, ends = _group_layout(labels, src, dst)
    ws = w[gorder]
    result = W.weight_code_batched(w, wmin, group_ids)

    groups, pooled = result["groups"], result["pooled"]
    flags = [1 if groups[str(g)]["choice"] == "own" else 0 for g in gids]
    bw = BitWriter()
    for f in flags:
        bw.write(f, 1)
    any_pooled = any(f == 0 for f in flags)
    if any_pooled:
        write_header(bw, pooled["family"], pooled["delta"],
                     pooled["params_q"])
    headers_per_group = []
    for g, f in zip(gids, flags):
        if f:
            e = groups[str(g)]
            write_header(bw, e["family"], e["delta"], e["params_q"])
            headers_per_group.append((e["family"], e["params_q"]))
        else:
            headers_per_group.append((pooled["family"], pooled["params_q"]))
    weights_header = bw.getvalue()
    _check_header_roundtrip(weights_header, flags, gids, groups,
                            pooled if any_pooled else None)

    shifted = [ws[s:e] - wmin + _shift_base(headers_per_group[i][0])
               for i, (s, e) in enumerate(zip(starts, ends))]
    level_streams, ideal_data_bits, n_symbols, deficit, n_floored = \
        encode_weights_stream(shifted, headers_per_group)

    header = {
        "tool": "czip",
        **meta,
        "model_id": meta["model_id"] + "+weights",
        "payload": "graph",
        "n_nodes": int(adj.shape[0]),
        "n_edges": int(src.shape[0]),
        "wmin": int(wmin),
        "n_weight_levels": len(level_streams),
        "source_digest": _source_digest(labels, src, dst, w=w),
        **({"fit_meta": _jsonable(fit_meta)} if fit_meta is not None else {}),
        "report": {
            "topology": topo_report,
            "weights": {
                "L_total_bits": float(result["L_total_bits"]),
                "ideal_data_bits": float(ideal_data_bits),
                "n_symbols": int(n_symbols),
                "n_groups": int(gids.size),
                "n_own": int(sum(flags)),
                "n_floored_symbols": int(n_floored),
                "deep_tail_deficit_bound_bits": float(deficit),
                "realized_header_bits": 8 * len(weights_header),
                "realized_data_bits": 32 * int(sum(
                    s.size for s in level_streams)),
            },
        },
    }
    gate = _require_gate(header)
    header["report"]["topology"]["bits_gate"] = gate["topology"]
    header["report"]["weights"]["bits_gate"] = gate["weights"]
    streams = {**topo_streams, "weights_header": weights_header}
    for i, s in enumerate(level_streams):
        streams[f"weights_level_{i}"] = np.asarray(s, dtype=np.uint32)
    return pack(header, streams)


def _decode_weights(header, streams, labels, src, dst):
    _, gorder, gids, starts, ends = _group_layout(labels, src, dst)
    br = BitReader(streams["weights_header"])
    flags = [br.read(1) for _ in range(gids.size)]
    pooled_hdr = read_header(br) if any(f == 0 for f in flags) else None
    headers_per_group = []
    for f in flags:
        if f:
            fam, _delta, par = read_header(br)
        else:
            fam, _delta, par = pooled_hdr
        headers_per_group.append((fam, par))
    level_streams = [streams[f"weights_level_{i}"]
                     for i in range(int(header["n_weight_levels"]))]
    group_sizes = [int(e - s) for s, e in zip(starts, ends)]
    shifted = decode_weights_stream(level_streams, group_sizes,
                                    headers_per_group)
    wmin = int(header["wmin"])
    ws = np.concatenate([
        sh + wmin - _shift_base(fam)
        for sh, (fam, _) in zip(shifted, headers_per_group)
    ]) if shifted else np.zeros(0, dtype=np.int64)
    w = np.empty(ws.size, dtype=np.int64)
    w[gorder] = ws
    return w


# Sanity bound on the counts a third-party header may declare, per container
# byte (with a floor for tiny containers). Generous — a 1 KB container legally
# describing 200k nodes measures ~200 nodes/byte — but it keeps a hostile
# header from driving math.comb/np.empty on a 2**60 node count before any
# digest check.
_MAX_DECLARED_PER_BYTE = 1 << 12
_MIN_DECLARED_BOUND = 1 << 16


def _check_declared_counts(header: dict, blob_len: int) -> None:
    limit = max(_MIN_DECLARED_BOUND, _MAX_DECLARED_PER_BYTE * blob_len)
    for key in ("n_nodes", "n_edges"):
        value = int(header[key])
        if value < 1:
            raise ValueError(f"bad header: {key}={value} (needs >= 1)")
        if value > limit:
            raise ValueError(
                f"bad header: {key}={value:,} exceeds the sanity bound "
                f"{limit:,} for a {blob_len:,}-byte container")


_MODELS = ("dcsbm", "dcsbm+weights", "nested-dcsbm", "nested-dcsbm+weights")


def _flat_message(streams: dict) -> dict:
    return {k: streams[k] for k in
            ("header", "partition_rank", "partition_words",
             "edges_payload", "degrees_payload", "adjacency_words")}


def _nested_message(header: dict, streams: dict, n: int) -> dict:
    """Rebuild nested_coder's message dict from the stream table.

    ``n_levels`` is a third-party count like n_nodes/n_edges, so it is bounded
    before it drives any allocation: each level strictly shrinks the
    item count, so a hierarchy over n nodes has at most n levels.
    """
    L = int(header.get("n_levels", 0))
    if not 1 <= L <= n:
        raise ValueError(
            f"bad header: n_levels={L:,} for a {n:,}-node graph (a canonical "
            "hierarchy has at least 1 and at most n levels)")
    names = ["header", "levels_header"]
    for l in range(L):
        names += [f"level_{l}_partition_rank", f"level_{l}_partition_words"]
    names += [f"level_{l}_expand_payload" for l in range(L - 1, 0, -1)]
    names += ["degrees_payload", "adjacency_words"]
    missing = [k for k in names if k not in streams]
    if missing:
        raise ValueError(f"container declares n_levels={L} but is missing "
                         f"stream(s) {missing}")
    return {k: streams[k] for k in names}


def decode(blob: bytes):
    """Decode a .cz blob -> (labels, csr adjacency).

    Verifies BOTH digests the header carries before returning anything: the
    ``params_digest`` recomputed over the decoded partition, and
    the ``source_digest`` over the decoded (labels, edges[, weights]).
    """
    header, streams = unpack(blob)
    model = header.get("model_id")
    if model not in _MODELS:
        raise ValueError(f"unsupported model/payload: "
                         f"{model}/{header.get('payload')}")
    _check_declared_counts(header, len(blob))
    n = int(header["n_nodes"])
    if model.startswith("nested-"):
        levels, src, dst = nested_coder.decode_nested_dcsbm(
            _nested_message(header, streams, n), n)
        labels = levels[0]
        params = _params_digest_nested(levels)
    else:
        labels, src, dst = sbm_coder.decode_dcsbm(_flat_message(streams), n)
        params = _params_digest(labels)
    src, dst, _ = _canonical_edges(src, dst)
    if params != header.get("params_digest"):
        declared = header.get("params_digest")
        raise ValueError(
            f"params_digest mismatch: decoded partition digests to "
            f"{params[:12]}…, header says "
            f"{(declared[:12] + '…') if declared else 'nothing'}")
    if model.endswith("+weights"):
        w = _decode_weights(header, streams, labels, src, dst)
        digest = _source_digest(labels, src, dst, w=w)
    else:
        w = np.ones(src.shape[0], dtype=np.int64)
        digest = _source_digest(labels, src, dst)
    if digest != header["source_digest"]:
        raise ValueError(f"source digest mismatch: decoded {digest[:12]}…, "
                         f"header says {header['source_digest'][:12]}…")
    adj = sp.csr_matrix((w, (src, dst)), shape=(n, n))
    return labels, adj


def verify_lossless(blob: bytes, adj: sp.spmatrix | None = None,
                    labels: np.ndarray | None = None) -> bytes:
    """Prove a blob decodes losslessly and stamp the proof into its header.

    The encoders skip the encode-side self-decode (the
    dominant cost at real-graph scale), which leaves ``report.lossless``
    null — so the product path owes one explicit decode. This runs it, and
    when BOTH ``adj`` and ``labels`` are given also compares the decoded
    arrays against the source directly (a check independent of the header's
    own source digest). A failed decode or a mismatch raises and nothing is
    stamped.

    Two distinct outcomes are recorded, never conflated:

    - ``digest_verified: true`` — always, once the decode and both header
      digests pass. This is a *self*-check: the container agrees with
      itself. It says nothing about the caller's source file.
    - ``lossless: true`` — only when the source arrays were supplied and
      matched entry-for-entry. A losslessness stamp requires a source to be
      lossless *with respect to*; ``verify_lossless(blob)`` alone leaves
      ``lossless`` null.

    The source weights get the same integrality guard the encoders use:
    without it a fractional source would silently compare equal to its
    own truncation.
    """
    dec_labels, dec_adj = decode(blob)
    header, streams = unpack(blob)
    if labels is not None:
        ref = np.unique(np.asarray(labels), return_inverse=True)[1] \
            .astype(np.int64)
        if not np.array_equal(dec_labels, ref):
            raise ValueError("verify: decoded partition differs from source")
    if adj is not None:
        ref_adj = sp.csr_matrix(adj)
        try:
            _integral_weights(ref_adj.data)
        except ValueError as exc:
            raise ValueError(f"verify: {exc}") from exc
        if header.get("payload") == "topology":
            ref_adj = (ref_adj > 0).astype(np.int64)
        ref_adj = ref_adj.astype(np.int64).tocsr()
        if ref_adj.shape != dec_adj.shape or (dec_adj != ref_adj).nnz:
            raise ValueError("verify: decoded adjacency differs from source")
    against_source = adj is not None and labels is not None
    rep = header.get("report", {})
    for sub in ((rep["topology"], rep["weights"]) if "topology" in rep
                else (rep,)):
        sub["digest_verified"] = True
        sub["lossless"] = True if against_source else sub.get("lossless")
    return pack(header, streams)


_EDGELIST_DELIMS = {".csv": ",", ".tsv": "\t"}


def load_edgelist(path, delimiter: str | None = None):
    """Load a (src, dst[, weight]) edge list into (CSR int64, node id array).

    Node ids are arbitrary tokens, mapped to matrix indices 0..n-1 in sorted
    order; the returned id array is that mapping (index -> original token).
    Duplicate (src, dst) pairs sum their weights (same aggregation convention
    as the neuPrint :ConnectsTo pulls). A missing weight column means weight
    1 per row. A header row is skipped only when a third column exists and is
    non-numeric (an unweighted file must be headerless — documented limit).
    Weights are integer synapse counts: ``4.0`` is accepted, ``2.7`` is an
    error naming the row, never rounded.
    """
    path = Path(path)
    if delimiter is None:
        delimiter = _EDGELIST_DELIMS.get(path.suffix.lower(), ",")
    triples: list[tuple[str, str, int]] = []
    with open(path, newline="") as f:
        for i, row in enumerate(csv.reader(f, delimiter=delimiter)):
            if not row or not "".join(row).strip():
                continue
            if len(row) < 2:
                raise ValueError(f"{path}:{i + 1}: need >=2 columns")
            if len(row) > 2:
                try:
                    value = float(row[2])
                except ValueError:
                    if i == 0:
                        continue  # header row
                    raise ValueError(
                        f"{path}:{i + 1}: non-numeric weight {row[2]!r}")
                # integrality is checked BEFORE any rounding: a
                # rounded weight would otherwise reach the digest, and the
                # container would certify a graph the source file never had
                w = _integral_weight(value, f"{path}:{i + 1}")
            else:
                w = 1
            triples.append((row[0].strip(), row[1].strip(), w))
    ids = sorted({t[0] for t in triples} | {t[1] for t in triples})
    index = {tok: k for k, tok in enumerate(ids)}
    n = len(ids)
    src = np.array([index[t[0]] for t in triples], dtype=np.int64)
    dst = np.array([index[t[1]] for t in triples], dtype=np.int64)
    w = np.array([t[2] for t in triples], dtype=np.int64)
    adj = sp.csr_matrix((w, (src, dst)), shape=(n, n), dtype=np.int64)
    adj.sum_duplicates()
    return adj, np.array(ids)


def _load_input(path: str):
    """Dispatch czip encode input by suffix: .npz CSR, else edge list."""
    p = Path(path)
    if p.suffix.lower() == ".npz":
        return sp.load_npz(path), None
    return load_edgelist(p)


def load_hierarchy(path):
    """Level list from a czip.sbm.save_hierarchy .npz (level_0 .. level_{k-1}).

    Returned raw (padded, as graph-tool wrote it); the encoders canonicalize.
    """
    with np.load(path) as z:
        return [np.asarray(z[f"level_{i}"], dtype=np.int64)
                for i in range(len(z.files))]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="czip", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_enc = sub.add_parser(
        "encode", help="encode a CSR .npz or edge-list CSV/TSV to .cz")
    p_enc.add_argument("input", help="scipy CSR .npz adjacency, or "
                                     "(src,dst[,weight]) CSV/TSV edge list")
    p_enc.add_argument("--partition", default=None,
                       help=".npy block labels (0..B-1, len n); "
                            "or use --hierarchy / --model auto")
    p_enc.add_argument("--hierarchy", default=None,
                       help=".npz hierarchy (level_0..level_{k-1}, the layout "
                            "czip.sbm.save_hierarchy writes) — encodes the "
                            "nested model instead of the flat partition")
    p_enc.add_argument("--model", choices=("auto",), default=None,
                       help="auto: fit + select the best covariate-free "
                            "rung (czip.czip_autofit) instead of --partition")
    p_enc.add_argument("--restarts", type=int, default=3,
                       help="--model auto: flat DC-SBM fit restarts")
    p_enc.add_argument("--nested-restarts", type=int, default=1,
                       help="--model auto: nested DC-SBM fit restarts. The "
                            "nested candidate is IN by default (1 restart); "
                            "0 excludes it, skipping that fit entirely")
    p_enc.add_argument("--seed0", type=int, default=100,
                       help="--model auto: first fit seed")
    p_enc.add_argument("-o", "--out", required=True)
    p_enc.add_argument("--wmin", type=int, default=None,
                       help="encode weights (model dcsbm+weights) with W as "
                            "the weight origin: the code transmits w - W, so "
                            "every weight must be >= W. Not a filter — a "
                            "smaller weight is an error, never dropped. Omit "
                            "for topology only")
    p_enc.add_argument("--allow-weight-drop", action="store_true")
    p_enc.add_argument("--no-verify", action="store_true",
                       help="skip the explicit decode of the written blob "
                            "(leaves report.lossless null / unproven)")

    p_dec = sub.add_parser("decode", help="decode a .cz to CSR .npz")
    p_dec.add_argument("input")
    p_dec.add_argument("-o", "--out", required=True)
    p_dec.add_argument("--labels-out", default=None)

    p_info = sub.add_parser("info", help="print the header as JSON")
    p_info.add_argument("input")

    args = parser.parse_args(argv)
    if args.cmd == "encode":
        chosen = [a for a in (args.partition, args.hierarchy, args.model)
                  if a is not None]
        if len(chosen) != 1:
            parser.error("exactly one of --partition / --hierarchy / "
                         "--model auto required")
        adj, node_ids = _load_input(args.input)
        # --wmin is the weight origin, not a filter: check it against the
        # data BEFORE any (minutes-long) fit
        if args.wmin is not None and adj.nnz:
            wmin_data = _integral_weights(adj.tocsr().data).min()
            if wmin_data < args.wmin:
                parser.error(
                    f"--wmin {args.wmin} exceeds the smallest weight in "
                    f"{args.input} ({wmin_data}); --wmin is the weight "
                    "origin (the code transmits w - wmin), not a filter — "
                    "threshold the graph yourself if you meant to drop edges")
        fit_meta = None
        hierarchy = labels = None
        if args.model == "auto":
            from czip.czip_autofit import fit_auto
            result = fit_auto(adj, seed0=args.seed0, restarts=args.restarts,
                              nested_restarts=args.nested_restarts)
            labels, fit_meta = result["labels"], result["fit_meta"]
            # a nested winner is emitted as the nested model; hierarchy[0] is
            # the base partition, so `labels` stays valid for the verify pass
            hierarchy = result["hierarchy"]
            shape = (f"levels_B={fit_meta['encoded_levels_B']}"
                     if hierarchy is not None
                     else f"B={fit_meta['encoded_B']}")
            print(f"auto-fit selected {result['selected']} ({shape}, "
                  f"{fit_meta['fit_wall_s']}s fit)")
        elif args.hierarchy is not None:
            # canonicalized once here so the verify pass below can compare the
            # decoded base partition against the same normal form the encoder
            # transmits
            hierarchy = nested_coder.canonical_hierarchy(
                load_hierarchy(args.hierarchy))
            labels = hierarchy[0]
        else:
            labels = np.load(args.partition)
        model_kw = ({"hierarchy": hierarchy} if hierarchy is not None
                    else {"labels": labels})
        if args.wmin is not None:
            blob = encode_weighted(adj, wmin=args.wmin, fit_meta=fit_meta,
                                   **model_kw)
        else:
            blob = encode_topology(adj,
                                   allow_weight_drop=args.allow_weight_drop,
                                   fit_meta=fit_meta, **model_kw)
        if not args.no_verify:
            blob = verify_lossless(blob, adj=adj, labels=labels)
        Path(args.out).write_bytes(blob)
        if node_ids is not None:
            ids_path = args.out + ".node_ids.npy"
            np.save(ids_path, node_ids)
            print(f"node id map (matrix index -> input id): {ids_path}")
        hdr, _ = unpack(blob)
        proof = ("decode-verified lossless" if not args.no_verify
                 else "NOT verified (--no-verify)")
        print(f"encoded {args.input} -> {args.out}: model "
              f"{hdr['model_id']}, {8 * len(blob)} container bits "
              f"({hdr['container_overhead_bits']} overhead), {proof}")
    elif args.cmd == "decode":
        labels, adj = decode(Path(args.input).read_bytes())
        sp.save_npz(args.out, adj)
        if args.labels_out:
            np.save(args.labels_out, labels)
        # a decode checks the container against ITSELF (both header
        # digests); calling that "lossless" would claim a comparison
        # against a source file this command never saw
        print(f"decoded {args.input} -> {args.out} "
              "(params + source digests verified)")
    elif args.cmd == "info":
        header, _ = unpack(Path(args.input).read_bytes())
        print(json.dumps(header, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
