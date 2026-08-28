"""Nested DC-SBM: hierarchy normal form, per-level ideal bits, expansion
codec, whole-message assembly.

Analytic layer, the expansion codec (`encode_expand`/`decode_expand`, one
level's E^{(l)} -> E^{(l-1)} step on top of czip.sbm_coder's composition
primitives), and the whole nested message
(`encode_nested_dcsbm`/`decode_nested_dcsbm`), which mirrors the flat
`czip.sbm_coder.encode_dcsbm`/`decode_dcsbm` pair segment for segment.

Formula authority: graph-tool via parity tests against
`gt.NestedBlockState.level_entropy` / `.levels[l].entropy` — the same rule as
czip.sbm_coder: nothing here transcribes a DL formula that a test has not
pinned (see tests/test_nested_coder.py).

Parity-pinned decomposition (graph-tool 3.6, directed, deg_corr base):

    state.entropy() == sum_l state.level_entropy(l)

    level 0     = adjacency + partition_dl + degree_dl
                  — the flat `edges_dl` is EXCLUDED: the hierarchy above
                    replaces it as the prior on e_rs.
    level l>=1  = expansion_l + partition_dl_l
                  — upper levels carry no degree term (deg_corr=False).

    partition_dl_l is exactly the flat partition code (sbm_coder) over the
    B_{l-1} blocks of the level below, and

    expansion_l = sum_{R,S ordered, incl. R==S}
                      log2 multiset(n_R * n_S, E^{(l)}_RS)

    with n_R the number of level-(l-1) blocks inside level-l block R and
    E^{(l)} the level-l aggregation of the base e_rs. multiset(k, m) =
    C(k + m - 1, m). The topmost expansion (where B == 1) is numerically the
    flat edges_dl of the level below, which is how the hierarchy pays for the
    term level 0 dropped.

Padding trap: `NestedBlockState.get_bs()` (and the .npz files
`czip.sbm.save_hierarchy` writes from it) are NOT dense — level arrays are
padded to the block-index space of the level below and carry raw labels up to
n. `canonical_hierarchy` densifies level by level; that is the normal form
every function here consumes.
"""

from __future__ import annotations

import math

import constriction
import numpy as np
import scipy.sparse as sp

from czip import sbm_coder


# ---------------------------------------------------------------------------
# Hierarchy normal form
# ---------------------------------------------------------------------------

def canonical_hierarchy(bs) -> list:
    """Dense, truncated hierarchy from raw `get_bs()`-style level arrays.

    Returns [labels_0 (length n, labels 0..B0-1), labels_1 (length B0), ...],
    truncated at the FIRST level with B == 1 (that level is kept; the
    all-zero trailing duplicates graph-tool pads on are dropped — they cost
    exactly zero nats). Blocks are numbered by first appearance at each
    level, so the normal form depends only on node order, never on
    graph-tool's internal block ids.

    Raises ValueError on a level array too short to index the level below, on
    negative labels, and on a hierarchy that never reaches a single block
    (an unterminated hierarchy is a *different* model: with no level above
    it, graph-tool charges the base level's edges_dl instead).
    """
    levels = [np.asarray(b, dtype=np.int64).reshape(-1) for b in bs]
    if not levels or levels[0].size == 0:
        raise ValueError("hierarchy must have a non-empty base level")
    out = []
    used = None                      # raw ids of the level below, dense order
    for l, raw in enumerate(levels):
        if used is None:
            vals = raw               # base level: one entry per node
        else:
            if int(used.max()) >= raw.shape[0]:
                raise ValueError(
                    f"level {l} has length {raw.shape[0]}, too short to index "
                    f"block {int(used.max())} of level {l - 1}")
            vals = raw[used]
        if int(vals.min()) < 0:
            raise ValueError(f"level {l} has negative block labels")
        used, dense = _dense_by_first_appearance(vals)
        sbm_coder._partition_counts(dense)   # defensive: no empty blocks
        out.append(dense)
        if used.shape[0] == 1:
            return out
    raise ValueError("hierarchy does not terminate in a single block "
                     f"(top level has B={used.shape[0]})")


def _dense_by_first_appearance(vals: np.ndarray):
    """(raw ids in dense order, dense 0..B-1 labels) for one level array."""
    uniq, first = np.unique(vals, return_index=True)
    order = np.argsort(first)                     # sorted-unique -> appearance
    rank = np.empty(uniq.shape[0], dtype=np.int64)
    rank[order] = np.arange(uniq.shape[0], dtype=np.int64)
    dense = rank[np.searchsorted(uniq, vals)]
    return uniq[order], dense.astype(np.int64)


def _check_canonical(hierarchy) -> list:
    """Validate a canonical hierarchy and return it as int64 arrays.

    Enforces exactly the normal form `canonical_hierarchy` emits: every level
    dense with no empty blocks, level l holding one label per block of level
    l-1, and a single block only at the (final) top level.
    """
    levels = [np.asarray(h, dtype=np.int64).reshape(-1) for h in hierarchy]
    if not levels:
        raise ValueError("empty hierarchy")
    prev_B = None
    for l, lab in enumerate(levels):
        _, n_l, B_l, _ = sbm_coder._partition_counts(lab)
        if prev_B is not None and n_l != prev_B:
            raise ValueError(
                f"level {l} has {n_l} items but level {l - 1} has "
                f"{prev_B} blocks")
        if B_l == 1 and l != len(levels) - 1:
            raise ValueError(
                f"level {l} already has B=1 but {len(levels) - 1 - l} levels "
                "follow; canonical hierarchies truncate there")
        prev_B = B_l
    if prev_B != 1:
        raise ValueError("hierarchy does not terminate in a single block "
                         f"(top level has B={prev_B})")
    return levels


# ---------------------------------------------------------------------------
# Per-level e_rs aggregation
# ---------------------------------------------------------------------------

def _aggregate(ers, parent: np.ndarray) -> np.ndarray:
    """Sum e_rs cells into the parent partition (directed, self-loops kept)."""
    Bp = int(parent.max()) + 1
    coo = sp.coo_matrix(ers)
    out = np.zeros((Bp, Bp), dtype=np.int64)
    np.add.at(out, (parent[coo.row], parent[coo.col]),
              coo.data.astype(np.int64))
    return out


def ers_levels(base, hierarchy) -> list:
    """E^{(l)} for every level of a canonical hierarchy.

    `base` is either the level-0 e_rs (dense array or scipy sparse, B0 x B0)
    or an (src, dst) edge-list pair, which is aggregated with the base labels
    exactly as czip.sbm_coder.encode_dcsbm builds it. Returns [E0, .., E_{L-1}];
    the top entry is the 1x1 matrix [e], which the decoder already knows from
    the message header.
    """
    levels = _check_canonical(hierarchy)
    labels = levels[0]
    B0 = int(labels.max()) + 1
    if isinstance(base, tuple):
        src, dst = (np.asarray(a, dtype=np.int64) for a in base)
        E0 = np.zeros((B0, B0), dtype=np.int64)
        np.add.at(E0, (labels[src], labels[dst]), 1)
    else:
        E0 = (base.toarray() if sp.issparse(base)
              else np.asarray(base)).astype(np.int64)
        if E0.shape != (B0, B0):
            raise ValueError(f"base e_rs has shape {E0.shape}, expected "
                             f"{(B0, B0)}")
    out = [E0]
    for l in range(1, len(levels)):
        out.append(_aggregate(out[-1], levels[l]))
    return out


# ---------------------------------------------------------------------------
# Per-level ideal bits
# ---------------------------------------------------------------------------

def expand_ideal_bits(ers_child, parent_labels) -> float:
    """Bits to expand one level: E^{(l)} -> E^{(l-1)} given the level-l split.

    Each ordered parent cell (R, S) holds E^{(l)}_RS edges spread uniformly
    over the n_R * n_S child cells it contains — a weak composition, so the
    cell costs log2 multiset(n_R n_S, E^{(l)}_RS). Empty cells cost nothing.
    """
    parent = np.asarray(parent_labels, dtype=np.int64).reshape(-1)
    _, _, Bp, n_R = sbm_coder._partition_counts(parent)
    E = _aggregate(ers_child, parent)
    bits = 0.0
    for R, S in np.argwhere(E > 0):
        m = int(E[R, S])
        k = int(n_R[R]) * int(n_R[S])
        bits += sbm_coder._log2_int(math.comb(k + m - 1, m))
    return bits


# ---------------------------------------------------------------------------
# Term codec: one level's expansion, E^{(l)} -> E^{(l-1)}
#
# Message convention — derived on both sides from already-known state, never
# transmitted: the ordered parent cells (R, S) with E^{(l)}_RS > 0 in
# row-major order, each carrying the weak composition of E^{(l)}_RS over its
# n_R * n_S child cells in row-major child-cell order (child blocks ascending
# by index within each parent block). The decoder holds E^{(l)} and the
# level-l labels of the level-(l-1) blocks before this stage runs: partitions
# travel bottom-up first, expansions top-down after.
#
# Branch, chosen WHOLESALE PER LEVEL on the level's edge total e (which the
# decoder reads off E^{(l)}, so no per-cell flag exists): e <=
# COMPOSITION_RANK_MAX_TOTAL uses sbm_coder's exact big-int weak-composition
# rank, above it sbm_coder's stars-and-bars slot walk in one range-coder
# stream. Ideal bits are the same either way, and equal expand_ideal_bits.
#
# The top level (B_l == 1) is a single parent cell over all B_{l-1}^2 child
# cells, so its emitted code is byte-identical to sbm_coder.encode_edges of
# the level below — the identity the ideal-bits parity tests already pin.
# ---------------------------------------------------------------------------

_WALK_CHUNK = 1 << 20   # slot-walk symbols buffered per incremental encode


def _expand_layout(parent_labels):
    """(parent, B_child, B_parent, n_R, order, starts) for one expansion.

    `order` sorts the child blocks by parent block (stable, so ascending
    child index within a parent), which makes each parent cell's child block
    a contiguous 2-D slice whose row-major flattening IS the transmitted
    child-cell order; `starts` holds that slice's offsets.
    """
    parent = np.asarray(parent_labels, dtype=np.int64).reshape(-1)
    _, B_child, B_parent, n_R = sbm_coder._partition_counts(parent)
    order = np.argsort(parent, kind="stable")
    starts = np.zeros(B_parent + 1, dtype=np.int64)
    np.cumsum(n_R, out=starts[1:])
    return parent, B_child, B_parent, n_R, order, starts


def _expand_walk_encode(P, starts, cells):
    """Slot-walk the nonzero parent cells into one range-coder stream.

    Walk arrays are buffered and flushed in bounded chunks (a memory knob at
    B_child^2 scale, not part of the format): the range coder is sequential,
    so chunked encode calls produce the same stream as one concatenated call
    — pinned by a test.
    """
    encoder = constriction.stream.queue.RangeEncoder()
    buf_bits, buf_probs = [], []
    n_buffered = 0
    n_symbols = 0

    def _flush():
        nonlocal n_buffered
        if n_buffered:
            encoder.encode(
                np.ascontiguousarray(np.concatenate(buf_bits), dtype=np.int32),
                sbm_coder._BERNOULLI_FAMILY,
                np.ascontiguousarray(np.concatenate(buf_probs),
                                     dtype=np.float64))
        buf_bits.clear()
        buf_probs.clear()
        n_buffered = 0

    for R, S in cells:
        comp = P[starts[R]:starts[R + 1], starts[S]:starts[S + 1]].reshape(-1)
        bits, probs = sbm_coder._walk_arrays(comp)
        if bits.size:
            buf_bits.append(bits)
            buf_probs.append(probs)
            n_buffered += int(bits.size)
            n_symbols += int(bits.size)
            if n_buffered >= _WALK_CHUNK:
                _flush()
    _flush()
    return encoder.get_compressed(), n_symbols


def encode_expand(ers_child, parent_labels, verify=True):
    """Code one level's expansion; returns (payload, report).

    The decoder needs only E^{(l)} (= the aggregation of `ers_child` into
    `parent_labels`, already decoded when this stage is read) and the level-l
    labels themselves. `verify=False` skips the encode-time self-decode; the
    caller must then prove losslessness with an explicit decode_expand pass
    (report["lossless"] is None until then).
    """
    parent, B_child, _, _, order, starts = _expand_layout(parent_labels)
    child = (ers_child.toarray() if sp.issparse(ers_child)
             else np.asarray(ers_child)).astype(np.int64)
    if child.shape != (B_child, B_child):
        raise ValueError(f"child e_rs has shape {child.shape}, expected "
                         f"{(B_child, B_child)}")
    E = _aggregate(child, parent)
    P = child[np.ix_(order, order)]
    cells = np.argwhere(E > 0)
    e = int(E.sum())

    if e <= sbm_coder.COMPOSITION_RANK_MAX_TOTAL:
        enc = sbm_coder.RankEncoder()
        for R, S in cells:
            comp = P[starts[R]:starts[R + 1],
                     starts[S]:starts[S + 1]].reshape(-1)
            m = int(E[R, S])
            enc.append(sbm_coder.weak_composition_rank(comp.tolist(), m),
                       math.comb(int(comp.size) + m - 1, m))
        payload = enc.to_bytes()
        n_symbols = 0
    else:
        payload, n_symbols = _expand_walk_encode(P, starts, cells)

    ideal = expand_ideal_bits(child, parent)
    realized = sbm_coder._payload_bits(payload)
    lossless = None
    if verify:
        lossless = bool(np.array_equal(
            decode_expand(payload, E, parent), child))
    return payload, {
        "ideal_bits": ideal,
        "realized_bits": realized,
        "overhead_bits": realized - ideal,
        "lossless": lossless,
        "n_symbols": n_symbols,
    }


def decode_expand(payload, ers_parent, parent_labels) -> np.ndarray:
    """Expand E^{(l)} into E^{(l-1)} — decoder-side inverse of encode_expand.

    Consumes only already-decoded state: the parent-level e_rs and the
    level-l labels of the level-(l-1) blocks (whose counts give n_R).
    """
    parent, B_child, B_parent, n_R, order, starts = _expand_layout(
        parent_labels)
    E = np.asarray(ers_parent, dtype=np.int64)
    if E.shape != (B_parent, B_parent):
        raise ValueError(f"parent e_rs has shape {E.shape}, expected "
                         f"{(B_parent, B_parent)}")
    e = int(E.sum())
    use_rank = e <= sbm_coder.COMPOSITION_RANK_MAX_TOTAL
    dec = sbm_coder.RankDecoder(payload) if use_rank else None
    walk_dec = (None if use_rank else
                constriction.stream.queue.RangeDecoder(
                    np.asarray(payload, dtype=np.uint32)))
    P = np.zeros((B_child, B_child), dtype=np.int64)
    for R, S in np.argwhere(E > 0):
        m = int(E[R, S])
        rows, cols = int(n_R[R]), int(n_R[S])
        k = rows * cols
        if use_rank:
            rank = dec.pop(math.comb(k + m - 1, m))
            comp = np.array(sbm_coder.weak_composition_unrank(rank, m, k),
                            dtype=np.int64)
        else:
            comp = sbm_coder._walk_decode(walk_dec, m, k)
        P[starts[R]:starts[R + 1],
          starts[S]:starts[S + 1]] = comp.reshape(rows, cols)
    out = np.zeros((B_child, B_child), dtype=np.int64)
    out[np.ix_(order, order)] = P
    return out


def nested_ideal_bits(src, dst, hierarchy) -> dict:
    """Itemized ideal bits of the whole nested message (uniform degree kind).

    Stages, in transmission order: the per-level partitions, the per-level
    expansions (top level down to the base e_rs), then the base degrees and
    the base adjacency. The base stages reuse czip.sbm_coder's flat ideal-bits
    functions verbatim — no formula is restated here.
    """
    levels = _check_canonical(hierarchy)
    labels = levels[0]
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    n = labels.shape[0]
    kout = np.bincount(src, minlength=n)
    kin = np.bincount(dst, minlength=n)
    E = ers_levels((src, dst), levels)

    part = [sbm_coder.partition_ideal_bits(lab) for lab in levels]
    expand = [0.0] + [expand_ideal_bits(E[l - 1], levels[l])
                      for l in range(1, len(levels))]
    deg = sbm_coder.degrees_uniform_ideal_bits(kout, kin, labels)
    adj = sbm_coder.adjacency_ideal_bits(src, dst, labels, kout, kin, E[0])

    total = sum(part) + sum(expand) + deg + adj
    return {
        "ideal_bits": total,
        "L_partition_bits": sum(part),
        "L_expansion_bits": sum(expand),
        "L_degree_bits": deg,
        "L_adjacency_bits": adj,
        "L_theta_bits": total - adj,
        "partition_bits_per_level": part,
        "expansion_bits_per_level": expand,
        "levels_B": [int(lab.max()) + 1 for lab in levels],
    }


def nested_total_ideal_bits(src, dst, hierarchy) -> float:
    """Total ideal bits of the nested message — parity target is
    `state.entropy(degree_dl_kind='uniform')` in bits."""
    return nested_ideal_bits(src, dst, hierarchy)["ideal_bits"]


# ---------------------------------------------------------------------------
# Full-message assembly: nested DC-SBM (degree term = 'uniform' kind)
#
# Layout, mirroring czip.sbm_coder.encode_dcsbm segment for segment, with the
# flat e_rs stage replaced by the hierarchy:
#
#   header(e) | header(L) | partitions bottom-up (l = 0..L-1)
#             | expansions top-down (l = L-1..1) | degrees | adjacency
#
# n is common knowledge; L is NOT — graph-tool charges nothing for the
# number of levels, so it travels as explicit Elias-gamma header bits that the
# report itemizes separately (header_levels_bits) instead of burying them.
#
# Every stage's model depends only on already-decoded quantities: the
# partitions fix every B_l and every group size, which is what the expansions
# need; the expansions run top-down from E^{(L-1)} = [[e]] (read off the
# header) to E^{(0)}, which is what the degree and adjacency stages need.
# ---------------------------------------------------------------------------

def encode_nested_dcsbm(src, dst, hierarchy, verify=True):
    """Encode (hierarchy, graph) as one itemized multi-segment message.

    `hierarchy` is accepted raw (padded `get_bs()`-style) and canonicalized
    here; the canonical form is what is transmitted and what
    `decode_nested_dcsbm` returns.

    verify=False skips the encode-time self-decodes of the expansion and
    adjacency stages (the dominant cost at real-graph scale); the caller must
    then prove losslessness with one explicit decode_nested_dcsbm pass
    (report["lossless"] is None until then).
    """
    levels = canonical_hierarchy(hierarchy)
    labels = levels[0]
    L = len(levels)
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    n = labels.shape[0]
    e = int(src.shape[0])
    kout = np.bincount(src, minlength=n)
    kin = np.bincount(dst, minlength=n)
    E = ers_levels((src, dst), levels)

    message = {
        "header": sbm_coder.elias_gamma_encode(e),
        "levels_header": sbm_coder.elias_gamma_encode(L),
    }
    stages: dict = {}
    reports = []

    def _stage(name, rep):
        stages[name] = {"ideal_bits": rep["ideal_bits"],
                        "realized_bits": rep["realized_bits"]}
        reports.append(rep)

    part = []
    for l, lab in enumerate(levels):
        rank, words, rep = sbm_coder.encode_partition(lab)
        message[f"level_{l}_partition_rank"] = rank
        message[f"level_{l}_partition_words"] = words
        part.append(rep["ideal_bits"])
        _stage(f"level_{l}_partition", rep)

    expand = [0.0] * L
    for l in range(L - 1, 0, -1):
        payload, rep = encode_expand(E[l - 1], levels[l], verify=verify)
        message[f"level_{l}_expand_payload"] = payload
        expand[l] = rep["ideal_bits"]
        _stage(f"level_{l}_expand", rep)

    deg_payload, deg_rep = sbm_coder.encode_degrees_uniform(kout, kin, labels)
    message["degrees_payload"] = deg_payload
    _stage("degrees", deg_rep)

    adj_words, adj_rep = sbm_coder.encode_adjacency(src, dst, labels, kout,
                                                    kin, E[0], verify=verify)
    message["adjacency_words"] = adj_words
    _stage("adjacency", adj_rep)

    header_realized = 8.0 * (len(message["header"])
                             + len(message["levels_header"]))
    ideal = sum(s["ideal_bits"] for s in stages.values())
    realized = header_realized + sum(s["realized_bits"]
                                     for s in stages.values())
    lossless_flags = [rep["lossless"] for rep in reports]
    message["report"] = {
        "ideal_bits": ideal,
        "realized_bits": realized,
        "overhead_bits": realized - ideal,
        "header_bits": float(sbm_coder.elias_gamma_bits(e)
                             + sbm_coder.elias_gamma_bits(L)),
        "header_e_bits": float(sbm_coder.elias_gamma_bits(e)),
        "header_levels_bits": float(sbm_coder.elias_gamma_bits(L)),
        "header_realized_bits": header_realized,
        "L_partition_bits": sum(part),
        "L_expansion_bits": sum(expand),
        "L_degree_bits": deg_rep["ideal_bits"],
        "L_adjacency_bits": adj_rep["ideal_bits"],
        "L_theta_bits": ideal - adj_rep["ideal_bits"],
        "partition_bits_per_level": part,
        "expansion_bits_per_level": expand,
        "levels_B": [int(lab.max()) + 1 for lab in levels],
        "n_levels": L,
        "stages": stages,
        "n_symbols": sum(rep["n_symbols"] for rep in reports),
        "n_floored_draws": adj_rep["n_floored_draws"],
        "tail_gap_bits": adj_rep["tail_gap_bits"],
        "lossless": (None if any(f is None for f in lossless_flags)
                     else bool(all(lossless_flags))),
    }
    return message


def decode_nested_dcsbm(message: dict, n: int):
    """Decode (hierarchy, src, dst) from the segments.

    n is common knowledge; the number of levels travels in the header. The
    returned hierarchy is the canonical (dense, truncated) form.
    """
    e = sbm_coder.elias_gamma_decode(message["header"])
    L = sbm_coder.elias_gamma_decode(message["levels_header"])
    levels = []
    items = n
    for l in range(L):
        lab = sbm_coder.decode_partition(
            message[f"level_{l}_partition_rank"],
            message[f"level_{l}_partition_words"], items)
        levels.append(lab)
        items = int(lab.max()) + 1
    ers = [None] * L
    ers[L - 1] = np.array([[e]], dtype=np.int64)
    for l in range(L - 1, 0, -1):
        ers[l - 1] = decode_expand(message[f"level_{l}_expand_payload"],
                                   ers[l], levels[l])
    kout, kin = sbm_coder.decode_degrees_uniform(message["degrees_payload"],
                                                 levels[0], ers[0])
    src, dst = sbm_coder.decode_adjacency(message["adjacency_words"],
                                          levels[0], kout, kin, ers[0])
    return levels, src, dst
