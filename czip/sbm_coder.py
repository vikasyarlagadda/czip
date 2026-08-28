"""Exact coder primitives for graph-tool's microcanonical DL terms.

Three primitives compose every DL term:

- P1 (here: RankEncoder/RankDecoder + rank/unrank bijections): factors that
  are "uniform over N objects" are transmitted as an exact bijective rank in
  [0, N), packed mixed-radix into one big integer so the whole stream costs
  ceil(log2 prod N) bits — ideal + <1 bit total (+ byte padding).
- P2 (urn categorical) and P3 (stub matching) live on top of czip.coder's
  constriction wrappers and are added by their own parity-tested codecs.

Formula authority: graph-tool via parity tests against czip.sbm._term —
nothing here transcribes a DL formula that hasn't been pinned by a test.
"""

from __future__ import annotations

import math

import numpy as np
import constriction

from czip import coder


# ---------------------------------------------------------------------------
# P1 — mixed-radix exact rank stream
# ---------------------------------------------------------------------------

class RankEncoder:
    """Pack a sequence of exact ranks (each uniform over its own N) into bytes.

    Little-endian mixed radix: the i-th factor is recoverable as
    value % N_i (then value //= N_i), so the decoder can learn each N_i
    online from already-decoded state, in the same order as encoding.
    """

    def __init__(self) -> None:
        self._acc = 0
        self._place = 1

    def append(self, rank: int, n_choices: int) -> None:
        if n_choices < 1:
            raise ValueError(f"n_choices must be >= 1, got {n_choices}")
        if not 0 <= rank < n_choices:
            raise ValueError(f"rank {rank} not in [0, {n_choices})")
        self._acc += rank * self._place
        self._place *= n_choices

    @property
    def ideal_bits(self) -> float:
        return _log2_int(self._place)

    def to_bytes(self) -> bytes:
        n_bytes = (self._place - 1).bit_length()  # bits needed for max acc
        n_bytes = (n_bytes + 7) // 8
        return self._acc.to_bytes(n_bytes, "little")


class RankDecoder:
    """Pop ranks in the same order they were appended by RankEncoder."""

    def __init__(self, payload: bytes) -> None:
        self._acc = int.from_bytes(payload, "little")

    def pop(self, n_choices: int) -> int:
        if n_choices < 1:
            raise ValueError(f"n_choices must be >= 1, got {n_choices}")
        rank = self._acc % n_choices
        self._acc //= n_choices
        return rank


def _log2_int(n: int) -> float:
    """log2 of a positive big int, exact-ish beyond float range."""
    if n < 1:
        raise ValueError("log2 of non-positive int")
    if n.bit_length() <= 900:  # comfortably inside float range
        return math.log2(n)
    # scale down by a power of two, exactly
    shift = n.bit_length() - 900
    return math.log2(n >> shift) + shift


# ---------------------------------------------------------------------------
# Combinatorial bijections (combinatorial number system)
# ---------------------------------------------------------------------------

def weak_composition_rank(comp, total: int) -> int:
    """Rank of a weak composition (nonneg parts summing to total) among all
    C(total + k - 1, total) weak compositions of `total` into k parts,
    ordered lexicographically by the part sequence."""
    comp = list(comp)
    if sum(comp) != total or min(comp, default=0) < 0:
        raise ValueError("not a weak composition of total")
    k = len(comp)
    rank = 0
    remaining = total
    for i, c in enumerate(comp[:-1]):
        p = k - i - 1  # cells left after this slot
        if c:
            # hockey-stick closed form of
            #   sum_{s<c} C(remaining - s + p - 1, p - 1)
            rank += (math.comb(remaining + p, p)
                     - math.comb(remaining - c + p, p))
        remaining -= c
    return rank


def weak_composition_unrank(rank: int, total: int, k: int) -> list:
    """Inverse of weak_composition_rank."""
    comp = []
    remaining = total
    for i in range(k - 1):
        p = k - i - 1  # cells left after this slot
        top = math.comb(remaining + p, p)

        def cum(c):
            # compositions with a value < c in this slot (hockey stick)
            return top - math.comb(remaining - c + p, p)

        # galloping search for the largest c with cum(c) <= rank
        # (typical part values are small, so this beats plain bisection)
        hi = 1
        while hi <= remaining and cum(hi) <= rank:
            hi *= 2
        hi = min(hi, remaining)
        lo = hi // 2  # cum(lo) <= rank by the gallop invariant
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if cum(mid) <= rank:
                lo = mid
            else:
                hi = mid - 1
        c = lo
        rank -= cum(c)
        comp.append(c)
        remaining -= c
    comp.append(remaining)
    return comp


def positive_composition_rank(comp) -> int:
    """Rank of a composition into positive parts among C(n-1, B-1)."""
    comp = list(comp)
    if min(comp) < 1:
        raise ValueError("parts must be positive")
    return weak_composition_rank([c - 1 for c in comp], sum(comp) - len(comp))


def positive_composition_unrank(rank: int, n: int, b: int) -> list:
    """Inverse of positive_composition_rank."""
    return [c + 1 for c in weak_composition_unrank(rank, n - b, b)]


# ---------------------------------------------------------------------------
# Scalable composition codec — stars-and-bars slot walk
#
# A weak composition of `total` into k parts is a string of `total` stars and
# k-1 bars over N = total + k - 1 slots. Walking the slots left to right and
# coding "bar here?" as a Bernoulli draw with P(bar) = bars_left / slots_left
# telescopes to exactly 1 / C(N, k-1) — the same multiset count the exact
# rank code realizes — so ideal bits are unchanged; realized bits carry only
# constriction quantization drift (within the acceptance criterion).
# Forced positions (bars_left == 0 or == slots_left) carry zero information
# and are skipped symmetrically on both sides.
#
# Above COMPOSITION_RANK_MAX_TOTAL the per-slot big-int combinatorics of the
# rank code are infeasible (superquadratic in total); the codecs below switch
# to this walk. The threshold is a format constant: both sides derive the
# branch from the already-decoded total, never from a transmitted flag.
# ---------------------------------------------------------------------------

COMPOSITION_RANK_MAX_TOTAL = 4096

_BERNOULLI_FAMILY = constriction.stream.model.Bernoulli(perfect=False)


def _walk_arrays(comp: np.ndarray):
    """(bits, probs) of the non-forced slot-walk positions for `comp`."""
    comp = np.asarray(comp, dtype=np.int64)
    k = comp.shape[0]
    total = int(comp.sum())
    N = total + k - 1
    if k <= 1 or N <= 0:
        return (np.array([], dtype=np.int32),
                np.array([], dtype=np.float64))
    bits = np.zeros(N, dtype=np.int32)
    bar_pos = np.cumsum(comp[:-1]) + np.arange(k - 1)
    bits[bar_pos] = 1
    bars_before = np.concatenate([[0], np.cumsum(bits[:-1])])
    b = (k - 1) - bars_before          # bars still to place (incl. here)
    t = N - np.arange(N)               # slots left (incl. here)
    mask = (b > 0) & (b < t)
    return bits[mask], b[mask].astype(np.float64) / t[mask].astype(np.float64)


def composition_walk_encode(comp):
    """Encode a weak composition via the slot walk. Returns (words, report)."""
    comp = np.asarray(comp, dtype=np.int64)
    k = comp.shape[0]
    total = int(comp.sum())
    bits, probs = _walk_arrays(comp)
    words = coder.encode_bernoulli(bits, probs)
    walk_bits = coder.ideal_bits_bernoulli(bits, probs)
    ideal = _log2_int(math.comb(total + k - 1, total)) if k else 0.0
    realized = float(32 * words.size)
    return words, {
        "ideal_bits": ideal,
        "walk_bits": walk_bits,
        "realized_bits": realized,
        "overhead_bits": realized - ideal,
        "n_symbols": int(bits.size),
    }


def _walk_decode(decoder, total: int, k: int) -> np.ndarray:
    """Decode one composition from an open RangeDecoder (slot-walk order)."""
    if k <= 0:
        raise ValueError("k must be >= 1")
    if k == 1:
        return np.array([total], dtype=np.int64)
    parts = []
    cur = 0
    b = k - 1
    t = total + k - 1
    while t > 0:
        if b == 0:                     # only stars remain
            cur += t
            break
        if b == t:                     # only bars remain
            parts.append(cur)
            cur = 0
            b -= 1
            t -= 1
            continue
        p = np.float64(b) / np.float64(t)
        sym = int(decoder.decode(_BERNOULLI_FAMILY,
                                 np.array([p], dtype=np.float64))[0])
        if sym:
            parts.append(cur)
            cur = 0
            b -= 1
        else:
            cur += 1
        t -= 1
    parts.append(cur)
    return np.array(parts, dtype=np.int64)


def composition_walk_decode(words: np.ndarray, total: int, k: int):
    decoder = constriction.stream.queue.RangeDecoder(
        np.asarray(words, dtype=np.uint32))
    return _walk_decode(decoder, total, k)


def _payload_bits(payload) -> float:
    """Realized size of a rank payload (bytes) or word stream (uint32)."""
    if isinstance(payload, (bytes, bytearray)):
        return 8.0 * len(payload)
    return 32.0 * np.asarray(payload).size


# ---------------------------------------------------------------------------
# P2 — decrementing-urn categorical (uniform over multiset arrangements)
# ---------------------------------------------------------------------------

_URN_CHUNK = 16384  # rows per vectorized prob-matrix chunk (memory bound)

_CATEGORICAL_FAMILY = constriction.stream.model.Categorical(perfect=False)


def urn_encode(labels: np.ndarray, counts: np.ndarray):
    """Encode a label sequence as a uniform arrangement of a known multiset.

    The product of urn probabilities is exactly prod(counts!)/n!, so
    ideal_bits = log2 multinomial(n; counts) — the partition-label factor of
    graph-tool's partition_dl. Returns (uint32 words, report dict).

    Probability rows are built vectorized in chunks of _URN_CHUNK and fed
    incrementally into one range-coder stream, so memory stays bounded at
    real-graph scale (n=139k, B~850 would need ~1 GB as a single matrix).
    The chunked rows are bit-identical to the per-step rows the decoder
    rebuilds (integer-valued float subtraction and the same IEEE division).
    """
    labels = np.asarray(labels, dtype=np.int64)
    counts = np.asarray(counts, dtype=np.int64)
    if labels.size != counts.sum():
        raise ValueError(
            f"{labels.size} labels but counts sum to {int(counts.sum())}")
    if labels.size == 0 or counts.size == 1:
        # empty, or single category: deterministic sequence, zero bits
        if counts.size == 1 and labels.max(initial=0) != 0:
            raise ValueError("labels out of range for single-category urn")
        return np.array([], dtype=np.uint32), {
            "ideal_bits": 0.0, "realized_bits": 0.0, "overhead_bits": 0.0,
            "n_symbols": int(labels.size)}
    n = labels.shape[0]
    k = counts.shape[0]
    if labels.min() < 0 or labels.max() >= k:
        raise ValueError("labels out of range for counts")
    encoder = constriction.stream.queue.RangeEncoder()
    remaining = counts.astype(np.float64)
    total = float(n)
    ideal = 0.0
    for a in range(0, n, _URN_CHUNK):
        lab_c = labels[a:a + _URN_CHUNK]
        m = lab_c.shape[0]
        onehot = np.zeros((m, k), dtype=np.float64)
        onehot[np.arange(m), lab_c] = 1.0
        cum = np.cumsum(onehot, axis=0)
        rem_before = remaining[None, :] - np.vstack(
            [np.zeros((1, k)), cum[:-1]])
        if rem_before[np.arange(m), lab_c].min() < 1.0:
            raise ValueError("labels exceed counts for some category")
        probs = rem_before / (total - np.arange(m, dtype=np.float64))[:, None]
        sym32 = np.ascontiguousarray(lab_c, dtype=np.int32)
        encoder.encode(sym32, _CATEGORICAL_FAMILY,
                       np.ascontiguousarray(probs))
        ideal += coder.ideal_bits_categorical(sym32, probs)
        remaining -= cum[-1]
        total -= m
    words = encoder.get_compressed()
    realized = float(32 * words.size)
    return words, {
        "ideal_bits": ideal,
        "realized_bits": realized,
        "overhead_bits": realized - ideal,
        "n_symbols": int(labels.size),
    }


def urn_decode(words: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Decode the label sequence; mirrors urn_encode's decrementing counts."""
    counts = np.asarray(counts, dtype=np.int64)
    n = int(counts.sum())
    out = np.empty(n, dtype=np.int64)
    if n == 0:
        return out
    if counts.size == 1:
        out[:] = 0
        return out
    family = constriction.stream.model.Categorical(perfect=False)
    decoder = constriction.stream.queue.RangeDecoder(
        np.asarray(words, dtype=np.uint32))
    remaining = counts.astype(np.float64).copy()
    total = float(n)
    for i in range(n):
        row = (remaining / total)[None, :]
        sym = int(decoder.decode(family, row)[0])
        out[i] = sym
        remaining[sym] -= 1.0
        total -= 1.0
    return out


# ---------------------------------------------------------------------------
# Term codec: partition_dl
# Parity-pinned decomposition (tests vs czip.sbm._term):
#   partition_dl = log2 n                    (B uniform over 1..n)
#               + log2 C(n-1, B-1)           (positive size composition)
#               + log2 n!/prod(n_r!)         (label sequence urn)
# ---------------------------------------------------------------------------

def _partition_counts(labels: np.ndarray):
    labels = np.asarray(labels, dtype=np.int64)
    n = labels.shape[0]
    B = int(labels.max()) + 1 if n else 0
    counts = np.bincount(labels, minlength=B)
    if counts.min() < 1:
        raise ValueError("labels must be dense 0..B-1 (no empty blocks)")
    return labels, n, B, counts


def partition_ideal_bits(labels: np.ndarray) -> float:
    labels, n, B, counts = _partition_counts(labels)
    multinom = math.factorial(n)
    for c in counts:
        multinom //= math.factorial(int(c))
    return (math.log2(n) + _log2_int(math.comb(n - 1, B - 1))
            + _log2_int(multinom))


def encode_partition(labels: np.ndarray):
    """Returns (rank_payload bytes, constriction words, report)."""
    labels, n, B, counts = _partition_counts(labels)
    enc = RankEncoder()
    enc.append(B - 1, n)                       # B uniform over 1..n
    enc.append(positive_composition_rank(counts.tolist()),
               math.comb(n - 1, B - 1))        # sizes composition
    words, urn_report = urn_encode(labels, counts)
    ideal = partition_ideal_bits(labels)
    realized = 8.0 * len(enc.to_bytes()) + 32.0 * words.size
    decoded = decode_partition(enc.to_bytes(), words, n)
    return enc.to_bytes(), words, {
        "ideal_bits": ideal,
        "realized_bits": realized,
        "overhead_bits": realized - ideal,
        "lossless": bool(np.array_equal(decoded, labels)),
        "n_symbols": n,
    }


def decode_partition(rank_payload: bytes, words: np.ndarray,
                     n: int) -> np.ndarray:
    dec = RankDecoder(rank_payload)
    B = dec.pop(n) + 1
    comp_rank = dec.pop(math.comb(n - 1, B - 1))
    counts = np.array(positive_composition_unrank(comp_rank, n, B),
                      dtype=np.int64)
    return urn_decode(words, counts)


# ---------------------------------------------------------------------------
# Term codec: edges_dl
# Parity-pinned decomposition: edges_dl = log2 multiset(B^2, e)
#   — the e_rs matrix is a weak composition of e over B^2 ordered cells.
# ---------------------------------------------------------------------------

def edges_ideal_bits(ers: np.ndarray) -> float:
    ers = np.asarray(ers, dtype=np.int64)
    B = ers.shape[0]
    e = int(ers.sum())
    return _log2_int(math.comb(B * B + e - 1, e))


def encode_edges(ers: np.ndarray):
    """Code the e_rs matrix; decoder needs (B, e) from earlier stages.

    e <= COMPOSITION_RANK_MAX_TOTAL: exact big-int rank (payload bytes,
    <1 bit + byte padding overhead). Above: slot-walk stream (uint32 words,
    quantization drift within the acceptance criterion). Same ideal bits
    either way.
    """
    ers = np.asarray(ers, dtype=np.int64)
    B = ers.shape[0]
    e = int(ers.sum())
    n_symbols = 0
    if e <= COMPOSITION_RANK_MAX_TOTAL:
        enc = RankEncoder()
        enc.append(weak_composition_rank(ers.reshape(-1).tolist(), e),
                   math.comb(B * B + e - 1, e))
        payload = enc.to_bytes()
    else:
        payload, walk_rep = composition_walk_encode(ers.reshape(-1))
        n_symbols = walk_rep["n_symbols"]
    ideal = edges_ideal_bits(ers)
    realized = _payload_bits(payload)
    decoded = decode_edges(payload, B, e)
    return payload, {
        "ideal_bits": ideal,
        "realized_bits": realized,
        "overhead_bits": realized - ideal,
        "lossless": bool(np.array_equal(decoded, ers)),
        "n_symbols": n_symbols,
    }


def decode_edges(payload, B: int, e: int) -> np.ndarray:
    if e <= COMPOSITION_RANK_MAX_TOTAL:
        dec = RankDecoder(payload)
        rank = dec.pop(math.comb(B * B + e - 1, e))
        flat = np.array(weak_composition_unrank(rank, e, B * B),
                        dtype=np.int64)
    else:
        flat = composition_walk_decode(payload, e, B * B)
    return flat.reshape(B, B)


# ---------------------------------------------------------------------------
# Term codec: degree_dl, degree_dl_kind='uniform'
# Parity-pinned decomposition: sum over blocks r of
#   log2 multiset(n_r, e_r_out) + log2 multiset(n_r, e_r_in)
#   — each block's out(in)-stub total spread over its nodes as a weak
#   composition, nodes in ascending index order.
# ---------------------------------------------------------------------------

def _block_stub_totals(kout, kin, labels):
    B = int(labels.max()) + 1
    er_out = np.zeros(B, dtype=np.int64)
    er_in = np.zeros(B, dtype=np.int64)
    np.add.at(er_out, labels, kout)
    np.add.at(er_in, labels, kin)
    return B, er_out, er_in


def degrees_uniform_ideal_bits(kout, kin, labels) -> float:
    kout = np.asarray(kout, dtype=np.int64)
    kin = np.asarray(kin, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    B, er_out, er_in = _block_stub_totals(kout, kin, labels)
    n_r = np.bincount(labels, minlength=B)
    bits = 0.0
    for r in range(B):
        bits += _log2_int(math.comb(int(n_r[r]) + int(er_out[r]) - 1,
                                    int(er_out[r])))
        bits += _log2_int(math.comb(int(n_r[r]) + int(er_in[r]) - 1,
                                    int(er_in[r])))
    return bits


def encode_degrees_uniform(kout, kin, labels):
    """Code per-block degree compositions (uniform kind).

    Decoder derives each block's stub totals from the already-decoded e_rs
    matrix, so only the within-block compositions are transmitted. Total
    stubs per direction (= e) <= COMPOSITION_RANK_MAX_TOTAL: exact big-int
    ranks; above: one slot-walk stream concatenating all blocks' out then in
    compositions in block order. Same ideal bits either way."""
    kout = np.asarray(kout, dtype=np.int64)
    kin = np.asarray(kin, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    B, er_out, er_in = _block_stub_totals(kout, kin, labels)
    e = int(er_out.sum())
    n_symbols = 0
    if e <= COMPOSITION_RANK_MAX_TOTAL:
        enc = RankEncoder()
        for r in range(B):
            nodes = np.where(labels == r)[0]
            for degs, er in ((kout, er_out[r]), (kin, er_in[r])):
                comp = degs[nodes].tolist()
                enc.append(weak_composition_rank(comp, int(er)),
                           math.comb(len(nodes) + int(er) - 1, int(er)))
        payload = enc.to_bytes()
    else:
        bits_parts, probs_parts = [], []
        for r in range(B):
            nodes = np.where(labels == r)[0]
            for degs in (kout, kin):
                bits, probs = _walk_arrays(degs[nodes])
                bits_parts.append(bits)
                probs_parts.append(probs)
        all_bits = np.concatenate(bits_parts)
        all_probs = np.concatenate(probs_parts)
        payload = coder.encode_bernoulli(all_bits, all_probs)
        n_symbols = int(all_bits.size)
    ideal = degrees_uniform_ideal_bits(kout, kin, labels)
    realized = _payload_bits(payload)
    dec_out, dec_in = _decode_degrees_from_totals(payload, labels,
                                                  er_out, er_in)
    return payload, {
        "ideal_bits": ideal,
        "realized_bits": realized,
        "overhead_bits": realized - ideal,
        "lossless": bool(np.array_equal(dec_out, kout)
                         and np.array_equal(dec_in, kin)),
        "n_symbols": n_symbols,
    }


def _decode_degrees_from_totals(payload, labels, er_out, er_in):
    labels = np.asarray(labels, dtype=np.int64)
    B = int(labels.max()) + 1
    n = labels.shape[0]
    e = int(np.asarray(er_out).sum())
    kout = np.zeros(n, dtype=np.int64)
    kin = np.zeros(n, dtype=np.int64)
    use_rank = e <= COMPOSITION_RANK_MAX_TOTAL
    dec = RankDecoder(payload) if use_rank else None
    walk_dec = (None if use_rank else
                constriction.stream.queue.RangeDecoder(
                    np.asarray(payload, dtype=np.uint32)))
    for r in range(B):
        nodes = np.where(labels == r)[0]
        for out_arr, er in ((kout, er_out[r]), (kin, er_in[r])):
            if use_rank:
                rank = dec.pop(math.comb(len(nodes) + int(er) - 1, int(er)))
                comp = weak_composition_unrank(rank, int(er), len(nodes))
            else:
                comp = _walk_decode(walk_dec, int(er), len(nodes))
            out_arr[nodes] = comp
    return kout, kin


def decode_degrees_uniform(payload, labels, ers):
    """Decode (kout, kin) given the partition and the e_rs matrix."""
    ers = np.asarray(ers, dtype=np.int64)
    return _decode_degrees_from_totals(payload, labels,
                                       ers.sum(axis=1), ers.sum(axis=0))


# ---------------------------------------------------------------------------
# Term codec: adjacency (deg-corr, directed multigraph ensemble)
# Parity-pinned decomposition: the microcanonical stub-matching probability
# factorizes into sequential multivariate-hypergeometric draws —
#   per block pair (r,s), pairs in row-major order:
#     1. source composition {c_u}: how many of the e_rs edges leave each
#        node u in r (drawn from r's remaining out-stubs);
#     2. target composition {d_v}: likewise for in-stubs of s;
#     3. pairing {A_uv}: per source u, its target multiset drawn from the
#        remaining target composition.
#   Every draw is hypergeometric: P(c) = C(k,c) C(K-k, m-c) / C(K, m) —
#   exact rationals, so the realized product telescopes to the ensemble
#   probability (verified against graph-tool by parity tests, incl. graphs
#   with parallel edges and self-loops; directed stubs make self-loops
#   need no special casing).
# ---------------------------------------------------------------------------

_ROW_CACHE: dict = {}
_ROW_CACHE_MAX = 1 << 20  # ~1M pmf vectors, mostly tiny; cleared when full


def _hyper_row_norm(k: int, K: int, m: int, lo: int, hi: int,
                    W: int) -> np.ndarray:
    """Normalized hypergeometric pmf (draw m of K items, k marked) placed at
    absolute positions [lo, hi] in a width-W row.

    Built with a log-space multiplicative recurrence — the SAME deterministic
    float path on encode and decode, so quantized rows can never desync.
    The pmf vector depends only on (k, K, m, lo, hi), not W, and is memoized
    verbatim (speed pass 2026-08-14): cache hits return the exact array the
    original code path produced, so the bitstream is unchanged."""
    key = (k, K, m, lo, hi)
    p = _ROW_CACHE.get(key)
    if p is None:
        cs = np.arange(lo, hi, dtype=np.float64)
        logr = (np.log(k - cs) + np.log(m - cs)
                - np.log(cs + 1.0) - np.log(K - k - m + cs + 1.0))
        logp = np.empty(cs.size + 1, dtype=np.float64)
        logp[0] = 0.0
        np.cumsum(logr, out=logp[1:])
        logp -= logp.max()
        p = np.exp(logp)
        p /= p.sum()
        if len(_ROW_CACHE) >= _ROW_CACHE_MAX:
            _ROW_CACHE.clear()
        _ROW_CACHE[key] = p
    row = np.zeros(W, dtype=np.float64)
    row[lo:hi + 1] = p
    return row


def _hyper_bits_exact(k: int, K: int, m: int, lo: int, hi: int,
                      value: int) -> float:
    """-log2 P(value) of the same hypergeometric draw, entirely in log space.

    Fallback for deep-tail draws (>~1074 bits below the row mode) where
    `_hyper_row_norm`'s np.exp underflows the true symbol to an exact float
    0.0 (first hit for real on dcsbm_ge1). Uses the identical multiplicative
    recurrence, normalized by log-sum-exp instead of exponentiating."""
    cs = np.arange(lo, hi, dtype=np.float64)
    logr = (np.log(k - cs) + np.log(m - cs)
            - np.log(cs + 1.0) - np.log(K - k - m + cs + 1.0))
    logp = np.concatenate([[0.0], np.cumsum(logr)])
    mx = logp.max()
    lse = mx + math.log(np.exp(logp - mx).sum())
    return float((lse - logp[value - lo]) / math.log(2.0))


def _adjacency_draws(labels, kout, kin, ers):
    """Draw state machine shared by encode/decode/ideal (send-driven).

    Yields (k, K, m, lo, hi, W, ctx) for every information-carrying draw and
    receives the drawn value via .send(); forced draws (single-point support)
    are resolved internally on both sides. Returns the (u, v) -> count dict.

    Scaling design (vs the original O(n*m) pass): nonzero block pairs only;
    within a stage, candidates are visited in remaining-stubs-descending
    order (deterministic, known to the decoder) so slots exhaust early and
    the forced tail is skipped; stage row widths W are derived from
    already-known state so encoder and decoder present identical row shapes.
    """
    labels = np.asarray(labels, dtype=np.int64)
    B = int(ers.shape[0])
    nodes_in = [np.flatnonzero(labels == r) for r in range(B)]
    kout_rem = np.asarray(kout, dtype=np.int64).copy()
    kin_rem = np.asarray(kin, dtype=np.int64).copy()
    A = {}

    def _composition(cand, rem, e_rs, kind, r, s):
        """Draw how the e_rs slots split over cand (multivariate hypergeom)."""
        vals = {}
        if len(cand) == 0:
            return vals
        order = cand[np.lexsort((cand, -rem[cand]))]
        W = int(min(int(rem[cand].max()), e_rs)) + 1
        K = int(rem[cand].sum())
        slots = e_rs
        for idx in range(len(order)):
            if slots == 0:
                break
            u = int(order[idx])
            k = int(rem[u])
            if K == slots:            # every remaining stub is used: forced
                for u2 in order[idx:]:
                    vals[int(u2)] = int(rem[u2])
                slots = 0
                break
            lo = max(0, slots - (K - k))
            hi = min(k, slots)
            if lo == hi:
                cu = lo
            else:
                cu = yield (k, K, slots, lo, hi, W, (kind, r, s, u))
            if cu:
                vals[u] = cu
            K -= k
            slots -= cu
        return vals

    for r, s in np.argwhere(ers > 0):
        r, s = int(r), int(s)
        e_rs = int(ers[r, s])
        blk_r, blk_s = nodes_in[r], nodes_in[s]
        c_vals = yield from _composition(
            blk_r[kout_rem[blk_r] > 0], kout_rem, e_rs, "src", r, s)
        d_vals = yield from _composition(
            blk_s[kin_rem[blk_s] > 0], kin_rem, e_rs, "dst", r, s)
        # pairing: per source u (largest c first), its target multiset is a
        # multivariate hypergeometric split of c_u over the remaining d_rem
        d_rem = dict(d_vals)
        for u, cu in sorted(c_vals.items(), key=lambda kv: (-kv[1], kv[0])):
            tgt = sorted((v for v, dv in d_rem.items() if dv > 0),
                         key=lambda v: (-d_rem[v], v))
            W = min(d_rem[tgt[0]], cu) + 1
            Tin = sum(d_rem[v] for v in tgt)
            cu_left = cu
            a_u = []
            for j, v in enumerate(tgt):
                if cu_left == 0:
                    break
                dv = d_rem[v]
                if Tin == cu_left:    # all remaining targets fill up: forced
                    for v2 in tgt[j:]:
                        a_u.append((v2, d_rem[v2]))
                    cu_left = 0
                    break
                lo = max(0, cu_left - (Tin - dv))
                hi = min(dv, cu_left)
                if lo == hi:
                    a = lo
                else:
                    a = yield (dv, Tin, cu_left, lo, hi, W,
                               ("pair", r, s, u, v))
                if a:
                    a_u.append((v, a))
                Tin -= dv
                cu_left -= a
            for v, a in a_u:
                A[(u, v)] = a
                d_rem[v] -= a
        for u, cv in c_vals.items():
            kout_rem[u] -= cv
        for v, dv in d_vals.items():
            kin_rem[v] -= dv
    return A


class _PairTruths:
    """Per-block-pair truth lookups from the grouped edge list (encode side).

    Edges are sorted once by (labels[src], labels[dst]); each pair's source
    counts, target counts, and (u, v) multiplicities are built lazily when
    the machine reaches that pair — never a scan of the whole edge list."""

    def __init__(self, src, dst, labels, B):
        src = np.asarray(src, dtype=np.int64)
        dst = np.asarray(dst, dtype=np.int64)
        keys = labels[src] * B + labels[dst]
        order = np.argsort(keys, kind="stable")
        self._src = src[order]
        self._dst = dst[order]
        self._keys = keys[order]
        self._B = B
        self._pair = None

    def _load(self, r, s):
        if self._pair == (r, s):
            return
        a, b = np.searchsorted(self._keys, [r * self._B + s,
                                            r * self._B + s + 1])
        ps, pd = self._src[a:b], self._dst[a:b]
        u_ids, u_cnt = np.unique(ps, return_counts=True)
        v_ids, v_cnt = np.unique(pd, return_counts=True)
        self._c = dict(zip(u_ids.tolist(), u_cnt.tolist()))
        self._d = dict(zip(v_ids.tolist(), v_cnt.tolist()))
        uv, uv_cnt = np.unique(np.stack([ps, pd]), axis=1,
                               return_counts=True)
        self._uv = {(int(u), int(v)): int(c)
                    for u, v, c in zip(uv[0], uv[1], uv_cnt)}
        self._pair = (r, s)

    def truth(self, ctx) -> int:
        kind, r, s = ctx[0], ctx[1], ctx[2]
        self._load(r, s)
        if kind == "src":
            return self._c.get(ctx[3], 0)
        if kind == "dst":
            return self._d.get(ctx[3], 0)
        return self._uv.get((ctx[3], ctx[4]), 0)


_ADJ_BATCH = 8192  # draws per incremental encode call

# constriction Categorical(perfect=False) hard-caps any symbol whose float
# probability is below ~2^-23 at ~23 bits (measured). Draws
# past the floor make realized bits undershoot the float ideal on heavy-tail
# data; the (n_floored, tail_gap) diagnostic quantifies that gap. REPORT-only.
QUANT_FLOOR_BITS = 23.0


def _adjacency_encode_pass(src, dst, labels, kout, kin, ers, encoder=None,
                           on_draw=None):
    """Run the draw machine with encode-side truths.

    Feeds same-width batches into `encoder` (if given) and returns
    (ideal_bits, n_symbols, (n_floored, tail_gap_bits), echoed edge dict).

    `on_draw`, if given, is called as on_draw(ctx, value, bits) for every
    information-carrying draw. It is a pure observer — it sees the same
    (ctx, value, bits) the encoder does and cannot influence the draw
    machine, so the bitstream is bit-identical with and without it."""
    labels = np.asarray(labels, dtype=np.int64)
    truths = _PairTruths(src, dst, labels, int(ers.shape[0]))
    gen = _adjacency_draws(labels, kout, kin, ers)
    ideal = 0.0
    n_symbols = 0
    n_floored = 0
    tail_gap = 0.0
    batch_syms, batch_rows, batch_w = [], [], None

    def _flush():
        if encoder is not None and batch_syms:
            encoder.encode(
                np.array(batch_syms, dtype=np.int32), _CATEGORICAL_FAMILY,
                np.ascontiguousarray(batch_rows, dtype=np.float64))
        batch_syms.clear()
        batch_rows.clear()

    try:
        req = next(gen)
        while True:
            k, K, m, lo, hi, W, ctx = req
            value = truths.truth(ctx)
            row = _hyper_row_norm(k, K, m, lo, hi, W)
            p_v = row[value]
            bits_v = (-math.log2(p_v) if p_v > 0.0
                      else _hyper_bits_exact(k, K, m, lo, hi, value))
            ideal += bits_v
            if bits_v > QUANT_FLOOR_BITS:
                n_floored += 1
                tail_gap += bits_v - QUANT_FLOOR_BITS
            n_symbols += 1
            if on_draw is not None:
                on_draw(ctx, value, bits_v)
            if encoder is not None:
                if W != batch_w or len(batch_syms) >= _ADJ_BATCH:
                    _flush()
                    batch_w = W
                batch_syms.append(value)
                batch_rows.append(row)
            req = gen.send(value)
    except StopIteration as stop:
        _flush()
        return ideal, n_symbols, (n_floored, tail_gap), stop.value


def _edge_dict(src, dst):
    A = {}
    for u, v in zip(np.asarray(src).tolist(), np.asarray(dst).tolist()):
        A[(u, v)] = A.get((u, v), 0) + 1
    return A


def adjacency_ideal_bits(src, dst, labels, kout, kin, ers) -> float:
    ideal, _, _, _ = _adjacency_encode_pass(src, dst, labels, kout, kin, ers)
    return float(ideal)


def adjacency_pair_bits(src, dst, labels, kout, kin, ers, kind_totals=None):
    """Per-ordered-pair bits attributed to the coder's *pairing* draws.

    The bits harvested here are the -log2 P(A_uv) of the single pairing draw that
    resolves the multiplicity of (u, v), in the coder's native row-major
    block-pair order. Composition (src/dst stub-split) draws attribute to
    *nodes*, not to edges, and are deliberately excluded.

    Returns (u, v, bits, total_ideal_bits). Only information-carrying draws
    appear: a pair whose multiplicity was forced by the remaining stubs cost
    the bitstream nothing and is simply absent from the result (callers score
    such edges at 0.0 attributed bits and report how many there were). Each
    (u, v) can be drawn at most once -- one block pair, one source pass -- so
    the returned keys are unique.

    `kind_totals`, if a dict is passed, is filled with the total bits spent on
    each draw kind ("src"/"dst" composition vs "pair"), because "bits
    attributed to edges" and "bits spent on pairing draws" are NOT the same
    number -- a pairing draw with value 0 spends bits asserting an *absence*
    inside a block pair, and no per-edge residual can carry those. Conflating
    the two is a real error, not a rounding difference.

    This is a read-only observation of the same pass `adjacency_ideal_bits`
    runs; it changes no draw and no bit.
    """
    us: list[int] = []
    vs: list[int] = []
    bs: list[float] = []

    def _on_draw(ctx, value, bits):
        if kind_totals is not None:
            kind_totals[ctx[0]] = kind_totals.get(ctx[0], 0.0) + bits
        if ctx[0] == "pair":
            us.append(ctx[3])
            vs.append(ctx[4])
            bs.append(bits)

    ideal, _, _, _ = _adjacency_encode_pass(src, dst, labels, kout, kin, ers,
                                            on_draw=_on_draw)
    return (np.asarray(us, dtype=np.int64), np.asarray(vs, dtype=np.int64),
            np.asarray(bs, dtype=np.float64), float(ideal))


def encode_adjacency(src, dst, labels, kout, kin, ers, verify=True):
    """Range-encode the edge multiset; decoder needs labels, degrees, e_rs."""
    encoder = constriction.stream.queue.RangeEncoder()
    ideal, n_symbols, (n_floored, tail_gap), echoed = _adjacency_encode_pass(
        src, dst, labels, kout, kin, ers, encoder)
    if echoed != _edge_dict(src, dst):
        raise AssertionError("encode pass did not echo the input multiset")
    words = encoder.get_compressed()
    realized = float(32 * words.size)
    if verify:
        dec_src, dec_dst = decode_adjacency(words, labels, kout, kin, ers)
        lossless = sorted(zip(dec_src.tolist(), dec_dst.tolist())) == \
            sorted(zip(np.asarray(src).tolist(), np.asarray(dst).tolist()))
    else:
        lossless = None
    return words, {
        "ideal_bits": float(ideal),
        "realized_bits": realized,
        "overhead_bits": realized - float(ideal),
        "lossless": lossless if lossless is None else bool(lossless),
        "n_symbols": n_symbols,
        "n_floored_draws": int(n_floored),
        "tail_gap_bits": float(tail_gap),
    }


def decode_adjacency(words, labels, kout, kin, ers):
    """Decode the edge multiset; returns (src, dst) in canonical order."""
    decoder = constriction.stream.queue.RangeDecoder(
        np.asarray(words, dtype=np.uint32))
    gen = _adjacency_draws(np.asarray(labels, dtype=np.int64),
                           kout, kin, ers)
    try:
        req = next(gen)
        while True:
            k, K, m, lo, hi, W, _ = req
            row = _hyper_row_norm(k, K, m, lo, hi, W)
            sym = int(decoder.decode(_CATEGORICAL_FAMILY, row[None, :])[0])
            req = gen.send(sym)
    except StopIteration as stop:
        A = stop.value
    src, dst = [], []
    for (u, v) in sorted(A):
        src.extend([u] * A[(u, v)])
        dst.extend([v] * A[(u, v)])
    return np.array(src, dtype=np.int64), np.array(dst, dtype=np.int64)


# ---------------------------------------------------------------------------
# Full-message assembly: flat DC-SBM (degree term = 'uniform' kind)
# Layout: header(e, Elias-gamma) | partition |
# e_rs matrix | degrees | adjacency. n is common knowledge (the node
# count is decoder side-information, never coded).
# Each stage's model depends only on already-decoded quantities.
# ---------------------------------------------------------------------------

def elias_gamma_encode(x: int) -> bytes:
    """Elias-gamma code for x >= 1, padded to whole bytes (zeros)."""
    if x < 1:
        raise ValueError("Elias-gamma needs x >= 1")
    nbits = x.bit_length()
    bits = "0" * (nbits - 1) + format(x, "b")
    bits += "0" * (-len(bits) % 8)
    return int(bits, 2).to_bytes(len(bits) // 8, "big")


def elias_gamma_decode(payload: bytes) -> int:
    bits = format(int.from_bytes(payload, "big"), f"0{8 * len(payload)}b")
    zeros = 0
    while bits[zeros] == "0":
        zeros += 1
    return int(bits[zeros:2 * zeros + 1], 2)


def elias_gamma_bits(x: int) -> int:
    return 2 * x.bit_length() - 1


def encode_dcsbm(src, dst, labels, verify=True):
    """Encode (partition, graph) as one itemized multi-segment message.

    verify=False skips the encode-time adjacency self-decode (the dominant
    cost at real-graph scale); the caller must then prove losslessness with
    one explicit decode_dcsbm pass (report["lossless"] is None until then).
    """
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    n = labels.shape[0]
    e = int(src.shape[0])
    kout = np.bincount(src, minlength=n)
    kin = np.bincount(dst, minlength=n)
    B = int(labels.max()) + 1
    ers = np.zeros((B, B), dtype=np.int64)
    np.add.at(ers, (labels[src], labels[dst]), 1)

    header = elias_gamma_encode(e)
    part_rank, part_words, part_rep = encode_partition(labels)
    edges_payload, edges_rep = encode_edges(ers)
    deg_payload, deg_rep = encode_degrees_uniform(kout, kin, labels)
    adj_words, adj_rep = encode_adjacency(src, dst, labels, kout, kin, ers,
                                          verify=verify)

    header_bits = float(elias_gamma_bits(e))
    ideal = (part_rep["ideal_bits"] + edges_rep["ideal_bits"]
             + deg_rep["ideal_bits"] + adj_rep["ideal_bits"])
    realized = (8.0 * len(header) + 8.0 * len(part_rank)
                + 32.0 * part_words.size + _payload_bits(edges_payload)
                + _payload_bits(deg_payload) + 32.0 * adj_words.size)
    return {
        "header": header,
        "partition_rank": part_rank,
        "partition_words": part_words,
        "edges_payload": edges_payload,
        "degrees_payload": deg_payload,
        "adjacency_words": adj_words,
        "report": {
            "ideal_bits": ideal,
            "realized_bits": realized,
            "overhead_bits": realized - ideal,
            "header_bits": header_bits,
            "L_partition_bits": part_rep["ideal_bits"],
            "L_edges_bits": edges_rep["ideal_bits"],
            "L_degree_bits": deg_rep["ideal_bits"],
            "L_adjacency_bits": adj_rep["ideal_bits"],
            "n_symbols": (part_rep["n_symbols"] + adj_rep["n_symbols"]
                          + edges_rep["n_symbols"] + deg_rep["n_symbols"]),
            "n_floored_draws": adj_rep["n_floored_draws"],
            "tail_gap_bits": adj_rep["tail_gap_bits"],
            "lossless": (None if adj_rep["lossless"] is None else
                         bool(part_rep["lossless"] and edges_rep["lossless"]
                              and deg_rep["lossless"]
                              and adj_rep["lossless"])),
        },
    }


def decode_dcsbm(message: dict, n: int):
    """Decode (labels, src, dst) from the segments; n is common knowledge."""
    e = elias_gamma_decode(message["header"])
    labels = decode_partition(message["partition_rank"],
                              message["partition_words"], n)
    B = int(labels.max()) + 1
    ers = decode_edges(message["edges_payload"], B, e)
    kout, kin = decode_degrees_uniform(message["degrees_payload"],
                                       labels, ers)
    src, dst = decode_adjacency(message["adjacency_words"],
                                labels, kout, kin, ers)
    return labels, src, dst
