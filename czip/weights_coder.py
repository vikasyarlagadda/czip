"""Realized entropy-coder round-trip for the per-group weight code.

Encodes the CSR weight values under the fitted OWN/POOLED per-group families
(a fit report: report.json + groups.json.gz) and proves

1. realized bits ≈ theoretical L_weights + explicit header extras + slack;
2. decode is lossless (exact weight recovery),

with the adjacency topology, the block partition, wmin and the code
conventions (PARAM_RANGES, the δ sweep grid 2⁰…2⁻¹⁵, K, group order) as
decoder side-information — the weights layer conditions on the transmitted
topology exactly as the theoretical accounting does.

Message layout (decode order):

1. flags — 1 raw bit per occupied group (OWN=1/POOLED=0), group order = sorted
   group ids (decodable from topology + partition).
2. pooled header (iff any POOLED group) + per-OWN-group headers, in group
   order: family (2 raw bits), δ exponent (4 raw bits), then one grid index
   per parameter (⌈log₂ n_grid⌉ raw bits, n_grid = #{k : lo < kδ ≤ hi‖clip}).
3. weight data — one constriction categorical stream, edges ordered by
   (group, CSR position). Each weight w is coded through **chained windows**:
   symbol s₀ = min(w', K) with P(s₀=j) = pmf(j)/1 for j<K and the exact tail
   mass P(w'≥K) on the escape symbol; escapes recurse on [K, 2K), … The
   chained probabilities telescope, so the ideal codelength of this scheme is
   *exactly* −log₂ pmf(w) — no approximation enters; realized excess is
   constriction quantization drift + word padding.

**The δ gap.** The theoretical L(θ) convention charges
log₂(range/δ) per parameter but never transmits δ itself; a real decoder needs
it. The realized code pays 4 bits per header (16-value sweep grid) and this
module itemizes that gap explicitly as ``L_delta_headers_bits``.
Header grid indices additionally pay
the integer-bit ceiling over the fractional theoretical charge (< 1 bit/param),
itemized as ``L_header_ceil_bits``.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
from scipy import special

from .weights import FAMILIES, PARAM_RANGES, _N1_SUPPORT, _from_transformed, log2_pmf

WINDOW = 128           # window width K (symbols/window = K + 1 incl. escape)
CHUNK_ROWS = 400_000   # categorical prob-matrix chunk (rows × (K+1) f64 ≤ ~0.4 GB)
DELTA_EXPS = 16        # δ sweep grid 2^0 … 2^-15 (fixed convention)
PRECISION_BITS = 24    # constriction quantization precision: symbols with ideal
                       # cost beyond this get floored probability mass, so the
                       # realized code can undershoot the ideal on deep tails
                       # (negative overhead; itemized, bounded)


# ------------------------------------------------------------------ bit I/O

class BitWriter:
    """MSB-first raw bit packer (headers/flags are uniform-code segments)."""

    def __init__(self) -> None:
        self._bits: list[int] = []

    def write(self, value: int, n_bits: int) -> None:
        if value < 0 or (n_bits < 64 and value >> n_bits):
            raise ValueError(f"value {value} does not fit in {n_bits} bits")
        for i in range(n_bits - 1, -1, -1):
            self._bits.append((value >> i) & 1)

    @property
    def n_bits(self) -> int:
        return len(self._bits)

    def getvalue(self) -> bytes:
        arr = np.array(self._bits, dtype=np.uint8)
        return np.packbits(arr).tobytes()


class BitReader:
    def __init__(self, data: bytes) -> None:
        self._bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
        self._pos = 0

    def read(self, n_bits: int) -> int:
        if self._pos + n_bits > self._bits.size:
            raise ValueError("bit stream exhausted")
        out = 0
        for _ in range(n_bits):
            out = (out << 1) | int(self._bits[self._pos])
            self._pos += 1
        return out


# ------------------------------------------------------------ header codec

def _grid_points(lo: float, hi: float, delta: float) -> tuple[int, int]:
    """Inclusive grid-index range [k_min, k_max] representable after clipping.

    Encoder quantizes t → k = round(t/δ) then clips k into [k_min, k_max];
    the decoded parameter is clip(kδ, lo+1e-9, hi) — identical on both sides.
    """
    k_min = math.floor((lo + 1e-9) / delta)      # first index whose clip is lo+1e-9
    k_max = math.ceil(hi / delta)
    return k_min, k_max


def _param_to_index(tr: str, value: float, lo: float, hi: float,
                    delta: float) -> int:
    t = {"id": lambda v: v, "log": math.log,
         "logit": lambda v: math.log(v / (1.0 - v))}[tr](value)
    k = round(t / delta)
    k_min, k_max = _grid_points(lo, hi, delta)
    return int(np.clip(k, k_min, k_max)) - k_min


def _index_to_param(tr: str, idx: int, lo: float, hi: float,
                    delta: float) -> float:
    k_min, _ = _grid_points(lo, hi, delta)
    t = float(np.clip((k_min + idx) * delta, lo + 1e-9, hi))
    return _from_transformed(tr, t)


def _header_bits(family: str, delta: float) -> tuple[int, float]:
    """(integer header payload bits, theoretical fractional param bits)."""
    total = 0
    frac = 0.0
    for lo, hi, _tr in PARAM_RANGES[family].values():
        k_min, k_max = _grid_points(lo, hi, delta)
        total += max(1, math.ceil(math.log2(k_max - k_min + 1)))
        frac += max(0.0, math.log2((hi - lo) / delta))
    return total, frac


def write_header(bw: BitWriter, family: str, delta: float,
                 params_q: dict[str, float]) -> None:
    bw.write(FAMILIES.index(family), 2)
    e = round(-math.log2(delta))
    if not (0 <= e < DELTA_EXPS) or 2.0 ** -e != delta:
        raise ValueError(f"delta {delta} not on the sweep grid")
    bw.write(e, 4)
    for name, (lo, hi, tr) in PARAM_RANGES[family].items():
        k_min, k_max = _grid_points(lo, hi, delta)
        n_bits = max(1, math.ceil(math.log2(k_max - k_min + 1)))
        bw.write(_param_to_index(tr, params_q[name], lo, hi, delta), n_bits)


def read_header(br: BitReader) -> tuple[str, float, dict[str, float]]:
    family = FAMILIES[br.read(2)]
    delta = 2.0 ** -br.read(4)
    params: dict[str, float] = {}
    for name, (lo, hi, tr) in PARAM_RANGES[family].items():
        k_min, k_max = _grid_points(lo, hi, delta)
        n_bits = max(1, math.ceil(math.log2(k_max - k_min + 1)))
        params[name] = _index_to_param(tr, br.read(n_bits), lo, hi, delta)
    return family, delta, params


# ------------------------------------------------- chained-window weight code

def _survivor(family: str, params: dict[str, float], k: int) -> float:
    """P(X ≥ k) on the family's shifted support, computed tail-accurately
    (no 1 − Σpmf cancellation): geometric via log1p, zeta via Hurwitz ζ,
    lognormal via erfc, negbinom via the regularized incomplete beta."""
    if family == "geometric":
        p = params["p"]
        return float(np.exp(k * np.log1p(-p))) if k > 0 else 1.0
    if family == "zeta":
        s = params["s"]
        return float(special.zeta(s, k) / special.zeta(s, 1)) if k > 1 else 1.0
    if family == "lognormal":
        mu, sigma = params["mu"], params["sigma"]

        def sf(x: float) -> float:
            return 0.5 * float(special.erfc(x / math.sqrt(2.0)))

        den = sf((math.log(0.5) - mu) / sigma)
        if k <= 1:
            return 1.0
        return sf((math.log(k - 0.5) - mu) / sigma) / den
    if family == "negbinom":
        r, p = params["r"], params["p"]
        if k <= 0:
            return 1.0
        # P(X ≥ k) = I_{1-p}(k, r) (regularized incomplete beta)
        return float(special.betainc(k, r, 1.0 - p))
    raise ValueError(f"unknown family {family!r}")


def _window_probs(family: str, params: dict[str, float], start: int,
                  wmin_shifted_base: int, window: int = WINDOW) -> np.ndarray:
    """Probability row over window [start, start+K) + escape, conditional on
    w_shifted ≥ start. ``wmin_shifted_base`` is 0 for ℕ₀ families, 1 for ℕ₁.

    Normalization uses survivor functions, so rows stay relative-accurate at
    any escape depth and the chained ideal telescopes to −log₂ pmf exactly
    (up to float rounding), never through a 1 − Σpmf subtraction."""
    ks = np.arange(start, start + window, dtype=float)
    with np.errstate(over="ignore"):
        pmf = np.exp2(log2_pmf(family, ks, params))
    pmf = np.where(np.isfinite(pmf), pmf, 0.0)
    total = _survivor(family, params, start)
    tail_mass = _survivor(family, params, start + window)
    if total <= 0:
        raise ValueError("degenerate window: zero total mass")
    row = np.concatenate([pmf, [tail_mass]])
    return row / total


def _shift_base(family: str) -> int:
    return 1 if family in _N1_SUPPORT else 0


def _row_table(headers: list[tuple[str, dict[str, float]]], level: int,
               group_ids: np.ndarray,
               window: int = WINDOW) -> tuple[np.ndarray, np.ndarray]:
    """Window-probability rows for one escape level, built only for the groups
    present in ``group_ids``. Returns (table, compact) where ``table[compact[i]]``
    is the row for position i."""
    ug, compact = np.unique(group_ids, return_inverse=True)
    table = np.empty((ug.size, window + 1), dtype=np.float64)
    for j, gi in enumerate(ug):
        fam, par = headers[int(gi)]
        base = _shift_base(fam)
        table[j] = _window_probs(fam, par, base + level * window, base, window)
    return table, compact


def encode_weights_stream(
    w_shifted_per_group: list[np.ndarray],
    headers: list[tuple[str, dict[str, float]]],
    window: int = WINDOW,
    chunk_rows: int = CHUNK_ROWS,
) -> tuple[list[np.ndarray], float, int, float, int]:
    """Encode all groups' shifted weights; returns (level_streams, ideal_bits,
    n_symbols).

    One constriction stream per escape level. Level 0 codes every edge in
    (group, CSR position) order; level L+1 codes exactly the level-L escapes,
    in the same order — so the decoder can infer every level's symbol count
    and group schedule from what it has already decoded. Within a level the
    encoder feeds chunks of ≤ CHUNK_ROWS prob rows to one RangeEncoder, so
    the (rows × (K+1)) matrix never exceeds ~0.4 GB.
    """
    values = [np.asarray(v, dtype=np.int64) for v in w_shifted_per_group]
    group_of = np.concatenate([
        np.full(v.size, gi, dtype=np.int64) for gi, v in enumerate(values)
    ]) if values else np.zeros(0, dtype=np.int64)
    vals = np.concatenate(values) if values else np.zeros(0, dtype=np.int64)

    import constriction
    from .coder import _CATEGORICAL_FAMILY

    streams: list[np.ndarray] = []
    ideal = 0.0
    deficit = 0.0
    n_symbols = 0
    n_floored = 0
    level = 0
    active = np.arange(vals.size)
    while active.size:
        table, compact = _row_table(headers, level, group_of[active], window)
        base = np.array([_shift_base(f) for f, _ in headers],
                        dtype=np.int64)[group_of[active]] + level * window
        sym = np.minimum(vals[active] - base, window).astype(np.int32)
        if sym.min() < 0:
            raise ValueError("value below window base — schedule broken")
        encoder = constriction.stream.queue.RangeEncoder()
        for s0 in range(0, sym.size, chunk_rows):
            sl = slice(s0, min(s0 + chunk_rows, sym.size))
            rows = table[compact[sl]]
            p = rows[np.arange(sl.stop - sl.start), sym[sl]]
            if np.any(p <= 0):
                raise ValueError("zero-probability symbol — window scheme broken")
            lp = -np.log2(p)
            ideal += float(lp.sum())
            floored = lp > PRECISION_BITS
            n_floored += int(floored.sum())
            deficit += float((lp[floored] - PRECISION_BITS).sum())
            encoder.encode(np.ascontiguousarray(sym[sl]), _CATEGORICAL_FAMILY,
                           np.ascontiguousarray(rows))
        streams.append(encoder.get_compressed())
        n_symbols += sym.size
        active = active[sym == window]
        level += 1
        if level > 10_000:  # pragma: no cover
            raise RuntimeError("runaway escape chain")
    return streams, ideal, n_symbols, deficit, n_floored


def decode_weights_stream(
    streams: list[np.ndarray],
    group_sizes: list[int],
    headers: list[tuple[str, dict[str, float]]],
    window: int = WINDOW,
    chunk_rows: int = CHUNK_ROWS,
) -> list[np.ndarray]:
    """Inverse of ``encode_weights_stream``; returns shifted values per group."""
    import constriction

    from .coder import _CATEGORICAL_FAMILY

    n_total = int(sum(group_sizes))
    group_of = np.concatenate([
        np.full(sz, gi, dtype=np.int64) for gi, sz in enumerate(group_sizes)
    ]) if group_sizes else np.zeros(0, dtype=np.int64)
    vals = np.zeros(n_total, dtype=np.int64)

    level = 0
    active = np.arange(n_total)
    while active.size:
        if level >= len(streams):
            raise ValueError("stream list exhausted before schedule completed")
        table, compact = _row_table(headers, level, group_of[active], window)
        decoder = constriction.stream.queue.RangeDecoder(streams[level])
        syms = np.empty(active.size, dtype=np.int64)
        for s0 in range(0, active.size, chunk_rows):
            sl = slice(s0, min(s0 + chunk_rows, active.size))
            rows = np.ascontiguousarray(table[compact[sl]])
            syms[sl] = decoder.decode(_CATEGORICAL_FAMILY, rows)
        base = np.array([_shift_base(f) for f, _ in headers],
                        dtype=np.int64)[group_of[active]] + level * window
        finished = syms < window
        vals[active[finished]] = base[finished] + syms[finished]
        active = active[~finished]
        level += 1
    if level != len(streams):
        raise ValueError("undecoded trailing streams — stream corrupt?")

    out = []
    start = 0
    for sz in group_sizes:
        out.append(vals[start:start + sz].copy())
        start += sz
    return out


# ----------------------------------------------------------- full round-trip

def weights_roundtrip(w: np.ndarray, wmin: int, group_ids: np.ndarray,
                      result: dict[str, Any]) -> dict[str, Any]:
    """Encode + decode the full weight message; report realized vs theoretical.

    ``result`` is the ``weight_code_batched`` output (or the deserialized
    report + groups artifacts). Theoretical reference = L_total_bits from the
    artifacts; explicit extras (δ headers, ceil rounding, word padding, drift)
    are itemized on top.
    """
    w = np.asarray(w)
    group_ids = np.asarray(group_ids)
    order = np.argsort(group_ids, kind="stable")
    ws, gs = w[order], group_ids[order]
    starts = np.flatnonzero(np.r_[True, gs[1:] != gs[:-1]])
    ends = np.r_[starts[1:], gs.size]
    gids = gs[starts]

    groups = result["groups"]
    if {str(g) for g in gids} != set(groups):
        raise ValueError("group set mismatch between data and artifacts")
    pooled = result["pooled"]

    # ---- flags + headers
    bw = BitWriter()
    flags = [1 if groups[str(g)]["choice"] == "own" else 0 for g in gids]
    for f in flags:
        bw.write(f, 1)
    l_flags = len(flags)

    n_headers = 0
    frac_bits = 0.0
    int_bits = 0
    any_pooled = any(f == 0 for f in flags)
    if any_pooled:
        write_header(bw, pooled["family"], pooled["delta"], pooled["params_q"])
        hb, fb = _header_bits(pooled["family"], pooled["delta"])
        n_headers += 1
        int_bits += hb
        frac_bits += fb
    for g, f in zip(gids, flags):
        if f:
            e = groups[str(g)]
            write_header(bw, e["family"], e["delta"], e["params_q"])
            hb, fb = _header_bits(e["family"], e["delta"])
            n_headers += 1
            int_bits += hb
            frac_bits += fb
    header_bytes = bw.getvalue()
    realized_header_bits = 8 * len(header_bytes)

    # ---- decode headers back (self-check of the header codec)
    br = BitReader(header_bytes)
    flags_dec = [br.read(1) for _ in flags]
    if flags_dec != flags:
        raise AssertionError("flag decode mismatch")
    headers_per_group: list[tuple[str, dict[str, float]]] = []
    pooled_hdr = read_header(br) if any_pooled else None
    own_headers = {}
    for g, f in zip(gids, flags):
        if f:
            own_headers[str(g)] = read_header(br)
    for g, f in zip(gids, flags):
        if f:
            fam, _delta, par = own_headers[str(g)]
        else:
            assert pooled_hdr is not None
            fam, _delta, par = pooled_hdr
        headers_per_group.append((fam, par))

    # header params must reproduce the artifact params exactly (same grid)
    for (fam, par), (g, f) in zip(headers_per_group, zip(gids, flags)):
        ref = groups[str(g)]["params_q"] if f else pooled["params_q"]
        for name, val in ref.items():
            if not math.isclose(par[name], val, rel_tol=0, abs_tol=1e-12):
                raise AssertionError(
                    f"header param mismatch group {g} {name}: {par[name]} vs {val}")

    # ---- weight data
    shifted = [ws[starts[i]:ends[i]] - wmin + _shift_base(headers_per_group[i][0])
               for i in range(gids.size)]
    t0 = time.time()
    streams, ideal_data_bits, n_symbols, deficit_bound, n_floored = \
        encode_weights_stream(shifted, headers_per_group)
    decoded = decode_weights_stream(streams,
                                    [int(e - s) for s, e in zip(starts, ends)],
                                    headers_per_group)
    enc_dec_s = time.time() - t0
    lossless = all(
        np.array_equal(d, s) for d, s in zip(decoded, shifted)
    )
    # 32-bit length prefix per level stream so the segments are self-delimiting
    realized_data_bits = 32 * int(sum(s.size for s in streams)) + 32 * len(streams)

    theoretical = float(result["L_total_bits"])
    l_delta_headers = 4.0 * n_headers
    l_family_bits = 2.0 * n_headers          # == log2(4), exact
    l_header_ceil = float(int_bits) - frac_bits
    header_padding = 8 * len(header_bytes) - (l_flags + n_headers * 6 + int_bits)
    realized_total = realized_header_bits + realized_data_bits
    # consistency: theoretical L_total = flags + headers(fractional) + data ideal
    theoretical_data = theoretical - l_flags - (l_family_bits + frac_bits)
    ideal_vs_theoretical_data = ideal_data_bits - theoretical_data
    # what the realized code *should* cost: theoretical + explicit extras
    expected = (theoretical + l_delta_headers + l_header_ceil + header_padding)
    # slack: word padding + length prefix per level stream, quantization drift
    slack = 64.0 * len(streams) + 3e-4 * n_symbols
    # two-sided window: flooring of sub-2^-24 symbols can push realized BELOW
    # the ideal by at most deficit_bound (each floored symbol still costs
    # ≥ PRECISION_BITS realized)
    lower = expected - deficit_bound - slack

    return {
        "lossless": bool(lossless),
        "n_edges": int(w.size),
        "n_symbols": int(n_symbols),
        "n_groups": int(gids.size),
        "n_headers": int(n_headers),
        "theoretical_bits": theoretical,
        "theoretical_data_bits": theoretical_data,
        "ideal_vs_theoretical_data_bits": ideal_vs_theoretical_data,
        "L_delta_headers_bits": l_delta_headers,
        "L_header_ceil_bits": l_header_ceil,
        "L_header_padding_bits": float(header_padding),
        "L_family_bits_in_theoretical": l_family_bits,
        "n_level_streams": len(streams),
        "realized_header_bits": realized_header_bits,
        "realized_data_bits": realized_data_bits,
        "realized_total_bits": realized_total,
        "ideal_data_bits": ideal_data_bits,
        "expected_realized_bits": expected,
        "excess_over_expected_bits": realized_total - expected,
        "slack_budget_bits": slack,
        "deep_tail_deficit_bound_bits": deficit_bound,
        "n_floored_symbols": int(n_floored),
        "within_slack": bool(lower <= realized_total <= expected + slack),
        "encode_decode_seconds": enc_dec_s,
        "window": WINDOW,
        "chunk_rows": CHUNK_ROWS,
    }


# ------------------------------------------------------------------------ CLI

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz", type=Path, required=True)
    p.add_argument("--partition", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True,
                   help="weight-fit output dir (report.json + groups.json.gz)")
    p.add_argument("--wmin", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)

    adj = sp.load_npz(args.npz).tocsr()
    part = np.load(args.partition)
    coo = adj.tocoo()
    b = part.astype(np.int64)
    n_blocks = int(b.max()) + 1
    group_ids = b[coo.row] * n_blocks + b[coo.col]
    w = coo.data.astype(np.int64)

    report = json.loads((args.run_dir / "report.json").read_text())
    with gzip.open(args.run_dir / "groups.json.gz", "rt") as f:
        art = json.load(f)
    result = {"L_total_bits": report["L_total_bits"],
              "groups": art["groups"], "pooled": art["pooled"]}

    out = weights_roundtrip(w, args.wmin, group_ids, result)
    out["inputs"] = {"npz": str(args.npz), "partition": str(args.partition),
                     "run_dir": str(args.run_dir), "wmin": args.wmin}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in
                      ("lossless", "theoretical_bits", "realized_total_bits",
                       "excess_over_expected_bits", "slack_budget_bits",
                       "within_slack")}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
