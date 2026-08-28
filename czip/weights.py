"""Per-group weight-distribution MDL.

Extends binary ladder rows to weighted totals. The code layout being priced:

    L_weights = L_structure (1 flag bit / occupied group)
              + [pooled header: log2(n_families) + L(θ_pooled), iff used]
              + Σ_groups ( OWN: log2(n_families) + L(θ_g) + L(w_g|θ_g)
                           POOLED: L(w_g|θ_pooled) )

All families are fitted on threshold-shifted values so ≥1 and ≥5 graphs use
identical machinery: ℕ₀-support families (geometric, negbinom) model
w' = w − wmin; ℕ₁-support families (zeta, lognormal) model w'' = w − wmin + 1.

L(θ) follows the standard convention: free params only, two-part
quantized code with a δ sweep; parameter ranges are fixed constants
(``PARAM_RANGES``) so the code is decodable. All reported codelengths are
exact −Σ log₂ pmf at the (quantized) parameters — no continuous
approximations enter any number (closed-form fits are used only to *locate*
parameters).
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy import special

LN2 = np.log(2.0)

FAMILIES = ("geometric", "zeta", "lognormal", "negbinom")

# family -> param -> (lo, hi, transform); quantization happens in the
# transformed space, L(θ) charges log2((hi-lo)/δ) per param.
PARAM_RANGES: dict[str, dict[str, tuple[float, float, str]]] = {
    "geometric": {"p": (-26.0, 26.0, "logit")},
    "zeta": {"s": (1.0, 26.0, "id")},
    "lognormal": {"mu": (-26.0, 26.0, "id"), "sigma": (1e-6, 26.0, "id")},
    "negbinom": {"r": (-26.0, 26.0, "log"), "p": (-26.0, 26.0, "logit")},
}

_N1_SUPPORT = {"zeta", "lognormal"}  # families on w'' = w - wmin + 1


def _shift(w: np.ndarray, wmin: int, family: str) -> np.ndarray:
    w = np.asarray(w)
    if w.size and int(w.min()) < wmin:
        raise ValueError(f"weights below wmin={wmin}")
    return w - wmin + 1 if family in _N1_SUPPORT else w - wmin


# ------------------------------------------------------------------ pmfs

def _norm_cdf(x: np.ndarray | float) -> np.ndarray | float:
    return 0.5 * (1.0 + special.erf(np.asarray(x, dtype=float) / np.sqrt(2.0)))


def log2_pmf(family: str, k: np.ndarray, params: dict[str, float]) -> np.ndarray:
    """log₂ pmf at shifted values ``k`` (ℕ₀ or ℕ₁ per family support)."""
    k = np.asarray(k, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        if family == "geometric":
            p = params["p"]
            lp = np.where(k == 0, np.log2(p), k * np.log2(1.0 - p) + np.log2(p))
        elif family == "zeta":
            s = params["s"]
            lp = np.where(k >= 1, -s * np.log2(np.maximum(k, 1.0))
                          - np.log2(special.zeta(s)), -np.inf)
        elif family == "lognormal":
            mu, sigma = params["mu"], params["sigma"]
            kk = np.maximum(k, 1.0)
            num = (_norm_cdf((np.log(kk + 0.5) - mu) / sigma)
                   - _norm_cdf((np.log(kk - 0.5) - mu) / sigma))
            den = 1.0 - _norm_cdf((np.log(0.5) - mu) / sigma)
            lp = np.where(k >= 1, np.log2(num) - np.log2(den), -np.inf)
        elif family == "negbinom":
            r, p = params["r"], params["p"]
            ln = (special.gammaln(k + r) - special.gammaln(r)
                  - special.gammaln(k + 1.0) + r * np.log(p)
                  + k * np.log(1.0 - p))
            lp = ln / LN2
        else:
            raise ValueError(f"unknown family {family!r}")
    return lp


def codelength_bits(family: str, w: np.ndarray, wmin: int,
                    params: dict[str, float]) -> float:
    """Exact −Σ log₂ P(w_e) in bits under ``family`` at ``params``."""
    k = _shift(w, wmin, family)
    return float(-np.sum(log2_pmf(family, k, params)))


# ---------------------------------------------------------------- fitting

def _clamp_transformed(family: str, params: dict[str, float]) -> dict[str, float]:
    out = {}
    for name, val in params.items():
        lo, hi, tr = PARAM_RANGES[family][name]
        t = _to_transformed(tr, val)
        t = float(np.clip(t, lo + 1e-9, hi))
        out[name] = _from_transformed(tr, t)
    return out


def _to_transformed(tr: str, v: float) -> float:
    if tr == "id":
        return float(v)
    if tr == "log":
        return float(np.log(v))
    if tr == "logit":
        v = min(max(float(v), 1e-300), 1.0 - 1e-16)
        return float(np.log(v / (1.0 - v)))
    raise ValueError(tr)


def _from_transformed(tr: str, t: float) -> float:
    if tr == "id":
        return float(t)
    if tr == "log":
        return float(np.exp(t))
    if tr == "logit":
        return float(1.0 / (1.0 + np.exp(-t)))
    raise ValueError(tr)


def fit_family(family: str, w: np.ndarray, wmin: int) -> dict[str, float]:
    """Deterministic MLE (or MLE-locating approximation; see module doc).

    ``scipy.optimize`` is imported here, not at module scope: fitting is an
    encode-side concern, and the decode closure (the czip tool-size term)
    charges every module it pulls in.
    """
    from scipy import optimize

    k = _shift(np.asarray(w), wmin, family).astype(float)
    if family == "geometric":
        params = {"p": 1.0 / (k.mean() + 1.0)}
    elif family == "zeta":
        slogk = float(np.sum(np.log2(k)))
        n = k.size

        def nll(s: float) -> float:
            return s * slogk + n * np.log2(special.zeta(s))

        lo, hi, _ = PARAM_RANGES["zeta"]["s"]
        res = optimize.minimize_scalar(nll, bounds=(lo + 1e-9, hi),
                                       method="bounded",
                                       options={"xatol": 1e-8})
        params = {"s": float(res.x)}
    elif family == "lognormal":
        lk = np.log(k)
        params = {"mu": float(lk.mean()), "sigma": float(max(lk.std(), 1e-6))}
    elif family == "negbinom":
        m, v = float(k.mean()), float(k.var())
        if v > m + 1e-12 and m > 0:
            r0 = m * m / (v - m)
            p0 = m / v
        else:  # under/equi-dispersed: near-Poisson start
            r0 = 100.0
            p0 = r0 / (r0 + max(m, 1e-9))

        def nll_t(t: np.ndarray) -> float:
            pr = {"r": _from_transformed("log", t[0]),
                  "p": _from_transformed("logit", t[1])}
            return float(-np.sum(log2_pmf("negbinom", k, pr)))

        t0 = np.array([_to_transformed("log", r0), _to_transformed("logit", p0)])
        res = optimize.minimize(nll_t, t0, method="Nelder-Mead",
                                options={"xatol": 1e-6, "fatol": 1e-9,
                                         "maxiter": 500})
        params = {"r": _from_transformed("log", float(res.x[0])),
                  "p": _from_transformed("logit", float(res.x[1]))}
    else:
        raise ValueError(f"unknown family {family!r}")
    return _clamp_transformed(family, params)


# ----------------------------------------------------------- MDL machinery

def _quantize(family: str, params: dict[str, float],
              delta: float) -> tuple[dict[str, float], float]:
    """Quantize in transformed space; return (params_q, L_theta_bits)."""
    q: dict[str, float] = {}
    l_theta = 0.0
    for name, val in params.items():
        lo, hi, tr = PARAM_RANGES[family][name]
        t = round(_to_transformed(tr, val) / delta) * delta
        t = float(np.clip(t, lo + 1e-9, hi))
        q[name] = _from_transformed(tr, t)
        l_theta += max(0.0, float(np.log2((hi - lo) / delta)))
    return q, l_theta


def _family_total_at_delta(family: str, w: np.ndarray, wmin: int,
                           params: dict[str, float], delta: float) -> dict[str, Any]:
    params_q, l_theta = _quantize(family, params, delta)
    l_w = codelength_bits(family, w, wmin, params_q)
    return {"delta": delta, "params_q": params_q, "L_theta_bits": l_theta,
            "L_w_bits": l_w, "L_total_bits": l_theta + l_w}


def mdl_quantize_family(family: str, w: np.ndarray, wmin: int) -> dict[str, Any]:
    """Fit, then δ-sweep the two-part code; return the total minimizer."""
    params = fit_family(family, w, wmin)
    best: dict[str, Any] | None = None
    for e in range(0, 16):
        cand = _family_total_at_delta(family, w, wmin, params, 2.0 ** -e)
        if best is None or cand["L_total_bits"] < best["L_total_bits"]:
            best = cand
    assert best is not None
    best["family"] = family
    return best


def select_family(w: np.ndarray, wmin: int) -> dict[str, Any]:
    """MDL family selection; total includes log2(n_families) selection bits."""
    best: dict[str, Any] | None = None
    for fam in FAMILIES:
        cand = mdl_quantize_family(fam, w, wmin)
        if best is None or cand["L_total_bits"] < best["L_total_bits"]:
            best = cand
    assert best is not None
    best = dict(best)
    best["L_sel_bits"] = float(np.log2(len(FAMILIES)))
    best["L_total_bits"] += best["L_sel_bits"]
    return best


def weight_code(w: np.ndarray, wmin: int,
                group_ids: np.ndarray | None = None) -> dict[str, Any]:
    """Full weight code: pooled-only, or per-group OWN/POOLED with flags.

    Decode order: flags (1 bit per group, group order fixed by the
    already-transmitted topology) → pooled header iff any POOLED → per-group
    OWN headers + data. Pooled θ is fitted on ALL edges (fixed convention,
    conservative; see design doc).
    """
    w = np.asarray(w)
    if w.size and int(w.min()) < wmin:
        raise ValueError(f"weights below wmin={wmin}")
    pooled = select_family(w, wmin)
    if group_ids is None:
        return {"L_total_bits": pooled["L_total_bits"],
                "L_structure_bits": 0.0,
                "L_pooled_bits": pooled["L_total_bits"],
                "pooled": pooled, "groups": {},
                "n_groups": 1, "n_own": 0, "n_pooled": 1}

    group_ids = np.asarray(group_ids)
    gids = np.unique(group_ids)
    pooled_header = pooled["L_sel_bits"] + pooled["L_theta_bits"]
    groups: dict[str, dict[str, Any]] = {}
    n_own = n_pooled = 0
    for gid in gids:
        wg = w[group_ids == gid]
        own = select_family(wg, wmin)
        pooled_cost = codelength_bits(pooled["family"], wg, wmin,
                                      pooled["params_q"])
        if own["L_total_bits"] < pooled_cost:
            n_own += 1
            groups[str(gid)] = {"choice": "own", "family": own["family"],
                                "params_q": own["params_q"],
                                "delta": own["delta"],
                                "L_bits": own["L_total_bits"],
                                "n_edges": int(wg.size)}
        else:
            n_pooled += 1
            groups[str(gid)] = {"choice": "pooled",
                                "L_bits": pooled_cost,
                                "n_edges": int(wg.size)}
    l_structure = float(len(gids))
    l_pooled = pooled_header if n_pooled > 0 else 0.0
    l_total = l_structure + l_pooled + sum(g["L_bits"] for g in groups.values())
    return {"L_total_bits": l_total, "L_structure_bits": l_structure,
            "L_pooled_bits": l_pooled, "pooled": pooled, "groups": groups,
            "n_groups": int(len(gids)), "n_own": n_own, "n_pooled": n_pooled}


# ------------------------------------------------- batched real-graph driver

def _own_lower_bound_bits() -> float:
    """A data-independent lower bound on any OWN group's total bits.

    OWN total = log2(n_families) + L(θ) + L(w|θ) with L(w|θ) ≥ 0 and, since
    the δ sweep starts at δ=1, L(θ) ≥ Σ_params log2(hi−lo) for the chosen
    family. Groups whose pooled cost is already below this bound can be
    flagged POOLED without fitting anything.
    """
    min_theta = min(
        sum(max(0.0, float(np.log2(hi - lo))) for (lo, hi, _) in fam.values())
        for fam in PARAM_RANGES.values())
    return float(np.log2(len(FAMILIES))) + min_theta


def weight_code_batched(w: np.ndarray, wmin: int, group_ids: np.ndarray,
                        cache_max_n: int = 64) -> dict[str, Any]:
    """Scalable equivalent of ``weight_code(w, wmin, group_ids)``.

    Same code layout and (up to float summation order, ~1e-12 rel) the same
    bits as the naive path; built for ~10⁵ occupied block pairs:

    - pooled per-edge bits are computed once over all edges and segment-summed
      (one vectorized pass instead of one ``codelength_bits`` call per group);
    - groups whose pooled cost is below ``_own_lower_bound_bits()`` skip the
      4-family OWN fit entirely (no fit can win there);
    - OWN fits for small groups (n ≤ ``cache_max_n``) are memoized on the
      sorted weight multiset — real graphs repeat tiny multisets like (5,) or
      (5,6) across thousands of block pairs.

    Extra keys vs ``weight_code``: ``n_bound_skipped``, ``n_cache_hits``.
    """
    w = np.asarray(w)
    if w.size == 0:
        raise ValueError("empty weight array")
    if int(w.min()) < wmin:
        raise ValueError(f"weights below wmin={wmin}")
    group_ids = np.asarray(group_ids)
    if group_ids.shape != w.shape:
        raise ValueError("group_ids must align with w")

    pooled = select_family(w, wmin)
    pooled_header = pooled["L_sel_bits"] + pooled["L_theta_bits"]

    order = np.argsort(group_ids, kind="stable")
    ws, gs = w[order], group_ids[order]
    starts = np.flatnonzero(np.r_[True, gs[1:] != gs[:-1]])
    ends = np.r_[starts[1:], gs.size]
    gids = gs[starts]

    lp = log2_pmf(pooled["family"], _shift(ws, wmin, pooled["family"]),
                  pooled["params_q"])
    with np.errstate(invalid="ignore"):
        pooled_costs = np.add.reduceat(-lp, starts)

    bound = _own_lower_bound_bits()
    cache: dict[bytes, dict[str, Any]] = {}
    groups: dict[str, dict[str, Any]] = {}
    n_own = n_pooled = n_bound_skipped = n_cache_hits = 0
    for i in range(gids.size):
        gid, pc = gids[i], float(pooled_costs[i])
        wg = ws[starts[i]:ends[i]]
        if pc <= bound:
            n_pooled += 1
            n_bound_skipped += 1
            groups[str(gid)] = {"choice": "pooled", "L_bits": pc,
                                "n_edges": int(wg.size)}
            continue
        own: dict[str, Any] | None = None
        key: bytes | None = None
        if wg.size <= cache_max_n:
            key = np.sort(wg).astype(np.int64).tobytes()
            own = cache.get(key)
            if own is not None:
                n_cache_hits += 1
        if own is None:
            own = select_family(wg, wmin)
            if key is not None:
                cache[key] = own
        if own["L_total_bits"] < pc:
            n_own += 1
            groups[str(gid)] = {"choice": "own", "family": own["family"],
                                "params_q": own["params_q"],
                                "delta": own["delta"],
                                "L_bits": own["L_total_bits"],
                                "n_edges": int(wg.size)}
        else:
            n_pooled += 1
            groups[str(gid)] = {"choice": "pooled", "L_bits": pc,
                                "n_edges": int(wg.size)}

    l_structure = float(gids.size)
    l_pooled = pooled_header if n_pooled > 0 else 0.0
    l_total = l_structure + l_pooled + sum(g["L_bits"] for g in groups.values())
    return {"L_total_bits": l_total, "L_structure_bits": l_structure,
            "L_pooled_bits": l_pooled, "pooled": pooled, "groups": groups,
            "n_groups": int(gids.size), "n_own": n_own, "n_pooled": n_pooled,
            "n_bound_skipped": n_bound_skipped, "n_cache_hits": n_cache_hits}


def recompute_weight_bits(w: np.ndarray, wmin: int, group_ids: np.ndarray | None,
                          result: dict[str, Any]) -> float:
    """Independent recompute: re-derive L_total from the data and the fit
    report alone.

    Uses only the transmitted quantities in ``result`` (per-group choice,
    OWN family/params_q, pooled family/params_q) — never the fitting code.
    """
    w = np.asarray(w)
    pooled = result["pooled"]
    if group_ids is None:
        return (pooled["L_sel_bits"] + pooled["L_theta_bits"]
                + codelength_bits(pooled["family"], w, wmin, pooled["params_q"]))
    group_ids = np.asarray(group_ids)
    order = np.argsort(group_ids, kind="stable")
    ws, gs = w[order], group_ids[order]
    starts = np.flatnonzero(np.r_[True, gs[1:] != gs[:-1]])
    ends = np.r_[starts[1:], gs.size]
    gids = gs[starts]
    if {str(g) for g in gids} != set(result["groups"]):
        raise ValueError("group set mismatch between data and artifacts")
    # pooled per-edge bits in one vectorized pass; OWN slices priced per group
    lp = log2_pmf(pooled["family"], _shift(ws, wmin, pooled["family"]),
                  pooled["params_q"])
    with np.errstate(invalid="ignore"):
        pooled_costs = np.add.reduceat(-lp, starts)
    total = float(gids.size)  # flag bits
    any_pooled = False
    for i in range(gids.size):
        entry = result["groups"][str(gids[i])]
        if entry["choice"] == "own":
            total += (float(np.log2(len(FAMILIES)))
                      + _theta_bits_at(entry["family"], entry["delta"])
                      + codelength_bits(entry["family"], ws[starts[i]:ends[i]],
                                        wmin, entry["params_q"]))
        else:
            any_pooled = True
            total += float(pooled_costs[i])
    if any_pooled:
        total += pooled["L_sel_bits"] + pooled["L_theta_bits"]
    return total


def _theta_bits_at(family: str, delta: float) -> float:
    return sum(max(0.0, float(np.log2((hi - lo) / delta)))
               for (lo, hi, _) in PARAM_RANGES[family].values())
