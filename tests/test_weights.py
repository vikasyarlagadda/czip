"""Tests for czip.weights — per-group weight-distribution MDL.

Reference codelengths are computed independently inside this file from the
raw pmf formulas (own gammaln/norm.cdf/zeta calls), never by calling the
module's own codelength helpers.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import special, stats

from czip import weights as W

RNG = np.random.default_rng(7)


# ---------------------------------------------------------------- pmf sanity

@pytest.mark.parametrize("family,params", [
    ("geometric", {"p": 0.3}),
    ("zeta", {"s": 2.5}),
    ("lognormal", {"mu": 0.7, "sigma": 1.1}),
    ("negbinom", {"r": 2.3, "p": 0.4}),
])
def test_pmf_sums_to_one(family, params):
    ks = np.arange(0, 200_000)
    lp = W.log2_pmf(family, ks, params)
    total = np.exp2(lp[np.isfinite(lp)]).sum()
    assert total == pytest.approx(1.0, abs=1e-3)


def test_support_shift_conventions():
    # ℕ₀ families put mass on w == wmin; ℕ₁ families on w'' = w - wmin + 1 = 1.
    for fam, params in [("geometric", {"p": 0.5}), ("negbinom", {"r": 1.0, "p": 0.5})]:
        w = np.array([5])
        assert np.isfinite(W.codelength_bits(fam, w, 5, params))
    for fam, params in [("zeta", {"s": 2.0}), ("lognormal", {"mu": 0.0, "sigma": 1.0})]:
        w = np.array([5])
        assert np.isfinite(W.codelength_bits(fam, w, 5, params))


# ------------------------------------------------------- exact codelengths

def test_geometric_codelength_closed_form():
    w = np.array([1, 2, 4, 1, 1])
    p = 0.6
    wp = w - 1
    ref = -np.sum(wp * np.log2(1 - p) + np.log2(p))
    assert W.codelength_bits("geometric", w, 1, {"p": p}) == pytest.approx(ref, rel=1e-12)


def test_zeta_codelength_closed_form():
    w = np.array([5, 6, 9, 5])
    s = 2.2
    k = w - 5 + 1
    ref = np.sum(s * np.log2(k) + np.log2(special.zeta(s)))
    assert W.codelength_bits("zeta", w, 5, {"s": s}) == pytest.approx(ref, rel=1e-12)


def test_lognormal_codelength_matches_norm_cdf():
    w = np.array([1, 2, 3, 10])
    mu, sigma = 0.5, 0.9
    k = w.astype(float)  # wmin=1 → w'' = w
    num = stats.norm.cdf((np.log(k + 0.5) - mu) / sigma) - stats.norm.cdf(
        (np.log(k - 0.5) - mu) / sigma)
    den = 1.0 - stats.norm.cdf((np.log(0.5) - mu) / sigma)
    ref = -np.sum(np.log2(num / den))
    got = W.codelength_bits("lognormal", w, 1, {"mu": mu, "sigma": sigma})
    assert got == pytest.approx(ref, rel=1e-10)


def test_negbinom_codelength_matches_gammaln():
    w = np.array([5, 5, 7, 12])
    r, p = 1.7, 0.35
    wp = (w - 5).astype(float)
    ln = (special.gammaln(wp + r) - special.gammaln(r) - special.gammaln(wp + 1)
          + r * np.log(p) + wp * np.log(1 - p))
    ref = -np.sum(ln) / np.log(2)
    got = W.codelength_bits("negbinom", w, 5, {"r": r, "p": p})
    assert got == pytest.approx(ref, rel=1e-10)


# ------------------------------------------------------------------- fitting

def test_fit_geometric_is_mle():
    w = np.array([1, 1, 2, 3, 1, 5])
    params = W.fit_family("geometric", w, 1)
    assert params["p"] == pytest.approx(1.0 / (np.mean(w - 1) + 1.0), rel=1e-12)


def test_fit_geometric_degenerate_all_wmin():
    params = W.fit_family("geometric", np.full(10, 5), 5)
    assert 0 < params["p"] <= 1
    bits = W.codelength_bits("geometric", np.full(10, 5), 5, params)
    assert bits == pytest.approx(0.0, abs=1e-6)


def test_fit_zeta_recovers_exponent():
    w = stats.zipf.rvs(2.5, size=20_000, random_state=1)
    params = W.fit_family("zeta", w, 1)
    assert params["s"] == pytest.approx(2.5, abs=0.1)


def test_fit_zeta_degenerate_all_ones_clamped():
    params = W.fit_family("zeta", np.ones(50, dtype=int), 1)
    lo, hi = W.PARAM_RANGES["zeta"]["s"][:2]
    assert lo < params["s"] <= hi


def test_fit_lognormal_closed_form():
    w = np.array([2, 3, 5, 9, 4])
    params = W.fit_family("lognormal", w, 1)
    lk = np.log(w.astype(float))
    assert params["mu"] == pytest.approx(lk.mean(), rel=1e-12)
    assert params["sigma"] == pytest.approx(lk.std(), rel=1e-9)


def test_fit_negbinom_recovers_params_roughly():
    w = stats.nbinom.rvs(3.0, 0.5, size=30_000, random_state=2) + 1  # wmin=1
    params = W.fit_family("negbinom", w, 1)
    assert params["r"] == pytest.approx(3.0, rel=0.2)
    assert params["p"] == pytest.approx(0.5, abs=0.1)


def test_fit_negbinom_underdispersed_fallback_valid():
    params = W.fit_family("negbinom", np.array([1, 2, 1, 2, 1, 2]), 1)
    assert params["r"] > 0 and 0 < params["p"] < 1
    assert np.isfinite(W.codelength_bits(
        "negbinom", np.array([1, 2, 1, 2, 1, 2]), 1, params))


# --------------------------------------------------------------- MDL machinery

def test_mdl_quantize_family_structure():
    w = stats.zipf.rvs(2.0, size=2_000, random_state=3)
    res = W.mdl_quantize_family("zeta", w, 1)
    assert res["L_total_bits"] == pytest.approx(
        res["L_theta_bits"] + res["L_w_bits"], rel=1e-12)
    # quantized param lies on the delta grid within range
    d = res["delta"]
    s_q = res["params_q"]["s"]
    lo, hi = W.PARAM_RANGES["zeta"]["s"][:2]
    assert lo < s_q <= hi
    # total is no worse than the finest-grid total
    fine = W._family_total_at_delta("zeta", w, 1, W.fit_family("zeta", w, 1), 2.0 ** -15)
    assert res["L_total_bits"] <= fine["L_total_bits"] + 1e-9


def test_select_family_is_min_plus_selection_bits():
    w = stats.zipf.rvs(1.8, size=3_000, random_state=4)
    sel = W.select_family(w, 1)
    per = {f: W.mdl_quantize_family(f, w, 1)["L_total_bits"] for f in W.FAMILIES}
    best = min(per, key=per.get)
    assert sel["family"] == best
    assert sel["L_total_bits"] == pytest.approx(per[best] + np.log2(len(W.FAMILIES)),
                                                rel=1e-12)


def test_weight_code_single_group_no_flags():
    w = stats.zipf.rvs(2.0, size=1_000, random_state=5)
    res = W.weight_code(w, 1)
    sel = W.select_family(w, 1)
    assert res["L_total_bits"] == pytest.approx(sel["L_total_bits"], rel=1e-12)
    assert res["L_structure_bits"] == 0.0


def test_weight_code_grouped_itemization_and_flags():
    # two clearly different groups, big enough that OWN pays for itself
    w1 = stats.zipf.rvs(3.0, size=4_000, random_state=6)
    w2 = stats.nbinom.rvs(5.0, 0.2, size=4_000, random_state=7) + 1
    w = np.concatenate([w1, w2])
    g = np.concatenate([np.zeros(4_000, int), np.ones(4_000, int)])
    res = W.weight_code(w, 1, group_ids=g)
    assert res["n_groups"] == 2
    assert res["L_structure_bits"] == 2.0  # one flag bit per group
    # itemization is additive
    parts = res["L_structure_bits"] + res["L_pooled_bits"] + sum(
        gr["L_bits"] for gr in res["groups"].values())
    assert res["L_total_bits"] == pytest.approx(parts, rel=1e-12)
    # grouped code should beat pooled-only on such heterogeneous data
    pooled_only = W.weight_code(w, 1)["L_total_bits"]
    assert res["L_total_bits"] < pooled_only
    assert res["n_own"] == 2 and res["n_pooled"] == 0
    # pooled θ not transmitted when nobody uses it
    assert res["L_pooled_bits"] == 0.0


def test_weight_code_tiny_groups_fall_back_to_pooled():
    w = stats.zipf.rvs(2.0, size=2_000, random_state=8)
    g = np.arange(2_000) % 500  # groups of 4 — OWN can't repay L(θ)
    res = W.weight_code(w, 1, group_ids=g)
    assert res["n_pooled"] > 0
    assert res["L_pooled_bits"] > 0.0
    # never worse than pooled-only by more than the flag bits
    pooled_only = W.weight_code(w, 1)["L_total_bits"]
    assert res["L_total_bits"] <= pooled_only + res["L_structure_bits"] + 1e-9


def test_weight_code_deterministic():
    w = stats.zipf.rvs(2.0, size=500, random_state=9)
    g = np.arange(500) % 3
    a = W.weight_code(w, 1, group_ids=g)
    b = W.weight_code(w, 1, group_ids=g)
    assert a["L_total_bits"] == b["L_total_bits"]
    assert a["groups"].keys() == b["groups"].keys()


def test_weight_code_rejects_below_wmin():
    with pytest.raises(ValueError):
        W.weight_code(np.array([4, 5, 6]), 5)


# ------------------------------------------------------------- batched driver

def _mixed_grouped_data():
    """Heterogeneous groups: 2 big OWN-worthy, many tiny, some duplicated."""
    w1 = stats.zipf.rvs(3.0, size=3_000, random_state=10)
    w2 = stats.nbinom.rvs(5.0, 0.2, size=3_000, random_state=11) + 1
    tiny_w = stats.zipf.rvs(2.0, size=800, random_state=12)
    w = np.concatenate([w1, w2, tiny_w])
    g = np.concatenate([np.zeros(3_000, int), np.ones(3_000, int),
                        2 + (np.arange(800) % 200)])  # 200 groups of 4
    return w, g


def test_batched_matches_naive_exactly():
    w, g = _mixed_grouped_data()
    naive = W.weight_code(w, 1, group_ids=g)
    batched = W.weight_code_batched(w, 1, g)
    assert batched["L_total_bits"] == pytest.approx(naive["L_total_bits"], rel=1e-9)
    assert batched["n_groups"] == naive["n_groups"]
    assert batched["n_own"] == naive["n_own"]
    assert batched["n_pooled"] == naive["n_pooled"]
    for gid, entry in naive["groups"].items():
        b = batched["groups"][gid]
        assert b["choice"] == entry["choice"]
        assert b["L_bits"] == pytest.approx(entry["L_bits"], rel=1e-9)
        if entry["choice"] == "own":
            assert b["family"] == entry["family"]


def test_batched_bound_skip_is_safe():
    # bound must be a true lower bound: every skipped group, fitted anyway,
    # would still have chosen POOLED.
    w, g = _mixed_grouped_data()
    res = W.weight_code_batched(w, 1, g)
    assert res["n_bound_skipped"] > 0
    bound = W._own_lower_bound_bits()
    for gid, entry in res["groups"].items():
        if entry["choice"] == "pooled" and entry["L_bits"] <= bound:
            wg = w[g == int(gid)]
            own = W.select_family(wg, 1)
            assert own["L_total_bits"] >= entry["L_bits"] - 1e-9


def test_batched_cache_hits_on_repeated_multisets():
    # one bulk group of 1s dominates the pooled fit, so the 100 repeated
    # single-edge (50,) groups are expensive under pooled (above the skip
    # bound) → they get fitted, and 99 of the fits are cache hits.
    w = np.concatenate([np.ones(2_000, int), np.full(100, 50)])
    g = np.concatenate([np.zeros(2_000, int), 1 + np.arange(100)])
    res = W.weight_code_batched(w, 1, g)
    assert res["n_cache_hits"] > 0
    naive = W.weight_code(w, 1, group_ids=g)
    assert res["L_total_bits"] == pytest.approx(naive["L_total_bits"], rel=1e-9)


def test_batched_input_validation():
    with pytest.raises(ValueError):
        W.weight_code_batched(np.array([], dtype=int), 1, np.array([], dtype=int))
    with pytest.raises(ValueError):
        W.weight_code_batched(np.array([1, 2]), 1, np.array([0]))
    with pytest.raises(ValueError):
        W.weight_code_batched(np.array([4, 5]), 5, np.array([0, 0]))


# ------------------------------------------------------- independent recompute

def test_recompute_matches_batched_grouped():
    w, g = _mixed_grouped_data()
    res = W.weight_code_batched(w, 1, g)
    got = W.recompute_weight_bits(w, 1, g, res)
    assert got == pytest.approx(res["L_total_bits"], rel=1e-9)


def test_recompute_matches_pooled_only():
    w = stats.zipf.rvs(2.0, size=1_000, random_state=13)
    res = W.weight_code(w, 1)
    got = W.recompute_weight_bits(w, 1, None, res)
    assert got == pytest.approx(res["L_total_bits"], rel=1e-9)


def test_recompute_rejects_group_mismatch():
    w, g = _mixed_grouped_data()
    res = W.weight_code_batched(w, 1, g)
    with pytest.raises(ValueError):
        W.recompute_weight_bits(w, 1, g + 1000, res)


def test_recompute_survives_json_round_trip():
    # artifacts go through json (string keys, plain floats) — recompute must
    # accept the deserialized form unchanged.
    import json
    w, g = _mixed_grouped_data()
    res = W.weight_code_batched(w, 1, g)
    slim = json.loads(json.dumps(
        {"pooled": {k: res["pooled"][k] for k in
                    ("family", "params_q", "L_sel_bits", "L_theta_bits")},
         "groups": res["groups"]}))
    got = W.recompute_weight_bits(w, 1, g, slim)
    assert got == pytest.approx(res["L_total_bits"], rel=1e-9)

