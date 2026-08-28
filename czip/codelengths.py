"""Analytic codelengths for the baseline ladder rungs M₀ (ER) and M₁
(directed configuration model).

Every function returns an itemized dict: ``L_theta`` (model cost),
``L_data`` (L(G|θ)) and ``L_total``, all in bits. Conventions:

- n is ALWAYS the full node universe (isolated nodes included); the pair
  space is N = n(n−1) ordered pairs (graphs are simple and self-loop-free).
- M₀ Bernoulli:       L(θ) = log₂(N+1) (encodes m, hence p̂ = m/N);
                      L(G|θ) = N·H₂(m/N).
- M₀ microcanonical:  L(θ) = log₂(N+1);  L(G|θ) = log₂ C(N, m).
- M₁ microcanonical:  L(G|θ) = log₂ m! − Σᵢ log₂ kᵢⁱⁿ! − Σᵢ log₂ kᵢᵒᵘᵗ!
  (stub-matching configuration count — the standard estimate; it counts
  multigraph configurations, so for a simple graph it upper-bounds the
  codelength of a uniform-over-simple-graphs code).
- M₁ Bernoulli (Poisson/Chung-Lu): pᵢⱼ = 1 − exp(−λᵢⱼ), λᵢⱼ = kᵢᵒᵘᵗkⱼⁱⁿ/m.
  Computed exactly in closed form: the non-edge mass Σ_{i≠j} λᵢⱼ has the
  closed form m − Σᵢ kᵢᵒᵘᵗkᵢⁱⁿ/m, so no O(n²) pass is needed.
- L(θ) for degree sequences: Elias-γ, Elias-δ (per node, k+1, canonical
  order) and an empirical-histogram code (distinct-degree count +
  (degree, count) pairs δ-coded + log₂ multinomial); the ladder rung uses
  the cheapest and itemizes all variants.

Disclosure (report-only — no behaviour change): picking the
cheapest of the three L(θ) variants is itself a choice a decoder would have
to be told, ~log₂3 ≈ 1.58 bits, and the ladder does NOT pay it. At FAFB scale
this is ~1e-9 of L(θ), but the rungs are one-and-a-half bits optimistic and
the omission is deliberate, not an oversight.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import scipy.sparse as sp
from scipy.special import gammaln

LN2 = math.log(2.0)
LOG2E = 1.0 / LN2


# ------------------------------------------------------------------ primitives


def log2_binom(n: int | float, k: int | float) -> float:
    """log₂ C(n, k) via lgamma (exact to well under a bit at FAFB scale)."""
    if k < 0 or k > n:
        raise ValueError(f"need 0 <= k <= n, got n={n}, k={k}")
    return float(gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)) * LOG2E


def _log2_factorial_sum(ks: np.ndarray) -> float:
    """Σᵢ log₂ kᵢ! for a nonnegative integer array."""
    return float(gammaln(np.asarray(ks, dtype=np.float64) + 1.0).sum()) * LOG2E


def elias_gamma_len(x: int) -> int:
    """Length in bits of the Elias-γ code for x ≥ 1."""
    if x < 1:
        raise ValueError("Elias codes are defined for integers >= 1")
    return 2 * int(math.floor(math.log2(x))) + 1


def elias_delta_len(x: int) -> int:
    """Length in bits of the Elias-δ code for x ≥ 1."""
    if x < 1:
        raise ValueError("Elias codes are defined for integers >= 1")
    b = int(math.floor(math.log2(x)))
    return b + elias_gamma_len(b + 1)


def _elias_len_vec(xs: np.ndarray, code: str) -> np.ndarray:
    xs = np.asarray(xs, dtype=np.int64)
    if np.any(xs < 1):
        raise ValueError("Elias codes are defined for integers >= 1")
    b = np.floor(np.log2(xs)).astype(np.int64)
    # guard fp edge cases at exact powers of two
    b = np.where(2 ** (b + 1) <= xs, b + 1, b)
    b = np.where(2**b > xs, b - 1, b)
    if code == "gamma":
        return 2 * b + 1
    if code == "delta":
        b2 = np.floor(np.log2(b + 1)).astype(np.int64)
        b2 = np.where(2 ** (b2 + 1) <= b + 1, b2 + 1, b2)
        b2 = np.where(2**b2 > b + 1, b2 - 1, b2)
        return b + 2 * b2 + 1
    raise ValueError(f"unknown code {code!r}")


# ------------------------------------------------------- degree-sequence codes


def degree_seq_cost_elias(ks: np.ndarray, *, code: str = "gamma") -> float:
    """Bits to encode a degree sequence as Elias-coded kᵢ+1, canonical order."""
    return float(_elias_len_vec(np.asarray(ks, dtype=np.int64) + 1, code).sum())


def degree_seq_cost_empirical(ks: np.ndarray) -> float:
    """Empirical-distribution code: δ(D) + Σ_k [δ(k+1) + δ(n_k)] over the D
    distinct degrees, plus log₂ n!/Π n_k! for the sequence given the histogram."""
    ks = np.asarray(ks, dtype=np.int64)
    n = ks.size
    values, counts = np.unique(ks, return_counts=True)
    hist_bits = float(elias_delta_len(len(values)))
    hist_bits += float(_elias_len_vec(values + 1, "delta").sum())
    hist_bits += float(_elias_len_vec(counts, "delta").sum())
    multinomial = (float(gammaln(n + 1)) - float(gammaln(counts + 1.0).sum())) * LOG2E
    return hist_bits + multinomial


# -------------------------------------------------------------------- ER model


def _pair_space(n: int) -> int:
    return n * (n - 1)


def er_bernoulli(n: int, m: int) -> dict[str, Any]:
    """M₀, Bernoulli variant."""
    N = _pair_space(n)
    l_theta = math.log2(N + 1)
    if m == 0 or m == N:
        l_data = 0.0
    else:
        p = m / N
        # log1p keeps the non-edge term exact for the tiny p of a sparse
        # connectome, where log2(1.0 - p) cancels.
        l_data = -(m * math.log2(p) + (N - m) * math.log1p(-p) * LOG2E)
    return _row("er", "bernoulli", n, m, l_theta, l_data)


def er_microcanonical(n: int, m: int) -> dict[str, Any]:
    """M₀, microcanonical (exact-m) variant."""
    N = _pair_space(n)
    return _row("er", "microcanonical", n, m, math.log2(N + 1), log2_binom(N, m))


# ------------------------------------------------------- configuration model


def _check_degrees(k_out: np.ndarray, k_in: np.ndarray) -> int:
    k_out = np.asarray(k_out)
    k_in = np.asarray(k_in)
    m_out, m_in = int(k_out.sum()), int(k_in.sum())
    if m_out != m_in:
        raise ValueError(f"degree sums disagree: sum(k_out)={m_out}, sum(k_in)={m_in}")
    return m_out


def cm_microcanonical(k_out: np.ndarray, k_in: np.ndarray) -> dict[str, Any]:
    """M₁, microcanonical: uniform over stub-matching configurations.

    L(G|θ) = log₂ m! − Σ log₂ k_outᵢ! − Σ log₂ k_inᵢ!  (L(θ) added by caller —
    the degree sequences themselves; see :func:`ladder_models`).
    """
    m = _check_degrees(k_out, k_in)
    l_data = (
        float(gammaln(m + 1)) * LOG2E
        - _log2_factorial_sum(k_out)
        - _log2_factorial_sum(k_in)
    )
    return _row("cm", "microcanonical", len(np.asarray(k_out)), m, 0.0, l_data)


def cm_bernoulli_poisson(
    k_out: np.ndarray,
    k_in: np.ndarray,
    edge_src: np.ndarray,
    edge_dst: np.ndarray,
) -> dict[str, Any]:
    """M₁, Bernoulli (Poisson/Chung-Lu): pᵢⱼ = 1 − exp(−k_outᵢ·k_inⱼ/m).

    Total bits = (Σ_{i≠j} λᵢⱼ − Σ_edges λ)·log₂e  +  Σ_edges −log₂(1−e^{−λ}),
    with Σ_{i≠j} λᵢⱼ = m − Σᵢ k_outᵢ·k_inᵢ/m in closed form.
    """
    k_out = np.asarray(k_out, dtype=np.float64)
    k_in = np.asarray(k_in, dtype=np.float64)
    m = _check_degrees(k_out, k_in)
    if m == 0:
        return _row("cm", "bernoulli_poisson", len(k_out), 0, 0.0, 0.0)
    lam_total = m - float((k_out * k_in).sum()) / m
    lam_edges = k_out[np.asarray(edge_src)] * k_in[np.asarray(edge_dst)] / m
    if np.any(lam_edges <= 0):
        raise ValueError("edge endpoint with zero degree — inconsistent input")
    edge_bits = float(-(np.log2(-np.expm1(-lam_edges))).sum())
    l_data = (lam_total - float(lam_edges.sum())) * LOG2E + edge_bits
    return _row("cm", "bernoulli_poisson", len(k_out), m, 0.0, l_data)


# ------------------------------------------------------------------ ladder API


def _row(
    model: str, variant: str, n: int, m: int, l_theta: float, l_data: float
) -> dict[str, Any]:
    return {
        "model": model,
        "variant": variant,
        "n": int(n),
        "m": int(m),
        "L_theta": float(l_theta),
        "L_data": float(l_data),
        "L_total": float(l_theta + l_data),
    }


def degree_theta_variants(k_out: np.ndarray, k_in: np.ndarray) -> dict[str, float]:
    """All degree-sequence encoding variants, bits for BOTH sequences."""
    return {
        "elias_gamma": degree_seq_cost_elias(k_out, code="gamma")
        + degree_seq_cost_elias(k_in, code="gamma"),
        "elias_delta": degree_seq_cost_elias(k_out, code="delta")
        + degree_seq_cost_elias(k_in, code="delta"),
        "empirical": degree_seq_cost_empirical(k_out)
        + degree_seq_cost_empirical(k_in),
    }


def ladder_models(adj: sp.spmatrix) -> list[dict[str, Any]]:
    """All M₀/M₁ variants for one adjacency matrix, fully itemized."""
    # copy: eliminate_zeros mutates in place, and an explicitly-stored 0 is
    # not an edge — leaving it in gave the ER rows m = nnz while the CM rows
    # counted Σk_out off the booleanised matrix, two m's in one table.
    a = sp.csr_matrix(adj, copy=True)
    a.eliminate_zeros()
    if (a.diagonal() != 0).any():
        raise ValueError("adjacency has self-loops; conventions assume none")
    n = a.shape[0]
    m = int(a.nnz)
    b = a.astype(bool).astype(np.int8)
    k_out = np.asarray(b.sum(axis=1)).ravel().astype(np.int64)
    k_in = np.asarray(b.sum(axis=0)).ravel().astype(np.int64)
    coo = b.tocoo()

    rows = [er_bernoulli(n, m), er_microcanonical(n, m)]

    theta_variants = degree_theta_variants(k_out, k_in)
    theta_code = min(theta_variants, key=theta_variants.get)
    theta_bits = theta_variants[theta_code]
    for r in (
        cm_microcanonical(k_out, k_in),
        cm_bernoulli_poisson(k_out, k_in, coo.row, coo.col),
    ):
        r["L_theta"] = theta_bits
        r["L_total"] = theta_bits + r["L_data"]
        r["theta_code"] = theta_code
        r["L_theta_variants"] = theta_variants
        r["n"] = n
        rows.append(r)
    return rows
