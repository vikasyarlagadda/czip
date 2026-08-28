"""Tests for czip/codelengths.py — analytic ER / configuration-model codelengths.

Expected values are hand-computed on a 4-node toy graph so any regression in
the bit accounting is caught against ground truth, not against the code itself.

Toy graph (directed, no self-loops): n=4, edges {0→1, 1→0, 2→3}, m=3.
Pair space N = n(n-1) = 12.
  k_out = [1, 1, 1, 0], k_in = [1, 1, 0, 1]
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import scipy.sparse as sp

from czip.codelengths import (
    cm_bernoulli_poisson,
    cm_microcanonical,
    degree_seq_cost_elias,
    degree_seq_cost_empirical,
    elias_delta_len,
    elias_gamma_len,
    er_bernoulli,
    er_microcanonical,
    ladder_models,
    log2_binom,
)

N_TOY = 4
M_TOY = 3
EDGES_TOY = [(0, 1), (1, 0), (2, 3)]
K_OUT = np.array([1, 1, 1, 0])
K_IN = np.array([1, 1, 0, 1])


def toy_adj() -> sp.csr_matrix:
    src, dst = zip(*EDGES_TOY)
    a = sp.csr_matrix(
        (np.ones(len(EDGES_TOY), dtype=np.int32), (src, dst)), shape=(N_TOY, N_TOY)
    )
    return a


# ------------------------------------------------------------------ primitives


def test_log2_binom_exact_small():
    # C(12, 3) = 220
    assert log2_binom(12, 3) == pytest.approx(math.log2(220), abs=1e-9)
    assert log2_binom(5, 0) == pytest.approx(0.0, abs=1e-12)
    assert log2_binom(5, 5) == pytest.approx(0.0, abs=1e-12)


def test_elias_gamma_known_lengths():
    # gamma(x): 2*floor(log2 x) + 1
    assert [elias_gamma_len(x) for x in [1, 2, 3, 4, 5, 8, 16]] == [1, 3, 3, 5, 5, 7, 9]


def test_elias_delta_known_lengths():
    # delta(x) = gamma(1 + floor(log2 x)) followed by floor(log2 x) bits
    assert [elias_delta_len(x) for x in [1, 2, 3, 4, 5, 8, 16, 32]] == [
        1, 4, 4, 5, 5, 8, 9, 10,
    ]


def test_elias_rejects_zero():
    with pytest.raises(ValueError):
        elias_gamma_len(0)
    with pytest.raises(ValueError):
        elias_delta_len(0)


# -------------------------------------------------------------------- ER model


def test_er_bernoulli_toy():
    r = er_bernoulli(N_TOY, M_TOY)
    # L(theta): encode m in 0..N -> log2(N+1) = log2(13)
    assert r["L_theta"] == pytest.approx(math.log2(13), abs=1e-9)
    # L(G|theta) = N * H2(m/N) = 12 * H2(0.25)
    h2 = -(0.25 * math.log2(0.25) + 0.75 * math.log2(0.75))
    assert r["L_data"] == pytest.approx(12 * h2, abs=1e-9)
    assert r["L_total"] == pytest.approx(r["L_theta"] + r["L_data"], abs=1e-12)


def test_er_microcanonical_toy():
    r = er_microcanonical(N_TOY, M_TOY)
    assert r["L_theta"] == pytest.approx(math.log2(13), abs=1e-9)
    assert r["L_data"] == pytest.approx(math.log2(220), abs=1e-9)


def test_er_micro_beats_bernoulli():
    # log2 C(N, m) <= N*H2(m/N) always
    b = er_bernoulli(N_TOY, M_TOY)
    m = er_microcanonical(N_TOY, M_TOY)
    assert m["L_data"] < b["L_data"]


def test_er_empty_graph():
    r = er_microcanonical(N_TOY, 0)
    assert r["L_data"] == pytest.approx(0.0, abs=1e-12)


# ------------------------------------------------------- configuration model


def test_cm_microcanonical_toy():
    # log2( m! / (prod k_in! * prod k_out!) ) = log2 3! = log2 6
    r = cm_microcanonical(K_OUT, K_IN)
    assert r["L_data"] == pytest.approx(math.log2(6), abs=1e-9)


def test_cm_microcanonical_with_multistub_degrees():
    # k_out = [2, 0], k_in = [0, 2], m=2: log2(2!/(2! * 2!)) = -1.0
    r = cm_microcanonical(np.array([2, 0]), np.array([0, 2]))
    assert r["L_data"] == pytest.approx(-1.0, abs=1e-9)


def test_cm_degree_sum_mismatch_raises():
    with pytest.raises(ValueError):
        cm_microcanonical(np.array([1, 0]), np.array([1, 1]))


def test_cm_bernoulli_poisson_toy_matches_bruteforce():
    # Independent brute-force per-pair sum: p_ij = 1 - exp(-k_out_i k_in_j / m)
    edges = set(EDGES_TOY)
    expected = 0.0
    for i in range(N_TOY):
        for j in range(N_TOY):
            if i == j:
                continue
            lam = K_OUT[i] * K_IN[j] / M_TOY
            if (i, j) in edges:
                expected += -math.log2(-math.expm1(-lam))
            else:
                expected += lam / math.log(2)
    src = np.array([e[0] for e in EDGES_TOY])
    dst = np.array([e[1] for e in EDGES_TOY])
    r = cm_bernoulli_poisson(K_OUT, K_IN, src, dst)
    assert r["L_data"] == pytest.approx(expected, abs=1e-9)


# ------------------------------------------------------- degree-sequence codes


def test_degree_seq_elias_gamma_toy():
    # k+1 for k_out = [2,2,2,1] -> gamma lens [3,3,3,1] = 10
    assert degree_seq_cost_elias(K_OUT, code="gamma") == pytest.approx(10.0)
    # k+1 for k_in = [2,2,1,2] -> 10
    assert degree_seq_cost_elias(K_IN, code="gamma") == pytest.approx(10.0)


def test_degree_seq_elias_delta_toy():
    # delta lens for [2,2,2,1] -> [4,4,4,1] = 13
    assert degree_seq_cost_elias(K_OUT, code="delta") == pytest.approx(13.0)


def test_degree_seq_empirical_toy():
    # k_out = [1,1,1,0]: histogram {0:1, 1:3}, D=2
    # L = delta(D) + sum_k [delta(k+1) + delta(n_k)] + log2(n!/prod n_k!)
    #   = delta(2) + [delta(1)+delta(1)] + [delta(2)+delta(3)] + log2(4!/(1!3!))
    #   = 4 + (1+1) + (4+4) + 2 = 16
    assert degree_seq_cost_empirical(K_OUT) == pytest.approx(16.0, abs=1e-9)


# ------------------------------------------------------------------ ladder API


def test_ladder_models_toy_structure():
    rows = ladder_models(toy_adj())
    names = {(r["model"], r["variant"]) for r in rows}
    assert ("er", "bernoulli") in names
    assert ("er", "microcanonical") in names
    assert ("cm", "microcanonical") in names
    assert ("cm", "bernoulli_poisson") in names
    for r in rows:
        assert r["L_total"] == pytest.approx(r["L_theta"] + r["L_data"], abs=1e-9)
        assert r["n"] == N_TOY and r["m"] == M_TOY
        # every CM row itemizes the degree-sequence code chosen for L(theta)
        if r["model"] == "cm":
            assert "theta_code" in r
            assert "L_theta_variants" in r


def test_ladder_cm_theta_uses_cheapest_code():
    rows = ladder_models(toy_adj())
    cm = [r for r in rows if r["model"] == "cm"][0]
    assert cm["L_theta"] == pytest.approx(min(cm["L_theta_variants"].values()), abs=1e-9)


def test_ladder_ignores_explicit_stored_zeros():
    """An explicitly-stored 0 is not an edge.

    `m = a.nnz` counted stored entries while the CM rows counted true edges
    off the booleanised matrix, so one loader that kept explicit zeros would
    have put two different m values in one table.
    """
    a = toy_adj().tolil()
    a[3, 0] = 0  # stored, but zero-valued: not an edge
    a = a.tocsr()
    a.data = np.append(a.data, 0)
    a.indices = np.append(a.indices, 0)
    a.indptr = np.array([0, 1, 2, 3, 4])
    assert a.nnz == M_TOY + 1  # the stored zero is counted by scipy
    rows = ladder_models(a)
    for r in rows:
        assert r["m"] == M_TOY, (r["model"], r["variant"])
        assert r["n"] == N_TOY
    # and the numbers match the clean toy graph exactly
    clean = {(r["model"], r["variant"]): r for r in ladder_models(toy_adj())}
    for r in rows:
        ref = clean[(r["model"], r["variant"])]
        assert r["L_total"] == pytest.approx(ref["L_total"], abs=1e-9)


def test_ladder_does_not_mutate_its_input():
    a = toy_adj()
    a.data = np.append(a.data, 0)
    a.indices = np.append(a.indices, 0)
    a.indptr = np.array([0, 1, 2, 3, 4])
    before = a.nnz
    ladder_models(a)
    assert a.nnz == before
