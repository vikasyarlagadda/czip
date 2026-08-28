"""Auto-fit ladder selection for `czip encode --model auto`.

For an arbitrary directed connectome with no hand-matched covariates, score
the covariate-free rungs and hand czip the best ENCODABLE partition:

- ``er``      — microcanonical ER (analytic; pair space n(n-1), or n^2 when
                self-loops are present, since plain ER cannot express them).
- ``degree``  — configuration model: directed stub-matching L(G|theta)
                (self-loops allowed by the math) + best degree-seq code,
                same convention as the analytic ladder.
- ``dcsbm``   — flat DC-SBM, budgeted seeded restarts of
                graph-tool minimize_blockmodel_dl (czip.sbm.fit_flat),
                scored by the itemized graph-tool description length.
- ``nested``  — nested DC-SBM, seeded restarts of
                graph-tool minimize_nested_blockmodel_dl (czip.sbm.fit_nested),
                scored by the itemized graph-tool description length. ON by
                default at one restart; ``nested_restarts=0`` opts out.

Selection = min itemized analytic L_total among {er, degree, dcsbm, nested}
— one metric, graph-tool's own entropy under its keyword defaults, for
every fitted rung. The encoded stream is the coder message of whichever
model won: a dcsbm winner hands czip its flat partition, a nested winner
hands czip the whole canonical hierarchy (czip.nested_coder, model
``nested-dcsbm``), and an er/degree winner is encoded as B=1 (whose
adjacency stream realizes the CM L(G|theta) exactly), with the
analytic winner recorded in fit_meta. Scores are
selection metadata; the codelength CLAIM for a .cz file is its own coder
report (realized + ideal), never these numbers.

A hierarchy graph-tool returns that cannot be put in
canonical form is still SCORED but marked non-encodable with the reason,
rather than dropped silently or allowed to fail the encode.

Reproducibility: encode and decode are deterministic, but the DC-SBM FIT is
not — graph-tool's minimize_blockmodel_dl is OpenMP-parallel and its result
depends on the thread count as well as the seed. So
``--model auto`` is NOT seed-reproducible: re-running the same seeds on a
different thread count can land on a different partition. ``fit_meta``
therefore records ``omp_threads`` alongside the seeds, and the header's
``params_digest`` is what VERIFIES the partition of a given .cz file — it
does not let you regenerate one.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import scipy.sparse as sp

from czip.codelengths import (cm_microcanonical, degree_theta_variants,
                             log2_binom)


def _binarize(adj: sp.spmatrix) -> sp.csr_matrix:
    b = (sp.csr_matrix(adj) > 0).astype(np.int64)
    b.eliminate_zeros()
    return b


def candidate_scores(adj: sp.spmatrix) -> dict[str, dict[str, Any]]:
    """Analytic itemized scores for the non-SBM rungs on the binarized graph."""
    b = _binarize(adj)
    n = b.shape[0]
    m = int(b.nnz)
    has_loops = bool((b.diagonal() != 0).any())
    N = n * n if has_loops else n * (n - 1)
    er = {
        "L_theta": math.log2(N + 1),
        "L_data": log2_binom(N, m),
        "pair_space": N,
        "self_loops_in_space": has_loops,
    }
    er["L_total"] = er["L_theta"] + er["L_data"]

    k_out = np.asarray(b.sum(axis=1)).ravel().astype(np.int64)
    k_in = np.asarray(b.sum(axis=0)).ravel().astype(np.int64)
    theta_variants = degree_theta_variants(k_out, k_in)
    theta_code = min(theta_variants, key=theta_variants.get)
    cm = cm_microcanonical(k_out, k_in)
    degree = {
        "L_theta": theta_variants[theta_code],
        "L_data": cm["L_data"],
        "theta_code": theta_code,
    }
    degree["L_total"] = degree["L_theta"] + degree["L_data"]
    return {"er": er, "degree": degree}


def fit_auto(adj: sp.spmatrix, seed0: int = 100, restarts: int = 3,
             nested_restarts: int = 1) -> dict[str, Any]:
    """Fit + select the best encodable covariate-free rung.

    Returns {"selected", "labels" (dense 0..B-1, the base partition),
    "hierarchy" (canonical levels when nested won, else None), "fit_meta"}.
    ``hierarchy[0]`` IS ``labels`` for a nested winner, so a caller that only
    understands flat partitions still gets a usable one. Topology-only:
    weights are binarized away for scoring and fitting (the weights layer is
    a separate czip concern via --wmin).

    ``nested_restarts=0`` excludes the nested candidate entirely (no fit, no
    score); it is the opt-out from a candidate that costs a second fit.
    """
    import graph_tool.all as gt

    from czip import nested_coder
    from czip.sbm import csr_to_graph_tool, fit_flat, fit_nested

    if restarts < 1:
        raise ValueError(f"restarts must be >= 1, got {restarts}")
    if nested_restarts < 0:
        raise ValueError(
            f"nested_restarts must be >= 0 (0 = no nested candidate), "
            f"got {nested_restarts}")

    b = _binarize(adj)
    scores: dict[str, dict[str, Any]] = candidate_scores(b)

    g = csr_to_graph_tool(b)
    best_state, best_bits = None, None
    per_restart = []
    seeds = [seed0 + i for i in range(restarts)]
    t0 = time.time()
    for seed in seeds:
        state, bits = fit_flat(g, seed=seed)
        per_restart.append({"seed": seed, "L_total": bits["L_total_bits"],
                            "B": bits["B"]})
        if best_bits is None or bits["L_total_bits"] < best_bits["L_total_bits"]:
            best_state, best_bits = state, bits
    fit_wall_s = time.time() - t0
    scores["dcsbm"] = {
        "L_theta": best_bits["L_theta_bits"],
        "L_data": best_bits["L_adjacency_bits"],
        "L_total": best_bits["L_total_bits"],
        "B": best_bits["B"],
    }

    nested_seeds = [seed0 + i for i in range(nested_restarts)]
    nested_per_restart: list[dict[str, Any]] = []
    hierarchy = None
    nested_wall_s = None
    if nested_seeds:
        t0 = time.time()
        best_nstate, best_nbits = None, None
        for seed in nested_seeds:
            nstate, nbits = fit_nested(g, seed=seed)
            nested_per_restart.append({"seed": seed,
                                       "L_total": nbits["L_total_bits"],
                                       "levels_B": nbits["levels_B"]})
            if best_nbits is None or \
                    nbits["L_total_bits"] < best_nbits["L_total_bits"]:
                best_nstate, best_nbits = nstate, nbits
        nested_wall_s = time.time() - t0
        scores["nested"] = {
            "L_theta": best_nbits["L_theta_bits"],
            "L_data": best_nbits["L_adjacency_bits"],
            "L_total": best_nbits["L_total_bits"],
            # graph-tool's own (padded) level list, as sbm.nested_sbm_bits
            # reports it; fit_meta["encoded_levels_B"] is the canonical,
            # truncated shape czip actually transmits
            "levels_B": best_nbits["levels_B"],
        }
        # the candidate is only encodable if its hierarchy has a normal form;
        # score it either way, but never select something czip cannot emit
        try:
            hierarchy = nested_coder.canonical_hierarchy(best_nstate.get_bs())
        except ValueError as exc:
            hierarchy = None
            scores["nested"]["encodable"] = False
            scores["nested"]["excluded_reason"] = str(exc)

    encodable = [k for k in ("er", "degree", "dcsbm", "nested")
                 if k in scores and scores[k].get("encodable", True)]
    selected = min(encodable, key=lambda k: scores[k]["L_total"])
    if selected == "nested":
        labels = hierarchy[0]
    elif selected == "dcsbm":
        hierarchy = None
        raw = best_state.get_blocks().a
        labels = np.unique(np.asarray(raw), return_inverse=True)[1] \
            .astype(np.int64)
    else:
        hierarchy = None
        labels = np.zeros(b.shape[0], dtype=np.int64)

    fit_meta = {
        "selected": selected,
        "selection_metric": "analytic itemized L_total, ladder "
                            "convention (see module docstring)",
        "scores": scores,
        "seeds": seeds,
        # the fit is OpenMP-parallel and thread-count dependent: seeds
        # alone do not pin it, so record the thread count with them
        "omp_threads": int(gt.openmp_get_num_threads()),
        "restarts": restarts,
        "per_restart": per_restart,
        "fit_wall_s": round(fit_wall_s, 3),
        "nested_restarts": nested_restarts,
        "nested_seeds": nested_seeds,
        "nested_per_restart": nested_per_restart,
        "nested_wall_s": (None if nested_wall_s is None
                          else round(nested_wall_s, 3)),
        "encoded_B": int(labels.max()) + 1,
        "encoded_levels_B": (None if hierarchy is None else
                             [int(lab.max()) + 1 for lab in hierarchy]),
    }
    return {"selected": selected, "labels": labels, "hierarchy": hierarchy,
            "fit_meta": fit_meta}
