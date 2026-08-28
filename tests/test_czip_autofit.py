"""Tests for czip.czip_autofit — ladder selection for `czip encode --model auto`.

Small synthetic graphs only (graph-tool fits are seconds at this scale).
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from czip import czip
from czip.czip_autofit import candidate_scores, fit_auto

pytest.importorskip("graph_tool.all")


@pytest.fixture(autouse=True)
def _single_threaded_fits():
    """Pin graph-tool to one OMP thread for every test in this module.

    Selection tests assert WHICH rung wins, and the winning margin between
    the flat and nested fits of the same graph is tens of bits — inside the
    jitter of graph-tool's OpenMP-parallel fit, which is
    thread-count dependent and was measured here to vary between two
    calls in ONE process at 2 threads. Seeds alone therefore do not pin a
    winner; seeds + thread count do. This narrows the environment so the
    assertions are reproducible — no assertion is loosened, and it also
    removes a pre-existing flake (the dcsbm-is-the-minimum check below now
    competes against nested, which lands ~1 bit away).
    """
    import graph_tool.all as gt

    keep = gt.openmp_get_num_threads()
    gt.openmp_set_num_threads(1)
    try:
        yield
    finally:
        gt.openmp_set_num_threads(keep)


def planted_two_block(n=60, p_in=0.5, p_out=0.02, seed=7):
    rng = np.random.default_rng(seed)
    half = n // 2
    blocks = np.r_[np.zeros(half, dtype=int), np.ones(n - half, dtype=int)]
    probs = np.where(blocks[:, None] == blocks[None, :], p_in, p_out)
    a = (rng.random((n, n)) < probs).astype(np.int64)
    np.fill_diagonal(a, 0)
    return sp.csr_matrix(a)


def planted_hierarchy(n_groups=12, per_group=8, n_super=3, p_in=0.7,
                      p_mid=0.10, p_out=0.005, seed=11):
    """Two-scale planted structure: groups inside supergroups.

    Picked so the nested fit beats the flat one by ~20-30 bits at every OMP
    thread count tried (1/2/4/8/14) — the flat-vs-nested margin is what the
    thread-dependent fit can move, so the test graph needs a gap
    wider than that jitter, not merely a positive one.
    """
    rng = np.random.default_rng(seed)
    n = n_groups * per_group
    grp = np.repeat(np.arange(n_groups), per_group)
    sup = grp // (n_groups // n_super)
    probs = np.where(grp[:, None] == grp[None, :], p_in,
                     np.where(sup[:, None] == sup[None, :], p_mid, p_out))
    a = (rng.random((n, n)) < probs).astype(np.int64)
    np.fill_diagonal(a, 0)
    return sp.csr_matrix(a)


def test_candidate_scores_has_er_and_degree():
    adj = planted_two_block()
    scores = candidate_scores(adj)
    assert set(scores) == {"er", "degree"}
    for row in scores.values():
        assert np.isfinite(row["L_total"]) and row["L_total"] > 0
        assert row["L_total"] == pytest.approx(
            row["L_theta"] + row["L_data"])


def test_candidate_scores_tolerates_self_loops():
    adj = planted_two_block().tolil()
    adj[0, 0] = 1
    scores = candidate_scores(sp.csr_matrix(adj))
    assert np.isfinite(scores["er"]["L_total"])
    assert np.isfinite(scores["degree"]["L_total"])


def test_fit_auto_planted_blocks_selects_dcsbm_and_roundtrips():
    adj = planted_two_block()
    result = fit_auto(adj, seed0=100, restarts=2)
    assert result["selected"] == "dcsbm"
    assert result["labels"].shape == (adj.shape[0],)
    assert len(np.unique(result["labels"])) >= 2
    meta = result["fit_meta"]
    assert meta["restarts"] == 2 and meta["seeds"] == [100, 101]
    assert {"er", "degree", "dcsbm"} <= set(meta["scores"])
    assert meta["scores"]["dcsbm"]["L_total"] == min(
        v["L_total"] for v in meta["scores"].values())
    # winner must actually encode + decode losslessly
    blob = czip.encode_topology(adj, result["labels"])
    _, dec = czip.decode(blob)
    assert (dec != adj).nnz == 0


def test_fit_auto_sparse_random_falls_back_to_encodable_labels():
    # near-structureless graph: whatever rung wins, labels must be a valid
    # dense partition the flat coder can encode
    rng = np.random.default_rng(3)
    n = 40
    a = sp.random(n, n, density=0.02, random_state=3,
                  data_rvs=lambda k: np.ones(k)).tocsr()
    a.setdiag(0)
    a.eliminate_zeros()
    a = a.astype(np.int64)
    result = fit_auto(a, seed0=100, restarts=1)
    labels = result["labels"]
    u = np.unique(labels)
    assert (u == np.arange(u.size)).all()  # dense 0..B-1
    if result["selected"] in ("er", "degree"):
        assert u.size == 1  # non-SBM winner encodes as B=1
    blob = czip.encode_topology(a, labels)
    _, dec = czip.decode(blob)
    assert (dec != a).nnz == 0


def test_cli_encode_model_auto_roundtrips_with_fit_meta(tmp_path):
    adj = planted_two_block()
    src_npz = tmp_path / "g.npz"
    sp.save_npz(src_npz, adj)
    cz = tmp_path / "g.cz"
    out_npz = tmp_path / "dec.npz"
    assert czip.main(["encode", str(src_npz), "-o", str(cz),
                      "--model", "auto", "--restarts", "1",
                      "--seed0", "100"]) == 0
    header, _ = czip.unpack(cz.read_bytes())
    assert header["model_id"] == "dcsbm"
    assert header["fit_meta"]["selected"] in ("er", "degree", "dcsbm")
    assert header["fit_meta"]["seeds"] == [100]
    assert czip.main(["decode", str(cz), "-o", str(out_npz)]) == 0
    dec = sp.load_npz(out_npz)
    assert (dec != adj).nnz == 0


def test_cli_encode_requires_partition_or_auto(tmp_path):
    adj = planted_two_block()
    src_npz = tmp_path / "g.npz"
    sp.save_npz(src_npz, adj)
    with pytest.raises(SystemExit):
        czip.main(["encode", str(src_npz), "-o", str(tmp_path / "g.cz")])


def test_cli_encode_model_auto_weighted(tmp_path):
    adj = planted_two_block()
    weighted = adj.copy()
    weighted.data = weighted.data * 6
    src_npz = tmp_path / "g.npz"
    sp.save_npz(src_npz, weighted)
    cz = tmp_path / "g.cz"
    out_npz = tmp_path / "dec.npz"
    assert czip.main(["encode", str(src_npz), "-o", str(cz),
                      "--model", "auto", "--restarts", "1",
                      "--seed0", "100", "--wmin", "5"]) == 0
    header, _ = czip.unpack(cz.read_bytes())
    assert header["model_id"] == "dcsbm+weights"
    assert "fit_meta" in header
    assert czip.main(["decode", str(cz), "-o", str(out_npz)]) == 0
    dec = sp.load_npz(out_npz)
    assert (dec != weighted).nnz == 0  # weight values preserved


def test_fit_auto_binarizes_weighted_input_for_topology_scores():
    adj = planted_two_block()
    weighted = adj.copy()
    weighted.data = weighted.data * 7
    r_bin = fit_auto(adj, seed0=100, restarts=1)
    r_w = fit_auto(weighted, seed0=100, restarts=1)
    assert r_bin["fit_meta"]["scores"]["er"]["L_total"] == pytest.approx(
        r_w["fit_meta"]["scores"]["er"]["L_total"])


# ------------------------------------------------- nested as a candidate


def test_fit_auto_nested_wins_on_hierarchical_graph_and_roundtrips():
    adj = planted_hierarchy()
    result = fit_auto(adj, seed0=100, restarts=1)   # nested on by default
    assert result["selected"] == "nested"
    hierarchy = result["hierarchy"]
    assert hierarchy is not None and len(hierarchy) >= 2
    # canonical normal form: level l labels the blocks of level l-1, top B=1
    assert hierarchy[0].shape == (adj.shape[0],)
    for lo, hi in zip(hierarchy, hierarchy[1:]):
        assert hi.shape == (int(lo.max()) + 1,)
    assert int(hierarchy[-1].max()) + 1 == 1
    # the returned labels ARE the hierarchy's base level
    assert np.array_equal(result["labels"], hierarchy[0])
    scores = result["fit_meta"]["scores"]
    assert scores["nested"]["L_total"] == min(
        v["L_total"] for v in scores.values())
    # winner must actually encode + decode losslessly as the nested model
    blob = czip.encode_topology(adj, hierarchy=hierarchy)
    header, _ = czip.unpack(blob)
    assert header["model_id"] == "nested-dcsbm"
    assert header["n_levels"] == len(hierarchy)
    _, dec = czip.decode(blob)
    assert (dec != adj).nnz == 0


def test_fit_auto_never_selects_a_non_encodable_nested_candidate(monkeypatch):
    """Scored, marked non-encodable, and NOT selected — even when it wins.

    graph-tool can hand back a hierarchy with no normal form; on this fixture
    it never does, so the failure is forced here. Without the forcing the
    encodability guard is only ever driven on its true branch, which is no
    coverage of the guard at all.
    """
    from czip import nested_coder

    def _no_normal_form(bs):
        raise ValueError("hierarchy does not terminate in a single block")

    monkeypatch.setattr(nested_coder, "canonical_hierarchy", _no_normal_form)
    adj = planted_hierarchy()               # the graph on which nested WINS
    result = fit_auto(adj, seed0=100, restarts=1)
    scores = result["fit_meta"]["scores"]
    # still fitted and still scored: it is excluded on encodability, not price
    assert scores["nested"]["L_total"] == min(
        v["L_total"] for v in scores.values())
    assert scores["nested"]["encodable"] is False
    assert "terminate" in scores["nested"]["excluded_reason"]
    assert result["selected"] == "dcsbm"
    assert result["hierarchy"] is None
    assert result["fit_meta"]["encoded_levels_B"] is None
    blob = czip.encode_topology(adj, result["labels"])
    header, _ = czip.unpack(blob)
    assert header["model_id"] == "dcsbm"


def test_fit_auto_flat_winner_still_returns_a_flat_partition():
    # nested is fitted and scored here too, but loses: behaviour unchanged
    adj = planted_two_block()
    result = fit_auto(adj, seed0=100, restarts=2)
    assert result["selected"] == "dcsbm"
    assert result["hierarchy"] is None
    scores = result["fit_meta"]["scores"]
    assert "nested" in scores
    assert scores["dcsbm"]["L_total"] < scores["nested"]["L_total"]
    blob = czip.encode_topology(adj, result["labels"])
    header, _ = czip.unpack(blob)
    assert header["model_id"] == "dcsbm"
    _, dec = czip.decode(blob)
    assert (dec != adj).nnz == 0


def test_fit_auto_nested_restarts_zero_excludes_nested():
    adj = planted_hierarchy()
    result = fit_auto(adj, seed0=100, restarts=1, nested_restarts=0)
    assert result["selected"] != "nested"
    assert result["hierarchy"] is None
    meta = result["fit_meta"]
    assert "nested" not in meta["scores"]
    assert meta["nested_restarts"] == 0
    assert meta["nested_seeds"] == [] and meta["nested_wall_s"] is None


def test_fit_auto_fit_meta_records_the_nested_fit():
    adj = planted_hierarchy()
    meta = fit_auto(adj, seed0=100, restarts=1,
                    nested_restarts=2)["fit_meta"]
    assert meta["nested_restarts"] == 2
    assert meta["nested_seeds"] == [100, 101]
    assert [r["seed"] for r in meta["nested_per_restart"]] == [100, 101]
    for r in meta["nested_per_restart"]:
        assert np.isfinite(r["L_total"]) and r["L_total"] > 0
        assert all(isinstance(b, int) for b in r["levels_B"])
    assert meta["nested_wall_s"] >= 0
    nested = meta["scores"]["nested"]
    assert nested["L_total"] == pytest.approx(
        nested["L_theta"] + nested["L_data"])
    assert nested["L_total"] == min(r["L_total"]
                                    for r in meta["nested_per_restart"])
    assert len(nested["levels_B"]) >= 2
    # the encoded hierarchy's shape is reported alongside encoded_B
    assert meta["encoded_levels_B"][-1] == 1
    assert meta["encoded_B"] == meta["encoded_levels_B"][0]


def test_fit_auto_is_reproducible_at_a_pinned_thread_count():
    """Seeds alone do NOT pin the nested fit — seeds + thread count do.

    Measured while writing this test: at 2 OMP threads two fit_auto calls
    with identical seeds returned different hierarchies (7 base blocks vs 5),
    so the honest claim is the one the module docstring already makes
    — pin the thread count and the fit repeats exactly. Do not weaken this to
    a scores-only comparison: the arrays are the model.
    """
    adj = planted_hierarchy()
    a = fit_auto(adj, seed0=100, restarts=1, nested_restarts=1)
    b = fit_auto(adj, seed0=100, restarts=1, nested_restarts=1)
    assert a["fit_meta"]["omp_threads"] == 1   # the module fixture's pin
    assert a["selected"] == b["selected"]
    assert np.array_equal(a["labels"], b["labels"])
    assert [x.tolist() for x in a["hierarchy"]] == \
        [x.tolist() for x in b["hierarchy"]]
    for k, v in a["fit_meta"]["scores"].items():
        assert v["L_total"] == pytest.approx(
            b["fit_meta"]["scores"][k]["L_total"])


def test_cli_encode_model_auto_emits_the_nested_model(tmp_path):
    adj = planted_hierarchy()
    src_npz = tmp_path / "g.npz"
    sp.save_npz(src_npz, adj)
    cz = tmp_path / "g.cz"
    out_npz = tmp_path / "dec.npz"
    assert czip.main(["encode", str(src_npz), "-o", str(cz),
                      "--model", "auto", "--restarts", "1",
                      "--seed0", "100"]) == 0
    header, _ = czip.unpack(cz.read_bytes())
    assert header["model_id"] == "nested-dcsbm"
    assert header["fit_meta"]["selected"] == "nested"
    assert header["n_levels"] == len(header["fit_meta"]["encoded_levels_B"])
    assert czip.main(["decode", str(cz), "-o", str(out_npz)]) == 0
    dec = sp.load_npz(out_npz)
    assert (dec != adj).nnz == 0


def test_cli_encode_model_auto_nested_restarts_zero_opts_out(tmp_path):
    adj = planted_hierarchy()
    src_npz = tmp_path / "g.npz"
    sp.save_npz(src_npz, adj)
    cz = tmp_path / "g.cz"
    assert czip.main(["encode", str(src_npz), "-o", str(cz),
                      "--model", "auto", "--restarts", "1", "--seed0", "100",
                      "--nested-restarts", "0"]) == 0
    header, _ = czip.unpack(cz.read_bytes())
    assert header["model_id"] == "dcsbm"
    assert "nested" not in header["fit_meta"]["scores"]


def test_cli_encode_model_auto_nested_weighted(tmp_path):
    adj = planted_hierarchy()
    weighted = adj.copy()
    weighted.data = weighted.data * 6
    src_npz = tmp_path / "g.npz"
    sp.save_npz(src_npz, weighted)
    cz = tmp_path / "g.cz"
    out_npz = tmp_path / "dec.npz"
    assert czip.main(["encode", str(src_npz), "-o", str(cz),
                      "--model", "auto", "--restarts", "1",
                      "--seed0", "100", "--wmin", "5"]) == 0
    header, _ = czip.unpack(cz.read_bytes())
    assert header["model_id"] == "nested-dcsbm+weights"
    assert czip.main(["decode", str(cz), "-o", str(out_npz)]) == 0
    dec = sp.load_npz(out_npz)
    assert (dec != weighted).nnz == 0
