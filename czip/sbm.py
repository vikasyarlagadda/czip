"""SBM fitting and description-length bit accounting.

Conventions:
- graph-tool `entropy()` is in nats; bits = nats / ln 2.
- L(theta)  = partition_dl + degree_dl + edges_dl
- L(G|theta) = adjacency term
- Additivity L_total = L(theta) + L(G|theta) is asserted on every computation.
- Nested states: L(G|theta) = base-level adjacency; L(theta) = total - that.
- entropy() keyword defaults are used everywhere (degree_dl_kind='distributed',
  multigraph=True, dense=False); note the simple-vs-multigraph caveat that
  implies.
"""

import math

import numpy as np
import scipy.sparse as sp

import graph_tool.all as gt

_LN2 = math.log(2)


def nats_to_bits(nats: float) -> float:
    return nats / _LN2


def csr_to_graph_tool(adj: sp.csr_matrix) -> gt.Graph:
    """Directed graph-tool graph from a scipy CSR adjacency (binary topology).

    All n rows become vertices, including isolated nodes: n is common
    knowledge and never shrinks.
    """
    n = adj.shape[0]
    coo = adj.tocoo()
    g = gt.Graph(directed=True)
    g.add_vertex(n)
    g.add_edge_list(np.column_stack([coo.row, coo.col]))
    return g


def _term(state, **kwargs) -> float:
    """One entropy term in nats, all other terms switched off."""
    args = dict(adjacency=False, dl=True, partition_dl=False,
                degree_dl=False, edges_dl=False)
    args.update(kwargs)
    return state.entropy(**args)


def sbm_bits(state) -> dict:
    """Itemized description length (bits) of a flat BlockState."""
    total = state.entropy()
    adj = state.entropy(adjacency=True, dl=False)
    part = _term(state, partition_dl=True)
    deg = _term(state, degree_dl=True)
    edges = _term(state, edges_dl=True)
    assert np.isclose(total, adj + part + deg + edges, rtol=1e-9), \
        f"entropy terms not additive: {total} != {adj + part + deg + edges}"
    theta = part + deg + edges
    return {
        "L_partition_bits": nats_to_bits(part),
        "L_degree_bits": nats_to_bits(deg),
        "L_edges_bits": nats_to_bits(edges),
        "L_theta_bits": nats_to_bits(theta),
        "L_adjacency_bits": nats_to_bits(adj),
        "L_total_bits": nats_to_bits(total),
        "B": int(state.get_B()),
        "Be": float(state.get_Be()),
        "deg_corrected": bool(state.deg_corr),
    }


def nested_sbm_bits(state) -> dict:
    """Itemized description length (bits) of a NestedBlockState.

    The whole hierarchy above the adjacency term is model cost.
    """
    total = state.entropy()
    adj = state.levels[0].entropy(adjacency=True, dl=False)
    theta = total - adj
    assert theta >= -1e-9, f"negative L(theta): {theta}"
    return {
        "L_theta_bits": nats_to_bits(theta),
        "L_adjacency_bits": nats_to_bits(adj),
        "L_total_bits": nats_to_bits(total),
        "levels_B": [int(s.get_B()) for s in state.get_levels()],
    }


def fit_flat(g: gt.Graph, seed: int, deg_corr: bool = True):
    """One restart of minimize_blockmodel_dl with pinned seeds."""
    np.random.seed(seed)
    gt.seed_rng(seed)
    state = gt.minimize_blockmodel_dl(g, state_args=dict(deg_corr=deg_corr))
    return state, sbm_bits(state)


def fit_nested(g: gt.Graph, seed: int, deg_corr: bool = True):
    """One restart of minimize_nested_blockmodel_dl with pinned seeds."""
    np.random.seed(seed)
    gt.seed_rng(seed)
    state = gt.minimize_nested_blockmodel_dl(
        g, state_args=dict(deg_corr=deg_corr))
    return state, nested_sbm_bits(state)


def fixed_partition_state(g: gt.Graph, labels: np.ndarray,
                          deg_corr: bool = True):
    """BlockState frozen to a given partition — entropy only, no fitting."""
    b = g.new_vertex_property("int", vals=np.asarray(labels, dtype=np.int64))
    state = gt.BlockState(g, b=b, deg_corr=deg_corr)
    return state, sbm_bits(state)


def partition_from_types(cell_types) -> np.ndarray:
    """Dense 0..B-1 block labels from a polars Series of type strings.

    Untyped (null) nodes share ONE extra block.
    """
    null_mask = cell_types.is_null().to_numpy()
    values = cell_types.to_numpy()
    labels = np.empty(len(values), dtype=np.int64)
    uniq, typed_codes = np.unique(values[~null_mask].astype(str),
                                  return_inverse=True)
    labels[~null_mask] = typed_codes
    labels[null_mask] = len(uniq)
    return labels


def save_partition(state, path) -> None:
    np.save(path, np.asarray(state.get_blocks().a, dtype=np.int64))


def save_hierarchy(state, path) -> None:
    """Save every level of a NestedBlockState's hierarchy to one .npz.

    Keys are level_0 (base partition) .. level_{k-1}, matching get_bs().
    save_partition keeps only the base level; this keeps the whole model
    so the complete nested DL can be recomputed from disk.
    """
    bs = state.get_bs()
    np.savez(path, **{f"level_{i}": np.asarray(b, dtype=np.int64)
                      for i, b in enumerate(bs)})


def load_nested_state(g: gt.Graph, path, deg_corr: bool = True):
    """Rebuild a NestedBlockState from a saved hierarchy; entropy must reproduce.

    Recomputes the DL from disk for nested runs, with no re-fitting.
    """
    with np.load(path) as z:
        bs = [z[f"level_{i}"] for i in range(len(z.files))]
    # graph-tool 3.6: base-state kwargs go via base_state_args (NOT state_args,
    # which is silently ignored with only a UserWarning)
    state = gt.NestedBlockState(g, bs=bs,
                                base_state_args=dict(deg_corr=deg_corr))
    return state, nested_sbm_bits(state)


def load_partition_state(g: gt.Graph, path, deg_corr: bool = True):
    """Rebuild a BlockState from a saved partition; entropy must reproduce.

    Recomputes the DL from disk without re-fitting.
    """
    labels = np.load(path)
    return fixed_partition_state(g, labels, deg_corr=deg_corr)
