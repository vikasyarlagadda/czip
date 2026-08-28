"""Backward compatibility: containers written by czip's original v1 writer.

`tests/data/legacy_flat_*.cz` were encoded by czip's ORIGINAL v1 container
writer — a release predating the per-container stream counts
(``n_word_streams`` / ``n_byte_streams``) that today's writer records in the
report, and predating the nested models entirely. They are frozen bytes: DO
NOT REGENERATE them with the current writer. Re-encoding them would turn this
file into a test of the current code against itself, which proves nothing
about compatibility.

`legacy_graph.npz` / `legacy_partition.npy` are the exact arrays those
containers encode, frozen alongside them, so a decode can be checked against
the real source rather than against a re-derived one.

Two model branches exist here because the original writer could only produce
two: flat `dcsbm` (topology) and flat `dcsbm+weights`. There is no legacy
nested fixture because nested containers did not exist yet.

What this file pins:

- the current decoder reads those bytes and reproduces the source arrays
  exactly (values, positions, partition);
- the bits gate still accepts them, falling back to its constant stream-count
  window because the header carries no counts of its own;
- the fixtures really are old-shaped — if someone regenerates them with a
  modern writer, `test_fixtures_predate_the_per_container_counts` fails.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from czip import czip

DATA = Path(__file__).parent / "data"

LEGACY_WEIGHTED = DATA / "legacy_flat_weighted.cz"
LEGACY_TOPOLOGY = DATA / "legacy_flat_topology.cz"
SOURCE_GRAPH = DATA / "legacy_graph.npz"
SOURCE_PARTITION = DATA / "legacy_partition.npy"

#: (container, model_id, weights are transmitted)
LEGACY_CONTAINERS = (
    (LEGACY_WEIGHTED, "dcsbm+weights", True),
    (LEGACY_TOPOLOGY, "dcsbm", False),
)
_IDS = ["flat-weighted", "flat-topology"]


def _source():
    """The frozen source graph and partition, as the encoder saw them."""
    adj = sp.load_npz(SOURCE_GRAPH).tocsr()
    adj.sort_indices()
    return adj, np.load(SOURCE_PARTITION)


def _topology_report(header: dict) -> dict:
    """The topology half of a report, whichever shape the header uses.

    A weighted container nests its topology numbers under ``topology``; a
    topology-only container puts them at the top of ``report``.
    """
    report = header["report"]
    return report.get("topology", report)


@pytest.mark.parametrize("container,model_id,weighted", LEGACY_CONTAINERS,
                         ids=_IDS)
def test_legacy_container_unpacks(container, model_id, weighted):
    header, _ = czip.unpack(container.read_bytes())
    assert header["format_version"] == czip.FORMAT_VERSION
    assert header["model_id"] == model_id
    assert header["payload"] == ("graph" if weighted else "topology")
    source_adj, source_labels = _source()
    assert header["n_nodes"] == source_adj.shape[0]
    assert header["n_edges"] == source_adj.nnz
    assert header["params_digest"] == czip._params_digest(source_labels)


@pytest.mark.parametrize("container,model_id,weighted", LEGACY_CONTAINERS,
                         ids=_IDS)
def test_fixtures_predate_the_per_container_counts(container, model_id,
                                                   weighted):
    """The guard on the fixtures themselves: these must be OLD bytes.

    Today's writer records ``n_word_streams`` / ``n_byte_streams`` in every
    report. Their absence is what makes these containers worth keeping, and a
    fixture quietly re-encoded by the current writer would carry them.
    """
    header, _ = czip.unpack(container.read_bytes())
    topology = _topology_report(header)
    assert "n_word_streams" not in topology
    assert "n_byte_streams" not in topology


@pytest.mark.parametrize("container,model_id,weighted", LEGACY_CONTAINERS,
                         ids=_IDS)
def test_legacy_container_decodes_to_the_frozen_source(container, model_id,
                                                       weighted):
    """The compatibility claim: old bytes in, the original graph out."""
    source_adj, source_labels = _source()
    if not weighted:
        # a topology-only container transmits the binarized graph
        source_adj = (source_adj > 0).astype(np.int64).tocsr()
        source_adj.sort_indices()

    labels, adj = czip.decode(container.read_bytes())  # both digests checked
    adj = adj.tocsr()
    adj.sort_indices()

    assert np.array_equal(labels, source_labels)
    assert adj.shape == source_adj.shape
    assert np.array_equal(adj.indptr, source_adj.indptr)
    assert np.array_equal(adj.indices, source_adj.indices)
    assert np.array_equal(adj.data.astype(np.int64),
                          source_adj.data.astype(np.int64))


@pytest.mark.parametrize("container,model_id,weighted", LEGACY_CONTAINERS,
                         ids=_IDS)
def test_legacy_container_still_passes_the_bits_gate(container, model_id,
                                                     weighted):
    """With no counts in the header, the gate uses its constant window."""
    header, _ = czip.unpack(container.read_bytes())
    gate = czip.bits_gate(header)
    assert gate["within_slack"] is True
    assert gate["topology"]["slack_budget_bits"] == pytest.approx(
        32.0 * 2 * czip._TOPOLOGY_WORD_STREAMS
        + 8.0 * czip._TOPOLOGY_BYTE_STREAMS
        + czip.GATE_DRIFT_BITS_PER_SYMBOL
        * _topology_report(header)["n_symbols"])


def test_current_writer_without_counts_gates_the_same_way(legacy_container):
    """Same fallback, reached from the other direction.

    A container the current writer just produced, with the two count fields
    removed from its header, must gate identically to one that never had
    them. This does not carry the compatibility claim — the frozen bytes
    above do — but it pins the fallback against changes to the current
    writer.
    """
    header, _ = czip.unpack(legacy_container)
    assert "n_word_streams" not in header["report"]["topology"]
    gate = czip.bits_gate(header)
    assert gate["within_slack"] is True
    labels, adj = czip.decode(legacy_container)
    assert labels.shape[0] == header["n_nodes"]
    assert adj.nnz == header["n_edges"]
