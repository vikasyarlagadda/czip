"""czip — lossless compression for connectome graphs.

A `.cz` container holds a directed graph and the model used to code it. The
container is a dispatcher, not a coder: the model bits come verbatim from the
message segments the coders emit, and the container adds no padding of its
own. Its own cost — the magic bytes, the version and header-length prefix,
and the header JSON — is measured as ``container_overhead_bits``. That cost
is itemized, not excluded: it is reported on its own line *and* included in
every whole-container bits-per-edge figure.

Encoding is exact. Weights are integer synapse counts and are never rounded;
``--wmin W`` is the weight ORIGIN (the code transmits ``w - W``), never a
filter. Decoding checks the container against itself — both header digests,
over the model parameters and over the source graph — before it returns
anything.

Typical use from the command line::

    czip encode graph.npz --partition labels.npy -o graph.cz
    czip info graph.cz
    czip decode graph.cz -o graph_decoded.npz --labels-out labels_out.npy

and from Python::

    import czip
    blob = czip.encode_weighted(adj, labels, wmin=1)
    labels_out, adj_out = czip.decode(blob)
"""

from czip import coder as coder
from czip import nested_coder as nested_coder
from czip import sbm_coder as sbm_coder
from czip import weights as weights
from czip import weights_coder as weights_coder
from czip.czip import (
    FORMAT_VERSION,
    MAGIC,
    bits_gate,
    decode,
    encode_topology,
    encode_weighted,
    load_edgelist,
    load_hierarchy,
    main,
    pack,
    unpack,
    verify_lossless,
)

__version__ = "0.1.0"

__all__ = [
    "FORMAT_VERSION",
    "MAGIC",
    "__version__",
    "bits_gate",
    "decode",
    "encode_topology",
    "encode_weighted",
    "load_edgelist",
    "load_hierarchy",
    "main",
    "pack",
    "unpack",
    "verify_lossless",
]
