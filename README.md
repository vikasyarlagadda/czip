# czip

Lossless compression for connectome-scale directed weighted graphs, with the
size of the decompressor counted.

[![CI](https://github.com/vikasyarlagadda/czip/actions/workflows/ci.yml/badge.svg)](https://github.com/vikasyarlagadda/czip/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](pyproject.toml)

<!-- add PyPI / DOI badges at public release -->

## What it is

`czip` compresses a directed, weighted graph (a sparse adjacency matrix or an
edge list) into a single `.cz` file, losslessly. It gets its compression from
the structure real connectomes have by fitting/accepting a degree-corrected
stochastic block model, flat or nested, and entropy-codes the graph under that
model with a range coder, so the bits spent on an edge are the bits that edge
actually costs given who its endpoints are. Decoding returns the exact same
matrix (indices, row pointers, integer weights) and `czip` proves it at
encode time by decoding the container it just wrote and comparing it against
your input before handing you the file. The container carries digests that are
re-checked at every decode. Furthermore, every size here is published with 
the container alone, and the container plus the measured bytes of the decoder
needed to open it.

## Headline numbers

The whole adult fly brain (FlyWire/FAFB v783, 139,255 neurons, 15,091,983
weighted connections at ≥1 synapse) encodes to a **15,348,001-byte container
(14.64 MB)** under the nested model. That is **8.14 bits per edge for the
whole container**, every byte of header and coded stream included.

The decoder that opens it is **50.03 MB** (52,464,017 B: a 48,921-byte code
archive plus the numpy, scipy and constriction wheels, whole and unpruned).
Counted together, **64.67 MB portable** for the complete fly brain connectome
and everything needed to read it back.

The lowest rate among the weighted containers is hemibrain v1.2.1 at ≥1
synapse: **7.69 bits per edge**. (Drop the weights and code topology alone and
FAFB at ≥5 synapses reaches **6.73 bits per edge**.) All sizes here are MiB
labelled MB (1024²).

## Install

```bash
pip install czip
```

That is all you need to **decode** anything, and to **encode** with a model
you supply (`--partition` or `--hierarchy`): numpy, scipy and constriction,
which pip pulls in for you.

Automatic model selection (`--model auto`) additionally needs `graph-tool`,
which cannot be installed with pip as it is a C++ library distributed through
conda-forge:

```bash
micromamba create -n czip -c conda-forge python=3.12 graph-tool
micromamba run -n czip pip install czip
```

`conda` and `mamba` work the same way. Nothing on the decode path ever imports
graph-tool, so you only need it on machines that fit models.

## Quickstart

A synthetic example generator ships with the repository, so you can round-trip
a graph without downloading any data:

```bash
python examples/make_example.py                     # 300 nodes, 6 blocks, 2,500 edges
czip encode example_graph.npz --partition example_partition.npy --wmin 1 -o example.cz
czip info example.cz                                # header: model, streams, coder report
czip decode example.cz -o example_decoded.npz --labels-out example_labels_out.npy
```

`--wmin 1` is the **weight origin**, not a threshold: the code transmits
`w - 1`, so every weight must be at least 1. It never drops an edge. Omit it
for a topology-only container. Full details in
[docs/MANUAL.md](docs/MANUAL.md).

## Results

Whole-container rates (total bytes × 8 ÷ edges, all overhead included) under
the nested degree-corrected SBM. Every container was decoded and compared
against its source.

| dataset | ≥1 synapse | ≥5 synapses |
|---|---|---|
| FAFB v783 (139,255 neurons) | 8.14 b/e | 11.13 b/e |
| hemibrain v1.2.1 (186,061) | 7.69 b/e | 10.78 b/e |
| MANC v1.2.3 (102,158) | 8.22 b/e | 10.93 b/e |
| BANC v888 (158,262) | 10.76 b/e at ≥3 synapses† | 12.12 b/e |
| male CNS v0.9 (176,571) | 8.28 b/e | 10.82 b/e |
| C. elegans (Cook 2019 hermaphrodite, 302)‡ | 19.14 b/e | 65.41 b/e |

† BANC's published connection file is pre-filtered at ≥3 synapses per pair, so
there is no ≥1-synapse row.
‡ An outer comparison only: a composite reconstruction with different
annotation semantics, and at 302 nodes its rate is dominated by fixed
container overhead. Not rank-compared against the fly rows.

Flat-model containers, per-container tool-size totals, wall times and dataset
citations: [docs/RESULTS.md](docs/RESULTS.md). Each of those containers costs
the same 50.03 MB decoder term on top.

**No connectome data ships with this repository.** What ships for the example
is the generator script, not its output so the only graph-data files in the
repository are about 8 KB of synthetic format-compatibility test fixtures,
built from a seeded random graph.

## How it works

A sparse graph is mostly a statement about *where* the edges are, and that is
where the bits go. `czip` describes the graph in stages, each one conditioned
on everything already decoded (a partition of the nodes into blocks, the
degree sequence, then the adjacency itself). Each stage is coded against a
distribution the decoder can reconstruct exactly from the previous stages, so
nothing has to be transmitted twice and no probability table travels in the
file.

The nested model does the same thing to the partition. It codes a hierarchy of
partitions (blocks of blocks of blocks) bottom-up, and then the expansions
top-down. On every connectome tested, the nested container is smaller than
the flat one. Fitting either model is graph-tool's job and coding it and proving
the round-trip, is `czip`'s.

The `.cz` container is a dispatcher, not a coder. It holds a canonical JSON
header (model id, digests, a stream table whose geometry is validated on read
rather than trusted, and an itemized coder report) followed by the coded
streams verbatim, with no padding anywhere so the `.cz` file is exactly its
10-byte prefix, its header, and its streams back to back. The container's own
cost, the prefix and the header, is measured and reported on its own line, and
it is counted in every bits-per-edge number here. Those are whole-container
rates, total bytes × 8 ÷ edges, with the overhead inside them.

## Guarantees

- **Lossless.** Decode reproduces the exact CSR adjacency.
  `encode` decodes what it just wrote and compares it against your source
  before returning, and stamps the result into the header. `--no-verify` opts
  out, and the header claim reads as unproven rather than as true.
- **Honest accounting.** Every stream reports realized against ideal (−log₂P)
  bits and the encode fails loudly if they disagree beyond a stated window.
  Container overhead is itemized and included in the published rates, never
  netted out of them. Tool size is measured and published beside every
  container size, including the encode-side cost.
- **Deterministic decode.** Encode and decode are CPU-deterministic; headers
  carry digests over the model parameters and the source arrays, and a
  mismatch raises at decode rather than being carried as an unchecked claim.
  The model *fit* behind `--model auto` is not deterministic as graph-tool's
  minimizer is thread-count dependent so the header records the thread count
  with the seeds, and the digest verifies the partition in the file.
- **A decoder you can audit.** Decoding needs seven modules of this package
  plus numpy, scipy and constriction, and the test suite pins that by decoding
  in a fresh interpreter with every encode-side package made unimportable.
  `python -m czip.decoder_artifact` builds that closure as a
  byte-reproducible archive.

## Citing

See [CITATION.cff](CITATION.cff). If you use the results tables, cite the
underlying datasets too — they are listed in
[docs/RESULTS.md](docs/RESULTS.md).

## License

MIT. See [LICENSE](LICENSE).
