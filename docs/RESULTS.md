# Results

These tables are measurements from my own research pipeline, frozen at
release. The measurement harness that produced them is part of those research
records and is not included in this repository — what is published here is the
numbers, not the machinery that generated them, and no connectome data ships
with the package. The only `.cz` files in the repository are two synthetic
format-compatibility test fixtures (about 8 KB with their source arrays, from
a seeded 120-node random graph, no biological content); they travel in the
source distribution, not the wheel, and none of the containers measured below
ships with anything.

Tool size was measured 2026-08-27; the round-trip runs were measured during
August 2026. Round-trips ran on an Apple M4 Max (single core) unless stated;
the tool-size table is measured against a frozen Linux x86_64 / CPython 3.12
base, which is stated where it matters.

Everything on this page is **realized container bytes** — the actual
entropy-coded bitstream on disk, including every byte of container overhead,
for a container that was decoded and compared against its source. No
theoretical or idealized codelengths appear here.

---

## 1. Round-trip results

Twenty-four containers, covering all five publicly released fly connectomes
(FAFB, BANC, MANC, hemibrain, male CNS) plus one C. elegans comparison row.
Every one of them is lossless: each container was decoded and compared
entry-for-entry against the source adjacency it was built from, and the
containers were re-verified under the released code — fresh decode, recomputed
accounting, and a byte-identical re-encode wherever the fit was retained.

`b/e` throughout is the **whole-container** rate: total container bytes × 8 ÷
edges, with all overhead included. It is not a per-stream figure and nothing
is excluded from it.

### Flat DC-SBM containers

| dataset | ≥1-synapse | ≥5-synapse (weighted) |
|---|---|---|
| FAFB v783 (139,255 neurons) | 8.64 b/e (15.1M edges) | 12.21 b/e (2.7M edges) |
| hemibrain v1.2.1 (186,061) | 8.07 b/e (7.1M edges) | 11.29 b/e (0.97M edges) |
| MANC v1.2.3 (102,158) | 8.36 b/e (6.5M edges) | 11.22 b/e (1.4M edges) |
| BANC v888 (158,262) | 11.55 b/e at ≥3 synapses (3.0M edges)† | 12.97 b/e (1.5M edges) |

The male CNS and C. elegans graphs have no flat container: on both of them,
`--model auto` selected the nested model outright.

### Nested DC-SBM containers

Smaller than the flat container on every dataset that has one:

| dataset | ≥1-synapse | ≥5-synapse (weighted) |
|---|---|---|
| FAFB v783 | 8.14 b/e | 11.13 b/e |
| hemibrain v1.2.1 | 7.69 b/e | 10.78 b/e |
| MANC v1.2.3 | 8.22 b/e | 10.93 b/e |
| BANC v888 | 10.76 b/e at ≥3 synapses† | 12.12 b/e |
| male CNS v0.9 (176,571) | 8.28 b/e (25.9M edges) | 10.82 b/e (6.3M edges) |
| C. elegans (Cook 2019 hermaphrodite, 302)‡ | 19.14 b/e (3,377 edges) | 65.41 b/e (652 edges) |

† BANC's published connection file is pre-filtered at ≥3 synapses per pair, so
BANC ships at ≥3 and ≥5 synapses; there is no ≥1-synapse BANC row to report.

‡ The C. elegans row is an outer engineering comparison of graph codelength
only. The source is a multi-animal composite TEM reconstruction with different
synapse-annotation semantics than the EM fly volumes; its ≥5 cut is a
cross-dataset sensitivity threshold rather than a confidence cutoff; and at 302
nodes its b/e is dominated by fixed container overhead. It is never
rank-compared against the fly rows.

The remaining two of the twenty-four are the FAFB ≥5 **topology-only**
containers (no weights coded): flat 8.01 b/e and nested 6.73 b/e. They are not
weighted rows and do not belong in the tables above.

Wall times, for scale — FAFB ≥1 weighted, single core: 1,569.6 s encode /
756.0 s decode for the flat container, 1,510.3 s / 650.2 s for the nested one.
The male CNS rows are the slowest in the set, and not the same row on both
ends: with a supplied hierarchy, ≥5 has the longest encode at 5,261.5 s
(1,220.5 s decode), while ≥1 has the longest decode at 1,302.1 s (2,858.2 s
encode).

---

## 2. Tool size

A `.cz` container is not self-decoding, so quoting its size alone charges
nothing for the program that reads it — and a large enough tool could hide
data inside itself. Three numbers are therefore reported for every container,
and the middle one is the one to quote:

- **S_container** — the container bytes alone.
- **S_portable** (primary) — container plus the hermetic decoder closure:
  `czip_decoder.zip` plus the three pinned wheels, whole, with nothing pruned.
  This is the "the decompressor is counted" number.
- **S_full** — container plus the encode-side closure on top, i.e. what it
  costs to *produce* a container rather than only to read one.

The frozen base is a lockfile environment (CPython 3.12.14, linux-64): CPython
and the operating system are assumed present and are not charged; every
package added on top of them is. Fitted parameters — partitions, degrees,
weight-model parameters — are entropy-coded inside the container and counted
exactly once, there. The decoder holds no fitted state, so it is not a
separate line item.

**Unit convention:** every "MB" on this page is a **MiB** (1024²). The byte
counts beside them are the authoritative figures.

### The decoder closure

`czip_decoder.zip` holds the 7 decode-path modules (156,892 B of source) and a
README. It is generated, not committed: `python -m czip.decoder_artifact --out
czip_decoder.zip` rebuilds it byte-identically from this release — 48,921 B,
sha256
`50c434eedede5c002b52709ac0214ec326df9b2271b99878d31ab5f9eaf1ea36`
— so you can regenerate the artifact these numbers are measured over and check
both the size and the digest yourself.

| item | version | bytes | MB | sha256 (first 16) |
|---|---|---:|---:|---|
| czip_decoder.zip | — | 48,921 | 0.05 | 50c434eedede5c00… |
| numpy-2.5.2 (cp312, manylinux x86_64) | 2.5.2 | 16,722,264 | 15.95 | 3cdec01fa790a186… |
| scipy-1.18.0 (cp312, manylinux x86_64) | 1.18.0 | 35,287,115 | 33.65 | 1f55797419e16e7f… |
| constriction-0.5.0 (cp312, manylinux x86_64) | 0.5.0 | 405,717 | 0.39 | d74dbfb7f0d41386… |
| **S_tool (portable)** | — | **52,464,017** | **50.03** | — |

Wheel basis: whole wheels as distributed by PyPI, with no pruning of unused
scipy submodules.

A secondary variant, used by no number on this page: the same three wheels
built for macOS arm64 come to 40,960,741 B (39.06 MB) — numpy 11,903,109 B,
scipy 28,681,889 B, constriction 375,743 B. The frozen base is Linux, so the
Linux table above is the one that counts.

**Encode-side closure** (the S_full extra): 75,545,040 B over 2 packages —
graph-tool 27,667 B, graph-tool-base 75,517,373 B, as distributed
(compressed). This is a **minimum**. It does not count the shared-library
closure graph-tool links against (boost, cairomm, gtk and their dependencies),
because separating "unique to graph-tool" from "already in the frozen base" is
a judgement call this record declines to make silently. The true encode-side
closure is therefore larger than this number, never smaller. It is also
measured from a local osx-arm64 conda cache, whose builds differ in size from
the linux-64 frozen base. S_tool (full) is 128,009,057 B (122.08 MB).

### Headline

Decoder closure **50.03 MB** (zip + wheels, compressed) plus container
**15.54 MB** decodes the full adult fly brain connectome at ≥1 synapse
losslessly — **65.58 MB portable in total**.

### Per-container totals

| dataset | model | container B | S_container (MB) | S_portable (MB) | S_full (MB) |
|---|---|---:|---:|---:|---:|
| BANC ≥3, weighted | flat (auto) | 4,383,307 | 4.18 | 54.21 | 126.26 |
| BANC ≥3, weighted | nested (given hierarchy) | 4,085,197 | 3.90 | 53.93 | 125.97 |
| BANC ≥5, weighted | flat (auto) | 2,478,094 | 2.36 | 52.40 | 124.44 |
| BANC ≥5, weighted | nested (given hierarchy) | 2,316,576 | 2.21 | 52.24 | 124.29 |
| C. elegans ≥1, weighted | nested (auto) | 8,078 | 0.01 | 50.04 | 122.09 |
| C. elegans ≥5, weighted | nested (auto) | 5,331 | 0.01 | 50.04 | 122.08 |
| FAFB ≥1, weighted | nested (given hierarchy) | 15,348,001 | 14.64 | 64.67 | 136.72 |
| FAFB ≥1, weighted | flat (given partition) | 16,299,521 | 15.54 | 65.58 | 137.62 |
| FAFB ≥5, topology only | nested (given hierarchy) | 2,271,953 | 2.17 | 52.20 | 124.25 |
| FAFB ≥5, topology only | flat (given partition) | 2,705,121 | 2.58 | 52.61 | 124.66 |
| FAFB ≥5, weighted | nested (given hierarchy) | 3,757,886 | 3.58 | 53.62 | 125.66 |
| FAFB ≥5, weighted | flat (given partition) | 4,121,107 | 3.93 | 53.96 | 126.01 |
| hemibrain ≥1, weighted | flat (auto) | 7,147,140 | 6.82 | 56.85 | 128.89 |
| hemibrain ≥1, weighted | nested (given hierarchy) | 6,809,347 | 6.49 | 56.53 | 128.57 |
| hemibrain ≥5, weighted | flat (auto) | 1,369,318 | 1.31 | 51.34 | 123.38 |
| hemibrain ≥5, weighted | nested (given hierarchy) | 1,307,201 | 1.25 | 51.28 | 123.33 |
| MANC ≥1, weighted | flat (auto) | 6,758,350 | 6.45 | 56.48 | 128.52 |
| MANC ≥1, weighted | nested (given hierarchy) | 6,646,245 | 6.34 | 56.37 | 128.42 |
| MANC ≥5, weighted | flat (auto) | 1,999,726 | 1.91 | 51.94 | 123.99 |
| MANC ≥5, weighted | nested (given hierarchy) | 1,947,647 | 1.86 | 51.89 | 123.94 |
| male CNS ≥1, weighted | nested (auto) | 26,805,640 | 25.56 | 75.60 | 147.64 |
| male CNS ≥1, weighted | nested (given hierarchy) | 26,782,926 | 25.54 | 75.58 | 147.62 |
| male CNS ≥5, weighted | nested (auto) | 8,544,038 | 8.15 | 58.18 | 130.23 |
| male CNS ≥5, weighted | nested (given hierarchy) | 8,502,802 | 8.11 | 58.14 | 130.19 |
| **all 24 containers** | — | **162,400,552** | **154.88** | **204.91** | **276.96** |

The last row is the honest total: the tool is paid once for all 24 containers,
not once each. Beside it — never instead of it — the amortized mean per
container is 8.54 MB portable (11.54 MB full). A single connectome pays the
whole tool term, which is what the per-container rows above already show.

The largest container in the set is the male CNS graph at ≥1 synapse:
26,805,640 B (25.56 MB), 25.9M edges, for 75.60 MB portable.

---

## 3. Datasets

**This repository contains no connectome data.** The numbers above are summary
statistics computed from publicly released datasets, each of which must be
obtained from, and cited to, its own source. The example graph is not bundled
either — `examples/make_example.py` generates it on your machine — and the
only graph-data files that do ship, the format-compatibility test fixtures,
are synthetic.

- **FlyWire / FAFB, materialization 783** — whole-brain connectivity data,
  Zenodo record 10676866, DOI [10.5281/zenodo.10676866](https://doi.org/10.5281/zenodo.10676866),
  CC BY 4.0. Cite Dorkenwald et al. (2024), *Neuronal wiring diagram of an
  adult brain*, Nature 634, and Schlegel et al. (2024), *Whole-brain
  annotation and multi-connectome cell typing of Drosophila*, Nature 634.
- **hemibrain v1.2.1** — Janelia hemibrain, obtained through neuPrint
  (`neuprint.janelia.org`, dataset `hemibrain:v1.2.1`). Cite Scheffer et al.
  (2020), *A connectome and analysis of the adult Drosophila central brain*,
  eLife 9:e57443.
- **MANC v1.2.3** — male adult nerve cord, obtained through neuPrint (dataset
  `manc:v1.2.3`), CC BY 4.0. Cite Takemura, S.-Y., et al. (2024), *A connectome
  of the male Drosophila ventral nerve cord*, eLife 13:RP97769,
  [doi:10.7554/eLife.97769.1](https://doi.org/10.7554/eLife.97769.1). Because
  the results above are whole-dataset, cite the companion annotation paper too:
  Marin, E. C., et al. (2024), *Systematic annotation of a complete adult male
  Drosophila nerve cord connectome reveals principles of functional
  organisation*, eLife 13:RP97766,
  [doi:10.7554/eLife.97766.1](https://doi.org/10.7554/eLife.97766.1). Both are
  eLife reviewed preprints — hence the RP numbers and versioned `.1` DOIs.
- **BANC v888** — brain-and-nerve-cord connectome, materialization 888,
  connection file pre-filtered at ≥3 synapses per pair. The
  project asks that both of these be cited, and both are CC BY 4.0: Bates,
  A. S., Phelps, J. S., Kim, M., et al.; BANC–FlyWire Consortium (2026),
  *Distributed control circuits across a brain-and-cord connectome*, Nature
  656, 957–970,
  [doi:10.1038/s41586-026-10735-w](https://doi.org/10.1038/s41586-026-10735-w);
  and the version-pinned deposit, *BANC v888 — Brain-and-nerve-cord connectome*
  [Data set], Harvard Dataverse,
  [doi:10.7910/DVN/7WTH1N](https://doi.org/10.7910/DVN/7WTH1N).
- **male CNS v0.9** — released via neuPrint (dataset `male-cns:v0.9`,
  2025-10-03), CC BY 4.0. Cite Berg, S., et al. (2025), *Sexual dimorphism in
  the complete connectome of the Drosophila male central nervous system*,
  bioRxiv 2025.10.09.680999,
  [doi:10.1101/2025.10.09.680999](https://doi.org/10.1101/2025.10.09.680999).
- **C. elegans** — adult hermaphrodite, directed chemical synapses only, 302
  named neurons; weights are counts of scored synaptic densities. Cite Cook et
  al. (2019), Nature 571:63–71, and the 2024 PLoS Biology supplementary
  synapse list, DOI [10.1371/journal.pbio.3002939](https://doi.org/10.1371/journal.pbio.3002939),
  CC BY 4.0.

The hemibrain, MANC and male CNS graphs were all obtained through neuPrint,
which has its own citation: Plaza, S. M., et al. (2022), *neuPrint: An open
access tool for EM connectomics*, Frontiers in Neuroinformatics 16:896292,
[doi:10.3389/fninf.2022.896292](https://doi.org/10.3389/fninf.2022.896292).

Where a dataset's license is stated by its publisher it is noted above; where
none is recorded, check with the publisher before redistributing anything
derived from it.
