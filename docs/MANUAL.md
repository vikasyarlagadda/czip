# czip manual

`czip` losslessly compresses directed, weighted graphs. You hand it a sparse
adjacency matrix or an edge list; it fits or accepts a block model, entropy-
codes the graph under that model, and writes a single versioned `.cz`
container. Decoding gives you back the exact same graph — the same indices,
the same row pointers, the same integer weights — and `czip` proves that at
encode time by decoding the container it just wrote and comparing it against
your input before it hands you the file.

It was built for connectomes, where the graphs are large, sparse, integer-
weighted, and strongly block-structured, but nothing in the format is specific
to neuroscience.

Three commands:

| command | does |
|---|---|
| `czip encode` | graph (`.npz` or CSV/TSV) → `.cz` |
| `czip decode` | `.cz` → graph (`.npz`), optionally the block labels too |
| `czip info` | print a container's header as JSON |

Every command is also reachable as `python -m czip …` if you would rather not
rely on the console entry point.

## Install

```bash
pip install czip
```

Encoding with an explicit model (`--partition` / `--hierarchy`) and **all**
decoding need only numpy, scipy and constriction, which pip installs for you.
Automatic model selection (`--model auto`) additionally needs `graph-tool`,
which is not pip-installable — see
[Automatic model selection](#automatic-model-selection---model-auto) below.

## Quickstart

### Encode, inspect, decode

The repository ships a small synthetic example so you can exercise the whole
path without downloading anything. This route needs no graph-tool.

```bash
# 1. Write a synthetic block-structured graph (300 nodes, 6 blocks,
#    2,500 weighted edges) plus its partition and an edge-list copy
python examples/make_example.py

# 2. Encode it under the given partition, with weights (weight origin 1)
czip encode example_graph.npz --partition example_partition.npy --wmin 1 \
    -o example.cz

# 3. Look at what was written: model id, streams, coder report
czip info example.cz

# 4. Decode, recovering the block labels alongside the graph
czip decode example.cz -o example_decoded.npz \
    --labels-out example_labels_out.npy

# 5. Check for yourself that it round-tripped
python - <<'PY'
import numpy as np, scipy.sparse as sp
a = sp.load_npz("example_graph.npz").tocsr()
b = sp.load_npz("example_decoded.npz").tocsr()
assert np.array_equal(a.indices, b.indices)
assert np.array_equal(a.indptr, b.indptr)
assert np.array_equal(a.data, b.data)
print("round-trip exact")
PY
```

Step 5 is belt-and-braces: step 2 already decoded the container and compared
it against the source, and stamped `report.lossless: true` into the header,
which `czip info` will show you.

### Encoding an edge list

`czip` also reads a `src,dst[,weight]` CSV or TSV. Node ids are arbitrary
tokens — including the long numeric ids connectome portals hand out — and the
id-to-index map is written next to the output. The example generator writes an
edge-list copy of the same graph, so you can encode it either way and compare:

```bash
czip encode example_edges.csv --partition example_partition.npy --wmin 1 \
    -o example_from_csv.cz
# also writes example_from_csv.cz.node_ids.npy — row i of the decoded matrix
# is node node_ids[i]
```

Details worth knowing:

- The encode writes **one extra file** the `.npz` path does not: an
  `<out>.node_ids.npy` id map, printed on the way past. Keep it — without it
  you get the graph back but not the names of its nodes.
- Node ids are mapped to matrix indices `0..n-1` in **sorted token order**
  (sorted as text, not numerically — so `"10"` sorts before `"9"`, and `"9"`
  after `"100"`). The partition file is indexed against those matrix rows, so
  build it against that order rather than against your raw ids. This is the
  usual way to waste bits on an edge list: a partition that does not line up
  with the loader's row order still encodes losslessly, it just describes the
  graph worse, and the container comes out bigger. The example generator
  sidesteps it by zero-padding its ids so that text order and numeric order
  agree — which is why this container comes out exactly the same size as the
  one encoded from `example_graph.npz` above.
- Duplicate `(src, dst)` rows sum their weights.
- A missing third column means weight 1 for every row.
- A header row is skipped only when a third column exists and is non-numeric.
  An unweighted two-column file must therefore be headerless — a documented
  limit, not an accident.

### Automatic model selection (`--model auto`)

If you have no partition to hand, `czip` will fit one:

```bash
czip encode graph.npz --model auto --wmin 1 -o graph.cz
```

**This is the one feature with a heavyweight dependency.** Fitting uses
`graph-tool`, which cannot be installed with pip — it is a C++ library with
its own build, distributed through conda-forge. Nothing else in `czip` needs
it, and the decode path never imports it, so you only need graph-tool on the
machine that *creates* containers this way:

```bash
micromamba create -n czip -c conda-forge python=3.12 graph-tool
micromamba run -n czip pip install czip
```

(`conda` or `mamba` work identically; substitute whichever you use.)

### Encoding a nested hierarchy

If you have a whole nested block hierarchy rather than a single partition:

```bash
czip encode graph.npz --hierarchy hierarchy.npz --wmin 1 -o graph.cz
```

The hierarchy file is an `.npz` with one array per level, keys `level_0` (one
label per node) through `level_{k-1}` — the layout graph-tool's `get_bs()`
produces. See [The nested models](#the-nested-models---hierarchy) for what
`czip` does with it.

## Inputs and their rules

Input is either a scipy CSR `.npz` adjacency (integer weights — for a
connectome, synapse counts) or a `src,dst[,weight]` CSV/TSV edge list. The
rules `encode` enforces:

- **At least one node and at least one edge.** The message header codes the
  edge count with Elias-gamma, which has no codeword for zero.
- **Exactly one model**, chosen from `--partition`, `--hierarchy`, or
  `--model auto`.
- **A partition carries exactly one block label per node**, with labels in
  `0..B-1`. A hierarchy is held to the nested normal form instead (below).
- **Weights must be integers.** A non-integral weight is an error, never
  rounded. That holds for both input kinds: the CSV/TSV loader checks each
  row's weight before any rounding and names the offending row (`4.0` is
  accepted, `2.7` is an error), matching the array guard on the `.npz` path.

### `--wmin` is the weight origin, not a threshold

`--wmin W` encodes weights (model `dcsbm+weights`, or `nested-dcsbm+weights`
with `--hierarchy`) with **W as the weight origin**: the code transmits
`w - W`, so every weight must be `>= W`. It is not a filter and it never drops
an edge — a smaller weight is a hard error, checked against your data before
any fit runs. If you mean to drop weak edges, threshold the graph yourself
first, then tell `czip` the origin of what remains.

Omit `--wmin` for a topology-only container. A weighted input is then refused
unless you pass `--allow-weight-drop` and say explicitly that you want the
weights discarded. Weights are never dropped silently.

### The losslessness stamp: `lossless` vs `digest_verified`

`encode` proves losslessness by decoding the blob it just built and comparing
it against the source, then stamps `report.lossless: true` into the header.
`--no-verify` opts out of that decode and leaves the claim unproven
(`lossless: null`) — the container is still written, but nothing has checked
it.

The verification records two *different* facts, and never conflates them:

| header field | set when | means |
|---|---|---|
| `digest_verified: true` | the blob decoded and both header digests (`params_digest`, `source_digest`) matched the decoded arrays | the container agrees with **itself** — a self-check, and no evidence at all about your input file |
| `lossless: true` | additionally, **both** source arrays (adjacency and labels) were supplied and matched the decode entry-for-entry | the decode reproduces **your source**, compared independently of the container's own digests |

A losslessness stamp requires a source to be lossless *with respect to*, so
verifying a blob with no source arrays leaves `lossless` null and records only
`digest_verified: true`. The CLI, run without `--no-verify`, passes the source
arrays, so it stamps `lossless: true`. Source weights are held to the same
integrality rule as the encoder's input — a fractional source array is an
error here too, never cast into agreement with its own truncation.

The two commands say different things on purpose, and it is worth reading them
literally. `encode` finishes with `decode-verified lossless`, because it still
had your source in hand to compare against. `decode` finishes with
`params + source digests verified` — never "lossless" — because a decode has
only the container: it can confirm the file is internally consistent and
untampered, and it cannot know what you originally fed the encoder.

## Model selection (`--model auto`)

For an arbitrary graph with no hand-matched covariates, `--model auto` scores a
ladder of covariate-free models — microcanonical Erdős–Rényi, the directed
configuration model (degree), and the flat degree-corrected SBM (budgeted
seeded restarts of graph-tool's `minimize_blockmodel_dl`, controlled by
`--restarts` and `--seed0`) — by itemized analytic description length, and
hands `czip` the best encodable partition. An Erdős–Rényi or degree winner is
encoded as `B=1`: its adjacency stream realizes the configuration model's
`L(G|θ)` exactly.

The nested DC-SBM is a fourth candidate, in by default with one fit restart
(`--nested-restarts`; `0` excludes it and skips that fit entirely). A nested
winner is emitted as a nested container, stamped `nested-dcsbm` or
`nested-dcsbm+weights` — the same layout `--hierarchy` encodes explicitly.
`--model auto` is therefore not a flat-model flag: it picks the family.

Fit scores are *selection metadata*, recorded in `fit_meta` in the header. The
codelength claim for a `.cz` file is always its own coder report (realized and
ideal bits), never the fit scores.

## The nested models (`--hierarchy`)

`--hierarchy PATH.npz` codes a whole DC-SBM hierarchy instead of a flat
partition, and stamps model id `nested-dcsbm` (topology only) or
`nested-dcsbm+weights` (when `--wmin` is also given).

The model id follows the model, not the flag. `--partition` always produces
the flat `dcsbm` / `dcsbm+weights` ids, because a flat partition is all you
gave it. `--model auto` produces either family: if the nested candidate wins
the selection, the fitted hierarchy goes to the nested encoder and the
container is stamped `nested-dcsbm` / `nested-dcsbm+weights`, exactly as if
you had passed that hierarchy to `--hierarchy` yourself. So a container's
`model_id` tells you which model was coded; it does not tell you which flag
produced it. Several of the containers in [RESULTS.md](RESULTS.md) are nested
containers that came out of `--model auto`.

`czip` canonicalizes the input hierarchy first: each level is densified to
`0..B_l−1` by first appearance, and the hierarchy is truncated at the first
level with `B = 1`. A hierarchy that never reaches a single block is refused
rather than padded into one — with no level above it, graph-tool charges the
base level's `edges_dl`, which makes it a different model, not a shorter
description of the same one. The dense level 0 *is* the base partition every
stage below keys on, the weights layer included: weights group edges by
ordered block pair on the base labels alone, so a nested model changes nothing
above it.

The message mirrors the flat one segment for segment, with the flat `e_rs`
stage replaced by the hierarchy:

```
header(e) | header(L) | partitions bottom-up (l = 0..L-1)
          | expansions top-down (l = L-1..1) | degrees | adjacency
```

The node count `n` is common knowledge; the level count `L` is not —
graph-tool charges nothing for the number of levels — so `L` travels as
explicit Elias-gamma header bits, itemized on its own
(`header_levels_bits`) instead of buried. Each level contributes the stream
pair `level_{l}_partition_rank` / `level_{l}_partition_words` (the flat
partition code, over the `B_{l−1}` blocks of the level below), and each level
above the base contributes `level_{l}_expand_payload`. The base
`degrees_payload` and `adjacency_words` streams are the flat coder's,
unchanged.

Every stage's model depends only on already-decoded quantities: the partitions
fix every `B_l` and every group size, and the expansions then run top-down from
`E^{(L−1)} = [[e]]` (read off the header) down to `E^{(0)}`, which is what the
degree and adjacency stages need.

The header carries `n_levels`. Decode bounds it against the node count before
allocating anything — a canonical hierarchy over `n` nodes has at least 1 and
at most `n` levels — and then requires exactly the streams that count implies,
naming any that are missing. `params_digest` for a nested container is SHA256
over the level count followed by every canonical level array, each serialized
`<i8`: digesting the base partition alone would give the same value to two
containers whose hierarchies differ only above level 0, and decode's partition
check would then pass on a swapped model. Decode recomputes the nested digest
over the decoded levels and raises before returning the graph, exactly as the
flat model's digest is recomputed over the decoded partition.

Both model families share `format_version` 1: `model_id` dispatches at decode,
so a container written under either model decodes with the same reader.

## The `.cz` v1 container

Little-endian layout:

```
magic "CZIP" | format_version u16 | header_len u32
| header (canonical JSON, sorted keys, UTF-8) | concatenated stream bytes
```

The header records the model id, the params digest, the payload kind
(`topology`, or topology + weights), `n_levels` for the nested models, a
`stream_table` of `(name, kind, offset, length)` per stream, the full coder
report, and `fit_meta`.

`kind` says how to read a stream's bytes, and nothing more: `u32` means a
constriction word stream, to be reinterpreted as little-endian `uint32`
words; `bytes` means an opaque byte payload, taken as-is. Plenty of segments
are `bytes` — the message header, the edge and degree payloads, the weights
header, and rank payloads among them — so `bytes` is not a synonym for "rank
payload". Which kind a given segment takes can also depend on the branch its
coder chose for this graph, so the table is always read, never assumed.

The stream table is validated as a geometry on read, never trusted: offsets
and lengths must be non-negative and must tile the stream region exactly —
monotonic, no gaps, no overlaps, no trailing bytes — and
`container_overhead_bits` is then the measured blob extent beyond that tiling.
`decode` also sanity-bounds the declared `n_nodes` / `n_edges` against the
container's byte length before allocating anything.

The container is a dispatcher, not a coder: model bits come verbatim from the
message segments, and the container adds no padding of its own — a `.cz` file
is exactly its 10-byte prefix, its header JSON, and the streams back to back,
with nothing between them. (Any alignment padding belongs to a coder's own
word stream, and is charged to that stream.)

The container's own cost — the magic bytes, the version and header-length
prefix, and the header JSON — is measured as `container_overhead_bits`: the
blob extent beyond what the stream table tiles, not a sum of self-declared
lengths. **That cost is itemized, not excluded.** It is reported on its own
line so you can see exactly what the container costs you, and it is *also*
inside every bits-per-edge figure published for `czip`, because those figures
are whole-container rates: total container bytes × 8 ÷ edges. Itemizing
overhead and excluding it are different things, and only the first one is
honest.

## The decoder closure

Decoding a `.cz` file needs exactly seven modules of this package plus numpy,
scipy and constriction. Nothing else — not graph-tool, not any fitting code,
not any of the model-selection machinery.

That is a pinned structural invariant rather than an observation. The test
suite decodes real containers in a fresh interpreter with the encode-side
packages made unimportable, and asserts that the loaded closure is exactly
numpy + scipy + constriction plus those seven modules. If a decode-path module
ever grew an encode-side import, the suite fails.

You can build that closure as a standalone archive:

```bash
python -m czip.decoder_artifact --out czip_decoder.zip
```

The archive is generated, not committed, and it is byte-reproducible: a fixed
timestamp, fixed permissions, a fixed member order, and a generated README
with no variable content mean two builds from the same source tree produce the
same bytes. That is what makes it honest to quote a size for the decoder —
see [RESULTS.md](RESULTS.md).

## Guarantees

- **Lossless.** Decode reproduces the exact CSR adjacency: indices, indptr,
  weights. The correctness gate is the round-trip test suite, unchanged by any
  optimization, and `encode` additionally decodes what it just wrote (see
  `--no-verify`).

- **Honest accounting.** Every stream reports realized versus ideal (−log₂P)
  bits. Container overhead is itemized rather than hidden — and counted, not
  excluded: every published bits-per-edge figure is a whole-container rate
  with that overhead inside it. Encode gates the
  comparison: each entropy-coded segment's realized bits must land inside
  `[ideal − floored-symbol deficit − slack, ideal + slack]`
  (`report.bits_gate`), and a violation fails the encode loudly. The slack's
  padding term is about *segment* padding — each word stream flushed to whole
  32-bit words, each byte payload to whole bytes; the container itself pads
  nothing — and it counts this message's own segments
  (`report.n_word_streams` / `report.n_byte_streams`, written by the encoder)
  rather than a constant: a nested message carries a partition stream pair per
  level plus the expansions, and several segments switch between a rank payload
  and a word stream depending on the edge total, so the count is a property of
  the message just built. The deficit is on the lower side because
  constriction's probability floor makes deep-tail symbols cost *less* than
  −log₂P — which is why real headers can carry a negative `overhead_bits`. The
  drift term is `3e-4 × n_symbols`, linear in message length rather than
  O(1): a tight engineering reconciliation window (0.06% of ideal bits at a
  million-edge scale), not an asymptotic bound.

- **Deterministic decode.** Encode and decode are CPU-deterministic, and `.cz`
  headers carry the params digest needed to verify the message. The DC-SBM
  *fit* behind `--model auto` is not deterministic: graph-tool's
  `minimize_blockmodel_dl` is OpenMP-parallel and thread-count dependent, so
  re-running the same seeds on a different thread count can land on a different
  partition. `fit_meta` records `omp_threads` alongside the seeds.
  `params_digest` verifies the partition inside a given `.cz`; it does not
  regenerate one. That verification happens at decode: `decode` recomputes
  SHA256 over the decoded partition's `<i8` bytes — the exact byte content the
  encoder digested, and for a nested container the level count plus every
  decoded level — and raises `params_digest mismatch` before returning the
  graph, so a tampered or absent field fails the decode rather than being
  carried as an unchecked claim. The `source_digest` over the decoded labels,
  edges and weights is checked in the same pass.

- **A stated degree-term convention.** Both topology coders realize
  graph-tool's `degree_dl_kind='uniform'` degree term, and that is the parity
  target the ideal-bits tests hold them to. graph-tool's default `distributed`
  kind is a *reference*, not a rung of the container: its delta against
  `uniform` is reported and never coded, because `q(m,n)` is asymptotically
  approximated by graph-tool above its `n=10,000` cache and so is not an exact
  codable count at connectome scale.
