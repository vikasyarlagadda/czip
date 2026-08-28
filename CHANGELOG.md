# Changelog

## 0.1.0 — 2026-08-28

Initial public release.

- Flat and nested degree-corrected stochastic block model coders, range-coded
  with `constriction`.
- Weighted and topology-only containers (`--wmin` sets the weight origin;
  omit it for topology only).
- The `.cz` v1 container format: canonical JSON header, validated stream
  table, params and source digests, itemized coder report.
- `czip encode` / `czip decode` / `czip info` command line, with automatic
  model selection (`--model auto`) and explicit models (`--partition`,
  `--hierarchy`).
- Encode verifies losslessness by decoding what it just wrote and comparing
  against the source (`--no-verify` opts out).
- The decoder-closure artifact: `python -m czip.decoder_artifact` builds a
  byte-reproducible archive of the seven modules a decode actually needs.
- Frozen results tables (`docs/RESULTS.md`) for five fly connectomes and one
  C. elegans comparison row, measured with this coder and published alongside
  the measured size of the tool itself.
