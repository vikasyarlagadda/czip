"""Entropy-coder round-trip harness (constriction range coder).

Ground truth for codelength claims: the realized
bitstream length of an actual encode must match the theoretical sum of
-log2 P to within quantization/word-granularity overhead, and decode must
be lossless. This module wraps constriction 0.5's range coder for the two
model shapes the ladder needs:

- categorical: per-symbol probability rows (n, k) — variable distributions,
  fixed alphabet (SBM block labels, weight buckets, ...);
- Bernoulli: per-position edge probability vector (n,) — edge-indicator
  codes (ER, GLM edge models).

Probabilities are quantized by constriction onto a fixed-precision grid, so
realized bits track the ideal under the *quantized* model; the excess over
the exact ideal is O(1) word padding plus O(n * 2^-precision) drift.
"""

from __future__ import annotations

import numpy as np
import constriction

_CATEGORICAL_FAMILY = constriction.stream.model.Categorical(perfect=False)
_BERNOULLI_FAMILY = constriction.stream.model.Bernoulli(perfect=False)


def _check_categorical(symbols: np.ndarray, probs: np.ndarray) -> None:
    if probs.ndim != 2:
        raise ValueError(f"probs must be 2-D (n, k), got shape {probs.shape}")
    if symbols.shape[0] != probs.shape[0]:
        raise ValueError(
            f"length mismatch: {symbols.shape[0]} symbols vs {probs.shape[0]} prob rows"
        )
    if symbols.size and (symbols.min() < 0 or symbols.max() >= probs.shape[1]):
        raise ValueError(
            f"symbols must lie in [0, {probs.shape[1]}); got range "
            f"[{symbols.min()}, {symbols.max()}]"
        )


def _check_bernoulli(bits: np.ndarray, probs: np.ndarray) -> None:
    if probs.ndim != 1:
        raise ValueError(f"probs must be 1-D (n,), got shape {probs.shape}")
    if bits.shape[0] != probs.shape[0]:
        raise ValueError(
            f"length mismatch: {bits.shape[0]} bits vs {probs.shape[0]} probs"
        )
    if bits.size and (bits.min() < 0 or bits.max() > 1):
        raise ValueError("bits must be 0/1")


def ideal_bits_categorical(symbols: np.ndarray, probs: np.ndarray) -> float:
    """Exact -sum log2 probs[i, symbols[i]] (theoretical codelength)."""
    _check_categorical(symbols, probs)
    if symbols.size == 0:
        return 0.0
    p = probs[np.arange(symbols.shape[0]), symbols]
    return float(-np.log2(p).sum())


def ideal_bits_bernoulli(bits: np.ndarray, probs: np.ndarray) -> float:
    """Exact -sum log2 P(bits[i]) under per-position Bernoulli probs."""
    _check_bernoulli(bits, probs)
    if bits.size == 0:
        return 0.0
    p = np.where(bits == 1, probs, 1.0 - probs)
    return float(-np.log2(p).sum())


def encode_categorical(symbols: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Range-encode symbols under per-row categorical probs; returns uint32 words."""
    _check_categorical(symbols, probs)
    encoder = constriction.stream.queue.RangeEncoder()
    if symbols.size:
        encoder.encode(
            np.ascontiguousarray(symbols, dtype=np.int32),
            _CATEGORICAL_FAMILY,
            np.ascontiguousarray(probs, dtype=np.float64),
        )
    return encoder.get_compressed()


def decode_categorical(words: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Decode a uint32 word stream back into probs.shape[0] symbols."""
    if probs.shape[0] == 0:
        return np.array([], dtype=np.int32)
    decoder = constriction.stream.queue.RangeDecoder(words)
    return decoder.decode(
        _CATEGORICAL_FAMILY, np.ascontiguousarray(probs, dtype=np.float64)
    )


def encode_bernoulli(bits: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Range-encode 0/1 bits under per-position Bernoulli probs."""
    _check_bernoulli(bits, probs)
    encoder = constriction.stream.queue.RangeEncoder()
    if bits.size:
        encoder.encode(
            np.ascontiguousarray(bits, dtype=np.int32),
            _BERNOULLI_FAMILY,
            np.ascontiguousarray(probs, dtype=np.float64),
        )
    return encoder.get_compressed()


def decode_bernoulli(words: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Decode a uint32 word stream back into probs.shape[0] bits."""
    if probs.shape[0] == 0:
        return np.array([], dtype=np.int32)
    decoder = constriction.stream.queue.RangeDecoder(words)
    return decoder.decode(
        _BERNOULLI_FAMILY, np.ascontiguousarray(probs, dtype=np.float64)
    )


def _report(
    realized_words: np.ndarray, decoded: np.ndarray, original: np.ndarray, ideal: float
) -> dict:
    realized = float(32 * realized_words.size)
    return {
        "realized_bits": realized,
        "ideal_bits": ideal,
        "overhead_bits": realized - ideal,
        "lossless": bool(np.array_equal(decoded, original)),
        "n_symbols": int(original.size),
    }


def roundtrip_categorical(symbols: np.ndarray, probs: np.ndarray) -> dict:
    """Encode, decode, and report realized vs ideal bits + losslessness."""
    words = encode_categorical(symbols, probs)
    decoded = decode_categorical(words, probs)
    return _report(words, decoded, symbols, ideal_bits_categorical(symbols, probs))


def roundtrip_bernoulli(bits: np.ndarray, probs: np.ndarray) -> dict:
    """Encode, decode, and report realized vs ideal bits + losslessness."""
    words = encode_bernoulli(bits, probs)
    decoded = decode_bernoulli(words, probs)
    return _report(words, decoded, bits, ideal_bits_bernoulli(bits, probs))
