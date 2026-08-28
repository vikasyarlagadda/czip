"""Tests for czip/coder.py — constriction entropy-coder round-trip harness.

Ground truth: realized bitstream length must match the
theoretical -log2 P sum to within quantization overhead, and decode must be
lossless.
"""

from __future__ import annotations

import numpy as np
import pytest

from czip.coder import (
    decode_bernoulli,
    decode_categorical,
    encode_bernoulli,
    encode_categorical,
    ideal_bits_bernoulli,
    ideal_bits_categorical,
    roundtrip_bernoulli,
    roundtrip_categorical,
)


def random_categorical(n: int, k: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    probs = rng.dirichlet(np.ones(k), size=n)
    symbols = np.array([rng.choice(k, p=p) for p in probs], dtype=np.int32)
    return symbols, probs


def test_ideal_bits_categorical_closed_form():
    probs = np.array([[0.5, 0.5], [0.25, 0.75]])
    symbols = np.array([0, 1], dtype=np.int32)
    # -log2(0.5) - log2(0.75)
    expected = 1.0 + -np.log2(0.75)
    assert ideal_bits_categorical(symbols, probs) == pytest.approx(expected)


def test_ideal_bits_bernoulli_closed_form():
    probs = np.array([0.5, 0.25])
    bits = np.array([1, 0], dtype=np.int32)
    expected = 1.0 + -np.log2(0.75)
    assert ideal_bits_bernoulli(bits, probs) == pytest.approx(expected)


def test_categorical_roundtrip_lossless():
    symbols, probs = random_categorical(5000, 7, seed=42)
    words = encode_categorical(symbols, probs)
    assert words.dtype == np.uint32
    decoded = decode_categorical(words, probs)
    np.testing.assert_array_equal(decoded, symbols)


def test_categorical_realized_matches_ideal():
    symbols, probs = random_categorical(10_000, 5, seed=7)
    report = roundtrip_categorical(symbols, probs)
    assert report["lossless"] is True
    ideal = report["ideal_bits"]
    realized = report["realized_bits"]
    # quantized model can't beat the exact ideal by more than rounding slack
    assert realized >= ideal - 1.0
    # word-granularity constant + tiny per-symbol quantization overhead
    assert realized - ideal <= 64.0 + 1e-3 * ideal


def test_bernoulli_roundtrip_lossless_and_tight():
    rng = np.random.default_rng(3)
    probs = rng.uniform(0.01, 0.99, size=20_000)
    bits = (rng.random(20_000) < probs).astype(np.int32)
    report = roundtrip_bernoulli(bits, probs)
    assert report["lossless"] is True
    assert report["realized_bits"] >= report["ideal_bits"] - 1.0
    assert report["realized_bits"] - report["ideal_bits"] <= 64.0 + 1e-3 * report["ideal_bits"]
    decoded = decode_bernoulli(encode_bernoulli(bits, probs), probs)
    np.testing.assert_array_equal(decoded, bits)


def test_bernoulli_er_adjacency_bitmap_roundtrip():
    # ER rung demo: constant-p bitmap over a small directed adjacency matrix
    rng = np.random.default_rng(11)
    n, p = 60, 0.07
    a = (rng.random((n, n)) < p).astype(np.int32)
    np.fill_diagonal(a, 0)
    bits = a.ravel()
    probs = np.full(bits.shape, p)
    report = roundtrip_bernoulli(bits, probs)
    assert report["lossless"] is True
    decoded = decode_bernoulli(encode_bernoulli(bits, probs), probs)
    np.testing.assert_array_equal(decoded.reshape(n, n), a)


def test_empty_sequence_roundtrip():
    symbols = np.array([], dtype=np.int32)
    probs = np.zeros((0, 4))
    words = encode_categorical(symbols, probs)
    decoded = decode_categorical(words, probs)
    assert decoded.size == 0
    assert ideal_bits_categorical(symbols, probs) == 0.0


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        encode_categorical(np.array([0, 1], dtype=np.int32), np.ones((3, 2)) / 2)
    with pytest.raises(ValueError):
        encode_bernoulli(np.array([0, 1], dtype=np.int32), np.array([0.5]))


def test_symbol_out_of_range_raises():
    probs = np.ones((2, 3)) / 3
    with pytest.raises(ValueError):
        encode_categorical(np.array([0, 3], dtype=np.int32), probs)
