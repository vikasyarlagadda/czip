"""Tests for the weights entropy-coder round-trip (czip.weights_coder)."""


import numpy as np
import pytest

from czip.weights import (
    codelength_bits,
    mdl_quantize_family,
    weight_code,
)
from czip.weights_coder import (
    WINDOW,
    BitReader,
    BitWriter,
    decode_weights_stream,
    encode_weights_stream,
    read_header,
    weights_roundtrip,
    write_header,
    _shift_base,
)


# ---------------------------------------------------------------- bit I/O

def test_bit_writer_reader_roundtrip():
    bw = BitWriter()
    vals = [(5, 3), (0, 1), (1, 1), (1023, 10), (7, 22), (0, 4)]
    for v, n in vals:
        bw.write(v, n)
    br = BitReader(bw.getvalue())
    for v, n in vals:
        assert br.read(n) == v


def test_bit_writer_rejects_overflow():
    bw = BitWriter()
    with pytest.raises(ValueError):
        bw.write(8, 3)


# ----------------------------------------------------------- header codec

def _fit_params_q(family, w, wmin):
    return mdl_quantize_family(family, np.asarray(w), wmin)


@pytest.mark.parametrize("family,w,wmin", [
    ("geometric", [5, 5, 6, 7, 9, 5, 8, 13], 5),
    ("zeta", [1, 1, 1, 2, 3, 8, 40, 1, 2], 1),
    ("lognormal", [5, 6, 8, 12, 20, 7, 6, 9, 33], 5),
    ("negbinom", [1, 2, 3, 1, 5, 2, 4, 2, 8, 3], 1),
])
def test_header_roundtrip_exact(family, w, wmin):
    fit = _fit_params_q(family, np.asarray(w), wmin)
    bw = BitWriter()
    write_header(bw, family, fit["delta"], fit["params_q"])
    fam2, delta2, params2 = read_header(BitReader(bw.getvalue()))
    assert fam2 == family
    assert delta2 == fit["delta"]
    for name, val in fit["params_q"].items():
        assert params2[name] == pytest.approx(val, rel=0, abs=1e-12)


def test_header_rejects_off_grid_delta():
    bw = BitWriter()
    with pytest.raises(ValueError):
        write_header(bw, "geometric", 0.3, {"p": 0.5})


# ---------------------------------------------- chained-window telescoping

@pytest.mark.parametrize("family,params,wmin", [
    ("geometric", {"p": 0.25}, 5),
    ("zeta", {"s": 1.5}, 1),
    ("lognormal", {"mu": 2.0, "sigma": 1.5}, 1),
    ("negbinom", {"r": 2.0, "p": 0.1}, 5),
])
def test_window_ideal_matches_exact_pmf(family, params, wmin):
    # values crossing several windows so escapes are exercised
    rng = np.random.default_rng(0)
    w = np.concatenate([
        rng.integers(wmin, wmin + 10, size=50),
        rng.integers(wmin + WINDOW - 2, wmin + WINDOW + 5, size=10),
        np.array([wmin + 3 * WINDOW + 7, wmin + 2 * WINDOW]),
    ]).astype(np.int64)
    headers = [(family, params)]
    shifted = [w - wmin + _shift_base(family)]
    streams, ideal, n_sym, deficit, n_floored = encode_weights_stream(shifted, headers)
    exact = codelength_bits(family, w, wmin, params)
    assert ideal == pytest.approx(exact, rel=1e-9)
    assert n_sym > w.size  # escapes happened

    decoded = decode_weights_stream(streams, [w.size], headers)
    assert np.array_equal(decoded[0], shifted[0])
    realized = 32 * sum(s.size for s in streams)
    slack = 64 * len(streams) + 3e-4 * n_sym + 32
    assert ideal - deficit - slack <= realized <= ideal + slack


def test_multi_group_stream_roundtrip():
    rng = np.random.default_rng(1)
    headers = [("geometric", {"p": 0.3}), ("zeta", {"s": 2.0}),
               ("lognormal", {"mu": 1.0, "sigma": 1.0})]
    shifted = [rng.integers(0, 40, size=200),
               rng.integers(1, 300, size=100),
               rng.integers(1, 60, size=150)]
    shifted = [np.asarray(s, dtype=np.int64) for s in shifted]
    streams, ideal, n_sym, deficit, n_floored = encode_weights_stream(shifted, headers)
    decoded = decode_weights_stream(streams, [200, 100, 150], headers)
    for d, s in zip(decoded, shifted):
        assert np.array_equal(d, s)
    assert ideal > 0


# ------------------------------------------------------- full round-trip

def _synthetic_grouped(seed=0, wmin=5, n_groups=6):
    rng = np.random.default_rng(seed)
    ws, gids = [], []
    for g in range(n_groups):
        n = int(rng.integers(3, 120))
        style = g % 3
        if style == 0:
            w = wmin + rng.geometric(0.4, size=n) - 1
        elif style == 1:
            w = wmin + np.round(rng.lognormal(1.5, 1.0, size=n)).astype(int)
        else:
            w = np.full(n, wmin) + rng.integers(0, 3, size=n)
        ws.append(w)
        gids.append(np.full(n, 100 + g))
    return np.concatenate(ws).astype(np.int64), np.concatenate(gids)


@pytest.mark.parametrize("seed", [0, 1])
def test_weights_roundtrip_synthetic(seed):
    wmin = 5
    w, gids = _synthetic_grouped(seed=seed, wmin=wmin)
    result = weight_code(w, wmin, gids)
    rep = weights_roundtrip(w, wmin, gids, result)
    assert rep["lossless"]
    assert rep["within_slack"], rep
    # the theoretical data part must equal the chained-window ideal exactly
    assert rep["ideal_vs_theoretical_data_bits"] == pytest.approx(0.0, abs=1e-6)
    # delta headers: 4 bits per transmitted header, by construction
    assert rep["L_delta_headers_bits"] == 4.0 * rep["n_headers"]
    assert rep["realized_total_bits"] >= rep["theoretical_bits"]


def test_weights_roundtrip_rejects_group_mismatch():
    wmin = 5
    w, gids = _synthetic_grouped(wmin=wmin)
    result = weight_code(w, wmin, gids)
    with pytest.raises(ValueError, match="group set mismatch"):
        weights_roundtrip(w, wmin, gids + 1, result)
