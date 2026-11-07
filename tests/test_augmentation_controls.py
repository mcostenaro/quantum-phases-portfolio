"""Controles nao aprendidos do protocolo v3 (main/docs/protocolo.md, Secao 6.2)."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from quantum_phases import augmentation_controls as ac


FEATURES = [str(i) for i in range(2, 23)]
CONTEXT = {"dataset": "H1", "n_train": 3, "seed": 11}


def make_train(*, n=3, labels=(0, 2), seed=0, low=-0.9, high=0.9):
    rng = np.random.default_rng(seed)
    rows = []
    for label in labels:
        for values in rng.uniform(low, high, size=(n, len(FEATURES))):
            rows.append({**dict(zip(FEATURES, values)), "labels": label})
    return pd.DataFrame(rows)


def config_for(method, sigma=0.02):
    return {"method": method, **({"sigma": sigma} if method == "jitter" else {})}


@pytest.mark.parametrize("method", ac.CONTROL_METHODS)
def test_controls_are_deterministic_and_keep_labels(method):
    train = make_train()
    context = {**CONTEXT, "arm": method}
    first = ac.generate(train, n_synth_per_class=10, config=config_for(method), seed=11,
                        unit_context=context)
    second = ac.generate(train, n_synth_per_class=10, config=config_for(method), seed=11,
                         unit_context=context)
    pd.testing.assert_frame_equal(first, second)
    assert list(first.columns) == FEATURES + ["labels"]
    assert first.groupby("labels").size().to_dict() == {0: 10, 2: 10}
    assert (first.dtypes[FEATURES] == np.float64).all()
    metadata = first.attrs["training_metadata"]
    assert [m["label"] for m in metadata] == [0, 2]
    assert all(m["method"] == method and m["n_real"] == 3 for m in metadata)
    without_seed = all(m["gen_seed"] is None for m in metadata)
    assert without_seed == (method not in ac.RANDOM_METHODS)


@pytest.mark.parametrize("method", sorted(ac.RANDOM_METHODS))
def test_random_controls_use_seeds_derived_per_unit(method):
    train = make_train()
    base = ac.generate(train, n_synth_per_class=10, config=config_for(method), seed=11,
                       unit_context={**CONTEXT, "arm": method})
    other_unit = ac.generate(train, n_synth_per_class=10, config=config_for(method), seed=11,
                             unit_context={**CONTEXT, "seed": 12, "arm": method})
    assert not np.array_equal(base[FEATURES].to_numpy(), other_unit[FEATURES].to_numpy())
    with pytest.raises(ValueError, match="differs from control"):
        ac.generate(train, n_synth_per_class=10, config=config_for(method), seed=11,
                    unit_context={**CONTEXT, "arm": "gan"})


def test_copies_follow_exact_rotation():
    train = make_train(n=3)
    synth = ac.generate_copies(train, n_synth_per_class=10)
    for label, group in train.groupby("labels"):
        expected = group[FEATURES].to_numpy()[np.arange(10) % 3]
        np.testing.assert_array_equal(synth.loc[synth.labels == label, FEATURES].to_numpy(), expected)


def test_box_center_is_the_midrange_of_each_coordinate():
    train = make_train(n=4)
    synth = ac.generate_box_center(train, n_synth_per_class=10)
    for label, group in train.groupby("labels"):
        X = group[FEATURES].to_numpy()
        expected = 0.5 * (X.min(axis=0) + X.max(axis=0))
        np.testing.assert_array_equal(synth.loc[synth.labels == label, FEATURES].to_numpy(),
                                      np.repeat(expected[None, :], 10, axis=0))


@pytest.mark.parametrize("n", [2, 4])
def test_interpolation_lies_on_a_segment_between_two_distinct_reals(n):
    train = make_train(n=n)
    synth = ac.generate_interpolation(train, n_synth_per_class=10, unit_context=CONTEXT)
    for label, group in train.groupby("labels"):
        X = group[FEATURES].to_numpy()
        for x in synth.loc[synth.labels == label, FEATURES].to_numpy():
            on_segment = False
            for i, j in itertools.permutations(range(n), 2):
                a, b = X[i], X[j]
                lam = float((x - b) @ (a - b) / ((a - b) @ (a - b)))
                if 0 <= lam <= 1 and np.allclose(x, lam * a + (1 - lam) * b, atol=1e-12):
                    on_segment = True
                    break
            assert on_segment


def test_jitter_is_clipped_to_the_physical_bounds():
    train = make_train(low=0.95, high=0.999)
    synth = ac.generate_jitter(train, n_synth_per_class=10, config={"sigma": 0.1},
                               unit_context=CONTEXT)
    values = synth[FEATURES].to_numpy()
    assert np.abs(values).max() <= 1.0 and (values == 1.0).any()
    with pytest.raises(ValueError, match="sigma"):
        ac.generate_jitter(train, n_synth_per_class=10, config={"sigma": 0})
    with pytest.raises(ValueError, match="sigma"):
        ac.generate_jitter(train, n_synth_per_class=10, config={})


@pytest.mark.parametrize("method", ac.CONTROL_METHODS)
def test_controls_preserve_the_bound_when_reals_respect_it(method):
    train = make_train(low=-1.0, high=1.0)
    synth = ac.generate(train, n_synth_per_class=10, config=config_for(method, sigma=0.1),
                        seed=3, unit_context={**CONTEXT, "arm": method})
    assert np.abs(synth[FEATURES].to_numpy()).max() <= 1.0


def test_dispatch_refuses_unknown_methods():
    with pytest.raises(ValueError, match="config\\['method'\\]"):
        ac.generate(make_train(), n_synth_per_class=10, config={"method": "gan"})
