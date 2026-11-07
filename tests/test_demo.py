import numpy as np
import pandas as pd
import pytest
import torch

from examples.make_demo_data import make_data, spin_operators
from examples.demo import GAN, DDPM
from C_GAN.src.experiment_generator import generate as gan
from C_difusion.src.experiment_generator import generate_per_phase as ddpm
from quantum_phases.split import train_test_split_per_class


def test_spin_algebra_and_ground_state_correlators():
    sx, sy, sz = spin_operators()
    np.testing.assert_allclose(sx @ sy - sy @ sx, 1j * sz, atol=1e-14)
    np.testing.assert_allclose(sx @ sx + sy @ sy + sz @ sz, 2*np.eye(3), atol=1e-14)
    data, parameters = make_data(n_per_regime=4)
    assert data.shape == (12, 10)
    assert np.isfinite(data.to_numpy()).all()
    assert np.abs(data.drop(columns="labels").to_numpy()).max() <= 1 + 1e-12
    # The polarized ferromagnetic doublet has Cx=Cy=0, Cz=1.
    ferro = data[data.labels == 0].drop(columns="labels").to_numpy()
    np.testing.assert_allclose(ferro[:, :6], 0, atol=1e-12)
    np.testing.assert_allclose(ferro[:, 6:], 1, atol=1e-12)
    assert (parameters.loc[parameters.labels == 0, "degeneracy"] == 2).all()


def test_split_has_no_overlap_and_uses_exact_count():
    data, _ = make_data(n_per_regime=5)
    data["row_id"] = np.arange(len(data))
    train, test = train_test_split_per_class(data, 3, seed=42)
    assert train.groupby("labels").size().to_dict() == {0: 3, 1: 3, 2: 3}
    assert set(train.row_id).isdisjoint(test.row_id)
    assert set(train.row_id) | set(test.row_id) == set(data.row_id)


@pytest.mark.parametrize("arm", ["gan", "ddpm"])
def test_generators_are_deterministic_bounded_and_record_training(arm):
    torch.set_num_threads(1)
    data, _ = make_data(n_per_regime=4)
    train, _ = train_test_split_per_class(data, 3, seed=42)
    context = dict(dataset="DEMO-L4", n_train=3, seed=42)
    def generate():
        if arm == "gan":
            return gan(train, n_synth_per_class=10, config={**GAN, "epochs": 3},
                       seed=42, unit_context=context)
        return ddpm(train, n_train=3, n_synth_per_class=10,
                    config={**DDPM, "min_train_steps": 3}, seed=42,
                    label_col="labels", unit_context=context)
    first, second = generate(), generate()
    pd.testing.assert_frame_equal(first, second)
    assert first.groupby("labels").size().to_dict() == {0: 10, 1: 10, 2: 10}
    assert np.isfinite(first.to_numpy()).all()
    assert np.abs(first.drop(columns="labels").to_numpy()).max() <= 1
    metadata = first.attrs["training_metadata"]
    assert len(metadata) == 3
    assert all(record["steps"] >= 3 for record in metadata)
