import json

import numpy as np
import pandas as pd
import pytest
from sklearn import get_config
from sklearn.neighbors import KNeighborsClassifier

from quantum_phases.classify import (
    _build_classifier,
    classifier_config_json,
    evaluate_real_vs_augmented,
)
from quantum_phases.neighbors import StableEuclideanKNN


@pytest.mark.parametrize("offset", [1e-14, 0.3])
def test_real_and_copies_select_the_closer_point_after_l2(offset):
    # Near-coincident regression and a well-separated control, both ED-21 size.
    train = np.zeros((2, 21))
    train[:, 0] = 1.0
    train[:, 1] = [offset, offset / 5]
    test = np.zeros((2, 21))
    test[:, 0] = 1.0
    test[1, 1] = offset
    df_train = pd.DataFrame(train).assign(labels=[7, 2])
    df_test = pd.DataFrame(test).assign(labels=[2, 7])
    copies = pd.concat([df_train] * 7, ignore_index=True)

    metrics = evaluate_real_vs_augmented(df_train, copies, df_test, k=1)

    for name in ("acc", "balanced_acc", "macro_f1"):
        assert metrics[f"{name}_real"] == 1.0
        assert metrics[f"{name}_aug"] == 1.0
        assert metrics[f"delta_{name}"] == 0.0


@pytest.mark.parametrize("algorithm", ["auto", "brute", "kd_tree", "ball_tree"])
def test_exact_distance_ties_follow_training_order_in_predict_and_proba(algorithm):
    # Label 7 wins by position, although class order would prefer label 2.
    model = StableEuclideanKNN(n_neighbors=1, algorithm=algorithm).fit(
        [[1.0, 0.0], [1.0, 0.0]], [7, 2]
    )
    before = get_config()["enable_cython_pairwise_dist"]
    assert model.predict([[1.0, 0.0]]).tolist() == [7]
    np.testing.assert_array_equal(model.predict_proba([[1.0, 0.0]]), [[0.0, 1.0]])
    assert get_config()["enable_cython_pairwise_dist"] == before


def test_k_boundary_ties_and_class_vote_ties_are_distinct():
    model = StableEuclideanKNN(n_neighbors=3).fit(
        [[1.0, 0.0]] * 4, [7, 2, 7, 2]
    )
    assert model.kneighbors([[1.0, 0.0]], return_distance=False).tolist() == [[0, 1, 2]]
    assert model.predict([[1.0, 0.0]]).tolist() == [7]
    model.set_params(n_neighbors=2)
    assert model.predict([[1.0, 0.0]]).tolist() == [2]


def test_direct_distance_preserves_tiny_nonzero_separations_across_blocks():
    model = StableEuclideanKNN(n_neighbors=1).fit(
        [[1.0, 1e-14], [1.0, 2e-15]], [7, 2]
    )
    model._distance_block_bytes = 16  # Force one query per block.
    distances, indices = model.kneighbors([[1.0, 0.0], [1.0, 1e-14]])
    np.testing.assert_array_equal(indices, [[1], [0]])
    np.testing.assert_allclose(distances, [[2e-15], [0.0]], rtol=1e-15, atol=0)
    np.testing.assert_array_equal(model.predict([[1.0, 0.0], [1.0, 1e-14]]), [2, 7])


@pytest.mark.parametrize("weights", ["uniform", "distance"])
def test_regular_k3_agrees_with_sklearn(weights):
    rng = np.random.default_rng(2026)
    train, test = rng.normal(size=(25, 21)), rng.normal(size=(7, 21))
    labels = np.arange(25) % 3
    params = dict(n_neighbors=3, weights=weights, metric="euclidean")
    stable = StableEuclideanKNN(**params).fit(train, labels)
    reference = KNeighborsClassifier(**params).fit(train, labels)
    np.testing.assert_array_equal(stable.predict(test), reference.predict(test))
    np.testing.assert_allclose(stable.predict_proba(test), reference.predict_proba(test))


def test_distance_weights_ignore_nonzero_neighbors_when_exact_match_exists():
    model = StableEuclideanKNN(n_neighbors=3, weights="distance").fit(
        [[1.0, 0.0], [1.0, 0.0], [1.0, 1e-14]], [7, 7, 2]
    )
    np.testing.assert_array_equal(model.predict_proba([[1.0, 0.0]]), [[0.0, 1.0]])


def test_leave_one_out_excludes_self_but_keeps_exact_duplicates():
    model = StableEuclideanKNN(n_neighbors=1).fit([[0.0], [0.0], [2.0]], [7, 2, 2])
    model._distance_block_bytes = 24
    assert model.kneighbors(return_distance=False).tolist() == [[1], [0], [0]]
    assert model.predict(None).tolist() == [2, 7, 7]


@pytest.mark.parametrize("k", [0, -1, 1.5, 3])
def test_invalid_neighbor_count_is_rejected(k):
    model = StableEuclideanKNN(n_neighbors=1).fit([[0.0], [1.0]], [0, 1])
    with pytest.raises(ValueError, match="n_neighbors"):
        model.kneighbors([[0.0]], n_neighbors=k)


def test_evaluator_still_caps_k_at_training_size():
    frame = pd.DataFrame({"x": [1.0], "labels": [7]})
    result = evaluate_real_vs_augmented(frame, frame.iloc[:0], frame, k=3)
    assert result["acc_real"] == result["acc_aug"] == 1.0


def test_non_euclidean_metric_remains_available():
    params = dict(n_neighbors=1, metric="manhattan")
    train, labels, query = [[0.0, 2.0], [1.1, 1.1]], [2, 7], [[0.0, 0.0]]
    stable = StableEuclideanKNN(**params).fit(train, labels)
    reference = KNeighborsClassifier(**params).fit(train, labels)
    np.testing.assert_array_equal(stable.predict(query), reference.predict(query))
    np.testing.assert_array_equal(stable.predict_proba(query), reference.predict_proba(query))


def test_shared_builder_and_serialized_config_identify_stable_evaluator():
    model = _build_classifier("knn", k_eff=1, random_state=42)
    assert isinstance(model, StableEuclideanKNN)
    config = json.loads(classifier_config_json("knn"))
    assert config["euclidean_evaluation"]["version"] == "direct_cdist_v1"
    assert "euclidean_evaluation" not in json.loads(classifier_config_json("svm"))
