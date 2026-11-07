"""Euclidean k-NN without cancellation between nearly coincident samples."""

from numbers import Integral

import numpy as np
from scipy.spatial.distance import cdist
from sklearn import config_context
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.validation import check_is_fitted, validate_data


class StableEuclideanKNN(KNeighborsClassifier):
    """Use coordinate differences and training order for exact distance ties.

    Voting (including class ties and distance weights) stays with sklearn.
    Other metrics retain sklearn's implementation. Queries are batched so the
    distance matrix does not grow with the full test set.
    """

    _distance_block_bytes = 8 * 1024 * 1024

    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        check_is_fitted(self)
        if self.effective_metric_ != "euclidean":
            return super().kneighbors(X, n_neighbors, return_distance)

        k = self.n_neighbors if n_neighbors is None else n_neighbors
        if not isinstance(k, Integral) or k < 1:
            raise ValueError("n_neighbors must be a positive integer.")
        query_is_train = X is None
        available = self.n_samples_fit_ - int(query_is_train)
        if k > available:
            raise ValueError(f"n_neighbors={k} exceeds available samples={available}.")
        queries = self._fit_X if query_is_train else validate_data(
            self, X, reset=False, dtype=np.float64, order="C"
        )
        n_queries = len(queries)
        indices = np.empty((n_queries, k), dtype=np.intp)
        distances = np.empty((n_queries, k)) if return_distance else None
        block_rows = max(1, self._distance_block_bytes // (8 * self.n_samples_fit_))
        for start in range(0, n_queries, block_rows):
            stop = min(start + block_rows, n_queries)
            # cdist sums squared coordinate differences; no norm/dot-product
            # identity and no tolerance that would merge distinct distances.
            block = cdist(queries[start:stop], self._fit_X, metric="euclidean")
            if query_is_train:
                block[np.arange(stop - start), np.arange(start, stop)] = np.inf
            if k == 1:
                selected = block.argmin(axis=1)[:, None]
            else:
                selected = np.argsort(block, axis=1, kind="stable")[:, :k]
            indices[start:stop] = selected
            if return_distance:
                distances[start:stop] = np.take_along_axis(block, selected, axis=1)
        return (distances, indices) if return_distance else indices

    def predict(self, X):
        check_is_fitted(self)
        if self.effective_metric_ != "euclidean":
            return super().predict(X)
        # sklearn's fused brute-force vote can bypass kneighbors entirely.
        # Disable that shortcut locally, retaining its validated voting rules.
        with config_context(enable_cython_pairwise_dist=False):
            return super().predict(X)

    def predict_proba(self, X):
        check_is_fitted(self)
        if self.effective_metric_ != "euclidean":
            return super().predict_proba(X)
        with config_context(enable_cython_pairwise_dist=False):
            return super().predict_proba(X)
