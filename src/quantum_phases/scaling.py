"""Escala fisica fixa do protocolo v3 (Secao 6.1).

Os geradores v3 trabalham em x = C / limite, com limites teoricos dos
correlacionadores e nenhum ajuste aos dados: ED usa |C| <= 1 em todas as
colunas; as representacoes DMRG usam |Tr[rho (lambda_a x lambda_a)]| <=
||lambda_a||^2, isto e, 1 para a = 1..7 e 4/3 para a = 8.
"""
from __future__ import annotations

import re

import numpy as np


ED_BOUND = 1.0
GELL_MANN_BOUNDS = (1.0,) * 7 + (4.0 / 3.0,)
_GELL_MANN_COLUMN = re.compile(r"^gell_mann_([1-8])$")


def physical_bounds(feature_cols) -> np.ndarray:
    """Limite teorico de cada coluna, na ordem recebida."""
    feature_cols = [str(column) for column in feature_cols]
    if not feature_cols:
        raise ValueError("No feature columns to scale.")
    bounds = []
    for column in feature_cols:
        match = _GELL_MANN_COLUMN.match(column)
        if match:
            bounds.append(GELL_MANN_BOUNDS[int(match.group(1)) - 1])
        elif column.isdigit():
            bounds.append(ED_BOUND)
        else:
            raise ValueError(
                f"No physical bound defined for feature column {column!r}."
            )
    kinds = {"dmrg" if _GELL_MANN_COLUMN.match(c) else "ed" for c in feature_cols}
    if len(kinds) != 1:
        raise ValueError("Mixed ED and Gell-Mann feature columns.")
    return np.asarray(bounds, dtype=np.float64)


class PhysicalScaler:
    """Escala fixa x = C / limite, com a interface usada pelos geradores."""

    def __init__(self, bounds):
        self.bounds = np.asarray(bounds, dtype=np.float64)
        if self.bounds.ndim != 1 or not np.all(self.bounds > 0):
            raise ValueError("Physical bounds must be a positive 1D array.")

    def transform(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != len(self.bounds):
            raise ValueError(
                f"Expected {len(self.bounds)} columns, got shape={X.shape}."
            )
        return X / self.bounds

    def inverse_transform(self, U) -> np.ndarray:
        U = np.asarray(U, dtype=np.float64)
        if U.ndim != 2 or U.shape[1] != len(self.bounds):
            raise ValueError(
                f"Expected {len(self.bounds)} columns, got shape={U.shape}."
            )
        return U * self.bounds

    def clip(self, X) -> np.ndarray:
        """Corta valores na escala fisica aos limites [-limite, limite]."""
        X = np.asarray(X, dtype=np.float64)
        return np.clip(X, -self.bounds, self.bounds)
