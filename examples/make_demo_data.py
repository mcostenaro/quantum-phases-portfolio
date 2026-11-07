"""Independent, small exact-diagonalization demonstration; no external data.

H = sum(Sx_i Sx_j + Sy_i Sy_j + Jz Sz_i Sz_j) + D sum(Sz_i**2).
Four spin-1 sites, periodic boundary; correlations anchored at site zero.
Labels identify sampled parameter regimes, not certified thermodynamic phases.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


FEATURES = [str(i) for i in range(2, 11)]
REGIMES = {0: "ferromagnetic-like", 1: "Haldane-like", 2: "large-D-like"}


def spin_operators():
    sx = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=complex) / np.sqrt(2)
    sy = np.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]], dtype=complex) / np.sqrt(2)
    sz = np.diag([1., 0., -1.]).astype(complex)
    return sx, sy, sz


def operators(length=4):
    result = []
    for spin in spin_operators():
        sites = []
        for site in range(length):
            operator = np.array([[1.]], dtype=complex)
            for position in range(length):
                operator = np.kron(operator, spin if position == site else np.eye(3))
            sites.append(operator)
        result.append(sites)
    return result


def make_data(n_per_regime=12):
    ops = operators()
    xy = sum(ops[a][i] @ ops[a][(i + 1) % 4] for a in (0, 1) for i in range(4))
    zz = sum(ops[2][i] @ ops[2][(i + 1) % 4] for i in range(4))
    anisotropy = sum(operator @ operator for operator in ops[2])
    correlations = [ops[a][0] @ ops[a][distance] for a in range(3) for distance in range(1, 4)]
    rows, parameters = [], []
    for label in REGIMES:
        for t in np.linspace(0, 1, n_per_regime):
            jz, d = ((-2.8 + .5*t, .1 + .2*t), (0.8 + .4*t, -.15 + .3*t),
                     (.8 + .4*t, 3.2 + .8*t))[label]
            h = xy + jz*zz + d*anisotropy
            energies, states = np.linalg.eigh(h)
            # Equal mixture over the full numerical ground eigenspace.
            ground = states[:, np.abs(energies - energies[0]) < 1e-9]
            values = [float(np.trace(ground.conj().T @ c @ ground).real / ground.shape[1])
                      for c in correlations]
            rows.append(dict(zip(FEATURES, values)) | {"labels": label})
            parameters.append({"labels": label, "Jz": jz, "D": d, "degeneracy": ground.shape[1]})
    return pd.DataFrame(rows), pd.DataFrame(parameters)
