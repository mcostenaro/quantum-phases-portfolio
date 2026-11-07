"""Controles nao aprendidos do protocolo v3 (Secao 6.2).

Cada controle recebe so o treino real da unidade, separado por fase, e produz
``n_synth_per_class`` amostras por fase com o rotulo de origem. A assinatura e
a mesma dos geradores (``generator_ref`` do runner). Este modulo nao importa
C_GAN nem C_difusion.

Seeds: com ``unit_context`` (dataset, n_train, seed), cada fase usa
``derive_generator_seed`` (protocolo v3, Secao 6.6). Sem contexto, para uso
direto em testes, a fase ``class_idx`` usa ``default_rng([seed, class_idx])``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import derive_generator_seed
from .scaling import PhysicalScaler, physical_bounds


CONTROL_METHODS = ("copias", "centro_caixa", "interpolacao", "jitter")
RANDOM_METHODS = frozenset({"interpolacao", "jitter"})


def _feature_columns(df: pd.DataFrame, label_col: str) -> list[str]:
    feature_cols = [column for column in df.columns if column != label_col]
    if not feature_cols:
        raise ValueError("No feature columns found.")
    return feature_cols


def _validate(df_train: pd.DataFrame, label_col: str, n_synth_per_class) -> int:
    if df_train.empty:
        raise ValueError("df_train is empty.")
    if label_col not in df_train.columns:
        raise ValueError(f"Missing label column '{label_col}'.")
    n_synth = int(n_synth_per_class)
    if n_synth < 1:
        raise ValueError(f"n_synth_per_class must be >= 1, got {n_synth}.")
    return n_synth


def _phase_seed(method: str, *, label, class_idx: int, seed, unit_context) -> int | None:
    if method not in RANDOM_METHODS:
        return None
    if unit_context is None:
        return None
    context_arm = unit_context.get("arm")
    if context_arm is not None and context_arm != method:
        raise ValueError(
            f"unit_context arm {context_arm!r} differs from control {method!r}."
        )
    return derive_generator_seed(
        arm=method,
        dataset=unit_context["dataset"],
        n_train=unit_context["n_train"],
        seed=unit_context["seed"],
        label=label,
    )


def _rng(gen_seed: int | None, *, seed, class_idx: int) -> np.random.Generator:
    if gen_seed is not None:
        return np.random.default_rng(int(gen_seed))
    return np.random.default_rng([int(seed), int(class_idx)])


def _copies(X: np.ndarray, n_synth: int, rng) -> np.ndarray:
    return X[np.arange(n_synth) % len(X)].copy()


def _box_center(X: np.ndarray, n_synth: int, rng) -> np.ndarray:
    center = 0.5 * (X.min(axis=0) + X.max(axis=0))
    return np.repeat(center[None, :], n_synth, axis=0)


def _interpolation(X: np.ndarray, n_synth: int, rng) -> np.ndarray:
    if len(X) < 2:
        raise ValueError("Interpolation needs at least two real samples per phase.")
    pairs = np.array(
        [rng.choice(len(X), size=2, replace=False) for _ in range(n_synth)]
    )
    lam = rng.uniform(0.0, 1.0, size=n_synth)[:, None]
    return lam * X[pairs[:, 0]] + (1.0 - lam) * X[pairs[:, 1]]


def _jitter(X: np.ndarray, n_synth: int, rng, *, sigma: float, scaler) -> np.ndarray:
    anchors = rng.integers(0, len(X), size=n_synth)
    noise = rng.standard_normal((n_synth, X.shape[1]))
    return scaler.clip(X[anchors] + float(sigma) * noise)


def _sigma(config) -> float:
    if config is None or "sigma" not in config:
        raise ValueError("Jitter requires config['sigma'].")
    sigma = float(config["sigma"])
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError(f"Jitter sigma must be > 0, got {sigma!r}.")
    return sigma


def _generate(
    method: str,
    df_train: pd.DataFrame,
    *,
    n_synth_per_class,
    config,
    seed,
    label_col,
    unit_context,
) -> pd.DataFrame:
    n_synth = _validate(df_train, label_col, n_synth_per_class)
    feature_cols = _feature_columns(df_train, label_col)
    scaler = PhysicalScaler(physical_bounds(feature_cols))
    sigma = _sigma(config) if method == "jitter" else None

    parts = []
    metadata = []
    for class_idx, (label, group) in enumerate(df_train.groupby(label_col, sort=True)):
        X = group[feature_cols].to_numpy(dtype=np.float64)
        gen_seed = _phase_seed(
            method, label=label, class_idx=class_idx, seed=seed, unit_context=unit_context
        )
        rng = _rng(gen_seed, seed=seed, class_idx=class_idx)
        if method == "copias":
            values = _copies(X, n_synth, rng)
        elif method == "centro_caixa":
            values = _box_center(X, n_synth, rng)
        elif method == "interpolacao":
            values = _interpolation(X, n_synth, rng)
        elif method == "jitter":
            values = _jitter(X, n_synth, rng, sigma=sigma, scaler=scaler)
        else:
            raise ValueError(f"Unknown control method {method!r}.")
        frame = pd.DataFrame(values, columns=feature_cols)
        frame[label_col] = label
        parts.append(frame)
        record = {
            "label": int(label),
            "method": method,
            "gen_seed": gen_seed,
            "n_real": int(len(X)),
        }
        if sigma is not None:
            record["sigma"] = sigma
        metadata.append(record)

    result = pd.concat(parts, ignore_index=True)[feature_cols + [label_col]]
    result.attrs["training_metadata"] = metadata
    return result


def generate_copies(df_train, *, n_synth_per_class, config=None, seed=0,
                    label_col="labels", unit_context=None) -> pd.DataFrame:
    """Rodizio exato: o sintetico j e o real j mod n_train da fase."""
    return _generate("copias", df_train, n_synth_per_class=n_synth_per_class,
                     config=config, seed=seed, label_col=label_col,
                     unit_context=unit_context)


def generate_box_center(df_train, *, n_synth_per_class, config=None, seed=0,
                        label_col="labels", unit_context=None) -> pd.DataFrame:
    """Ponto (min + max)/2 de cada coordenada, repetido."""
    return _generate("centro_caixa", df_train, n_synth_per_class=n_synth_per_class,
                     config=config, seed=seed, label_col=label_col,
                     unit_context=unit_context)


def generate_interpolation(df_train, *, n_synth_per_class, config=None, seed=0,
                           label_col="labels", unit_context=None) -> pd.DataFrame:
    """x = lambda*a + (1 - lambda)*b, com a != b reais da fase e lambda ~ U(0, 1)."""
    return _generate("interpolacao", df_train, n_synth_per_class=n_synth_per_class,
                     config=config, seed=seed, label_col=label_col,
                     unit_context=unit_context)


def generate_jitter(df_train, *, n_synth_per_class, config, seed=0,
                    label_col="labels", unit_context=None) -> pd.DataFrame:
    """x = a + sigma*eps, com corte nos limites fisicos."""
    return _generate("jitter", df_train, n_synth_per_class=n_synth_per_class,
                     config=config, seed=seed, label_col=label_col,
                     unit_context=unit_context)


def generate(df_train, *, n_synth_per_class, config, seed=0, label_col="labels",
             unit_context=None) -> pd.DataFrame:
    """Entrada unica do runner: ``config['method']`` escolhe o controle."""
    method = None if config is None else config.get("method")
    if method not in CONTROL_METHODS:
        raise ValueError(f"config['method'] must be one of {CONTROL_METHODS}, got {method!r}.")
    return _generate(method, df_train, n_synth_per_class=n_synth_per_class,
                     config=config, seed=seed, label_col=label_col,
                     unit_context=unit_context)
