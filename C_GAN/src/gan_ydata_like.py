from __future__ import annotations

import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset


DEFAULT_GAN_CONFIG = {
    "latent_dim": 64,
    "hidden_dim": 64,
    "batch_size": 2,
    "epochs": 300,
    "lr": 2e-4,
    "beta1": 0.5,
    "beta2": 0.9,
    # Chaves do protocolo v3 (Secao 6.3); os padroes reproduzem a v2.
    "scaling": "minmax_per_phase",
    "output_activation": "sigmoid",
    "d_logits": False,
}
SCALINGS = ("minmax_per_phase", "physical")
OUTPUT_ACTIVATIONS = {"sigmoid": nn.Sigmoid, "tanh": nn.Tanh}
# Bateria fixa de latentes para o movimento da saida (protocolo v3, Secao 6.3).
MOVEMENT_BATTERY_SIZE = 256


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Generator(nn.Module):
    """PyTorch version of the ydata_synthetic VanillaGAN generator."""

    def __init__(
        self,
        latent_dim: int,
        n_features: int,
        hidden_dim: int,
        output_activation: str = "sigmoid",
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, n_features),
            OUTPUT_ACTIVATIONS[output_activation](),
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    """PyTorch version of the ydata_synthetic VanillaGAN discriminator.

    Com ``logits=True`` (v3), a saida e o logit, sem Sigmoid.
    """

    def __init__(self, n_features: int, hidden_dim: int, logits: bool = False):
        super().__init__()
        layers = [
            nn.Linear(n_features, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        ]
        if not logits:
            layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _merge_config(gan_config: dict | None) -> dict:
    config = DEFAULT_GAN_CONFIG.copy()
    if gan_config:
        config.update(gan_config)
    if config["scaling"] not in SCALINGS:
        raise ValueError(f"scaling must be one of {SCALINGS}, got {config['scaling']!r}.")
    if config["output_activation"] not in OUTPUT_ACTIVATIONS:
        raise ValueError(
            "output_activation must be one of "
            f"{tuple(OUTPUT_ACTIVATIONS)}, got {config['output_activation']!r}."
        )
    if config["scaling"] == "physical" and config["output_activation"] != "tanh":
        raise ValueError("Physical scaling requires output_activation='tanh'.")
    if config["scaling"] == "minmax_per_phase" and config["output_activation"] != "sigmoid":
        raise ValueError("Min-Max scaling requires output_activation='sigmoid'.")
    config["d_logits"] = bool(config["d_logits"])
    return config


def _feature_columns(df: pd.DataFrame, label_col: str) -> list[str]:
    feature_cols = [col for col in df.columns if col != label_col]
    if not feature_cols:
        raise ValueError("No feature columns found.")
    return feature_cols


def _window_mean(values: list[float], *, last: bool) -> float:
    """Media dos primeiros ou ultimos 10% dos passos (pelo menos um)."""
    width = max(1, len(values) // 10)
    window = values[-width:] if last else values[:width]
    return float(np.mean(window))


def _fit_gan_for_class(
    X: np.ndarray,
    gan_config: dict,
    seed: int,
    *,
    scaler=None,
):
    """Treina a GAN de uma fase; devolve gerador, escala e metadados.

    Com ``scaling='minmax_per_phase'`` o caminho e o da v2, sem metadados.
    Com ``scaling='physical'`` (v3) a escala fixa vem em ``scaler`` e os
    metadados de treino sao registrados sem consumir o RNG global.
    """
    _seed_all(seed)

    physical = gan_config.get("scaling", "minmax_per_phase") == "physical"
    X = np.asarray(X, dtype=np.float64 if physical else np.float32)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape={X.shape}.")
    if len(X) < 2:
        raise ValueError("A GAN class split needs at least 2 samples.")

    latent_dim = int(gan_config["latent_dim"])
    hidden_dim = int(gan_config["hidden_dim"])
    batch_size = min(int(gan_config["batch_size"]), len(X))
    epochs = int(gan_config["epochs"])
    lr = float(gan_config["lr"])
    betas = (float(gan_config["beta1"]), float(gan_config["beta2"]))
    d_logits = bool(gan_config.get("d_logits", False))

    if batch_size < 2:
        raise ValueError(f"batch_size must be >= 2, got {batch_size}.")
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}.")

    device_name = gan_config.get("device")
    device = (
        torch.device(device_name)
        if device_name
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    if physical:
        if scaler is None:
            raise ValueError("Physical scaling requires a fixed scaler.")
        X_scaled = scaler.transform(X).astype(np.float32)
    else:
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X).astype(np.float32)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_scaled)),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    generator = Generator(
        latent_dim,
        X.shape[1],
        hidden_dim,
        output_activation=gan_config.get("output_activation", "sigmoid"),
    ).to(device)
    discriminator = Discriminator(X.shape[1], hidden_dim, logits=d_logits).to(device)

    opt_g = Adam(generator.parameters(), lr=lr, betas=betas)
    opt_d = Adam(discriminator.parameters(), lr=lr, betas=betas)
    criterion = nn.BCEWithLogitsLoss() if d_logits else nn.BCELoss()

    if physical:
        # Gerador proprio de numeros aleatorios: a bateria nao altera o RNG global.
        battery_rng = torch.Generator(device="cpu").manual_seed(int(seed) + 2)
        battery = torch.randn(
            MOVEMENT_BATTERY_SIZE, latent_dim, generator=battery_rng
        ).to(device)
        with torch.no_grad():
            start_output = generator(battery).cpu().numpy().astype(np.float64)
        d_losses, g_losses, d_accuracies = [], [], []
        threshold = 0.0 if d_logits else 0.5

    for _ in range(epochs):
        for (real_x,) in loader:
            real_x = real_x.to(device)
            current_batch = real_x.size(0)
            real_label = torch.ones(current_batch, 1, device=device)
            fake_label = torch.zeros(current_batch, 1, device=device)

            opt_d.zero_grad()
            real_out = discriminator(real_x)
            real_loss = criterion(real_out, real_label)
            z = torch.randn(current_batch, latent_dim, device=device)
            fake_x = generator(z).detach()
            fake_out = discriminator(fake_x)
            fake_loss = criterion(fake_out, fake_label)
            d_loss = 0.5 * (real_loss + fake_loss)
            d_loss.backward()
            opt_d.step()

            opt_g.zero_grad()
            z = torch.randn(current_batch, latent_dim, device=device)
            fake_x = generator(z)
            g_loss = criterion(discriminator(fake_x), real_label)
            g_loss.backward()
            opt_g.step()

            if physical:
                d_losses.append(float(d_loss.item()))
                g_losses.append(float(g_loss.item()))
                correct = (real_out.detach() > threshold).sum() + (
                    fake_out.detach() <= threshold
                ).sum()
                d_accuracies.append(float(correct.item()) / (2 * current_batch))

    metadata = None
    if physical:
        with torch.no_grad():
            end_output = generator(battery).cpu().numpy().astype(np.float64)
        bounds = np.asarray(scaler.bounds, dtype=np.float64)
        movement = float(np.mean(np.abs(end_output - start_output) * bounds))
        metadata = {
            "steps": int(len(d_losses)),
            "loss_d_start": _window_mean(d_losses, last=False),
            "loss_d_end": _window_mean(d_losses, last=True),
            "loss_g_start": _window_mean(g_losses, last=False),
            "loss_g_end": _window_mean(g_losses, last=True),
            "d_accuracy_end": _window_mean(d_accuracies, last=True),
            "output_movement": movement,
        }
    return generator, scaler, metadata


def _train_gan_for_class(
    X: np.ndarray,
    gan_config: dict,
    seed: int,
):
    """Interface historica (v2): gerador e MinMaxScaler ajustado na fase."""
    generator, scaler, _metadata = _fit_gan_for_class(X, gan_config, seed)
    return generator, scaler


def _generate_samples(
    generator: Generator,
    scaler,
    *,
    n_samples: int,
    latent_dim: int,
    feature_cols: list[str],
    seed: int,
    clip_unit_box: bool = True,
    dtype=np.float32,
) -> pd.DataFrame:
    _seed_all(seed)
    generator.eval()
    device = next(generator.parameters()).device

    z = torch.randn(int(n_samples), int(latent_dim), device=device)
    with torch.no_grad():
        scaled_features = generator(z).cpu().numpy()

    if clip_unit_box:
        scaled_features = np.clip(scaled_features, 0.0, 1.0)
    else:
        scaled_features = scaled_features.astype(np.float64)
    features = scaler.inverse_transform(scaled_features).astype(dtype)
    return pd.DataFrame(features, columns=feature_cols)


def _class_seed(*, seed, class_idx: int, label, unit_context) -> int:
    if unit_context is None:
        return int(seed + 2002 * class_idx)
    from quantum_phases.config import derive_generator_seed

    return derive_generator_seed(
        arm="gan",
        dataset=unit_context["dataset"],
        n_train=unit_context["n_train"],
        seed=unit_context["seed"],
        label=label,
    )


def generate_by_class(
    df_train: pd.DataFrame,
    *,
    n_synth_per_class,
    gan_config,
    seed=42,
    label_col="labels",
    unit_context=None,
) -> pd.DataFrame:
    """Uma GAN por fase.

    Sem ``unit_context``, a fase ``class_idx`` usa ``seed + 2002 * class_idx``
    (v2). Com contexto da unidade, usa a seed derivada do protocolo v3.
    Com ``scaling='physical'``, grava metadados por fase em
    ``attrs['training_metadata']``.
    """
    if df_train.empty:
        raise ValueError("df_train is empty.")
    if label_col not in df_train.columns:
        raise ValueError(f"Missing label column '{label_col}'.")

    n_synth_per_class = int(n_synth_per_class)
    if n_synth_per_class < 1:
        raise ValueError(f"n_synth_per_class must be >= 1, got {n_synth_per_class}.")

    config = _merge_config(gan_config)
    feature_cols = _feature_columns(df_train, label_col)
    physical = config["scaling"] == "physical"
    scaler = None
    if physical:
        from quantum_phases.scaling import PhysicalScaler, physical_bounds

        scaler = PhysicalScaler(physical_bounds(feature_cols))

    synth_parts = []
    training_metadata = []
    for class_idx, (label, group) in enumerate(df_train.groupby(label_col)):
        class_seed = _class_seed(
            seed=seed, class_idx=class_idx, label=label, unit_context=unit_context
        )
        X_class = group[feature_cols].to_numpy(
            dtype=np.float64 if physical else np.float32
        )
        generator, class_scaler, metadata = _fit_gan_for_class(
            X_class, config, class_seed, scaler=scaler
        )
        df_synth = _generate_samples(
            generator,
            class_scaler,
            n_samples=n_synth_per_class,
            latent_dim=int(config["latent_dim"]),
            feature_cols=feature_cols,
            seed=class_seed + 1,
            clip_unit_box=not physical,
            dtype=np.float64 if physical else np.float32,
        )
        df_synth[label_col] = label
        synth_parts.append(df_synth)
        if metadata is not None:
            training_metadata.append({"label": int(label), "seed": int(class_seed), **metadata})

    result = pd.concat(synth_parts, ignore_index=True)[feature_cols + [label_col]]
    if training_metadata:
        result.attrs["training_metadata"] = training_metadata
    return result
