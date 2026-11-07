from __future__ import annotations

import math
import random
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset


DEFAULT_DIFFUSION_CONFIG = {
    "batch_size": 8,
    "epochs": 100,
    "min_train_steps": 0,
    "time_emb_dim": 32,
    "phase_emb_dim": 8,
    "t_max": 100,
    "beta_start": 1e-4,
    "beta_end": 0.02,
    "zero_terminal_snr": True,
    "lr": 1e-4,
    # Chaves do protocolo v3 (Secao 6.4); os padroes reproduzem a v2.
    "scaling": "minmax_per_phase",
    "schedule": "linear",
    "cosine_s": 0.008,
    "parameterization": "eps",
    "sampling_variance": "beta",
    "clip": "unit_box_final",
    "dropout": 0.0,
}

_CONFIG_ALIASES = {
    "batch_size": "batch_size",
    "epochs": "epochs",
    "min_train_steps": "min_train_steps",
    "time_emb_dim": "time_emb_dim",
    "phase_emb_dim": "phase_emb_dim",
    "t_max": "t_max",
    "beta_start": "beta_start",
    "beta_end": "beta_end",
    "zero_terminal_snr": "zero_terminal_snr",
    "lr": "lr",
    "scaling": "scaling",
    "schedule": "schedule",
    "cosine_s": "cosine_s",
    "parameterization": "parameterization",
    "sampling_variance": "sampling_variance",
    "clip": "clip",
    "dropout": "dropout",
    "BATCH_SIZE": "batch_size",
    "EPOCHS": "epochs",
    "MIN_TRAIN_STEPS": "min_train_steps",
    "TIME_EMB_DIM": "time_emb_dim",
    "PHASE_EMB_DIM": "phase_emb_dim",
    "T_MAX": "t_max",
    "BETA_START": "beta_start",
    "BETA_END": "beta_end",
    "ZERO_TERMINAL_SNR": "zero_terminal_snr",
    "LR": "lr",
}

_CHOICES = {
    "scaling": ("minmax_per_phase", "physical"),
    "schedule": ("linear", "cosine"),
    "parameterization": ("eps", "v", "x0"),
    "sampling_variance": ("beta", "posterior"),
    "clip": ("unit_box_final", "denoised_physical"),
}


# -----------------------------------------------------------------------------
# Validation and data helpers
# -----------------------------------------------------------------------------
def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _feature_columns(df: pd.DataFrame, label_col: str) -> list[str]:
    feature_cols = [col for col in df.columns if col != label_col]
    if not feature_cols:
        raise ValueError("No feature columns found.")
    return feature_cols


def _encode_labels(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    classes = np.unique(y)
    class_to_idx = {label: idx for idx, label in enumerate(classes)}
    y_enc = np.vectorize(class_to_idx.get)(y).astype(np.int64)
    return y_enc, classes


def _resolve_samples_per_label(
    classes: np.ndarray,
    n_synth_per_class,
) -> dict[Any, int]:
    if isinstance(n_synth_per_class, Mapping):
        resolved = {}
        for label in classes:
            if label in n_synth_per_class:
                n_samples = n_synth_per_class[label]
            elif str(label) in n_synth_per_class:
                n_samples = n_synth_per_class[str(label)]
            else:
                raise KeyError(f"Missing n_synth_per_class entry for class={label}.")

            n_samples = int(n_samples)
            if n_samples < 1:
                raise ValueError(
                    f"n_synth_per_class values must be >= 1, got {n_samples} "
                    f"for class={label}."
                )
            resolved[label] = n_samples
        return resolved

    n_samples = int(n_synth_per_class)
    if n_samples < 1:
        raise ValueError(f"n_synth_per_class must be >= 1, got {n_samples}.")
    return {label: n_samples for label in classes}


def _as_bool(value: Any, *, key: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    raise ValueError(f"{key} must be a boolean value, got {value!r}.")


def _normalize_config(diffusion_config: Mapping[str, Any] | None) -> dict[str, Any]:
    merged = DEFAULT_DIFFUSION_CONFIG.copy()
    if diffusion_config:
        for key, value in diffusion_config.items():
            if key == "device":
                continue
            target = _CONFIG_ALIASES.get(key)
            if target is None:
                raise ValueError(f"Unknown diffusion_config key: {key}")
            merged[target] = value

    cfg = {
        "batch_size": int(merged["batch_size"]),
        "epochs": int(merged["epochs"]),
        "min_train_steps": int(merged["min_train_steps"]),
        "time_emb_dim": int(merged["time_emb_dim"]),
        "phase_emb_dim": int(merged["phase_emb_dim"]),
        "t_max": int(merged["t_max"]),
        "beta_start": float(merged["beta_start"]),
        "beta_end": float(merged["beta_end"]),
        "zero_terminal_snr": _as_bool(
            merged["zero_terminal_snr"],
            key="zero_terminal_snr",
        ),
        "lr": float(merged["lr"]),
        "scaling": str(merged["scaling"]),
        "schedule": str(merged["schedule"]),
        "cosine_s": float(merged["cosine_s"]),
        "parameterization": str(merged["parameterization"]),
        "sampling_variance": str(merged["sampling_variance"]),
        "clip": str(merged["clip"]),
        "dropout": float(merged["dropout"]),
    }

    if cfg["batch_size"] < 1:
        raise ValueError(f"batch_size must be >= 1, got {cfg['batch_size']}.")
    if cfg["epochs"] < 1:
        raise ValueError(f"epochs must be >= 1, got {cfg['epochs']}.")
    if cfg["min_train_steps"] < 0:
        raise ValueError(
            f"min_train_steps must be >= 0, got {cfg['min_train_steps']}."
        )
    if cfg["time_emb_dim"] < 2 or cfg["time_emb_dim"] % 2:
        raise ValueError("time_emb_dim must be an even integer >= 2.")
    if cfg["phase_emb_dim"] < 1:
        raise ValueError(f"phase_emb_dim must be >= 1, got {cfg['phase_emb_dim']}.")
    if cfg["t_max"] < 2:
        raise ValueError(f"t_max must be >= 2, got {cfg['t_max']}.")
    if not 0 < cfg["beta_start"] < cfg["beta_end"] < 1:
        raise ValueError(
            "Expected 0 < beta_start < beta_end < 1, got "
            f"{cfg['beta_start']} and {cfg['beta_end']}."
        )
    if cfg["lr"] <= 0:
        raise ValueError(f"lr must be > 0, got {cfg['lr']}.")
    for key, choices in _CHOICES.items():
        if cfg[key] not in choices:
            raise ValueError(f"{key} must be one of {choices}, got {cfg[key]!r}.")
    if (cfg["scaling"] == "physical") != (cfg["clip"] == "denoised_physical"):
        raise ValueError(
            "Use scaling='physical' with clip='denoised_physical' and "
            "scaling='minmax_per_phase' with clip='unit_box_final'."
        )
    if cfg["clip"] == "unit_box_final" and (
        cfg["parameterization"] != "eps" or cfg["sampling_variance"] != "beta"
    ):
        raise ValueError("clip='unit_box_final' is the v2 sampler (eps, beta).")
    if cfg["schedule"] == "cosine" and cfg["zero_terminal_snr"]:
        raise ValueError("The cosine schedule is used without zero_terminal_snr.")
    if not cfg["cosine_s"] > 0:
        raise ValueError(f"cosine_s must be > 0, got {cfg['cosine_s']}.")
    if not 0 <= cfg["dropout"] < 1:
        raise ValueError(f"dropout must be in [0, 1), got {cfg['dropout']}.")

    return cfg


def _uses_v2_sampler(cfg: Mapping[str, Any]) -> bool:
    return cfg["clip"] == "unit_box_final"


def _validate_xy(
    X: np.ndarray,
    y: np.ndarray,
    *,
    dtype=np.float32,
) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=dtype)
    y = np.asarray(y)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D. Got shape={X.shape}.")
    if len(X) != len(y):
        raise ValueError(f"X and y must have same length. Got {len(X)} vs {len(y)}.")
    if len(X) == 0:
        raise ValueError("X and y are empty.")
    return X, y


# -----------------------------------------------------------------------------
# Conditional DDPM model
# -----------------------------------------------------------------------------
def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    base = torch.log(torch.tensor(10000.0, device=t.device))
    freqs = torch.exp(torch.arange(half, device=t.device) * -(base / max(half - 1, 1)))
    angles = t[:, None].float() * freqs[None, :]
    return torch.cat((torch.sin(angles), torch.cos(angles)), dim=1)


class PhaseConditionalDenoiser(nn.Module):
    def __init__(
        self,
        x_dim: int,
        n_phases: int,
        time_dim: int,
        phase_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.time_dim = time_dim
        self.phase_embedding = nn.Embedding(n_phases, phase_dim)
        input_dim = x_dim + time_dim + phase_dim

        # Sem dropout, a rede e identica a da v2 (mesma ordem de inicializacao).
        layers: list[nn.Module] = []
        for in_dim, out_dim in ((input_dim, 64), (64, 128), (128, 64)):
            layers.extend([nn.Linear(in_dim, out_dim), nn.SiLU()])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(64, x_dim))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        phase: torch.Tensor,
    ) -> torch.Tensor:
        t_emb = sinusoidal_embedding(t, self.time_dim)
        phase_emb = self.phase_embedding(phase)
        return self.net(torch.cat([x_t, t_emb, phase_emb], dim=1))


# -----------------------------------------------------------------------------
# Diffusion process
# -----------------------------------------------------------------------------
def make_noise_schedule(
    t_max: int,
    *,
    beta_start: float,
    beta_end: float,
    zero_terminal_snr: bool = True,
    schedule: str = "linear",
    cosine_s: float = 0.008,
) -> tuple[torch.Tensor, torch.Tensor]:
    if schedule == "cosine":
        if zero_terminal_snr:
            raise ValueError("The cosine schedule is used without zero_terminal_snr.")
        return cosine_noise_schedule(t_max, s=cosine_s)
    if schedule != "linear":
        raise ValueError(f"Unknown noise schedule {schedule!r}.")
    betas = torch.linspace(beta_start, beta_end, t_max)
    if zero_terminal_snr:
        betas = _rescale_zero_terminal_snr(betas)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alpha_bars


def cosine_noise_schedule(
    t_max: int,
    *,
    s: float = 0.008,
    max_beta: float = 0.999,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Agenda cosseno de Nichol e Dhariwal (2021), protocolo v3 Secao 6.4.

    alpha_bar(t) = f(t)/f(0), f(t) = cos^2(((t/T) + s)/(1 + s) * pi/2);
    beta_t = min(1 - alpha_bar_t/alpha_bar_{t-1}, max_beta); alpha_bar e
    recalculado a partir dos beta cortados.
    """
    steps = torch.arange(int(t_max) + 1, dtype=torch.float64)
    f = torch.cos(((steps / int(t_max)) + float(s)) / (1.0 + float(s)) * math.pi / 2) ** 2
    alpha_bar = f / f[0]
    betas = torch.clamp(1.0 - alpha_bar[1:] / alpha_bar[:-1], min=0.0, max=max_beta)
    alpha_bars = torch.cumprod(1.0 - betas, dim=0)
    return betas.float(), alpha_bars.float()


def _rescale_zero_terminal_snr(
    betas: torch.Tensor,
    *,
    max_beta: float = 0.999,
) -> torch.Tensor:
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    alpha_bars_sqrt = torch.sqrt(alpha_bars)

    first = alpha_bars_sqrt[0].clone()
    last = alpha_bars_sqrt[-1].clone()
    if torch.isclose(first, last).item():
        raise ValueError("Cannot rescale diffusion schedule with degenerate alpha bars.")

    alpha_bars_sqrt = (alpha_bars_sqrt - last) * first / (first - last)
    alpha_bars = alpha_bars_sqrt.square()
    prev_alpha_bars = torch.cat(
        [
            torch.ones(1, dtype=alpha_bars.dtype, device=alpha_bars.device),
            alpha_bars[:-1],
        ]
    )
    alphas = alpha_bars / torch.clamp(prev_alpha_bars, min=1e-12)
    return torch.clamp(1.0 - alphas, min=1e-8, max=max_beta)


def add_noise(
    x_0: torch.Tensor,
    t: torch.Tensor,
    alpha_bars: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    alpha_bar_t = alpha_bars[t].unsqueeze(1)
    noise = torch.randn_like(x_0)
    x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1.0 - alpha_bar_t) * noise
    return x_t, noise


def v_target(
    x_0: torch.Tensor,
    noise: torch.Tensor,
    alpha_bar_t: torch.Tensor,
) -> torch.Tensor:
    """Predicao v (Salimans e Ho, 2022): v = sqrt(ab)*eps - sqrt(1 - ab)*x0."""
    return torch.sqrt(alpha_bar_t) * noise - torch.sqrt(1.0 - alpha_bar_t) * x_0


def x0_from_v(
    x_t: torch.Tensor,
    v: torch.Tensor,
    alpha_bar_t: torch.Tensor,
) -> torch.Tensor:
    """x0 = sqrt(ab)*x_t - sqrt(1 - ab)*v."""
    return torch.sqrt(alpha_bar_t) * x_t - torch.sqrt(1.0 - alpha_bar_t) * v


def eps_from_v(
    x_t: torch.Tensor,
    v: torch.Tensor,
    alpha_bar_t: torch.Tensor,
) -> torch.Tensor:
    """eps = sqrt(1 - ab)*x_t + sqrt(ab)*v."""
    return torch.sqrt(1.0 - alpha_bar_t) * x_t + torch.sqrt(alpha_bar_t) * v


def denoise_step(
    model: PhaseConditionalDenoiser,
    x_t: torch.Tensor,
    t: torch.Tensor,
    phase: torch.Tensor,
    betas: torch.Tensor,
    alpha_bars: torch.Tensor,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    eps_pred = model(x_t, t, phase)
    beta_t = betas[t].unsqueeze(1)
    alpha_t = 1.0 - beta_t
    alpha_bar_t = alpha_bars[t].unsqueeze(1)
    z = torch.randn_like(x_t) if noise is None else noise

    denom = torch.sqrt(torch.clamp(1.0 - alpha_bar_t, min=1e-8))
    mean = (x_t - (beta_t / denom) * eps_pred) / torch.sqrt(
        torch.clamp(alpha_t, min=1e-8)
    )
    return mean + torch.sqrt(torch.clamp(beta_t, min=1e-8)) * z


# -----------------------------------------------------------------------------
# Training and sampling
# -----------------------------------------------------------------------------
def train_one_epoch(
    model: PhaseConditionalDenoiser,
    dataloader: DataLoader,
    betas: torch.Tensor,
    alpha_bars: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    parameterization: str = "eps",
) -> float:
    model.train()
    mse = nn.MSELoss()
    total_loss = 0.0
    total_items = 0
    t_max = betas.shape[0]

    for x_0, phase in dataloader:
        x_0 = x_0.to(device)
        phase = phase.to(device)
        t = torch.randint(0, t_max, (x_0.shape[0],), device=device, dtype=torch.long)

        x_t, noise = add_noise(x_0, t, alpha_bars)
        if parameterization == "v":
            target = v_target(x_0, noise, alpha_bars[t].unsqueeze(1))
        elif parameterization == "x0":
            target = x_0
        else:
            target = noise
        loss = mse(model(x_t, t, phase), target)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite diffusion loss encountered during training.")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x_0.shape[0]
        total_items += x_0.shape[0]

    return total_loss / max(total_items, 1)


def sample_phase(
    model: PhaseConditionalDenoiser,
    *,
    n_samples: int,
    phase_idx: int,
    x_dim: int,
    betas: torch.Tensor,
    alpha_bars: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    x_t = torch.randn(int(n_samples), x_dim, device=device)
    phase = torch.full((int(n_samples),), phase_idx, device=device, dtype=torch.long)

    with torch.no_grad():
        for step in reversed(range(betas.shape[0])):
            t = torch.full((int(n_samples),), step, device=device, dtype=torch.long)
            noise = None if step > 0 else torch.zeros_like(x_t)
            x_t = denoise_step(model, x_t, t, phase, betas, alpha_bars, noise=noise)

    return x_t


def sample_phase_posterior(
    model: PhaseConditionalDenoiser,
    *,
    n_samples: int,
    phase_idx: int,
    x_dim: int,
    betas: torch.Tensor,
    alpha_bars: torch.Tensor,
    device: torch.device,
    parameterization: str = "v",
    sampling_variance: str = "posterior",
) -> torch.Tensor:
    """Amostragem do protocolo v3 (Secao 6.4) na escala dividida pelos limites.

    Em cada passo, x0_hat e estimado e cortado em [-1, 1]; x_{t-1} usa a media
    do posterior de Ho et al. (2020). No ultimo passo devolve x0_hat sem ruido.
    """
    model.eval()
    n_samples = int(n_samples)
    x_t = torch.randn(n_samples, x_dim, device=device)
    phase = torch.full((n_samples,), phase_idx, device=device, dtype=torch.long)
    one = torch.ones((), dtype=alpha_bars.dtype, device=alpha_bars.device)

    with torch.no_grad():
        for step in reversed(range(betas.shape[0])):
            t = torch.full((n_samples,), step, device=device, dtype=torch.long)
            output = model(x_t, t, phase)
            alpha_bar_t = alpha_bars[step]
            alpha_bar_prev = alpha_bars[step - 1] if step > 0 else one
            beta_t = betas[step]
            if parameterization == "v":
                x0_hat = x0_from_v(x_t, output, alpha_bar_t)
            elif parameterization == "x0":
                x0_hat = output
            else:
                x0_hat = (x_t - torch.sqrt(1.0 - alpha_bar_t) * output) / torch.sqrt(
                    alpha_bar_t
                )
            x0_hat = torch.clamp(x0_hat, -1.0, 1.0)
            if step == 0:
                return x0_hat
            coef_x0 = torch.sqrt(alpha_bar_prev) * beta_t / (1.0 - alpha_bar_t)
            coef_xt = torch.sqrt(1.0 - beta_t) * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
            mean = coef_x0 * x0_hat + coef_xt * x_t
            if sampling_variance == "posterior":
                variance = beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
            else:
                variance = beta_t
            x_t = mean + torch.sqrt(variance) * torch.randn_like(x_t)

    return x_t


def train_and_sample(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_synth_per_class,
    diffusion_config: Mapping[str, Any] | None = None,
    seed: int = 42,
    device: torch.device | str | None = None,
    bounds=None,
) -> dict[str, Any]:
    cfg = _normalize_config(diffusion_config)
    physical = cfg["scaling"] == "physical"
    X, y = _validate_xy(X, y, dtype=np.float64 if physical else np.float32)
    _seed_all(int(seed))

    device = torch.device(device) if device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    y_enc, classes = _encode_labels(y)
    samples_by_label = _resolve_samples_per_label(classes, n_synth_per_class)

    if physical:
        if bounds is None:
            raise ValueError("Physical scaling requires the feature bounds.")
        from quantum_phases.scaling import PhysicalScaler

        scaler = PhysicalScaler(bounds)
        X_scaled = scaler.transform(X).astype(np.float32)
    else:
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X).astype(np.float32)
    dataset = TensorDataset(
        torch.tensor(X_scaled, dtype=torch.float32),
        torch.tensor(y_enc, dtype=torch.long),
    )
    batch_size = max(1, min(cfg["batch_size"], len(dataset)))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    steps_per_epoch = max(1, len(dataloader))
    min_epochs_from_steps = int(np.ceil(cfg["min_train_steps"] / steps_per_epoch))
    effective_epochs = max(cfg["epochs"], min_epochs_from_steps)

    betas, alpha_bars = make_noise_schedule(
        cfg["t_max"],
        beta_start=cfg["beta_start"],
        beta_end=cfg["beta_end"],
        zero_terminal_snr=cfg["zero_terminal_snr"],
        schedule=cfg["schedule"],
        cosine_s=cfg["cosine_s"],
    )
    betas = betas.to(device)
    alpha_bars = alpha_bars.to(device)

    model = PhaseConditionalDenoiser(
        x_dim=X.shape[1],
        n_phases=len(classes),
        time_dim=cfg["time_emb_dim"],
        phase_dim=cfg["phase_emb_dim"],
        dropout=cfg["dropout"],
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    loss_history = []
    for _ in range(effective_epochs):
        loss = train_one_epoch(
            model,
            dataloader,
            betas,
            alpha_bars,
            optimizer,
            device,
            parameterization=cfg["parameterization"],
        )
        loss_history.append(loss)

    X_syn_parts = []
    y_syn_parts = []
    for phase_idx, label in enumerate(classes):
        n_samples = int(samples_by_label[label])
        if _uses_v2_sampler(cfg):
            synth_scaled = sample_phase(
                model,
                n_samples=n_samples,
                phase_idx=phase_idx,
                x_dim=X.shape[1],
                betas=betas,
                alpha_bars=alpha_bars,
                device=device,
            )
            synth_scaled_np = np.clip(synth_scaled.cpu().numpy(), 0.0, 1.0)
            synth = scaler.inverse_transform(synth_scaled_np).astype(np.float32)
        else:
            synth_scaled = sample_phase_posterior(
                model,
                n_samples=n_samples,
                phase_idx=phase_idx,
                x_dim=X.shape[1],
                betas=betas,
                alpha_bars=alpha_bars,
                device=device,
                parameterization=cfg["parameterization"],
                sampling_variance=cfg["sampling_variance"],
            )
            synth = scaler.inverse_transform(
                synth_scaled.cpu().numpy().astype(np.float64)
            )
        X_syn_parts.append(synth)
        y_syn_parts.append(np.full(n_samples, label, dtype=classes.dtype))

    X_syn = np.vstack(X_syn_parts)
    y_syn = np.concatenate(y_syn_parts)

    return {
        "model": model,
        "scaler": scaler,
        "classes": classes,
        "config": cfg,
        "loss_history": loss_history,
        "X_syn": X_syn,
        "y_syn": y_syn,
        "effective_epochs": effective_epochs,
        "steps_per_epoch": steps_per_epoch,
        "train_steps_total": int(effective_epochs * steps_per_epoch),
    }


def _window_mean(values, *, last: bool) -> float:
    """Media dos primeiros ou ultimos 10% das epocas (pelo menos uma)."""
    values = list(values)
    width = max(1, len(values) // 10)
    window = values[-width:] if last else values[:width]
    return float(np.mean(window))


# -----------------------------------------------------------------------------
# Public experiment API
# -----------------------------------------------------------------------------
def generate_by_class(
    df_train: pd.DataFrame,
    *,
    n_synth_per_class,
    diffusion_config: Mapping[str, Any] | None = None,
    seed=42,
    label_col="labels",
) -> pd.DataFrame:
    """
    Train one phase-conditional DDPM and return synthetic rows by class.

    With ``scaling='physical'`` (protocol v3) the fixed physical bounds of the
    feature columns replace the Min-Max scaler, and the training metadata adds
    the phase label and the mean loss of the first and last 10% of the epochs
    (one epoch per step with a full batch).
    """
    if df_train.empty:
        raise ValueError("df_train is empty.")
    if label_col not in df_train.columns:
        raise ValueError(f"Missing label column '{label_col}'.")

    feature_cols = _feature_columns(df_train, label_col)
    physical = _normalize_config(diffusion_config)["scaling"] == "physical"
    extra = {}
    if physical:
        from quantum_phases.scaling import physical_bounds

        extra["bounds"] = physical_bounds(feature_cols)
    X_train = df_train[feature_cols].to_numpy(
        dtype=np.float64 if physical else np.float32
    )
    y_train = df_train[label_col].to_numpy()

    device_name = None if diffusion_config is None else diffusion_config.get("device")
    out = train_and_sample(
        X_train,
        y_train,
        n_synth_per_class=n_synth_per_class,
        diffusion_config=diffusion_config,
        seed=int(seed),
        device=device_name,
        **extra,
    )

    df_synth = pd.DataFrame(out["X_syn"], columns=feature_cols)
    df_synth[label_col] = out["y_syn"]
    df_synth = df_synth[feature_cols + [label_col]].reset_index(drop=True)
    record = {
        "labels": out["classes"].tolist(),
        "seed": int(seed),
        "effective_epochs": int(out["effective_epochs"]),
        "steps_per_epoch": int(out["steps_per_epoch"]),
        "train_steps_total": int(out["train_steps_total"]),
    }
    if physical:
        classes = out["classes"].tolist()
        if len(classes) == 1:
            record["label"] = int(classes[0])
        record.update({
            "steps": int(out["train_steps_total"]),
            "loss_start": _window_mean(out["loss_history"], last=False),
            "loss_end": _window_mean(out["loss_history"], last=True),
        })
    df_synth.attrs["training_metadata"] = [record]
    return df_synth
