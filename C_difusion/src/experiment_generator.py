from __future__ import annotations

import pandas as pd

from C_difusion.src.difusion_core import generate_by_class


def _resolve_n_synth(rule, *, n_train: int) -> int:
    if isinstance(rule, str) and rule == "n_train":
        return int(n_train)
    return int(rule)


def _resolve_config(*, config: dict | None, diffusion_config: dict | None) -> dict:
    if config is None:
        if diffusion_config is None:
            raise ValueError("Missing diffusion config.")
        return dict(diffusion_config)

    if diffusion_config is not None and dict(diffusion_config) != dict(config):
        raise ValueError("Use either config or diffusion_config, not conflicting values.")

    return dict(config)


def generate(
    df_train: pd.DataFrame,
    *,
    n_synth_per_class,
    config: dict,
    seed=42,
    label_col="labels",
) -> pd.DataFrame:
    return generate_by_class(
        df_train,
        n_synth_per_class=n_synth_per_class,
        diffusion_config=config,
        seed=seed,
        label_col=label_col,
    )


def generate_per_phase(
    df_train: pd.DataFrame,
    *,
    n_train: int,
    n_synth_per_class,
    config: dict | None = None,
    diffusion_config: dict | None = None,
    seed: int,
    label_col: str,
    unit_context: dict | None = None,
) -> pd.DataFrame:
    """
    Train one diffusion model per phase and return synthetic rows by phase.

    This mirrors the active GAN strategy: each class split is modeled
    independently, then all synthetic class batches are concatenated.
    Without ``unit_context`` the phase seed is ``seed + 2002 * class_idx``
    (protocol v2); with it, the derived seed of protocol v3 (Section 6.6).
    """
    if df_train.empty:
        raise ValueError("df_train is empty.")
    if label_col not in df_train.columns:
        raise ValueError(f"Missing label column '{label_col}'.")

    resolved_config = _resolve_config(config=config, diffusion_config=diffusion_config)

    synth_parts = []
    training_metadata = []
    for class_idx, (label, group) in enumerate(df_train.groupby(label_col)):
        n_synth = _resolve_n_synth(n_synth_per_class, n_train=int(n_train))
        if unit_context is None:
            class_seed = int(seed + 2002 * class_idx)
        else:
            from quantum_phases.config import derive_generator_seed

            class_seed = derive_generator_seed(
                arm="ddpm",
                dataset=unit_context["dataset"],
                n_train=unit_context["n_train"],
                seed=unit_context["seed"],
                label=label,
            )
        df_synth = generate_by_class(
            group.reset_index(drop=True),
            n_synth_per_class=n_synth,
            diffusion_config=resolved_config,
            seed=class_seed,
            label_col=label_col,
        )
        synth_parts.append(df_synth)
        training_metadata.extend(df_synth.attrs.get("training_metadata", []))

    if not synth_parts:
        raise ValueError("No phase groups were found in df_train.")

    result = pd.concat(synth_parts, ignore_index=True)
    if training_metadata:
        result.attrs["training_metadata"] = training_metadata
    return result
