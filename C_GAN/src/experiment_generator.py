from __future__ import annotations

import pandas as pd

from C_GAN.src.gan_ydata_like import generate_by_class

GAN_CONFIG_KEYS = ("latent_dim", "hidden_dim", "batch_size", "epochs", "lr", "beta1", "beta2")
# Chaves do protocolo v3, repassadas quando presentes.
GAN_OPTIONAL_CONFIG_KEYS = ("scaling", "output_activation", "d_logits")


def generate(
    df_train: pd.DataFrame,
    *,
    n_synth_per_class,
    config: dict,
    seed=42,
    label_col="labels",
    unit_context: dict | None = None,
) -> pd.DataFrame:
    """Uma GAN por fase; ``unit_context`` ativa as seeds derivadas da v3."""
    return generate_by_class(
        df_train,
        n_synth_per_class=n_synth_per_class,
        gan_config=config,
        seed=seed,
        label_col=label_col,
        unit_context=unit_context,
    )


def _phase_label(label) -> int:
    try:
        return int(float(label))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid phase label: {label!r}") from exc


def _config_for_phase(label, *, base: dict, by_phase: dict[int, dict]) -> dict:
    config = dict(base)
    config.update(by_phase.get(_phase_label(label), {}))
    return config


def _gan_config_only(config: dict) -> dict:
    return {
        **{key: config[key] for key in GAN_CONFIG_KEYS},
        **{key: config[key] for key in GAN_OPTIONAL_CONFIG_KEYS if key in config},
    }


def _n_synth_for_phase(config: dict, *, n_train: int, default_n_synth) -> int:
    rule = config.get("n_synth_per_class", default_n_synth)
    if isinstance(rule, str) and rule == "n_train":
        return int(n_train)
    return int(rule)


def generate_by_phase(
    df_train: pd.DataFrame,
    *,
    n_train: int,
    n_synth_per_class,
    config: dict,
    seed: int,
    label_col: str,
) -> pd.DataFrame:
    """Treina uma GAN por fase e concatena os lotes sinteticos.

    Espelha generate_per_phase do DDPM (uma GAN por grupo de fase,
    class_seed = seed + 2002*class_idx). `config` traz duas chaves:
    "base" (config GAN comum) e "by_phase" ({label: overrides}, podendo
    incluir "n_synth_per_class"; o valor "n_train" usa n_train).
    """
    if df_train.empty:
        raise ValueError("df_train is empty.")
    if label_col not in df_train.columns:
        raise ValueError(f"Missing label column '{label_col}'.")

    base = dict(config.get("base") or {})
    by_phase = {
        _phase_label(label): dict(phase_config)
        for label, phase_config in (config.get("by_phase") or {}).items()
    }

    synth_parts = []
    for class_idx, (label, group) in enumerate(df_train.groupby(label_col)):
        phase_config = _config_for_phase(label, base=base, by_phase=by_phase)
        n_synth = _n_synth_for_phase(
            phase_config,
            n_train=int(n_train),
            default_n_synth=n_synth_per_class,
        )
        class_seed = int(seed + 2002 * class_idx)
        synth_parts.append(
            generate_by_class(
                group.reset_index(drop=True),
                n_synth_per_class=n_synth,
                gan_config=_gan_config_only(phase_config),
                seed=class_seed,
                label_col=label_col,
            )
        )

    return pd.concat(synth_parts, ignore_index=True)
