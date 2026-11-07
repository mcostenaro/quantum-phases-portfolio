from __future__ import annotations

import hashlib

from .data import DATASETS


LABEL_COL = "labels"
DROP_COLS = ("0", "1")
EXPLORATORY_SEED_NAMESPACE = "quantum_phases_experiments_v1"
MAIN_SEED_NAMESPACE = "quantum_phases_main_v2"
# Protocolo v3, Secao 6.5: seeds exclusivas do ajuste leave-one-dataset-out.
TUNING_SEED_NAMESPACE = "quantum_phases_tuning_v1"
CUSTOM_SEED_NAMESPACE = "custom"
# Alias retrocompativel: scripts auxiliares que apenas chamam generate_seeds()
# continuam fora do conjunto reservado para a rodada principal.
DEFAULT_SEED_NAMESPACE = EXPLORATORY_SEED_NAMESPACE
MAX_SEED = 2_147_483_647
N_SIM = 100
N_TUNING_SEEDS = 30


def generate_seeds(
    n_sim: int,
    *,
    namespace: str = DEFAULT_SEED_NAMESPACE,
) -> list[int]:
    n_sim = int(n_sim)
    if n_sim < 1:
        raise ValueError(f"n_sim must be >= 1, got {n_sim}.")

    seeds = []
    seen = set()
    index = 0
    while len(seeds) < n_sim:
        payload = f"{namespace}:{index}".encode("utf-8")
        digest = hashlib.blake2b(payload, digest_size=8).digest()
        seed = int.from_bytes(digest, "big") % MAX_SEED + 1
        if seed not in seen:
            seeds.append(seed)
            seen.add(seed)
        index += 1
    return seeds


def identify_seed_namespace(seeds) -> str:
    """Identify a canonical seed prefix, otherwise mark it as custom."""
    normalized = [int(seed) for seed in seeds]
    if not normalized:
        return CUSTOM_SEED_NAMESPACE
    for namespace in (
        MAIN_SEED_NAMESPACE,
        EXPLORATORY_SEED_NAMESPACE,
        TUNING_SEED_NAMESPACE,
    ):
        if normalized == generate_seeds(len(normalized), namespace=namespace):
            return namespace
    return CUSTOM_SEED_NAMESPACE


def _label_text(label) -> str:
    """Rotulo inteiro da fase em texto; outros rotulos sao recusados."""
    try:
        value = float(label)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Phase label must be an integer, got {label!r}.") from exc
    if not value.is_integer():
        raise ValueError(f"Phase label must be an integer, got {label!r}.")
    return str(int(value))


def derive_generator_seed(*, arm: str, dataset: str, n_train, seed, label) -> int:
    """Seed por fase do protocolo v3 (Secao 6.6), desacoplada entre bracos.

    A seed nao depende do candidato do ajuste: todos os candidatos de um braco
    recebem as mesmas seeds por unidade e fase.
    """
    payload = (
        f"v3|{arm}|{dataset}|{int(n_train)}|{int(seed)}|{_label_text(label)}"
    ).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big") % MAX_SEED + 1


V3_SEED_DERIVATION_RULE = (
    "blake2b('v3|{arm}|{dataset}|{n_train}|{seed}|{label}', digest_size=8) "
    "big-endian % 2147483647 + 1"
)


EXPLORATORY_SEEDS = generate_seeds(
    N_SIM,
    namespace=EXPLORATORY_SEED_NAMESPACE,
)
MAIN_SEEDS = generate_seeds(
    N_SIM,
    namespace=MAIN_SEED_NAMESPACE,
)
TUNING_SEEDS = generate_seeds(
    N_TUNING_SEEDS,
    namespace=TUNING_SEED_NAMESPACE,
)
if not set(EXPLORATORY_SEEDS).isdisjoint(MAIN_SEEDS):
    raise RuntimeError("Exploratory and main seed sets must be disjoint.")
if not set(TUNING_SEEDS).isdisjoint({*MAIN_SEEDS, *EXPLORATORY_SEEDS}):
    raise RuntimeError(
        "Tuning seeds must be disjoint from main and exploratory seeds."
    )

# Nomes curtos aceitos pelas CLIs (--seed-namespace).
SEED_NAMESPACES = {
    "main_v2": MAIN_SEED_NAMESPACE,
    "exploratory_v1": EXPLORATORY_SEED_NAMESPACE,
    "tuning_v1": TUNING_SEED_NAMESPACE,
}
SEEDS_BY_NAMESPACE = {
    "main_v2": MAIN_SEEDS,
    "exploratory_v1": EXPLORATORY_SEEDS,
    "tuning_v1": TUNING_SEEDS,
}

# Alias retrocompativel para diagnosticos e scripts historicamente exploratorios.
SEEDS = EXPLORATORY_SEEDS
N_TRAIN_VALUES = [2, 3, 4, 5, 6]
N_SYNTH_PER_CLASS = 10
K = 1
TORCH_NUM_THREADS = 1

__all__ = [
    "DATASETS",
    "LABEL_COL",
    "DROP_COLS",
    "EXPLORATORY_SEED_NAMESPACE",
    "MAIN_SEED_NAMESPACE",
    "TUNING_SEED_NAMESPACE",
    "CUSTOM_SEED_NAMESPACE",
    "DEFAULT_SEED_NAMESPACE",
    "MAX_SEED",
    "N_SIM",
    "N_TUNING_SEEDS",
    "generate_seeds",
    "identify_seed_namespace",
    "derive_generator_seed",
    "V3_SEED_DERIVATION_RULE",
    "EXPLORATORY_SEEDS",
    "MAIN_SEEDS",
    "TUNING_SEEDS",
    "SEED_NAMESPACES",
    "SEEDS_BY_NAMESPACE",
    "SEEDS",
    "N_TRAIN_VALUES",
    "N_SYNTH_PER_CLASS",
    "K",
    "TORCH_NUM_THREADS",
]
