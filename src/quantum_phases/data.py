from __future__ import annotations

from pathlib import Path

import pandas as pd

from .paths import (
    DATA_DMRG_PROCESSED_DIR,
    DATA_LABELED_DIR,
    DATA_MAHLOW_PROCESSED_DIR,
    DATA_RAW_DIR,
    REPO_ROOT,
)

# Registro historico dos tres datasets produzidos por diagonalizacao exata.
# Mantemos DATASETS como alias retrocompativel porque varios diagnosticos
# assumem especificamente o schema ED com features "2".."26".
ED_DATASETS = [
    ("H1", DATA_LABELED_DIR / "H1_labeled.csv"),
    ("H2", DATA_LABELED_DIR / "H2_labeled.csv"),
    ("H3", DATA_LABELED_DIR / "H3_labeled.csv"),
]
DATASETS = ED_DATASETS

DMRG_DATASETS = [
    (
        "H3_DMRG",
        DATA_DMRG_PROCESSED_DIR / "h3_bilinear_biquadratic_dmrg_labeled.csv",
    ),
]

# Representacoes DMRG fornecidas em data/mahlow. Os CSVs preparados seguem o
# mesmo schema de oito correlacionadores de Gell-Mann do H3_DMRG, sem o indice
# da planilha nem os parametros deltas/theta.
MAHLOW_DATASETS = [
    ("H1_m", DATA_MAHLOW_PROCESSED_DIR / "h1_m_labeled.csv"),
    ("H3_m", DATA_MAHLOW_PROCESSED_DIR / "h3_m_labeled.csv"),
]

# Registro de quatro datasets mantido para os experimentos auxiliares que ja
# foram validados com H3_DMRG, mas ainda nao com as representacoes Mahlow.
MAIN_EXPERIMENT_DATASETS = [*ED_DATASETS, *DMRG_DATASETS]

# Registro completo disponível aos experimentos generativos. Os entrypoints
# principais usam ED_DATASETS por padrão e consultam este registro quando o
# usuário seleciona representações DMRG pela CLI.
GENERATIVE_EXPERIMENT_DATASETS = [*MAIN_EXPERIMENT_DATASETS, *MAHLOW_DATASETS]

# Schemas aos quais nao se aplica a selecao ED-only de correlacionadores locais.
GELL_MANN_DATASETS = [*DMRG_DATASETS, *MAHLOW_DATASETS]

ED_LOCAL_FEATURE_SET = "local_correlators"
ED_ALL_FEATURE_SET = "all_correlators"
DMRG_FEATURE_SET = "gell_mann_correlators"
ED_LOCAL_N_FEATURES = 21
ED_ALL_N_FEATURES = 25
DMRG_N_FEATURES = 8

_ED_DATASET_NAMES = frozenset(name for name, _path in ED_DATASETS)
_ED_DATASET_PATHS = frozenset(Path(path).resolve() for _name, path in ED_DATASETS)
_GELL_MANN_DATASET_NAMES = frozenset(name for name, _path in GELL_MANN_DATASETS)
_GELL_MANN_DATASET_PATHS = frozenset(
    Path(path).resolve() for _name, path in GELL_MANN_DATASETS
)


def registered_dataset_kind(dataset_name: str, path) -> str | None:
    """Return the registered feature schema for a dataset, if known."""
    resolved_path = Path(path).resolve()
    if (
        dataset_name in _GELL_MANN_DATASET_NAMES
        or resolved_path in _GELL_MANN_DATASET_PATHS
    ):
        return "dmrg"
    if dataset_name in _ED_DATASET_NAMES or resolved_path in _ED_DATASET_PATHS:
        return "ed"
    return None


def dataset_feature_policy(
    dataset_name: str,
    path,
    *,
    ed_local_correlators_only: bool,
) -> dict:
    """Resolve ED-21/ED-25 without applying the ED rule to DMRG schemas."""
    dataset_kind = registered_dataset_kind(dataset_name, path)
    if dataset_kind == "dmrg":
        return {
            "local_correlators_only": False,
            "feature_set": DMRG_FEATURE_SET,
            "expected_n_features": DMRG_N_FEATURES,
        }
    if ed_local_correlators_only:
        return {
            "local_correlators_only": True,
            "feature_set": ED_LOCAL_FEATURE_SET,
            "expected_n_features": (
                ED_LOCAL_N_FEATURES if dataset_kind == "ed" else None
            ),
        }
    return {
        "local_correlators_only": False,
        "feature_set": ED_ALL_FEATURE_SET,
        "expected_n_features": ED_ALL_N_FEATURES if dataset_kind == "ed" else None,
    }


def dataset_path(dataset_name: str) -> Path:
    for name, path in GENERATIVE_EXPERIMENT_DATASETS:
        if name == dataset_name:
            return path
    valid = ", ".join(name for name, _path in GENERATIVE_EXPERIMENT_DATASETS)
    raise ValueError(f"Unknown dataset: {dataset_name!r}. Use one of: {valid}.")


def validate_local_correlator_selection(datasets, *, enabled: bool) -> None:
    """Legacy strict guard retained for external callers.

    New experiment code should use :func:`dataset_feature_policy`, which applies
    the ED selection per dataset and preserves DMRG-8 in mixed runs.
    """
    if not enabled:
        return

    unsupported = [
        name
        for name, path in datasets
        if registered_dataset_kind(name, path) == "dmrg"
    ]
    if unsupported:
        names = ", ".join(unsupported)
        raise ValueError(
            "local_correlators_only is defined only for the ED schema "
            f"(features '2'..'26'); it cannot be applied to: {names}. "
            "Pass ED_DATASETS explicitly or disable local_correlators_only."
        )


def load_dataset(path, *, label_col="labels", drop_cols=("0", "1")) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Dataset is empty: {path}")
    if label_col not in df.columns:
        raise ValueError(f"Missing label column '{label_col}' in {path}")

    df = df.drop(columns=list(drop_cols), errors="ignore")
    return df.reset_index(drop=True)


# Nos datasets atuais (features "2".."26" apos load_dataset), a ultima coluna
# de feature ("26") e a soma das tres anteriores ("23"+"24"+"25", verificado
# numericamente em H1/H3 com residuo ~1e-8) e essas tres sao correlacionadores
# nao locais. As 21 colunas restantes ("2".."22") formam o subconjunto
# historicamente denominado ``local_correlators``. O mapeamento físico dos
# cabeçalhos posicionais deve ser documentado antes de afirmar acessibilidade
# experimental.
N_NONLOCAL_TAIL_COLS = 4


def select_local_correlators(
    df: pd.DataFrame,
    *,
    label_col="labels",
    n_tail_cols: int = N_NONLOCAL_TAIL_COLS,
) -> pd.DataFrame:
    """Keep only the local spin correlators, dropping the last feature columns.

    Drops the trailing `n_tail_cols` feature columns (non-local correlators
    plus their sum), preserving `label_col` and the column order.
    """
    n_tail_cols = int(n_tail_cols)
    if n_tail_cols < 0:
        raise ValueError(f"n_tail_cols must be >= 0, got {n_tail_cols}.")
    if label_col not in df.columns:
        raise ValueError(f"Missing label column '{label_col}'.")

    feature_cols = [col for col in df.columns if col != label_col]
    if n_tail_cols >= len(feature_cols):
        raise ValueError(
            f"Cannot drop {n_tail_cols} trailing feature columns; "
            f"dataframe has only {len(feature_cols)} feature columns."
        )
    return df.drop(columns=feature_cols[len(feature_cols) - n_tail_cols :])


# -----------------------------------------------------------------------------
# Emenda de 05/11/2026 (main/docs/protocolo.md, Secao 6.1): exclusao, na
# leitura, das linhas ED com algum |C| > 1 nas features do ED-21. Opcao
# explicita e desligada por padrao: sem ela, a v2 e as runs de 03 a 05/11/2026
# continuam reproduziveis. data/ nao e alterado.
# -----------------------------------------------------------------------------
ANOMALOUS_ROWS_RULE_ID = "ed21_abs_gt_1"
ANOMALOUS_ROWS_COLUMNS = tuple(str(index) for index in range(2, 23))
ANOMALOUS_ROWS_LIMIT = 1.0
ANOMALOUS_ROWS_EXPECTED_COUNTS = {"H1": 1, "H2": 0, "H3": 6}


def anomalous_rows_rule() -> dict:
    """Descricao da regra gravada no manifesto."""
    return {
        "rule_id": ANOMALOUS_ROWS_RULE_ID,
        "rule": (
            "exclude ED rows with any |x| > 1 in the ED-21 feature columns "
            "'2'..'22', without tolerance"
        ),
        "columns": list(ANOMALOUS_ROWS_COLUMNS),
        "limit": ANOMALOUS_ROWS_LIMIT,
        "applies_to": "registered ED datasets; Gell-Mann (DMRG) datasets keep every row",
        "row_index": "0-based position after load_dataset (CSV line with header = index + 2)",
        "expected_counts": dict(ANOMALOUS_ROWS_EXPECTED_COUNTS),
        "protocol": "main/docs/protocolo.md, Secao 6.1 (emenda de 05/11/2026)",
    }


def find_anomalous_rows(df: pd.DataFrame) -> list[int]:
    """Posicoes (desde 0, como em ``load_dataset``) com algum |C| > 1 no ED-21.

    So as colunas "2".."22" entram na regra, tambem num frame ED-25: "26" e
    uma soma que passa de 1 sem violar limite fisico.
    """
    missing = [column for column in ANOMALOUS_ROWS_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"The anomalous-row rule needs the ED-21 columns; missing {missing}.")
    if not df.index.equals(pd.RangeIndex(len(df))):
        raise ValueError("find_anomalous_rows expects the 0-based index of load_dataset.")
    mask = (df[list(ANOMALOUS_ROWS_COLUMNS)].abs() > ANOMALOUS_ROWS_LIMIT).any(axis=1)
    return [int(index) for index in df.index[mask.to_numpy()]]


def exclude_anomalous_rows(df: pd.DataFrame, dataset_name: str, path, *, label_col="labels"):
    """Aplica a regra a um dataset recem-lido; devolve ``(frame, registro)``.

    O frame devolvido tem indice novo, desde 0, como o de ``load_dataset``.
    Datasets DMRG ficam inteiros, com ``applies = False``.
    """
    kind = registered_dataset_kind(dataset_name, path)
    if kind == "dmrg":
        return df, {
            "rule_id": ANOMALOUS_ROWS_RULE_ID, "applies": False,
            "n_rows_read": int(len(df)), "n_rows_kept": int(len(df)),
            "excluded_rows": [], "excluded_labels": [],
        }
    if kind != "ed":
        raise ValueError(
            "The anomalous-row rule is defined only for registered ED and DMRG "
            f"datasets, not {dataset_name!r} ({path})."
        )
    rows = find_anomalous_rows(df)
    kept = df.drop(index=rows).reset_index(drop=True)
    return kept, {
        "rule_id": ANOMALOUS_ROWS_RULE_ID, "applies": True,
        "n_rows_read": int(len(df)), "n_rows_kept": int(len(kept)),
        "excluded_rows": rows,
        "excluded_labels": df.loc[rows, label_col].tolist() if label_col in df else [],
        "expected_count": ANOMALOUS_ROWS_EXPECTED_COUNTS.get(str(dataset_name)),
    }


def anomalous_rows_exclusion(datasets, *, label_col="labels", drop_cols=("0", "1")) -> dict:
    """Le cada dataset uma vez e monta o registro do manifesto.

    Nao confere a contagem: ``check_anomalous_rows_counts`` faz isso dentro da
    run, para que uma contagem errada deixe a run como ``failed``.
    """
    by_dataset = {}
    for dataset_name, path in datasets:
        frame = load_dataset(path, label_col=label_col, drop_cols=tuple(drop_cols))
        _kept, record = exclude_anomalous_rows(frame, dataset_name, path, label_col=label_col)
        by_dataset[str(dataset_name)] = record
    return {**anomalous_rows_rule(), "by_dataset": by_dataset}


def check_anomalous_rows_counts(exclusion: dict) -> None:
    """Interrompe se a contagem de algum dataset ED diferir da esperada."""
    expected = exclusion["expected_counts"]
    problems = []
    for dataset_name, record in exclusion["by_dataset"].items():
        if not record["applies"]:
            continue
        found = len(record["excluded_rows"])
        if dataset_name not in expected:
            problems.append(f"{dataset_name}: no expected count")
        elif found != int(expected[dataset_name]):
            problems.append(
                f"{dataset_name}: found {found} (rows {record['excluded_rows']}), "
                f"expected {expected[dataset_name]}"
            )
    if problems:
        raise ValueError(
            "Anomalous-row count differs from the protocol (Secao 6.1): "
            + "; ".join(problems)
        )


def apply_recorded_row_exclusion(df: pd.DataFrame, dataset_name: str, path, recorded,
                                 *, label_col="labels") -> pd.DataFrame:
    """Reaplica a exclusao gravada num manifesto, conferindo as mesmas linhas.

    ``recorded`` vazio ou ``None`` (runs sem a opcao) devolve o frame intacto.
    """
    if not recorded:
        return df
    if recorded.get("rule_id") != ANOMALOUS_ROWS_RULE_ID:
        raise ValueError(f"{dataset_name}: unknown row-exclusion rule {recorded.get('rule_id')!r}.")
    kept, record = exclude_anomalous_rows(df, dataset_name, path, label_col=label_col)
    recorded_rows = [int(row) for row in recorded.get("excluded_rows", [])]
    if record["applies"] != bool(recorded.get("applies")) or record["excluded_rows"] != recorded_rows:
        raise ValueError(
            f"{dataset_name}: excluded rows {record['excluded_rows']} differ from the "
            f"recorded {recorded_rows}."
        )
    return kept
