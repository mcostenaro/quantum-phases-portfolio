"""Run one paired demonstration with the research generator implementations."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np
import pandas as pd
import scipy
import sklearn
import torch

from C_GAN.src.experiment_generator import generate as gan
from C_difusion.src.experiment_generator import generate_per_phase as ddpm
from quantum_phases.augmentation_controls import CONTROL_METHODS, generate as control
from quantum_phases.classify import evaluate_real_vs_augmented
from quantum_phases.split import train_test_split_per_class
from examples.make_demo_data import make_data


GAN = dict(latent_dim=16, hidden_dim=64, batch_size=64, epochs=40,
           lr=2e-4, beta1=.5, beta2=.999, scaling="physical",
           output_activation="tanh", d_logits=True, device="cpu")
DDPM = dict(batch_size=64, epochs=1, min_train_steps=40, time_emb_dim=32,
            phase_emb_dim=4, t_max=100, zero_terminal_snr=False,
            scaling="physical", schedule="cosine", cosine_s=.008,
            parameterization="x0", sampling_variance="posterior",
            clip="denoised_physical", lr=1e-3, dropout=0., device="cpu")


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    data, parameters = make_data()
    data.to_csv(output / "demo_data.csv", index=False)
    parameters.to_csv(output / "demo_parameters.csv", index=False)
    train, test = train_test_split_per_class(data, 3, seed=42)
    context = {"dataset": "DEMO-L4", "n_train": 3, "seed": 42}
    results, metadata = [], {}
    for arm in (*CONTROL_METHODS, "gan", "ddpm"):
        if arm == "gan":
            synth = gan(train, n_synth_per_class=10, config=GAN, seed=42, unit_context=context)
        elif arm == "ddpm":
            synth = ddpm(train, n_train=3, n_synth_per_class=10, config=DDPM,
                         seed=42, label_col="labels", unit_context=context)
        else:
            config = {"method": arm, "sigma": .005}
            synth = control(train, n_synth_per_class=10, config=config, seed=42,
                            unit_context={**context, "arm": arm})
        metadata[arm] = synth.attrs.get("training_metadata", [])
        assert synth.groupby("labels").size().to_dict() == {0: 10, 1: 10, 2: 10}
        assert np.isfinite(synth.to_numpy()).all()
        assert np.abs(synth.drop(columns="labels").to_numpy()).max() <= 1 + 1e-12
        synth.to_csv(output / f"synthetic_{arm}.csv", index=False)
        scores = evaluate_real_vs_augmented(train, synth, test)
        results.append({"arm": arm, **{key: value for key, value in scores.items()
                                      if not key.startswith("report_")}})
    results = pd.DataFrame(results)
    assert results.macro_f1_real.nunique() == 1
    copies = results.loc[results.arm == "copias"].iloc[0]
    assert all(copies[key] == 0 for key in ("delta_acc", "delta_balanced_acc", "delta_macro_f1"))
    results.to_csv(output / "metrics.csv", index=False)
    manifest = {
        "kind": "execution demonstration; not the v3 scientific grid",
        "data": "independent L=4 spin-1 ED; 9 correlators; 36 points; 3 parameter regimes",
        "n_train_per_regime": 3, "n_test": len(test), "seed": 42,
        "gan_config": GAN, "ddpm_config": DDPM, "jitter_sigma": .005,
        "training_metadata": metadata,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scipy": scipy.__version__,
                     "scikit-learn": sklearn.__version__, "torch": torch.__version__},
        "hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.glob("*.csv"))},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(results[["arm", "macro_f1_real", "macro_f1_aug", "delta_macro_f1"]].to_string(index=False))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New output directory; never overwrites a run")
    run(parser.parse_args().output)
