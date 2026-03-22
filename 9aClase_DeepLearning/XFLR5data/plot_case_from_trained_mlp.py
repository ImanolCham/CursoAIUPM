from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_PATH = Path("mlp_cp_model.joblib")
DATA_PATH = Path("xflr5_pressures_combined.pkl")
OUTPUT_PATH = Path("cp_curves_Re0p5e6_flap0_alpha5.png")

REYNOLDS = 0.5e6
ALPHA_DEG = 5.0
FLAP_DEG = 0.0


def main() -> None:
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    x_scaler = bundle["x_scaler"]
    y_scaler = bundle["y_scaler"]
    feature_cols = bundle["feature_cols"]
    target_cols = bundle["target_cols"]

    x_percent = np.arange(0.0, 101.0, 1.0)
    X_plot = pd.DataFrame(
        {
            "reynolds": REYNOLDS,
            "alpha_deg": ALPHA_DEG,
            "flap_deg": FLAP_DEG,
            "x_percent": x_percent,
        }
    )[feature_cols]

    y_pred = y_scaler.inverse_transform(model.predict(x_scaler.transform(X_plot)))
    pred_df = pd.DataFrame(y_pred, columns=target_cols)
    pred_df["x_percent"] = x_percent

    actual_df = pd.read_pickle(DATA_PATH)
    actual_case = actual_df[
        (actual_df["reynolds"] == REYNOLDS)
        & (actual_df["alpha_deg"] == ALPHA_DEG)
        & (actual_df["flap_deg"] == FLAP_DEG)
    ][["x_percent", "cp_intrados", "cp_extrados"]].copy()

    plt.figure(figsize=(10, 6))
    plt.plot(
        pred_df["x_percent"],
        pred_df["cp_intrados"],
        label="Intrados MLP",
        color="#d1495b",
        linewidth=2.5,
    )
    plt.plot(
        pred_df["x_percent"],
        pred_df["cp_extrados"],
        label="Extrados MLP",
        color="#00798c",
        linewidth=2.5,
    )

    if not actual_case.empty:
        plt.plot(
            actual_case["x_percent"],
            actual_case["cp_intrados"],
            "--",
            label="Intrados datos",
            color="#d1495b",
            alpha=0.7,
        )
        plt.plot(
            actual_case["x_percent"],
            actual_case["cp_extrados"],
            "--",
            label="Extrados datos",
            color="#00798c",
            alpha=0.7,
        )

    plt.gca().invert_yaxis()
    plt.xlabel("x/c (%)")
    plt.ylabel("Cp")
    plt.title("Curvas de presion predichas por la MLP\nRe=0.5e6, flap=0 deg, alpha=5 deg")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=180)

    print(f"Figura guardada en: {OUTPUT_PATH.resolve()}")
    print(f"Predicciones generadas: {len(pred_df)} puntos")
    print(f"Caso real disponible: {not actual_case.empty}")


if __name__ == "__main__":
    main()
