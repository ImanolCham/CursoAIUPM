from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import pandas as pd


FILE_PATTERN = re.compile(
    r"^Re_(?P<re_million>\d+(?:\.\d+)?)_(?P<flap_deg>-?\d+(?:\.\d+)?)\.csv$"
)


@dataclass(frozen=True)
class FileMeta:
    source_file: str
    reynolds_million: float
    flap_deg: float


def parse_numeric_row(line: str, expected_items: int) -> list[float] | None:
    parts = [part.strip() for part in line.split(",")]
    if len(parts) != expected_items:
        return None
    try:
        return [float(part) for part in parts]
    except ValueError:
        return None


def parse_file(file_path: Path) -> pd.DataFrame:
    match = FILE_PATTERN.match(file_path.name)
    if not match:
        raise ValueError(f"Nombre de fichero no reconocido: {file_path.name}")

    file_meta = FileMeta(
        source_file=file_path.name,
        reynolds_million=float(match.group("re_million")),
        flap_deg=float(match.group("flap_deg")),
    )

    lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
    airfoil_name = ""
    for line in lines:
        stripped = line.strip()
        if stripped and stripped.lower() != "xflr5 v6.58" and not stripped.startswith("Reynolds ="):
            airfoil_name = stripped
            break

    records: list[dict[str, float | int | str]] = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line.startswith("Reynolds ="):
            i += 1
            continue

        meta_parts = [part.strip() for part in lines[i].split(",")]
        if len(meta_parts) < 6:
            raise ValueError(f"Cabecera de Reynolds no válida en {file_path.name}: {lines[i]!r}")

        reynolds = float(meta_parts[1])
        mach = float(meta_parts[3])
        ncrit = float(meta_parts[5])

        if i + 2 >= len(lines):
            raise ValueError(f"Bloque incompleto en {file_path.name} cerca de la línea {i + 1}")

        alpha_values = parse_numeric_row(lines[i + 2], expected_items=8)
        if alpha_values is None:
            raise ValueError(
                f"Fila de coeficientes no válida en {file_path.name} cerca de la línea {i + 3}: {lines[i + 2]!r}"
            )
        alpha_deg, cd, cl, cm, xtr1, xtr2, tehmom, cpmn = alpha_values

        data_header_index = i + 3
        if data_header_index >= len(lines) or lines[data_header_index].strip() != "Cpi,Cpv":
            raise ValueError(
                f"Cabecera de presiones no encontrada en {file_path.name} cerca de la línea {data_header_index + 1}"
            )

        point_index = 0
        j = data_header_index + 1
        while j < len(lines):
            raw_line = lines[j].strip()
            if not raw_line:
                break
            if raw_line.startswith("Reynolds ="):
                break

            cp_values = parse_numeric_row(raw_line, expected_items=2)
            if cp_values is None:
                raise ValueError(
                    f"Fila Cpi/Cpv no válida en {file_path.name} cerca de la línea {j + 1}: {lines[j]!r}"
                )

            cp_intrados, cp_extrados = cp_values
            records.append(
                {
                    "source_file": file_meta.source_file,
                    "airfoil_name": airfoil_name,
                    "reynolds_million": file_meta.reynolds_million,
                    "reynolds": reynolds,
                    "mach": mach,
                    "ncrit": ncrit,
                    "flap_deg": file_meta.flap_deg,
                    "alpha_deg": alpha_deg,
                    "cd": cd,
                    "cl": cl,
                    "cm": cm,
                    "xtr1": xtr1,
                    "xtr2": xtr2,
                    "tehmom": tehmom,
                    "cpmn": cpmn,
                    "point_index": point_index,
                    "cp_intrados": cp_intrados,
                    "cp_extrados": cp_extrados,
                }
            )
            point_index += 1
            j += 1

        i = j + 1

    if not records:
        raise ValueError(f"No se han encontrado datos en {file_path.name}")

    df = pd.DataFrame.from_records(records)
    df["points_in_block"] = df.groupby(
        ["source_file", "alpha_deg"], sort=False
    )["point_index"].transform("count")
    denominator = (df["points_in_block"] - 1).where(df["points_in_block"] > 1, 1)
    df["x_percent"] = 100.0 * df["point_index"] / denominator
    return df


def build_dataframe(root_dir: Path) -> pd.DataFrame:
    csv_files = sorted(root_dir.glob("Re_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No se han encontrado ficheros Re_*.csv en {root_dir}")

    parts = [parse_file(file_path) for file_path in csv_files]
    combined = pd.concat(parts, ignore_index=True)
    combined = combined.sort_values(
        ["reynolds", "flap_deg", "alpha_deg", "point_index"], ignore_index=True
    )
    return combined


def main() -> None:
    root_dir = Path(__file__).resolve().parent
    combined = build_dataframe(root_dir)

    pickle_path = root_dir / "xflr5_pressures_combined.pkl"
    csv_path = root_dir / "xflr5_pressures_combined.csv"

    combined.to_pickle(pickle_path)
    combined.to_csv(csv_path, index=False)

    cases = combined[["source_file", "alpha_deg"]].drop_duplicates()
    summary = {
        "files": combined["source_file"].nunique(),
        "alpha_cases": len(cases),
        "rows": len(combined),
        "reynolds_min": int(combined["reynolds"].min()),
        "reynolds_max": int(combined["reynolds"].max()),
        "flap_min": combined["flap_deg"].min(),
        "flap_max": combined["flap_deg"].max(),
        "alpha_min": combined["alpha_deg"].min(),
        "alpha_max": combined["alpha_deg"].max(),
        "points_per_block_min": int(combined["points_in_block"].min()),
        "points_per_block_max": int(combined["points_in_block"].max()),
    }

    print("DataFrame combinado generado")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"pickle: {pickle_path.name}")
    print(f"csv: {csv_path.name}")
    print()
    print(combined.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
