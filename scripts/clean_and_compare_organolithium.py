#!/usr/bin/env python3
"""First-pass cleaning and manuscript-style comparison for organolithium datasets."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import PercentFormatter


ROOT = Path("/Users/zhaowenyuan/Projects/FlowFigTabMiner")
COMPARE_DIR = ROOT / "data" / "final_output" / "dataset_comparison"

MY_ENRICHED_XLSX = COMPARE_DIR / "ALL_organolithium_normalized_20260327_1022_enriched.xlsx"
USPTO_RICH_CSV = ROOT / "data" / "input" / "uspto_organolithium_extracted_rich.csv"
ORD_RICH_CSV = ROOT / "other dataset" / "ord" / "organolithium" / "ord_organolithium_reactions_rich.csv"

MY_CLEANED_XLSX = COMPARE_DIR / "ALL_organolithium_normalized_20260327_1022_cleaned_v3.xlsx"
MY_CLEANING_JSON = COMPARE_DIR / "my_dataset_cleaning_v3_summary.json"
DATASET_COMPARE_CSV = COMPARE_DIR / "organolithium_dataset_comparison_cleaned_v3.csv"
DATASET_COMPARE_MD = COMPARE_DIR / "organolithium_dataset_comparison_cleaned_v3.md"
DATASET_COMPARE_TEX = COMPARE_DIR / "organolithium_dataset_comparison_cleaned_v3.tex"
PLOT_COMPLETENESS = COMPARE_DIR / "organolithium_dataset_comparison_cleaned_v3.png"
PLOT_OUTCOMES = COMPARE_DIR / "organolithium_yield_time_temperature_grid_cleaned_v3.png"


FLOW_REACTOR_PATTERN = re.compile(
    r"(?i)(flow|microflow|microreactor|micromixer|t-mixer|t-shaped|coil|capillary|packed bed|tube reactor|static mixer|disc reactor)"
)
BATCH_PATTERN = re.compile(r"(?i)\bbatch\b|\bflask\b|\bround-bottom\b")


def coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def keep_in_range(series: pd.Series, low: float | None = None, high: float | None = None) -> tuple[pd.Series, pd.Series]:
    cleaned = series.copy()
    invalid = pd.Series(False, index=series.index)
    present = series.notna()
    if low is not None:
        invalid |= present & (series < low)
    if high is not None:
        invalid |= present & (series > high)
    cleaned[invalid] = pd.NA
    return cleaned, invalid


def percent_weights(series: pd.Series) -> pd.Series:
    if len(series) == 0:
        return pd.Series(dtype=float)
    return pd.Series([100.0 / len(series)] * len(series), index=series.index, dtype=float)


def histogram_percent_max(series: pd.Series, bins) -> float:
    if len(series) == 0:
        return 0.0
    counts, _ = np.histogram(series, bins=bins)
    return float(counts.max() * 100.0 / len(series)) if len(series) else 0.0


def add_panel_note(ax, n_value: int, coverage_pct: float) -> None:
    ax.text(
        0.98,
        0.95,
        f"n={n_value:,}\ncoverage={coverage_pct:.1f}%",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.8, "edgecolor": "#cccccc"},
    )


def plot_percent_hist(ax, series: pd.Series, bins, color: str) -> None:
    weights = np.full(len(series), 100.0 / len(series), dtype=float)
    ax.hist(series.to_numpy(dtype=float), bins=bins, weights=weights, color=color, edgecolor="white", linewidth=0.6)


def classify_my_flow(row: pd.Series) -> str:
    reactor = str(row.get("reactor_type") or "")
    notes = str(row.get("notes") or "")
    source = str(row.get("source_table_or_figure") or "")
    has_flow_numeric = any(
        pd.notna(row.get(col))
        for col in [
            "cleaned_residence_time_s",
            "cleaned_flow_rate_mL_min",
            "cleaned_flow_rate_stream1_mL_min",
            "cleaned_flow_rate_stream2_mL_min",
        ]
    )
    joined = " ".join([reactor, notes, source])
    if has_flow_numeric or FLOW_REACTOR_PATTERN.search(joined):
        return "flow"
    if pd.notna(row.get("batch_yield_pct")) or BATCH_PATTERN.search(joined):
        return "batch"
    return "unknown"


def ord_extract_temperature_value(conditions_json: str | None) -> float | None:
    if not isinstance(conditions_json, str) or not conditions_json:
        return None
    try:
        data = json.loads(conditions_json)
    except Exception:
        return None
    temp = data.get("temperature", {})
    for key in ("setpoint", "control"):
        node = temp.get(key)
        if isinstance(node, dict) and isinstance(node.get("value"), (int, float)):
            return float(node["value"])
    return None


def ord_extract_time_seconds(outcomes_json: str | None) -> float | None:
    if not isinstance(outcomes_json, str) or not outcomes_json:
        return None
    try:
        outcomes = json.loads(outcomes_json)
    except Exception:
        return None
    factors = {"SECOND": 1.0, "SECONDS": 1.0, "MINUTE": 60.0, "MINUTES": 60.0, "HOUR": 3600.0, "HOURS": 3600.0}
    for outcome in outcomes:
        rt = outcome.get("reaction_time")
        if isinstance(rt, dict) and isinstance(rt.get("value"), (int, float)):
            factor = factors.get(rt.get("units"))
            if factor:
                return float(rt["value"]) * factor
    return None


def ord_has_yield(outcomes_json: str | None) -> bool:
    if not isinstance(outcomes_json, str) or not outcomes_json:
        return False
    try:
        outcomes = json.loads(outcomes_json)
    except Exception:
        return False
    for outcome in outcomes:
        for product in outcome.get("products", []):
            for measurement in product.get("measurements", []):
                if measurement.get("type") == "YIELD":
                    return True
    return False


def clean_my_dataset() -> tuple[pd.DataFrame, dict]:
    df = pd.read_excel(MY_ENRICHED_XLSX, sheet_name=0)

    numeric_cols = [
        "yield_pct",
        "batch_yield_pct",
        "conversion_pct",
        "selectivity_pct",
        "temperature_C",
        "residence_time_s",
        "flow_rate_mL_min",
        "flow_rate_stream1_mL_min",
        "flow_rate_stream2_mL_min",
        "pressure_bar",
    ]
    for col in numeric_cols:
        df[col] = coerce_numeric(df[col])

    cleaning_summary = {"input_file": str(MY_ENRICHED_XLSX), "output_file": str(MY_CLEANED_XLSX), "rules": {}}

    percent_cols = ["yield_pct", "batch_yield_pct", "conversion_pct", "selectivity_pct"]
    for col in percent_cols:
        cleaned, invalid = keep_in_range(df[col], 0, 100)
        df[f"cleaned_{col}"] = cleaned
        df[f"qc_{col}_out_of_range"] = invalid
        cleaning_summary["rules"][col] = {"invalid_rows": int(invalid.sum()), "range": [0, 100]}

    temp_cleaned, temp_invalid = keep_in_range(df["temperature_C"], -120, 150)
    df["cleaned_temperature_C"] = temp_cleaned
    df["qc_temperature_C_out_of_range"] = temp_invalid
    cleaning_summary["rules"]["temperature_C"] = {"invalid_rows": int(temp_invalid.sum()), "range": [-120, 150]}

    time_cleaned, time_invalid = keep_in_range(df["residence_time_s"], 0.00001, 86400)
    df["cleaned_residence_time_s"] = time_cleaned
    df["qc_residence_time_s_out_of_range"] = time_invalid
    cleaning_summary["rules"]["residence_time_s"] = {"invalid_rows": int(time_invalid.sum()), "range": [0.00001, 86400]}

    for col in ["flow_rate_mL_min", "flow_rate_stream1_mL_min", "flow_rate_stream2_mL_min"]:
        cleaned, invalid = keep_in_range(df[col], 0.0001, None)
        df[f"cleaned_{col}"] = cleaned
        df[f"qc_{col}_nonpositive"] = invalid
        cleaning_summary["rules"][col] = {"invalid_rows": int(invalid.sum()), "range": [0.0001, None]}

    pressure_cleaned, pressure_invalid = keep_in_range(df["pressure_bar"], 0, 500)
    df["cleaned_pressure_bar"] = pressure_cleaned
    df["qc_pressure_bar_out_of_range"] = pressure_invalid
    cleaning_summary["rules"]["pressure_bar"] = {"invalid_rows": int(pressure_invalid.sum()), "range": [0, 500]}

    df["cleaned_outcome_primary"] = df["cleaned_yield_pct"].combine_first(df["cleaned_conversion_pct"]).combine_first(
        df["cleaned_selectivity_pct"]
    )
    df["cleaned_outcome_type"] = pd.NA
    df.loc[df["cleaned_yield_pct"].notna(), "cleaned_outcome_type"] = "yield_pct"
    df.loc[df["cleaned_outcome_type"].isna() & df["cleaned_conversion_pct"].notna(), "cleaned_outcome_type"] = "conversion_pct"
    df.loc[df["cleaned_outcome_type"].isna() & df["cleaned_selectivity_pct"].notna(), "cleaned_outcome_type"] = "selectivity_pct"

    df["flow_batch_label_v1"] = df.apply(classify_my_flow, axis=1)
    df["clean_model_ready_v1"] = (
        df["cleaned_outcome_primary"].notna()
        & df["cleaned_temperature_C"].notna()
        & df["cleaned_residence_time_s"].notna()
    )
    df["clean_structure_ready_v1"] = df["reaction_smiles_enriched"].notna()

    cleaning_summary["rows"] = int(len(df))
    cleaning_summary["post_clean_counts"] = {
        "cleaned_yield_pct": int(df["cleaned_yield_pct"].notna().sum()),
        "cleaned_conversion_pct": int(df["cleaned_conversion_pct"].notna().sum()),
        "cleaned_selectivity_pct": int(df["cleaned_selectivity_pct"].notna().sum()),
        "cleaned_temperature_C": int(df["cleaned_temperature_C"].notna().sum()),
        "cleaned_residence_time_s": int(df["cleaned_residence_time_s"].notna().sum()),
        "clean_model_ready_v1": int(df["clean_model_ready_v1"].sum()),
        "clean_structure_ready_v1": int(df["clean_structure_ready_v1"].sum()),
        "flow_rows": int((df["flow_batch_label_v1"] == "flow").sum()),
        "batch_rows": int((df["flow_batch_label_v1"] == "batch").sum()),
        "unknown_rows": int((df["flow_batch_label_v1"] == "unknown").sum()),
    }

    with pd.ExcelWriter(MY_CLEANED_XLSX, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="All Records", index=False)
    MY_CLEANING_JSON.write_text(json.dumps(cleaning_summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return df, cleaning_summary


def summarize_my_dataset(df: pd.DataFrame) -> dict:
    return {
        "dataset": "My 130-paper dataset",
        "source_type": "literature-derived curated dataset",
        "topic_specificity": "high",
        "rows": int(len(df)),
        "source_units": int(df["paper_basename"].nunique(dropna=True)),
        "unique_reactions": int(df["reaction_smiles_enriched"].nunique(dropna=True)),
        "structure_complete_pct": float(df["reaction_smiles_enriched"].notna().mean() * 100),
        "outcome_label_pct": float(df["cleaned_outcome_primary"].notna().mean() * 100),
        "yield_label_pct": float(df["cleaned_yield_pct"].notna().mean() * 100),
        "temperature_pct": float(df["cleaned_temperature_C"].notna().mean() * 100),
        "time_pct": float(df["cleaned_residence_time_s"].notna().mean() * 100),
        "solvent_pct": float(df["solvent"].notna().mean() * 100),
        "catalyst_pct": float(df["catalyst"].notna().mean() * 100),
        "notes_pct": float(df["notes"].notna().mean() * 100),
        "flow_pct": float((df["flow_batch_label_v1"] == "flow").mean() * 100),
        "batch_pct": float((df["flow_batch_label_v1"] == "batch").mean() * 100),
        "unknown_mode_pct": float((df["flow_batch_label_v1"] == "unknown").mean() * 100),
        "model_ready_pct": float(df["clean_model_ready_v1"].mean() * 100),
    }


def summarize_uspto_dataset() -> tuple[dict, pd.DataFrame]:
    df = pd.read_csv(USPTO_RICH_CSV, low_memory=False)
    df["temperature_C"] = coerce_numeric(df["temperature_C"])
    df["reaction_time_s"] = coerce_numeric(df["reaction_time_s"])
    df["cleaned_temperature_C"], df["qc_temperature_invalid"] = keep_in_range(df["temperature_C"], -120, 150)
    df["cleaned_reaction_time_s"], df["qc_time_invalid"] = keep_in_range(df["reaction_time_s"], 0.00001, 86400)
    summary = {
        "dataset": "USPTO-LLM organolithium",
        "source_type": "patent-derived transformed corpus",
        "topic_specificity": "medium",
        "rows": int(len(df)),
        "source_units": int(df["source_file"].nunique(dropna=True)),
        "unique_reactions": int(df["reaction_id"].nunique(dropna=True)),
        "structure_complete_pct": float(df["reaction_smiles_raw"].notna().mean() * 100),
        "outcome_label_pct": 0.0,
        "yield_label_pct": 0.0,
        "temperature_pct": float(df["cleaned_temperature_C"].notna().mean() * 100),
        "time_pct": float(df["cleaned_reaction_time_s"].notna().mean() * 100),
        "solvent_pct": float(df["solvent"].notna().mean() * 100),
        "catalyst_pct": float(df["catalyst"].notna().mean() * 100),
        "notes_pct": 0.0,
        "flow_pct": 0.0,
        "batch_pct": 0.0,
        "unknown_mode_pct": 100.0,
        "model_ready_pct": 0.0,
    }
    return summary, df


def summarize_ord_dataset() -> tuple[dict, pd.DataFrame]:
    df = pd.read_csv(ORD_RICH_CSV, low_memory=False)
    df["temperature_C_extracted"] = df["conditions_json"].map(ord_extract_temperature_value)
    df["time_s_extracted"] = df["outcomes_json"].map(ord_extract_time_seconds)
    df["has_yield_extracted"] = df["outcomes_json"].map(ord_has_yield)
    df["has_flow_extracted"] = df["conditions_json"].fillna("").str.contains('"flow"', regex=False)
    df["cleaned_temperature_C_extracted"], df["qc_temperature_invalid"] = keep_in_range(df["temperature_C_extracted"], -120, 150)
    df["cleaned_time_s_extracted"], df["qc_time_invalid"] = keep_in_range(df["time_s_extracted"], 0.00001, 86400)
    summary = {
        "dataset": "ORD organolithium",
        "source_type": "public structured reaction database",
        "topic_specificity": "medium",
        "rows": int(len(df)),
        "source_units": int(df["doi"].nunique(dropna=True)),
        "unique_reactions": int(df["reaction_id"].nunique(dropna=True)),
        "structure_complete_pct": float(df["reaction_smiles"].notna().mean() * 100),
        "outcome_label_pct": float(df["has_outcomes"].fillna(False).mean() * 100),
        "yield_label_pct": float(df["has_yield_extracted"].mean() * 100),
        "temperature_pct": float(df["cleaned_temperature_C_extracted"].notna().mean() * 100),
        "time_pct": float(df["cleaned_time_s_extracted"].notna().mean() * 100),
        "solvent_pct": float(df["inputs_json"].fillna("").str.contains('"reaction_role": "SOLVENT"', regex=False).mean() * 100),
        "catalyst_pct": float(df["inputs_json"].fillna("").str.contains('"reaction_role": "CATALYST"', regex=False).mean() * 100),
        "notes_pct": float(df["has_notes"].fillna(False).mean() * 100),
        "flow_pct": float(df["has_flow_extracted"].mean() * 100),
        "batch_pct": 0.0,
        "unknown_mode_pct": float((~df["has_flow_extracted"]).mean() * 100),
        "model_ready_pct": float((df["has_yield_extracted"] & df["cleaned_temperature_C_extracted"].notna() & df["cleaned_time_s_extracted"].notna()).mean() * 100),
    }
    return summary, df


def save_tables_and_plots(my_df: pd.DataFrame, uspto_df: pd.DataFrame, ord_df: pd.DataFrame) -> None:
    my_summary = summarize_my_dataset(my_df)
    uspto_summary, uspto_df = summarize_uspto_dataset()
    ord_summary, ord_df = summarize_ord_dataset()
    summary_df = pd.DataFrame([my_summary, uspto_summary, ord_summary])
    summary_df.to_csv(DATASET_COMPARE_CSV, index=False)

    md_lines = [
        "# Cleaned Dataset Comparison (v1)",
        "",
        "| Dataset | Source type | Specificity | Rows | Sources | Unique reactions | Structure % | Outcome % | Yield % | Temp % | Time % | Solvent % | Catalyst % | Flow % | Batch % | Model-ready % |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary_df.iterrows():
        md_lines.append(
            f"| {row['dataset']} | {row['source_type']} | {row['topic_specificity']} | {int(row['rows'])} | {int(row['source_units'])} | {int(row['unique_reactions'])} | "
            f"{row['structure_complete_pct']:.1f} | {row['outcome_label_pct']:.1f} | {row['yield_label_pct']:.1f} | {row['temperature_pct']:.1f} | {row['time_pct']:.1f} | "
            f"{row['solvent_pct']:.1f} | {row['catalyst_pct']:.1f} | {row['flow_pct']:.1f} | {row['batch_pct']:.1f} | {row['model_ready_pct']:.1f} |"
        )
    DATASET_COMPARE_MD.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    tex = r"""\begin{table*}[t]
\centering
\small
\begin{tabular}{p{2.8cm}p{2.8cm}p{1.1cm}p{0.8cm}p{0.9cm}p{1.0cm}p{0.9cm}p{0.9cm}p{0.9cm}p{0.9cm}p{0.9cm}p{0.9cm}p{0.9cm}p{0.9cm}}
\hline
Dataset & Source type & Rows & Sources & Struct. & Outcome & Yield & Temp. & Time & Solvent & Catalyst & Flow & Batch & Model-ready \\
\hline
Our dataset & literature-derived curated dataset & %d & %d & %.1f\%% & %.1f\%% & %.1f\%% & %.1f\%% & %.1f\%% & %.1f\%% & %.1f\%% & %.1f\%% & %.1f\%% & %.1f\%% \\
USPTO-LLM organolithium & patent-derived transformed corpus & %d & %d & %.1f\%% & %.1f\%% & %.1f\%% & %.1f\%% & %.1f\%% & %.1f\%% & %.1f\%% & %.1f\%% & %.1f\%% & %.1f\%% \\
ORD organolithium & public structured reaction database & %d & %d & %.1f\%% & %.1f\%% & %.1f\%% & %.1f\%% & %.1f\%% & %.1f\%% & %.1f\%% & %.1f\%% & %.1f\%% & %.1f\%% \\
\hline
\end{tabular}
\caption{First-pass cleaned comparison of the literature-derived organolithium flow dataset against public organolithium subsets from USPTO-LLM and ORD. Model-ready indicates records with an outcome label together with numeric temperature and time information under the corresponding dataset schema.}
\label{tab:organolithium_cleaned_comparison_v1}
\end{table*}
""" % (
            my_summary["rows"], my_summary["source_units"], my_summary["structure_complete_pct"], my_summary["outcome_label_pct"], my_summary["yield_label_pct"], my_summary["temperature_pct"], my_summary["time_pct"], my_summary["solvent_pct"], my_summary["catalyst_pct"], my_summary["flow_pct"], my_summary["batch_pct"], my_summary["model_ready_pct"],
            uspto_summary["rows"], uspto_summary["source_units"], uspto_summary["structure_complete_pct"], uspto_summary["outcome_label_pct"], uspto_summary["yield_label_pct"], uspto_summary["temperature_pct"], uspto_summary["time_pct"], uspto_summary["solvent_pct"], uspto_summary["catalyst_pct"], uspto_summary["flow_pct"], uspto_summary["batch_pct"], uspto_summary["model_ready_pct"],
            ord_summary["rows"], ord_summary["source_units"], ord_summary["structure_complete_pct"], ord_summary["outcome_label_pct"], ord_summary["yield_label_pct"], ord_summary["temperature_pct"], ord_summary["time_pct"], ord_summary["solvent_pct"], ord_summary["catalyst_pct"], ord_summary["flow_pct"], ord_summary["batch_pct"], ord_summary["model_ready_pct"],
        )
    DATASET_COMPARE_TEX.write_text(tex, encoding="utf-8")

    plot_df = summary_df.melt(
        id_vars=["dataset"],
        value_vars=[
            "structure_complete_pct",
            "outcome_label_pct",
            "yield_label_pct",
            "temperature_pct",
            "time_pct",
            "solvent_pct",
            "catalyst_pct",
            "flow_pct",
            "batch_pct",
            "model_ready_pct",
        ],
        var_name="metric",
        value_name="percent",
    )
    plt.figure(figsize=(14, 6))
    sns.barplot(data=plot_df, x="metric", y="percent", hue="dataset")
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Coverage (%)")
    plt.xlabel("")
    plt.title("Cleaned Comparison of Organolithium Datasets")
    plt.tight_layout()
    plt.savefig(PLOT_COMPLETENESS, dpi=220)
    plt.close()

    ord_yield_values = []
    for text in ord_df.loc[ord_df["has_yield_extracted"], "outcomes_json"].dropna():
        try:
            outcomes = json.loads(text)
        except Exception:
            continue
        found = False
        for outcome in outcomes:
            for product in outcome.get("products", []):
                for measurement in product.get("measurements", []):
                    if measurement.get("type") == "YIELD":
                        pct = measurement.get("percentage", {}).get("value")
                        if isinstance(pct, (int, float)):
                            ord_yield_values.append(float(pct))
                            found = True
                            break
                if found:
                    break
            if found:
                break

    ord_yield_series = pd.Series(ord_yield_values, dtype=float)
    ord_yield_series, _ = keep_in_range(ord_yield_series, 0, 100)
    ord_yield_series = ord_yield_series.dropna()

    grid_data = [
        {
            "label": "My dataset",
            "color": "#1f77b4",
            "yield": pd.to_numeric(my_df["cleaned_yield_pct"], errors="coerce").dropna(),
            "time": pd.to_numeric(my_df["cleaned_residence_time_s"], errors="coerce").dropna(),
            "temp": pd.to_numeric(my_df["cleaned_temperature_C"], errors="coerce").dropna(),
            "rows": len(my_df),
        },
        {
            "label": "ORD",
            "color": "#2ca02c",
            "yield": ord_yield_series,
            "time": pd.to_numeric(ord_df["cleaned_time_s_extracted"], errors="coerce").dropna(),
            "temp": pd.to_numeric(ord_df["cleaned_temperature_C_extracted"], errors="coerce").dropna(),
            "rows": len(ord_df),
        },
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    column_titles = ["Yield Distribution", "Residence Time Distribution", "Temperature Distribution"]
    for col, title in enumerate(column_titles):
        axes[0, col].set_title(title)

    yield_xlim = (0, 100)
    temp_xlim = (-120, 150)
    all_time = pd.concat([info["time"] for info in grid_data if len(info["time"]) > 0], ignore_index=True)
    positive_time = all_time[all_time > 0]
    time_xlim = None
    if len(positive_time) > 0:
        time_xlim = (positive_time.min(), positive_time.max())
        time_bins = np.logspace(np.log10(time_xlim[0]), np.log10(time_xlim[1]), 30)
    else:
        time_bins = 30

    yield_bins = np.linspace(yield_xlim[0], yield_xlim[1], 26)
    temp_bins = np.linspace(temp_xlim[0], temp_xlim[1], 28)

    yield_ymax = max(histogram_percent_max(info["yield"], yield_bins) for info in grid_data)
    time_ymax = max(histogram_percent_max(info["time"][info["time"] > 0], time_bins) for info in grid_data)
    temp_ymax = max(histogram_percent_max(info["temp"], temp_bins) for info in grid_data)

    yield_ymax = max(yield_ymax * 1.15, 5)
    time_ymax = max(time_ymax * 1.15, 5)
    temp_ymax = max(temp_ymax * 1.15, 5)

    for row_idx, info in enumerate(grid_data):
        y = info["yield"]
        t = info["time"]
        temp = info["temp"]
        total_rows = info["rows"]

        if len(y) > 0:
            plot_percent_hist(axes[row_idx, 0], y, yield_bins, info["color"])
            add_panel_note(axes[row_idx, 0], len(y), len(y) * 100.0 / total_rows)
        else:
            axes[row_idx, 0].text(
                0.5,
                0.5,
                "No explicit yield labels",
                ha="center",
                va="center",
                transform=axes[row_idx, 0].transAxes,
            )
            add_panel_note(axes[row_idx, 0], 0, 0.0)
        axes[row_idx, 0].set_ylabel(f"{info['label']}\nPercentage (%)")
        axes[row_idx, 0].set_xlabel("Yield (%)")
        axes[row_idx, 0].set_xlim(*yield_xlim)
        axes[row_idx, 0].set_ylim(0, yield_ymax)
        axes[row_idx, 0].yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))

        t = t[t > 0]
        if len(t) > 0:
            plot_percent_hist(axes[row_idx, 1], t, time_bins, info["color"])
            axes[row_idx, 1].set_xscale("log")
            add_panel_note(axes[row_idx, 1], len(t), len(t) * 100.0 / total_rows)
        else:
            add_panel_note(axes[row_idx, 1], 0, 0.0)
        axes[row_idx, 1].set_xlabel("Time (s, log scale)")
        axes[row_idx, 1].set_ylabel("Percentage (%)")
        if time_xlim is not None:
            axes[row_idx, 1].set_xlim(*time_xlim)
        axes[row_idx, 1].set_ylim(0, time_ymax)
        axes[row_idx, 1].yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))

        if len(temp) > 0:
            plot_percent_hist(axes[row_idx, 2], temp, temp_bins, info["color"])
            add_panel_note(axes[row_idx, 2], len(temp), len(temp) * 100.0 / total_rows)
        else:
            add_panel_note(axes[row_idx, 2], 0, 0.0)
        axes[row_idx, 2].set_xlabel("Temperature (C)")
        axes[row_idx, 2].set_ylabel("Percentage (%)")
        axes[row_idx, 2].set_xlim(*temp_xlim)
        axes[row_idx, 2].set_ylim(0, temp_ymax)
        axes[row_idx, 2].yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))

    plt.tight_layout()
    plt.savefig(PLOT_OUTCOMES, dpi=220)
    plt.close()


def main() -> None:
    sns.set_theme(style="whitegrid")
    COMPARE_DIR.mkdir(parents=True, exist_ok=True)
    my_df = pd.read_excel(MY_CLEANED_XLSX)
    uspto_df = pd.read_csv(USPTO_RICH_CSV, low_memory=False)
    ord_df = pd.read_csv(ORD_RICH_CSV, low_memory=False)
    save_tables_and_plots(my_df, uspto_df, ord_df)

    print(f"Used cleaned workbook: {MY_CLEANED_XLSX}")
    print(f"Reference cleaning summary: {MY_CLEANING_JSON}")
    print(f"Saved comparison CSV: {DATASET_COMPARE_CSV}")
    print(f"Saved comparison Markdown: {DATASET_COMPARE_MD}")
    print(f"Saved comparison LaTeX: {DATASET_COMPARE_TEX}")
    print(f"Saved completeness plot: {PLOT_COMPLETENESS}")
    print(f"Saved outcome plot: {PLOT_OUTCOMES}")


if __name__ == "__main__":
    main()
