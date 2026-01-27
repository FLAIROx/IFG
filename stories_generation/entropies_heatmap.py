import json
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

# Style configuration (match entropies_table.py)
sns.set_context("paper")
sns.set_style("whitegrid")

rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42
rcParams["font.size"] = 14
rcParams["axes.labelsize"] = 16
rcParams["axes.titlesize"] = 16
rcParams["xtick.labelsize"] = 14
rcParams["ytick.labelsize"] = 14
rcParams["legend.fontsize"] = 14
rcParams["axes.grid"] = True
rcParams["grid.alpha"] = 0.3
rcParams["grid.linestyle"] = "--"
rcParams["figure.figsize"] = [9, 5]
rcParams["figure.dpi"] = 300
rcParams["savefig.dpi"] = 300

EPS = 1e-9
METRICS = ["H(i)", "H(r)", "H(r|i)", "H(i|r)"]


def load_and_process_data(file_path):
    """Load semantic entropy data and process into a DataFrame."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for story_id, story_data in data.get("metrics_by_prompt", {}).items():
        # Handle both old format (intent_description_metrics) and new format (style_description_metrics)
        metrics_group = story_data.get("style_description_metrics", {})
        if not metrics_group:
            metrics_group = story_data.get("intent_description_metrics", {})

        for style_name, metrics in metrics_group.items():
            h_i = metrics.get("entropy_intents", 0.0)
            h_r = metrics.get("entropy_responses", 0.0)
            h_r_given_i = metrics.get("conditional_entropy_responses_given_intents", 0.0)
            h_i_given_r = metrics.get("conditional_entropy_intents_given_responses", 0.0)

            rows.append(
                {
                    "Story": story_id,
                    "Style": style_name,
                    "H(i)": h_i,
                    "H(r)": h_r,
                    "H(r|i)": h_r_given_i,
                    "H(i|r)": h_i_given_r,
                }
            )

    df = pd.DataFrame(rows)

    # Clean epsilons (values < 1e-5 become 0.0)
    for col in METRICS:
        df[col] = df[col].apply(lambda x: 0.0 if x < 1e-5 else x)

    return df


def parse_temps_from_filename(filename):
    """Extract (intent_temp, response_temp) from file name."""
    match = re.search(r"semantic_entropy_gemma_12b_([0-9.]+)_([0-9.]+)\.json$", filename)
    if not match:
        return None
    return float(match.group(1)), float(match.group(2))


def summarize_metrics_by_style(df):
    """Return mean metrics per style across stories."""
    return df.groupby("Style")[METRICS].mean()


def collect_metrics_by_temp_and_style(data_dir):
    """Load all files and return mapping: style -> (intent, response) -> metric means."""
    metrics_by_style = {}
    for fname in os.listdir(data_dir):
        temps = parse_temps_from_filename(fname)
        if temps is None:
            continue
        intent_temp, response_temp = temps
        file_path = os.path.join(data_dir, fname)
        df = load_and_process_data(file_path)
        style_means = summarize_metrics_by_style(df)
        for style, metrics in style_means.iterrows():
            metrics_by_style.setdefault(style, {})[(intent_temp, response_temp)] = metrics
    return metrics_by_style


def build_heatmap_data(metrics_by_temp, fixed_intent=None, fixed_response=None):
    """Create heatmap data with rows as temps and columns as metrics."""
    rows = []
    temps = []
    for (intent_temp, response_temp), metrics in metrics_by_temp.items():
        if fixed_intent is not None and intent_temp != fixed_intent:
            continue
        if fixed_response is not None and response_temp != fixed_response:
            continue
        if fixed_intent is not None:
            temps.append(response_temp)
        else:
            temps.append(intent_temp)
        rows.append(metrics.values)

    if not rows:
        raise ValueError("No data found for the requested temperature slice.")

    # Sort rows by temperature
    sort_idx = np.argsort(temps)
    temps_sorted = [temps[i] for i in sort_idx]
    rows_sorted = [rows[i] for i in sort_idx]

    heatmap_df = pd.DataFrame(rows_sorted, columns=METRICS)
    heatmap_df.insert(0, "Temp", temps_sorted)
    return heatmap_df


def plot_heatmap(heatmap_df, title, y_label, output_path):
    """Plot a heatmap for the metrics across temperature values."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    data = heatmap_df[METRICS].to_numpy()
    sns.heatmap(
        data,
        ax=ax,
        cmap="viridis",
        vmin=0.0,
        vmax=2.6,
        annot=True,
        fmt=".3f",
        cbar_kws={"label": "Entropy"},
    )
    ax.set_xticklabels(METRICS, rotation=0)
    ax.set_yticklabels([f"{t:.1f}" for t in heatmap_df["Temp"]], rotation=0)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved heatmap to: {output_path}")


def main():
    data_dir = "/home/s5e/uljad.s5e/IFG/stories_generation/outputs/new_semantic_ent"
    output_dir = "/home/s5e/uljad.s5e/IFG/stories_generation/outputs"

    metrics_by_style = collect_metrics_by_temp_and_style(data_dir)

    for style, metrics_by_temp in metrics_by_style.items():
        safe_style = re.sub(r"[^A-Za-z0-9_.-]+", "_", style).strip("_").lower()
        
        style_shorter = style
        if "Elaborate" in style:
            style_shorter = "Elaborate, multi sentence"
        elif "Highly" in style:
            style_shorter = "Semantically Compressed"
        elif "Single" in style:
            style_shorter = "Single-token"

        # Heatmap 1: response temps on y-axis for intent temp = 0.3
        try:
            heatmap_resp = build_heatmap_data(metrics_by_temp, fixed_intent=0.3)
            plot_heatmap(
                heatmap_resp,
                title=f"Semantic Entropy (Intent Temp = 0.3) \n {style_shorter} Response",
                y_label="Response Temperature",
                output_path=os.path.join(
                    output_dir, f"entropy_heatmap_intent_0.3_{safe_style}.png"
                ),
            )
        except ValueError:
            print(f"Skipping intent=0.3 heatmap for style '{style}' (no data).")

        # Heatmap 2: intent temps on y-axis for response temp = 0.3
        try:
            heatmap_intent = build_heatmap_data(metrics_by_temp, fixed_response=0.3)
            plot_heatmap(
                heatmap_intent,
                title=f"Semantic Entropy (Response Temp = 0.3) \n {style_shorter} Intent",
                y_label="Intent Temperature",
                output_path=os.path.join(
                    output_dir, f"entropy_heatmap_response_0.3_{safe_style}.png"
                ),
            )
        except ValueError:
            print(f"Skipping response=0.3 heatmap for style '{style}' (no data).")


if __name__ == "__main__":
    main()
