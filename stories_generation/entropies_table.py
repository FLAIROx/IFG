import dataclasses
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import textwrap
from matplotlib import rcParams
import seaborn as sns
import tyro

# Style configuration
sns.set_context("paper")
sns.set_style("whitegrid")

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
rcParams['pdf.fonttype'] = 42
rcParams['ps.fonttype'] = 42
rcParams['font.size'] = 14
rcParams['axes.labelsize'] = 16
rcParams['axes.titlesize'] = 16
rcParams['xtick.labelsize'] = 16
rcParams['ytick.labelsize'] = 16
rcParams['legend.fontsize'] = 14
rcParams['axes.grid'] = True
rcParams['grid.alpha'] = 0.3
rcParams['grid.linestyle'] = '--'
rcParams['figure.figsize'] = [7, 5]
rcParams['figure.dpi'] = 300
rcParams['savefig.dpi'] = 300

EPS = 1e-9


@dataclasses.dataclass
class Config:
    file_path: str = "stories_generation/outputs/new_semantic_ent/semantic_entropy_gemma_12b_1.3_0.3.json"
    output_viz_path: str = "stories_generation/outputs/entropy_visualization.png"


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

            rows.append({
                "Story": story_id,
                "Style": style_name,
                "H(i)": h_i,
                "H(r)": h_r,
                "H(r|i)": h_r_given_i,
                "H(i|r)": h_i_given_r,
            })

    df = pd.DataFrame(rows)

    # Clean epsilons (values < 1e-5 become 0.0)
    numeric_cols = ["H(i)", "H(r)", "H(r|i)", "H(i|r)"]
    for col in numeric_cols:
        df[col] = df[col].apply(lambda x: 0.0 if x < 1e-5 else x)

    return df


def create_averaged_table(df):
    """Create table of entropies averaged across stories for each style."""
    numeric_cols = ["H(i)", "H(r)", "H(r|i)", "H(i|r)"]
    
    # Group by Style and compute mean and std
    grouped = df.groupby("Style")[numeric_cols].agg(["mean", "std"]).reset_index()
    
    # Flatten column names
    grouped.columns = ["Style"] + [f"{col}_{stat}" for col, stat in grouped.columns[1:]]
    
    # Create a cleaner table with just means
    mean_table = df.groupby("Style")[numeric_cols].mean().reset_index()
    
    # Count stories per style
    counts = df.groupby("Style").size().reset_index(name="n_stories")
    mean_table = mean_table.merge(counts, on="Style")
    
    return mean_table, grouped


def verify_entropy_identity(df):
    """Verify the entropy identity: H(r) = H(i) + H(r|i) - H(i|r)."""
    df = df.copy()
    df["RHS"] = df["H(i)"] + df["H(r|i)"] - df["H(i|r)"]
    df["Diff"] = df["H(r)"] - df["RHS"]
    
    print("\n--- Entropy Identity Verification ---")
    print("H(r) should equal H(i) + H(r|i) - H(i|r)")
    print(f"Mean absolute difference: {df['Diff'].abs().mean():.6f}")
    print(f"Max absolute difference: {df['Diff'].abs().max():.6f}")
    
    return df


def visualize_entropies(mean_table, output_path):
    """Create bar chart visualization of entropies by style."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(mean_table))
    width = 0.2
    
    # Shorten labels for display
    short_labels = [textwrap.fill(s, 25) for s in mean_table["Style"]]
    
    bars1 = ax.bar(x - 1.5*width, mean_table["H(i)"], width, label='$H(\\mathbf{i})$', color='#1f77b4')
    bars2 = ax.bar(x - 0.5*width, mean_table["H(r)"], width, label='$H(\\mathbf{r})$', color='#ff7f0e')
    bars3 = ax.bar(x + 0.5*width, mean_table["H(r|i)"], width, label='$H(\\mathbf{r}|\\mathbf{i})$', color='#2ca02c')
    bars4 = ax.bar(x + 1.5*width, mean_table["H(i|r)"], width, label='$H(\\mathbf{i}|\\mathbf{r})$', color='#d62728')
    
    ax.set_ylabel('Entropy (nats)')
    ax.set_title('Semantic Entropy Metrics by Intent Style\n(Averaged Across Stories)')
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=9)
    ax.legend(loc='upper right')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"\nVisualization saved to: {output_path}")


def print_latex_table(mean_table):
    """Print LaTeX formatted table with styles as rows and entropies as columns."""
    print("\n--- LaTeX Table Output ---")
    
    # Table: rows = styles, columns = entropies
    table_df = mean_table[["Style", "H(i)", "H(r)", "H(r|i)", "H(i|r)"]].copy()
    table_df.columns = ["Intent Style", "$H(\\mathbf{i})$", "$H(\\mathbf{r})$", 
                        "$H(\\mathbf{r}|\\mathbf{i})$", "$H(\\mathbf{i}|\\mathbf{r})$"]
    
    latex = table_df.to_latex(
        index=False, 
        float_format="%.3f",
        escape=False,
        caption="Semantic Entropy Metrics Averaged Across Stories",
        label="tab:entropy_metrics",
        column_format="l" + "c" * 4,  # left-align style, center entropies
    )
    print(latex)
    
    # Also print a simple markdown table
    print("\n--- Markdown Table ---")
    print("| Intent Style | H(i) | H(r) | H(r|i) | H(i|r) |")
    print("|--------------|------|------|--------|--------|")
    for _, row in mean_table.iterrows():
        print(f"| {row['Style'][:40]} | {row['H(i)']:.3f} | {row['H(r)']:.3f} | {row['H(r|i)']:.3f} | {row['H(i|r)']:.3f} |")
    
    return table_df


def main(cfg: Config) -> None:
    file_path = cfg.file_path
    output_viz_path = cfg.output_viz_path

    try:
        print(f"Loading data from: {file_path}")
        df = load_and_process_data(file_path)
        
        print(f"\nLoaded {len(df)} rows across {df['Style'].nunique()} styles and {df['Story'].nunique()} stories")
        
        # Create averaged table
        mean_table, full_stats = create_averaged_table(df)
        
        print("\n--- Averaged Entropy Metrics by Style ---")
        print("(Rows: Intent Styles, Columns: Entropy Metrics)")
        print(mean_table[["Style", "H(i)", "H(r)", "H(r|i)", "H(i|r)"]].to_string(index=False))
        
        # Verify entropy identity on averaged data
        verify_entropy_identity(mean_table)
        
        # Visualize
        visualize_entropies(mean_table, output_viz_path)
        
        # Print LaTeX table
        print_latex_table(mean_table)
        
    except Exception as e:
        print(f"An error occurred: {e}")
        raise


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
