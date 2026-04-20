import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import rcParams
import os

# ── Settings ────────────────────────────────────────────────────────────────
CSV_PATH   = "feature_importance_for_plot.csv"   # change to your CSV path
OUTPUT_DIR = "plot_feature_importances"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Font
rcParams["font.family"] = "Arial"

# ── Load data ────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
wavenumbers   = df.iloc[:, 0]          # first column  → x-axis
fg_columns    = df.columns[1:]         # remaining 40 columns → functional groups

# ── Plot each functional group ───────────────────────────────────────────────
for fg in fg_columns:
    fig, ax = plt.subplots(figsize=(8,5)) # 5:3 aspect ratio can use 10,6 or 15,9

    # Line plot
    ax.plot(wavenumbers, df[fg], linewidth=1.2, label=fg)

    # ── Border: all 4 spines visible, 2 pt width ────────────────────────────
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2)

    # ── X-axis: reversed 4000→500, major tick every 875, outside ──────────
    ax.set_xlim(4000, 500)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(875))
    ax.xaxis.set_minor_locator(ticker.NullLocator())

    # ── Y-axis: 0→5, major tick every 1, inside ─────────────────────────
    ax.set_ylim(0, 5)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_minor_locator(ticker.NullLocator())

    # ── Ticks: x outside, y inside ──────────────────────────────────────
    ax.tick_params(
        axis="x",
        which="major",
        top=False, bottom=True,
        direction="out",               # outside
        width=1.5,
        labelsize=16,
    )
    ax.tick_params(
        axis="y",
        which="major",
        left=True, right=False,
        direction="in",                # inside
        width=1.5,
        labelsize=16,
    )

    # ── Axis labels ─────────────────────────────────────────────────────────
    ax.set_xlabel("Wavenumber (cm⁻¹)", fontsize=20, fontfamily="Arial")
    ax.set_ylabel("Feature Importance",  fontsize=20, fontfamily="Arial")

    # ── Tick label font ──────────────────────────────────────────────────────
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily("Arial")
        label.set_fontsize(16)

    # ── Legend inside plot, 1 pt border ─────────────────────────────────────
    legend = ax.legend(
        loc="upper right",
        fontsize=14,
        prop={"family": "Arial", "size": 14},
        frameon=True,
        edgecolor="black",
        framealpha=1,
    )
    legend.get_frame().set_linewidth(1)

    plt.tight_layout()

    # Safe filename (replace special characters)
    safe_name = str(fg).replace("/", "_").replace("\\", "_").replace(" ", "_")
    out_path  = os.path.join(OUTPUT_DIR, f"{safe_name}.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    #fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

print(f"\nDone! {len(fg_columns)} plots saved to '{OUTPUT_DIR}/'")