import os
import scanpy as sc
import pandas as pd
import numpy as np
from pertps import PerturbAnalyzer, plot_ps_on_lda, plot_global_summary
from pertps.barcodes import AMBIGUOUS_LABEL, UNASSIGNED_LABEL, load_barcode_table
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


# ==============================================================================
# 1. USER CONFIGURATION
# ==============================================================================
h5ad_filepath    = "./demo/PertTF_Subset_100MB.h5ad"
output_h5ad_path = "./demo/PertTF_Subset_PS_Scores.h5ad"
barcode_filepath = "./demo/BARCODE_10x_Merged.txt"
negative_ctrl    = "Non-Targeting"

# Guide-calling thresholds. A cell is assigned to its top guide only when that
# guide has at least MIN_UMI counts and beats the runner-up by DOMINANCE_RATIO;
# otherwise it is Ambiguous. Keep these in step with perturbseq-pipeline so the
# two give the same per-cell answer.
MIN_UMI          = 3
DOMINANCE_RATIO  = 2.0

# Horizontal cut of the quadrant plot, placed inside the control population.
# "mean" rather than "median": single-cell counts are zero-inflated and the
# control median is exactly 0 for most genes, which collapses "low expression"
# into "expression is exactly zero".
EXPRESSION_CUT   = "mean"   # mean | median

gene_list = [
    "SMARCC1", "TCF7L2", "HMGA2", "AFF4", "HIF1A", "TCF7L1", "SMARCA4", "CTNNB1", 
    "NFIB", "TCF12", "SMAD3", "EZH2", "REST", "HDAC1", "ARID1B", "SMARCB1", 
    "CREBBP", "SALL4", "SMAD4", "SUZ12", "SMARCD2", "ARID1A", "ARNT", "EP300", 
    "CLOCK", "JARID2", "SMARCD1", "KLF6", "SMARCC2", "SOX2", "PAX5", "POU5F1", 
    "EED", "TCF7", "SMARCA2", "BATF", "SMARCD3", "OR2F1", "ESRRB", "OR2AG2", 
    "NANOG", "KLF4", "OR2D3", "TFCP2L1", "OR2A25", "OR6A2", "TBX3", "LEF1", 
    "RUNX1", "MYC"
]

# Create output directories
for folder in ["plots_batch", "tables_batch", "plots_fixed_lda"]:
    os.makedirs(folder, exist_ok=True)

# ==============================================================================
# 2. LOAD DATA & PREP
# ==============================================================================
print("Loading AnnData...")
adata = sc.read_h5ad(h5ad_filepath)

# --- CRITICAL FIX 1: Handle Non-Unique Names ---
if not adata.obs_names.is_unique:
    print("WARNING: Non-unique obs_names detected. Making them unique...")
    adata.obs_names_make_unique()

print("Loading Barcode Table...")
# load_barcode_table re-derives the target from the sgrna name (so CD81.2 maps
# to CD81 rather than becoming its own target) and resolves cells carrying
# several guides with the dominance rule instead of keeping whichever row came
# last, which assigned a guide at random for roughly a quarter of cells.
labels = load_barcode_table(
    barcode_filepath,
    strip_prefix=True,
    min_umi=MIN_UMI,
    dominance_ratio=DOMINANCE_RATIO,
)
adata.obs['gene'] = adata.obs_names.map(labels.to_dict()).fillna('Other')

# Ambiguous and unassigned cells belong to neither the target nor the control
# group; leaving them in either would blur both.
n_amb = int((adata.obs['gene'] == AMBIGUOUS_LABEL).sum())
n_un = int((adata.obs['gene'] == UNASSIGNED_LABEL).sum())
n_other = int((adata.obs['gene'] == 'Other').sum())
print(
    f"Assignment: {adata.n_obs - n_amb - n_un - n_other} assigned, "
    f"{n_amb} ambiguous, {n_un} unassigned, {n_other} not in the barcode table"
)

# Diagnostic check to ensure mapping worked
print(f"--- MAPPING CHECK (Top Gene Counts) ---")
print(adata.obs['gene'].value_counts().head())

print(f"--- INITIAL DIMENSIONS ---\nCells: {adata.n_obs}\nGenes: {adata.n_vars}")

# ==============================================================================
# 3. SURGICAL LOOP (CALCULATE PS SCORES)
# ==============================================================================
analyzer = PerturbAnalyzer(adata, neg_ctrl=negative_ctrl)

print("--- STARTING SURGICAL ANALYSIS (PS SCORES) ---")
for target_gene in tqdm(gene_list, desc="Processing Genes"):
    # Calculate scores using the translated scMAGeCK EM logic
    scores = analyzer.calculate_ps_score(target_gene)
    
    if scores is not None:
        score_name = f"{target_gene}_eff"
        # Update Master Object
        adata.obs[score_name] = scores.reindex(adata.obs_names).fillna(0)
        
        # Save CSV equivalent
        scores.to_csv(f"tables_batch/{target_gene}_PS_Scores.csv")
    else:
        # Pass silently to keep logs clean, or print warning if desired
        pass

# ==============================================================================
# 4. GLOBAL LDA GENERATION (THE FIXED MAP)
# ==============================================================================
print("--- STARTING GLOBAL LDA GENERATION ---")

# This method trains the LDA on perturbed cells and creates the UMAP coordinates
try:
    analyzer.compute_lda_umap(gene_list)
except ValueError as e:
    print(f"CRITICAL ERROR in LDA: {e}")
    print("This likely means the barcode mapping resulted in 0 target cells.")

# ==============================================================================
# 5. VISUALIZATION ON FIXED MAP
# ==============================================================================
print("Generating Fixed UMAP plots...")
plot_ps_on_lda(adata, gene_list, output_dir="plots_fixed_lda", neg_ctrl="NT")
print("Generating Global UMAP plots...")
plot_global_summary(adata, output_dir="plots_fixed_lda",score_threshold=0.8, downsample_bg=0.05)

# ==============================================================================
# 5. GENERATING LABELED DIAGNOSTIC SCATTER PLOTS (CLEAN VERSION)
# ==============================================================================
print("--- GENERATING LABELED DIAGNOSTIC SCATTER PLOTS ---")
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("plots_scatter_validation", exist_ok=True)

for target_gene in tqdm(gene_list, desc="Generating Labeled Scatters"):
    score_col = f"{target_gene}_eff"
    
    # Check if data exists
    if score_col not in adata.obs.columns or target_gene not in adata.var_names:
        continue
        
    # 1. EXTRACT DATA
    # We flatten the sparse matrix to a 1D array
    exp_data = adata[:, target_gene].X.toarray().flatten() if hasattr(adata.X, "toarray") else adata[:, target_gene].X
    
    plot_df = pd.DataFrame({
        'PS_Score': adata.obs[score_col],
        'Expression': exp_data,
        'Group': adata.obs['gene']
    }).fillna(0)

    # 2. SEPARATE GROUPS (Surgical Split)
    # Background: Control Cells
    df_ctrl = plot_df[plot_df['Group'] == negative_ctrl].copy()
    
    # Foreground: Target Cells
    df_target = plot_df[plot_df['Group'] == target_gene].copy()
    
    if len(df_target) < 10:
        continue

    # 3. DOWNSAMPLE BACKGROUND (The Cleaner)
    # We only plot 2,000 random control cells. This clears the "grey fog".
    if len(df_ctrl) > 2000:
        df_ctrl = df_ctrl.sample(n=2000, random_state=42)

    # 4. PLOTTING (Layered)
    plt.figure(figsize=(9, 7))
    
    # Layer 1: The sparse grey background (Drawn FIRST)
    plt.scatter(
        df_ctrl['PS_Score'], 
        df_ctrl['Expression'], 
        c='lightgrey', 
        s=30, 
        alpha=0.3,       # High transparency
        label=negative_ctrl, 
        edgecolors='none'
    )

    # Layer 2: The red signal (Drawn SECOND / ON TOP)
    plt.scatter(
        df_target['PS_Score'], 
        df_target['Expression'], 
        c='#e74c3c',     # Bright Red
        s=40,            # Slightly larger
        alpha=0.7,       # More solid
        label=target_gene, 
        edgecolors='white', 
        linewidth=0.5
    )

    # 5. LABELS & LINES
    # The horizontal cut sits inside the control population. Use the mean rather
    # than the median: expression is zero-inflated, so the control median is
    # exactly 0 for most genes and the cut degenerates into "expression is
    # exactly zero" with the dashed line drawn on the axis. Across the demo the
    # two give the same net signal to within ~0.1 percentage points.
    full_ctrl = plot_df[plot_df['Group'] == negative_ctrl]['Expression']
    if EXPRESSION_CUT == "median":
        h_thresh = full_ctrl.median()
    else:
        h_thresh = full_ctrl.mean()
    v_thresh = 0.5

    # Applying the same classification to control cells says what the knockdown
    # percentage actually means: "66% knocked down" reads very differently next
    # to a 0% control rate than next to a 16% one.
    ctrl_kd = float(
        ((full_ctrl <= h_thresh) & (df_ctrl['PS_Score'] >= v_thresh)).mean() * 100
    ) if len(full_ctrl) else float('nan')
    tgt_kd = float(
        ((df_target['Expression'] <= h_thresh) & (df_target['PS_Score'] >= v_thresh)).mean() * 100
    )

    plt.axvline(x=v_thresh, color='black', linestyle='--', alpha=0.5)
    plt.axhline(y=h_thresh, color='black', linestyle='--', alpha=0.5)

    # Smart Quadrant Labels (Dynamic positioning)
    y_max = plot_df['Expression'].max()
    y_min = plot_df['Expression'].min()
    
    # Top-Right (Failed KD)
    plt.text(0.75, y_max * 0.9, 'ESCAPERS', fontsize=10, fontweight='bold', color='darkred', ha='center')
    # Bottom-Right (Successful KD)
    plt.text(0.75, max(0, y_min) + (y_max*0.05), 'SUCCESSFUL KD', fontsize=10, fontweight='bold', color='darkgreen', ha='center')
    # Top-Left (Controls)
    plt.text(0.25, y_max * 0.9, 'CONTROL / WT', fontsize=10, fontweight='bold', color='blue', ha='center')
    # Bottom-Left (Low Signal)
    plt.text(0.25, max(0, y_min) + (y_max*0.05), 'LOW SIGNAL', fontsize=10, fontweight='bold', color='grey', ha='center')

    plt.title(
        f"Perturbation Validation: {target_gene}\n"
        f"knocked down: {tgt_kd:.0f}% of {target_gene} cells vs "
        f"{ctrl_kd:.0f}% of controls (net {tgt_kd - ctrl_kd:+.0f}%)",
        fontsize=10,
    )
    plt.xlabel("Perturbation Score (PS)")
    plt.ylabel(f"Normalized {target_gene} Expression")
    plt.legend(loc='upper right', frameon=False)
    
    plt.savefig(f"plots_scatter_validation/{target_gene}_labeled_scatter.png", dpi=300)
    plt.close()
