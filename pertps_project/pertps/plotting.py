import scanpy as sc
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd


def knockdown_fraction(frame, h_thresh, v_thresh=0.5, expression_col="Expression",
                       score_col="PS_Score"):
    """Percentage of cells in the successful-knockdown quadrant.

    A cell counts when its expression is at or below ``h_thresh`` *and* its
    perturbation score is at or above ``v_thresh``. Expression is on the same
    log-normalized scale as ``adata.X``; the score is the normalized PS score
    in ``[0, 1]``.

    Both columns are read from a single frame on purpose. Taking them from two
    frames lets pandas align the ``&`` on the union of two different indexes,
    which silently divides a sample-sized numerator by a population-sized
    denominator. Returns NaN for an empty frame.
    """
    if len(frame) == 0:
        return float("nan")
    hit = (frame[expression_col] <= h_thresh) & (frame[score_col] >= v_thresh)
    return float(hit.mean() * 100)


def plot_ps_on_lda(adata, gene_list, output_dir="plots_fixed_lda", neg_ctrl="Non-Targeting", basis="X_lda_umap"):
    """
    Generates 50 individual plots:
    - Background: NT cells (Light Grey)
    - Foreground: Target Gene (Blue Gradient by PS Score)
    """
    os.makedirs(output_dir, exist_ok=True)
    sc.set_figure_params(dpi=150, frameon=False, facecolor='white')

    if basis not in adata.obsm.keys():
        print(f"Error: Basis '{basis}' not found in adata.obsm")
        return

    for gene in gene_list:
        score_col = f"{gene}_eff"
        if score_col not in adata.obs.columns:
            continue

        # Setup Layers
        bg_mask = adata.obs['lda_label'] != gene
        fg_mask = adata.obs['lda_label'] == gene
        
        bg_cells = adata[bg_mask].copy()
        fg_cells = adata[fg_mask].copy()
        
        # Sort foreground
        if fg_cells.n_obs > 0:
            fg_cells = fg_cells[fg_cells.obs[score_col].sort_values().index]

        # Plotting
        fig, ax = plt.subplots(figsize=(8, 7))

        # Layer 1: Grey Background
        if bg_cells.n_obs > 0:
            ax.scatter(
                bg_cells.obsm[basis][:, 0],
                bg_cells.obsm[basis][:, 1],
                c="lightgrey",
                s=15, 
                alpha=0.4, 
                edgecolor='none',
                label=neg_ctrl
            )

        # Layer 2: Blue Foreground
        if fg_cells.n_obs > 0:
            sc.pl.embedding(
                fg_cells,
                basis=basis,
                color=score_col,
                cmap="Blues",
                ax=ax,
                show=False,
                frameon=False,
                title=f"LDA-Supervised UMAP: {gene}",
                s=25,
                vmin=0.0, vmax=1.0,
                sort_order=True
            )

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel("LDA-UMAP 1")
        ax.set_ylabel("LDA-UMAP 2")
        
        outfile = os.path.join(output_dir, f"{gene}_Fixed_LDA.png")
        plt.tight_layout()
        plt.savefig(outfile, bbox_inches='tight', dpi=300)
        plt.close()

    print(f"Individual plots saved to: {output_dir}")

def plot_global_summary(adata, output_dir="plots_fixed_lda", basis="X_lda_umap", score_threshold=0.8, downsample_bg=0.1):
    """
    Generates a SURGICAL 'All-in-One' map:
    1. Filters for high-confidence hits (Score > 0.8).
    2. Throws away 90% of the grey background to reduce clutter.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Reset Labels
    adata.obs['plot_label'] = 'Background'
    
    # 2. Strict Filtering (High Confidence Only)
    print(f"Filtering: Keeping only cells with PS Score > {score_threshold}...")
    genes_found = 0
    
    for gene in adata.obs['lda_label'].unique():
        if gene in ['NT', 'Non-Targeting', 'Other', 'Background']:
            continue
            
        score_col = f"{gene}_eff"
        if score_col in adata.obs.columns:
            mask = (adata.obs['lda_label'] == gene) & (adata.obs[score_col] > score_threshold)
            adata.obs.loc[mask, 'plot_label'] = gene
            genes_found += 1

    # 3. Surgical Downsampling of Background
    # We create a mask that keeps ALL colored cells, but only 10% (0.1) of grey cells
    bg_full_mask = adata.obs['plot_label'] == 'Background'
    bg_indices = np.where(bg_full_mask)[0]
    
    # Randomly select a subset of background indices
    keep_n = int(len(bg_indices) * downsample_bg)
    keep_bg_indices = np.random.choice(bg_indices, keep_n, replace=False)
    
    # Combine: Keep ALL foreground + Subsampled Background
    fg_mask = adata.obs['plot_label'] != 'Background'
    fg_indices = np.where(fg_mask)[0]
    final_indices = np.concatenate([keep_bg_indices, fg_indices])
    
    # Create a temporary subset object for plotting
    adata_plot = adata[final_indices].copy()
    
    # 4. PLOT
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Layer 1: The Sparse Grey Cloud
    sc.pl.embedding(
        adata_plot[adata_plot.obs['plot_label'] == 'Background'],
        basis=basis,
        ax=ax,
        show=False,
        frameon=False,
        size=15,             # Larger dots since there are fewer of them
        color_map=None,
        palette=['lightgrey'], 
        color='plot_label',
        title="",
        legend_loc=None,
        alpha=0.3
    )

    # Layer 2: The High-Confidence Hits
    # We increase dot size here to make them "pop" like islands
    fg_data = adata_plot[adata_plot.obs['plot_label'] != 'Background']
    
    if fg_data.n_obs > 0:
        sc.pl.embedding(
            fg_data,
            basis=basis,
            color='plot_label',
            ax=ax,
            show=False,
            frameon=False,
            title=f"Global Summary (Score > {score_threshold})",
            size=30,             # BIGGER dots for the hits
            legend_loc='on data',
            legend_fontsize=9,
            legend_fontweight='bold',
            palette='tab20'      
        )

    # Clean axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    outfile = os.path.join(output_dir, "Global_Summary_LDA_map.png")
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    print(f"Surgical Summary saved to {outfile}")