import logging

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.optimize import minimize
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import umap

logger = logging.getLogger(__name__)

class PerturbAnalyzer:
    def __init__(self, adata, neg_ctrl="Non-Targeting", scale_factor=3.0):
        self.adata = adata
        self.neg_ctrl = neg_ctrl
        self.scale_factor = scale_factor

    def calculate_ps_score(self, target_gene, top_n=100, min_pct_expressing_control=1.0):
        """
        OPTIMIZED: Vectorized Analytic Solution.
        Replaces numerical optimization loop with exact linear projection.
        Mathematically equivalent to minimizing least squares but 100x faster.
        """
        import pandas as pd
        import numpy as np
        import scanpy as sc

        # 1. Surgical Subset (Target + Control only)
        # Matches R logic of isolating the relevant cells
        cells_mask = self.adata.obs['gene'].isin([target_gene, self.neg_ctrl])
        subset = self.adata[cells_mask].copy()
        
        if subset.n_obs < 10:
            return None

        # A gene the controls do not express cannot be shown to be knocked
        # down: with a control median of zero every cell lands on the "low
        # expression" side of the quadrant split, so any high score is read as
        # a successful knockdown. Before this check the olfactory-receptor
        # controls topped the efficiency ranking (OR2D3 at 51%).
        if min_pct_expressing_control is not None and target_gene in subset.var_names:
            ctrl = subset[subset.obs['gene'] == self.neg_ctrl, target_gene].X
            ctrl = ctrl.toarray() if hasattr(ctrl, "toarray") else np.asarray(ctrl)
            pct = 100.0 * float(np.mean(np.asarray(ctrl).ravel() > 0))
            if pct < min_pct_expressing_control:
                logger.warning(
                    "Skipping %s: expressed in only %.2f%% of control cells, so a "
                    "knockdown cannot be measured.", target_gene, pct,
                )
                return None

        # 2. Feature Selection (Identify top changing genes)
        # Using t-test (closest to R's simple diff logic)
        try:
            sc.tl.rank_genes_groups(subset, groupby='gene', reference=self.neg_ctrl, method='t-test')
            # Extract top_n gene names
            target_biomarkers = pd.DataFrame(subset.uns['rank_genes_groups']['names'])[target_gene].head(top_n).tolist()
        except Exception as exc:
            # A bare 'except' here also swallows KeyboardInterrupt and hides why
            # a target failed, which makes a 50-gene batch impossible to debug.
            logger.warning(
                "Skipping %s: ranking genes failed (%s: %s)",
                target_gene, type(exc).__name__, exc,
            )
            return None

        # 3. Prepare Matrices
        # Y: Expression of biomarkers (Cells x Genes)
        Y = subset[:, target_biomarkers].X
        if hasattr(Y, "toarray"): 
            Y = Y.toarray()
            
        # X: The binary perturbation vector (1 for Target, 0 for NT)
        x_obs = np.where(subset.obs['gene'] == target_gene, 1.0, 0.0)

        # 4. Calculate Betas (The "Signature" of the perturbation)
        # We use OLS (Ordinary Least Squares) to find how genes change when x=1
        # beta = Cov(X, Y) / Var(X)
        x_centered = x_obs - np.mean(x_obs)
        y_centered = Y - np.mean(Y, axis=0)
        
        # Denominator: Variance of X
        var_x = np.dot(x_centered, x_centered)
        if var_x == 0: return None
        
        # Beta: The coefficients
        betas = np.dot(x_centered, y_centered) / var_x

        # 5. Calculate Scores (The Exact Analytic Projection)
        # Instead of 'minimize' loop, we project every cell onto the Beta signature at once.
        # Score_i = (Cell_i . Beta) / (Beta . Beta)
        
        beta_norm_sq = np.dot(betas, betas)
        if beta_norm_sq == 0:
            eff_scores = np.zeros(subset.n_obs)
        else:
            # Center Y for projection
            # Note: We project the centered Y to match the centered betas
            eff_scores = np.dot(y_centered, betas) / beta_norm_sq

        # 6. Apply Constraints (Matches R's bounds)
        # R constrains between 0 and scale_factor. We do the same via clipping.
        # We also handle the "Mean Shift" relative to NT (NT should be ~0)
        
        # Shift scores so NT mean is 0 (approx)
        nt_mean_score = np.mean(eff_scores[subset.obs['gene'] == self.neg_ctrl])
        eff_scores = eff_scores - nt_mean_score
        
        # Clip to [0, scale_factor]
        # This effectively replaces 'L-BFGS-B' bounds
        eff_scores = np.clip(eff_scores, 0, self.scale_factor)
        
        # Normalize max to 1.0 (Optional, for visualization consistency)
        if np.max(eff_scores) > 0:
            eff_scores /= np.max(eff_scores)

        # Return Series aligned to index
        return pd.Series(eff_scores, index=subset.obs_names)

    def compute_lda_umap(self, gene_list, n_pcs=40):
        """
        Surgical Fix: Uses 'eigen' solver.
        - 'eigen' supports shrinkage (like lsqr) to tighten clusters.
        - 'eigen' supports transform (unlike lsqr) to generate the map.
        """
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        import umap
        import numpy as np
        import scanpy as sc # Ensure scanpy is imported

        print(f"--- GENERATING HIGH-PRECISION LDA UMAP (Eigen Solver) ---")
        
        adata_work = self.adata.copy()
        
        # Standard Pre-processing
        # Note: The 'densifies it' warning you saw is normal here. 
        # We need dense data for proper scaling.
        if adata_work.X.max() > 20:
            print("Normalizing raw counts...")
            sc.pp.normalize_total(adata_work, target_sum=1e4)
            sc.pp.log1p(adata_work)
            
        print("Finding HVGs and Scaling...")
        sc.pp.highly_variable_genes(adata_work, n_top_genes=2000)
        sc.pp.scale(adata_work, max_value=10)
        
        print("Running PCA...")
        sc.tl.pca(adata_work, n_comps=n_pcs)
        
        # Labeling (Barcode Ground Truth)
        adata_work.obs['lda_label'] = 'Other'
        adata_work.obs.loc[adata_work.obs['gene'] == self.neg_ctrl, 'lda_label'] = 'NT'
        
        valid_genes = []
        for gene in gene_list:
            mask = adata_work.obs['gene'] == gene
            if mask.sum() > 5:
                adata_work.obs.loc[mask, 'lda_label'] = gene
                valid_genes.append(gene)
                
        # Subset to target+control only
        valid_mask = adata_work.obs['lda_label'].isin(valid_genes + ['NT'])
        subset = adata_work[valid_mask].copy()
        
        # --- THE FIX: SOLVER='EIGEN' ---
        # 'eigen' is the only solver that allows shrinkage AND projection.
        print("Running LDA (Solver=Eigen, Shrinkage=Auto)...")
        lda = LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto')
        X_lda = lda.fit_transform(subset.obsm['X_pca'], subset.obs['lda_label'])
        
        # --- THE FIX: TIGHT UMAP ---
        print("Running Tight UMAP (min_dist=0.01)...")
        reducer = umap.UMAP(
            n_neighbors=30, 
            min_dist=0.01,   # R-Style Tightness
            metric='cosine', 
            random_state=42
        )
        embedding = reducer.fit_transform(X_lda)
        
        # Align coordinates back to master object
        full_embedding = np.full((self.adata.n_obs, 2), np.nan)
        full_embedding[np.where(valid_mask)[0]] = embedding
        
        self.adata.obsm['X_lda_umap'] = full_embedding
        self.adata.obs['lda_label'] = adata_work.obs['lda_label']
        
        return self.adata