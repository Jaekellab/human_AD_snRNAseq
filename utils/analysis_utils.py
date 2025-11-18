import anndata as ad
import numpy as np

def cluster_cells_by_gene_expression(adata: ad.AnnData, threshold: float, gene_name: str = "MT-ND4") -> None:
    """
    Clusters cells in the AnnData object based on gene expression threshold.
    
    Parameters:
        adata (AnnData): The input AnnData object containing cell data.
        threshold (float): The expression threshold for clustering.
        gene_name (str): The name of the gene used for clustering. Default is "MT-ND4".
        
    Returns:
        None: The function updates `adata.obs` with a new column `MT_ND4_cluster`.
    """
    if gene_name not in adata.var_names:
        raise ValueError(f"Gene '{gene_name}' not found in `adata.var_names`.")
    
    # Get the expression values for MT-ND4
    gene_expression = adata[:, gene_name].X.toarray().flatten() if hasattr(adata[:, gene_name].X, 'toarray') else adata[:, gene_name].X
    
    # Cluster cells based on the threshold
    clusters = np.where(gene_expression >= threshold, "High", "Low")
    
    # Add the cluster information to `adata.obs`
    adata.obs[f"{gene_name}_cluster"] = clusters

    return adata