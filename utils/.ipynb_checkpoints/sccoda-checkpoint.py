import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import os
import sys
from pathlib import Path

import scanpy as sc
import anndata as ad

# compositional packages
import pertpy as pt

RESULTS_DIR = Path('/lustre/groups/ml01/projects/2023_ad_jaekel/AD1_2025/results/compositional')
DATA_DIR = Path('/lustre/groups/ml01/projects/2023_ad_jaekel/AD1_2025/adatas')

# set verbosity levels
sc.settings.verbosity = 2

# Accessed on: 17 June 2025
# Modified from https://github.com/theislab/scCODA/blob/887955e5f968960e2112fdab4258a205596540ee/sccoda/util/cell_composition_data.py#L4
def sccoda_from_scanpy(
        adata,
        cell_type_identifier: str,
        sample_identifier: str,
        covariate_key = None,
        covariate_df  = None
):

    """
    Creates a compositional analysis dataset from a single anndata object, as it is produced by e.g. scanpy.

    The anndata object needs to have a column in adata.obs that contains the cell type assignment,
    and one column that specifies the grouping into samples.
    Covariates can either be specified via a key in adata.uns, or as a separate DataFrame.

    NOTE: The order of samples in the returned dataset is determined by the first occurence of cells from each sample in `adata`

    Parameters
    ----------
    adata
        list of scanpy data sets
    cell_type_identifier
        column name in adata.obs that specifies the cell types
    sample_identifier
        column name in adata.obs that specifies the sample
    covariate_key
        key for adata.uns, where covariate values are stored
    covariate_df
        DataFrame with covariates

    Returns
    -------
    A compositional analysis data set

    data
        A compositional analysis data set

    """

    groups = adata.obs.value_counts([sample_identifier, cell_type_identifier])
    count_data = groups.unstack(level=cell_type_identifier)
    count_data = count_data.fillna(0)

    if covariate_key is not None:
        covariate_df = pd.DataFrame(adata.uns[covariate_key])
    elif covariate_df is None:
        print("No covariate information specified!")
        covariate_df = pd.DataFrame(index=count_data.index)

    if set(covariate_df.index) != set(count_data.index):
        raise ValueError("anndata sample names and covariate_df index do not have the same elements!")
    covs_ord = covariate_df.reindex(count_data.index)
    covs_ord.index = covs_ord.index.astype(str)

    var_dat = count_data.sum(axis=0).rename("n_cells").to_frame()
    var_dat.index = var_dat.index.astype(str)

    return ad.AnnData(X=count_data.values,
                      var=var_dat,
                      obs=covs_ord)

def save_sccoda_data(sccoda_data, reference_cell_type):
    df = pd.DataFrame(columns = ['Covariate', 'Reference cell type', 'Cell Type', 'Final Parameter', 'SD', 'Inclusion probability', 'log2-fold change'])
    varm = sccoda_data["coda"].varm
    keys_with_effect = [key for key in varm.keys() if "effect_df" in key]
    
    df_list = []
    for key in keys_with_effect:
        effect_df = sccoda_data['coda'].varm[key][['Final Parameter', 'SD', 'Inclusion probability', 'log2-fold change']]
        # Reset index to get cell type as a column
        effect_df = effect_df.reset_index().rename(columns={'index': 'Cell Type'})
        
        # Add the constant columns
        effect_df['Covariate'] = key
        effect_df['Reference cell type'] = reference_cell_type
        
        # Reorder columns as desired
        effect_df = effect_df[
            ['Covariate', 'Reference cell type', 'Cell Type', 
             'Final Parameter', 'SD', 'Inclusion probability', 'log2-fold change']
        ]
        df_list.append(effect_df)
    
    # Concatenate all pieces into a single DataFrame
    df = pd.concat(df_list, ignore_index=True)
    
    return df

def sccoda_across_cell_types(sccoda_model, sccoda_data, FORMULA = 'condition', FDR = 0.2, save_path = None):
    cell_types = sccoda_data["coda"].var.index
    results_cycle = pd.DataFrame(index=cell_types, columns=["times_credible"]).fillna(0)
    df_list = []
    
    for reference_ct in cell_types:
        print(f"Reference: {reference_ct}")
    
        # Run inference
        sccoda_data = sccoda_model.prepare(
            sccoda_data,
            modality_key="coda",
            formula=FORMULA,
            reference_cell_type=reference_ct,
        )
        sccoda_model.run_nuts(sccoda_data, modality_key="coda")
        sccoda_model.set_fdr(sccoda_data, modality_key="coda", est_fdr=FDR)

        df = save_sccoda_data(sccoda_data, reference_ct)
        df_list.append(df)
    
        # Select credible effects
        cred_eff = sccoda_model.credible_effects(sccoda_data, modality_key="coda")
        print(cred_eff)
        cred_eff.index = cred_eff.index.droplevel(level=0)
    
        # add up credible effects
        #results_cycle["times_credible"] += cred_eff.astype("int")

        #heatmap_sccoda_result(sccoda_data)

    df = pd.concat(df_list, ignore_index=True)

    if not save_path is None:
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(Path(save_path, f'effect_df_{FORMULA}_{FDR}.csv'))
        print(f'Saved to {save_dir}.')

    return results_cycle, sccoda_data, df

def sccoda_prepare_subset(adata, condition, sub_adata_obs, obs_variable, cell_type_label = 'cell_type_sea', plotting = True):
    sub_adata = adata[adata.obs[sub_adata_obs] == obs_variable].copy()
    sccoda_model = pt.tl.Sccoda()
    sccoda_data_sub = sccoda_model.load(
        sub_adata,
        type="cell_level",
        generate_sample_level=True,
        cell_type_identifier=cell_type_label,
        sample_identifier="sample",
        covariate_obs=[condition],
    )
    if plotting:
        pt.pl.coda.boxplots(
            sccoda_data_sub,
            modality_key="coda",
            feature_name=condition,
            figsize=(12, 5),
            add_dots=True,
            args_swarmplot={"palette": ["red"]},
        )
        pt.pl.coda.stacked_barplot(
            sccoda_data_sub, modality_key="coda", feature_name=condition, figsize=(4, 2)
        )
        plt.show()

    return sccoda_model, sccoda_data_sub

def heatmap_df_results(df, metric = 'log2-fold change'):
    avg_log2fc = df.groupby(['Covariate', 'Cell Type'])[metric].mean()
    
    # If your avg_log2fc is a Series with MultiIndex, unstack it:
    heatmap_data = avg_log2fc.unstack('Cell Type')
    
    # Plot the heatmap
    plt.figure(figsize=(10, 2))
    sns.heatmap(heatmap_data, cmap='vlag', annot=True, fmt=".2f", center=0)
    plt.title('Average log2-fold change')
    plt.xlabel('Cell Type')
    plt.ylabel('Covariate')
    plt.tight_layout()
    plt.show()

def heatmap_sccoda_result(df, metric: str = "log2-fold change"):
    covariates = df['covariate']
    
    rows = []
    row_labels = []
    
    for key in keys_with_effect:
        df = varm[key]
    
        # If it's a DataFrame with index = cell types:
        if hasattr(df, 'columns'):
            log2_fc = df[metric].values
        else:
            log2_fc = df[metric]  # adjust if it's a structured ndarray
    
        rows.append(log2_fc)
        # Clean up the label: remove the prefix
        clean_label = key.replace("effect_df_condition_area", "").lstrip("_")
        row_labels.append(clean_label)
    
    heatmap_df = pd.DataFrame(rows, index=row_labels, columns=cell_types)
    plt.figure(figsize=(len(cell_types)*0.5, len(row_labels)*0.5))
    sns.heatmap(heatmap_df, cmap="RdBu_r", center=0,
                cbar_kws={f'label': {metric}},
                xticklabels=True, yticklabels=True)
    plt.title(metric)
    plt.xlabel("Cell types")
    plt.ylabel("Condition + Area")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    RESULTS_DIR = Path('/lustre/groups/ml01/projects/2023_ad_jaekel/AD1_2025/results/compositional')
    DATA_DIR = Path('/lustre/groups/ml01/projects/2023_ad_jaekel/AD1_2025/adatas')

    adata = sc.read_h5ad(Path(DATA_DIR / "april22_adata_final.h5ad"))
    print(adata)
    adata.X = adata.layers['raw']
    print(adata.X[:10, :10])
    cell_type_label = 'cell_type_sea'

    OPTION = 1

    if OPTION == 1:
        FDR = 0.4
        for AREA, name in [('prefrontal cortex', 'PFC'), ('temporal cortex', 'TC'), ('visual cortex', 'VC')]:
            sub_adata = adata[adata.obs['area'] == AREA]
            data = sccoda_from_scanpy(
                sub_adata,
                cell_type_identifier=cell_type_label,
                sample_identifier="sample"
            )
        
            for formula in ['condition']:
                sccoda_model = pt.tl.Sccoda()
                sccoda_data = sccoda_model.load(
                    adata,
                    type="cell_level",
                    generate_sample_level=True,
                    cell_type_identifier=cell_type_label,
                    sample_identifier="sample",
                    covariate_obs=[formula],
                )
                RESULTS_DIR = Path(f'/lustre/groups/ml01/projects/2023_ad_jaekel/AD1_2025/results/compositional/{name}')
                results_cycle, sccoda_data_cond, df = sccoda_across_cell_types(sccoda_model, sccoda_data, FORMULA = formula, FDR = FDR, save_path = RESULTS_DIR)

    else: 
        
        for formula in ['condition', 'area', 'condition_area']:
            sccoda_model = pt.tl.Sccoda()
            sccoda_data = sccoda_model.load(
                adata,
                type="cell_level",
                generate_sample_level=True,
                cell_type_identifier=cell_type_label,
                sample_identifier="sample",
                covariate_obs=[formula],
            )
            results_cycle, sccoda_data_cond, df = sccoda_across_cell_types(sccoda_model, sccoda_data, FORMULA = formula, FDR = 0.2, save_path = RESULTS_DIR)
    
