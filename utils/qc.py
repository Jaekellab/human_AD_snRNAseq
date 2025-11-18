import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
from typing import List
from scipy.stats import spearmanr
from scipy.special import logit
#from statannotations.Annotator import Annotator

NUCLEAR_GENE_SET = [
    "MALAT1", "NEAT1", "FTX", "FOXP1", "RBMS3", "ZBTB20", "LRMDA", "PBX1",
    "ITPR2", "AUTS2", "TTC28", "BNC2", "EXOC4", "RORA", "PRKG1", "ARID1B",
    "PARD3B", "GPHN", "N4BP2L2", "PKHD1L1", "EXOC6B", "FBXL7", "MED13L",
    "TBC1D5", "IMMP2L", "SYNE1", "RERE", "MBD5", "EXT1", "WWOX"
]



# def statistical_test_boxplot(df, 
#                              x, 
#                              features, 
#                              pairs, 
#                             fig_size = (6, 4)):
#     """
#     Parameters:
#     ------------
#         df: pandas.Dataframe 
#             Column names must be equal to features and x and each row is a cell
#         x: string
#         features: list
#             must be present in df column names
#     """

#     fig, ax = plt.subplots(1,2, figsize=fig_size)
#     flatax = ax.flatten()
    
#     for i in range(len(flatax)):
#         ax = sns.boxplot(data=df, x=x, y=features[i], ax = flatax[i])
        
#         # Add statistical annotation
#         annot = Annotator(
#             flatax[i],
#             pairs=pairs,
#             data=df,
#             x=x,
#             y=features[i]
#         )
#         annot.configure(
#             test="Mann-Whitney",
#             loc="outside",  # Use "outside" if space is tight
#             text_format="star",
#             show_test_name=False,
#             verbose=1,
#             comparisons_correction=None,  # Or None if no multiple testing correction
#             fontsize=10
#         )
#         annot.apply_test()
#         _, test_results = annot.annotate()
        
#         # Final layout and save
#         fig.tight_layout()

"""
Adapted from: https://github.com/linnalab/qclus/blob/main/qclus/utils.py#L92
Accessed: 02.June 2025
"""
def get_qc_metrics(
    adata: sc.AnnData,
    gene_set: List[str],
    key_name: str,
    normlog: bool = False,
    scale: bool = False,
) -> None:
    """
    Calculate QC metrics for a given gene set and add them to the AnnData object.

    Parameters:
        adata (sc.AnnData): AnnData object containing single-cell data.
        gene_set (List[str]): List of genes to calculate QC metrics for.
        key_name (str): Key name under which to store the metrics.
        normlog (bool, optional): Whether to normalize and log-transform the data.
        scale (bool, optional): Whether to scale the data.
    """
    # Check if gene_set is a list
    if not isinstance(gene_set, list):
        raise TypeError(f"gene_set must be a list, got {type(gene_set)}.")

    # Check if genes are in adata.var_names
    missing_genes = [gene for gene in gene_set if gene not in adata.var_names]
    if missing_genes:
        print(f"Warning: The following genes are not in adata.var_names and will be ignored: {missing_genes}")

    # Create a boolean mask for the genes in the gene_set
    adata.var[key_name] = adata.var_names.isin(gene_set)

    # Create a layer to store the modified data
    adata.layers[key_name] = adata.X.copy()

    # Use the new layer for normalization and scaling
    if normlog:
        sc.pp.normalize_total(adata, target_sum=1e4, layer=key_name)
        sc.pp.log1p(adata, layer=key_name)

    if scale:
        sc.pp.scale(adata, max_value=10, layer=key_name)

    # Calculate QC metrics on the specified layer
    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=[key_name],
        percent_top=None,
        log1p=False,
        inplace=True,
        layer=key_name,
    )
    sc.tl.score_genes(adata, gene_list=gene_set, score_name=f"score_{key_name}")

    # Remove the layer to save memory
    del adata.layers[key_name]

"""
Adapted from: https://github.com/HCA-integration/scAtlasTb/blob/main/workflow/qc/scripts/qc_utils.py#L112
Accessed: 19.May 2025
"""
def plot_qc_joint(
    df: pd.DataFrame,
    x: str,
    y: str,
    log_x: int = 1,
    log_y: int = 1,
    hue: str = None,
    main_plot_function=None,
    marginal_hue=None,
    x_threshold=None,
    y_threshold=None,
    title='',
    return_df=False,
    marginal_kwargs: dict=None,
    show: bool = False,
    palette: [list, str, dict] =  None,
    **kwargs,
):
    """
    Plot scatter plot with marginal histograms from df columns.

    :param df: observation dataframe
    :param x: df column for x axis
    :param y: df column for y axis
    :param log: log base for transforming values. Default 1, no transformation
    :param hue: df column with annotations for color coding scatter plot points
    :param marginal_hue: df column with annotations for color coding marginal plot distributions
    :param x_threshold: tuple of upper and lower filter thresholds for x axis
    :param y_threshold: tuple of upper and lower filter thresholds for y axis
    :param title: Title text for plot
    :param palette: color palette
    :return:
        seaborn plot (and df dataframe with updated values, if `return_df=True`)
    """
    import seaborn as sns
    
    if main_plot_function is None:
        main_plot_function = sns.scatterplot
    if not x_threshold:
        x_threshold=(0, np.inf)
    if not y_threshold:
        y_threshold=(0, np.inf)

    def log1p_base(_x, base):
        return np.log1p(_x) / np.log(base)

    if log_x > 1:
        x_log = f'log{log_x} {x}'
        df[x_log] = log1p_base(df[x], log_x)
        x_threshold = log1p_base(x_threshold, log_x)
        x = x_log
    
    if log_y > 1:
        y_log = f'log{log_y} {y}'
        df[y_log] = log1p_base(df[y], log_y)
        y_threshold = log1p_base(y_threshold, log_y)
        y = y_log
        
    if marginal_kwargs is None:
        marginal_kwargs = dict(legend=False, )
    
    if marginal_hue in df.columns:
        marginal_hue = None if df[marginal_hue].nunique() > 100 else marginal_hue
    use_marg_hue = marginal_hue is not None
    
    if not use_marg_hue:
         marginal_kwargs.pop('palette', None)
    
    g = sns.JointGrid(
        data=df,
        x=x,
        y=y,
        xlim=(0, df[x].max()),
        ylim=(0, df[y].max()),
    )
    
    # main plot
    g.plot_joint(
        main_plot_function,
        data=df.sample(frac=1),
        hue=hue,
        palette=palette,
        **kwargs,
    )
    
    # marginal hist plot
    g.plot_marginals(
        sns.histplot,
        data=df,
        hue=marginal_hue,
        element='step' if use_marg_hue else 'bars',
        fill=False,
        bins=100,
        palette=palette,
        **marginal_kwargs,
    )

    g.fig.suptitle(title, fontsize=12)
    # workaround for patchworklib
    g._figsize = g.fig.get_size_inches()

    # x threshold
    for t, t_def in zip(x_threshold, (0, np.inf)):
        if t != t_def:
            g.ax_joint.axvline(x=t, color='red')
            g.ax_marg_x.axvline(x=t, color='red')

    # y threshold
    for t, t_def in zip(y_threshold, (0, np.inf)):
        if t != t_def:
            g.ax_joint.axhline(y=t, color='red')
            g.ax_marg_y.axhline(y=t, color='red')

    # Suppress automatic showing
    if not show:
        import matplotlib.pyplot as plt
        plt.close(g.fig)

    if return_df:
        return g, df
    return g

def plot_mt_vs_nuclear_fraction(adata,
                                obs_covariate: str,
                               obs_mt: str,
                               obs_nuclear: str):
    """Plot changes of MT and nuclear fraction across covariates, e.i. braak stage"""
    
    if obs_nuclear not in adata.obs:
        nucl_gene_set = [
            "MALAT1", "NEAT1", "FTX", "FOXP1", "RBMS3", "ZBTB20", "LRMDA", "PBX1",
            "ITPR2", "AUTS2", "TTC28", "BNC2", "EXOC4", "RORA", "PRKG1", "ARID1B",
            "PARD3B", "GPHN", "N4BP2L2", "PKHD1L1", "EXOC6B", "FBXL7", "MED13L",
            "TBC1D5", "IMMP2L", "SYNE1", "RERE", "MBD5", "EXT1", "WWOX"
        ]
        get_qc_metrics(adata_filtered, nucl_gene_set, obs_nuclear, normlog=True)
    
    if obs_mt not in adata.obs:
        adata.var["mt"] = adata.var_names.str.startswith("MT-")
        mt_gene_index = np.where(adata.var["mt"])[0]
        adata.obs[obs_mt] = np.array(adata.X[:, mt_gene_index].sum(axis=1)) / np.array(adata.X.sum(axis=1))
    
    df = adata.obs[[obs_covariate, obs_mt, obs_nuclear]].copy()
    df.columns = ['Covariate', 'percent_mt', 'nuclear']
    
    # Nuclear fraction
    plt.subplot(1, 2, 1)
    sns.boxplot(x='Covariate', y='percent_mt', data=df, whis=1.5, showfliers=False, boxprops={'facecolor':'None'})
    sns.stripplot(x='Covariate', y='percent_mt', data=df, color='black', jitter=0.25, size=2.5, alpha=0.5)
    rho_mt, pval_mt = spearmanr(df['Covariate'].astype(float), df['percent_mt'])
    plt.title(f'Spearman r={rho_mt:.2f}, p={pval_mt:.2e}')
    plt.xlabel('Covariate')
    plt.ylabel('MT fraction (%)')
    
    # Nuclear fraction
    plt.subplot(1, 2, 2)
    sns.boxplot(x='Covariate', y='nuclear', data=df, whis=1.5, showfliers=False, boxprops={'facecolor':'None'})
    sns.stripplot(x='Covariate', y='nuclear', data=df, color='black', jitter=0.25, size=2.5, alpha=0.5)
    rho_nuclear, pval_nuclear = spearmanr(df['Covariate'].astype(float), df['nuclear'])
    plt.title(f'Spearman r={rho_nuclear:.2f}, p={pval_nuclear:.2e}')
    plt.xlabel('Braak stage')
    plt.ylabel('Nuclear fraction')
    
    plt.tight_layout()
    plt.show()

def plot_scattered_boxplot_mito(adata, count_col = 'total_counts_mt', sample_col='sample_id', group_col='braak_stage', 
                               figsize=(6, 5), save_path=None):
    """
    Create scattered boxplot for logit-transformed mitochondrial fraction by condition.
    
    Parameters:
    -----------
    adata : AnnData object
        Single cell data object
    count_col: str
        Column name in .obs for counts to consider (either mitochondrial or nuclear)
    sample_col : str
        Column name for sample identification
    group_col : str  
        Column name for grouping (e.g., 'braak_stage')
    figsize : tuple
        Figure size (width, height)
    save_path : str, optional
        Path to save the figure
        
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    if 'mt' in count_col:
        name = 'Mitochondrial'
    else:
        name = 'Nuclear'
    
    # Aggregate data at sample level
    sample_data = adata.obs.groupby(sample_col).agg({
        count_col: 'sum',
        'total_counts': 'sum', 
        group_col: 'first'  # assuming all cells from same sample have same condition
    }).reset_index()
    
    # Calculate logit transformation at sample level
    sample_data['logit'] = logit((sample_data[count_col] + 1) / 
                                     (sample_data['total_counts'] + 2))
    
    # Define colors: blue for braak_stage 0,1,2 and orange for others
    def get_color(braak_stage):
        return '#3C3CDD' if braak_stage in [0, 1, 2] else '#FFAE1C'
    
    sample_data['color'] = sample_data[group_col].apply(get_color)
    
    # Define custom tick positions and labels for logit scale
    tick_positions = logit(np.array([0.001, 0.003, 0.01, 0.03, 0.1, 0.5]))
    tick_labels = ["0.1%", "0.3%", "1%", "3%", "10%", "50%"]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get unique groups and sort them
    groups = sorted(sample_data[group_col].unique())
    
    # Prepare data for boxplot
    box_data = []
    colors_for_groups = []
    
    for group in groups:
        group_data = sample_data[sample_data[group_col] == group]
        box_data.append(group_data['logit'].values)
        # Use the most common color for this group for the box
        colors_for_groups.append(group_data['color'].iloc[0])
    
    # Create boxplot
    bp = ax.boxplot(box_data, positions=range(len(groups)), 
                    patch_artist=True, showfliers=False,
                    boxprops=dict(alpha=0.6),
                    medianprops=dict(color='black', linewidth=2))
    
    # Color the boxes based on the group
    for patch, color in zip(bp['boxes'], colors_for_groups):
        patch.set_facecolor(color)
        patch.set_alpha(0.3)  # Make boxes semi-transparent
    
    # Add scattered dots for individual samples
    for i, group in enumerate(groups):
        group_data = sample_data[sample_data[group_col] == group]
        
        # Add some jitter to x-axis for better visualization
        n_samples = len(group_data)
        if n_samples == 1:
            x_positions = [i]
        else:
            # Create jitter around the group position
            jitter_width = 0.15
            x_positions = np.random.uniform(i - jitter_width, i + jitter_width, n_samples)
        
        # Plot individual sample points with their colors
        for j, (_, row) in enumerate(group_data.iterrows()):
            ax.scatter(x_positions[j], row['logit'], 
                      c=row['color'], s=60, alpha=0.8,
                      edgecolors='white', linewidth=1, zorder=3)
    
    # Customize the plot
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f'Braak {g}' for g in groups], rotation=45)
    ax.set_xlabel('Braak Stage', fontsize=12)
    ax.set_ylabel(f'{name} fraction (%)', fontsize=12)
    
    # Apply custom y-axis ticks and labels for logit scale
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    
    # Set title
    ax.set_title(f'{name} Fraction by Braak Stage\n(Sample-level Analysis)', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    # Print summary statistics
    print("\nSample counts by group:")
    print(sample_data[group_col].value_counts().sort_index())
    
    print("\nSummary statistics by group:")
    for group in groups:
        group_data = sample_data[sample_data[group_col] == group]['logit']
        print(f"Braak {group}: n={len(group_data)}, mean={group_data.mean():.3f}, std={group_data.std():.3f}")
    
    return fig, ax, sample_data


def simple_logit_boxplot(adata, 
                         count_col = 'total_counts_mt',
                         sample_col='sample_id', 
                         group_col='braak_stage', 
                         hue_col='area', 
                         colors=None, 
                         figsize=(12, 5), 
                         save_path=None):
    """
    Simple function that exactly replicates your working seaborn approach
    but with logit transformation and custom y-axis labels.
    """
    if 'mt' in count_col:
        name = 'Mitochondrial'
    else:
        name = 'Nuclear'
    
    # Aggregate data at sample level
    sample_data = adata.obs.groupby(sample_col).agg({
        count_col: 'sum',
        'total_counts': 'sum', 
        group_col: 'first',
        hue_col: 'first'
    }).reset_index()
    
    # Calculate logit transformation
    sample_data['logit_mito'] = logit((sample_data[count_col] + 1) / 
                                     (sample_data['total_counts'] + 2))
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Your exact working code:
    sns.boxplot(data=sample_data, x=group_col, y='logit_mito', hue=hue_col, 
                palette=colors, ax=ax, showfliers=False)
    
    # Add scattered dots
    sns.stripplot(data=sample_data, x=group_col, y='logit_mito', hue=hue_col,
                 palette=colors, size=6, alpha=0.8, ax=ax, dodge=True,
                 edgecolor='white', linewidth=0.5)
    
    # Custom y-axis with percentage labels
    tick_positions = logit(np.array([0.001, 0.003, 0.01, 0.03, 0.1, 0.5]))
    tick_labels = ["0.1%", "0.3%", "1%", "3%", "10%", "50%"]
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)
    
    # Labels and styling to match previous boxplot style
    ax.set_xlabel("Braak Stage", fontsize=12)
    ax.set_ylabel(f"{name} fraction (%)", fontsize=12)
    
    # Add grid for better readability (like previous plots)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    
    # Style legend to match previous plots
    legend = ax.get_legend()
    if legend:
        legend.set_title('Area')
        legend.set_bbox_to_anchor((1.05, 1))
        legend.set_loc('upper left')
        # Make legend frame more subtle
        legend.get_frame().set_alpha(0.9)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_edgecolor('gray')
        legend.get_frame().set_linewidth(0.5)
    
    # Set title to match previous style
    ax.set_title(f'{name} Fraction by Braak Stage and Brain Area\n(Sample-level Analysis)', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Style the plot to match previous scattered boxplots
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('gray')
    ax.spines['bottom'].set_color('gray')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    # Print summary statistics like previous functions
    print("\nSample counts by group and area:")
    crosstab = pd.crosstab(sample_data[group_col], sample_data[hue_col], margins=True)
    print(crosstab)
    
    return fig, ax, sample_data

# Even simpler - just the data prep function
def prepare_sample_data_for_plotting(adata, sample_col='sample_id', group_col='braak_stage', hue_col='area'):
    """
    Just prepare the sample-level data so you can use your exact seaborn code.
    """
    
    # Aggregate data at sample level
    sample_data = adata.obs.groupby(sample_col).agg({
        'total_counts_mt': 'sum',
        'total_counts': 'sum', 
        group_col: 'first',
        hue_col: 'first'
    }).reset_index()
    
    # Calculate logit transformation
    sample_data['logit_mito'] = logit((sample_data['total_counts_mt'] + 1) / 
                                     (sample_data['total_counts'] + 2))
    
    return sample_data