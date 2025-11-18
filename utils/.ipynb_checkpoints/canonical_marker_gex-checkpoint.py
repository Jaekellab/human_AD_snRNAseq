import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LinearRegression
from scipy.stats import t
import scanpy as sc
import warnings
warnings.filterwarnings('ignore')

def plot_gene_expression_by_area(adata, gene_name, cell_type, 
                                x_col='Age', area_col='area', 
                                tech_sample_col='tech_sample',
                                cell_type_col='cell_type',
                                legend_loc = False,
                                figsize=(6, 5), 
                                 save_path=None):
    """
    Plot gene expression vs Age/Braak stage with separate lines per brain area.
    Similar to the R ggplot but with areas instead of cell types.
    
    Parameters:
    -----------
    adata : AnnData object
        Single-cell data object
    gene_name : str
        Name of the gene to plot (must be in adata.var_names)
    cell_type : str | None
        Cell type to analyze. If None, all cell types are considered.
    x_col : str
        Column name for x-axis (e.g., 'Age', 'braak_stage')
    area_col : str
        Column name for brain area
    tech_sample_col : str
        Column name for technical sample
    cell_type_col : str
        Column name for cell type
    figsize : tuple
        Figure size (width, height)
    save_fig : str, optional
        Where to save fig (.png) or not
    """
    
    # Check if gene exists
    if gene_name not in adata.var_names:
        print(f"Error: Gene '{gene_name}' not found in data")
        return
    
    # Filter for specific cell type
    if cell_type:
        cell_mask = adata.obs[cell_type_col] == cell_type
    else:
        cell_mask = np.ones(len(adata), dtype=bool)  # include all cells
    if not np.any(cell_mask):
        print(f"Error: Cell type '{cell_type}' not found in data")
        return
    
    # Get gene expression data
    gene_idx = adata.var_names.get_loc(gene_name)
    if hasattr(adata.X, 'toarray'):
        expression = adata.X[cell_mask, gene_idx].toarray().flatten()
    else:
        expression = adata.X[cell_mask, gene_idx]
    
    # Get corresponding metadata
    cell_obs = adata.obs[cell_mask].copy()
    cell_obs['expression'] = expression
    
    # Check if required columns exist
    required_cols = [x_col, area_col, tech_sample_col]
    missing_cols = [col for col in required_cols if col not in cell_obs.columns]
    if missing_cols:
        print(f"Error: Missing columns: {missing_cols}")
        return
    
    # Group by tech_sample and calculate mean expression
    sample_summary = cell_obs.groupby(tech_sample_col).agg({
        'expression': 'mean',
        x_col: 'first',
        area_col: 'first',
        cell_type_col: 'first'
    }).reset_index()
    
    # Create the plot
    plt.figure(figsize=figsize)
    
    # Get unique areas and use colors from adata.uns['area_colors']
    areas = sample_summary[area_col].unique()
    if f'{area_col}_colors' in adata.uns:
        area_color_map = adata.uns[f'{area_col}_colors']
        # Filter to only include areas present in the data
        area_color_map = {area: area_color_map[i] for i, area in enumerate(areas)}
        # Add default colors for any missing areas
        missing_areas = [area for area in areas if area not in area_color_map]
        if missing_areas:
            default_colors = sns.color_palette("Set2", len(missing_areas))
            for i, area in enumerate(missing_areas):
                area_color_map[area] = default_colors[i]
    else:
        # Fallback to default colors if area_colors not found
        colors = sns.color_palette("Set2", len(areas))
        area_color_map = dict(zip(areas, colors))
        print("Warning: 'area_colors' not found in adata.uns, using default colors")
    
    # Plot points and trend lines for each area
    for area in areas:
        area_data = sample_summary[sample_summary[area_col] == area]

        if pd.api.types.is_categorical_dtype(area_data[x_col]):
            area_data[x_col] = area_data[x_col].cat.as_ordered()
        
        # Scatter plot
        plt.scatter(area_data[x_col], area_data['expression'], 
                   c=[area_color_map[area]], label=area, 
                   s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        # Add trend line if we have enough points
        if len(area_data) > 1:
            # Sort by x-axis for smooth line
            area_data_sorted = area_data.sort_values(x_col)
            
            # Fit linear regression
            X = area_data_sorted[x_col].values.reshape(-1, 1)
            y = area_data_sorted['expression'].values
            
            # Check if there's variation in x values
            if len(np.unique(X)) > 1:
                reg = LinearRegression().fit(X, y)
                
                # Generate prediction line
                x_range = np.linspace(area_data[x_col].min(), area_data[x_col].max(), 100)
                y_pred = reg.predict(x_range.reshape(-1, 1))
                
                # Plot trend line
                plt.plot(x_range, y_pred, color=area_color_map[area], 
                        linewidth=2, alpha=0.8)
                
                # Calculate R-squared and p-value
                r_squared = reg.score(X, y)
                
                # Calculate p-value for the slope
                if len(area_data) > 2:  # Need at least 3 points for meaningful p-value
                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                        area_data[x_col], area_data['expression'])
    
    # Customize plot
    plt.xlabel(x_col, fontsize=12, fontweight='bold')
    plt.ylabel('Norm. Gene Expression', fontsize=12, fontweight='bold')
    plt.title(f'{gene_name} - {cell_type}', fontsize=14, fontweight='bold', style='italic')
    
    # Customize legend
    if legend_loc:
        plt.legend(title='Brain Area', title_fontsize=12, fontsize=10, 
                  loc='upper right', frameon=True, fancybox=True, shadow=True)
    
    # Grid and styling
    plt.grid(True, alpha=0.3)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        save_path = Path(save_path, f'{cell_type}_{gene_name}_expr_by_{area_col}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()
    
    # Print summary statistics
    print(f"\nSummary for {gene_name} in {cell_type}:")
    print(f"Total samples: {len(sample_summary)}")
    print(f"Brain areas: {len(areas)}")
    print(f"Samples per area:")
    for area in areas:
        n_samples = len(sample_summary[sample_summary[area_col] == area])
        print(f"  {area}: {n_samples}")
    
    # Overall correlation
    overall_corr = np.corrcoef(sample_summary[x_col], sample_summary['expression'])[0, 1]
    print(f"Overall correlation: {overall_corr:.3f}")
    
    return sample_summary




def plot_marker_gene_expression_comparison(adata, marker_genes_dict, 
                                         condition_col='condition', 
                                         celltype_col='cell_type', 
                                         region_col='region',
                                         ad_label='AD', ctrl_label='Ctrl',
                                         figsize_per_subplot=(4, 3),
                                         save_path=None, dpi=300):
    """
    Create a comprehensive box plot comparing marker gene expression between AD and Control
    across different cell types and cortex regions.
    
    Parameters:
    -----------
    adata : AnnData
        Annotated data object containing gene expression data
    marker_genes_dict : dict
        Dictionary with cell types as keys and list of marker genes as values
        Example: {'Neurons': ['SYN1', 'MAP2'], 'Astrocytes': ['GFAP', 'AQP4']}
    condition_col : str
        Column name in adata.obs containing condition information (AD/Ctrl)
    celltype_col : str
        Column name in adata.obs containing cell type information
    region_col : str
        Column name in adata.obs containing cortex region information
    ad_label : str
        Label for AD samples in the condition column
    ctrl_label : str
        Label for control samples in the condition column
    figsize_per_subplot : tuple
        Size of each individual subplot (width, height)
    save_path : str, optional
        Path to save the figure
    dpi : int
        Resolution for saving the figure
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The generated figure object
    stats_df : pandas.DataFrame
        Statistical comparison results
    """
    
    # Get unique cell types and regions
    cell_types = list(marker_genes_dict.keys())
    regions = adata.obs[region_col].unique()
    
    # Filter for available cell types and regions in the data
    available_cell_types = [ct for ct in cell_types if ct in np.unique(adata.obs[celltype_col])]
    available_regions = [reg for reg in regions if reg in adata.obs[region_col].unique()]
    
    if not available_cell_types:
        raise ValueError("No specified cell types found in the data")
    if not available_regions:
        raise ValueError("No regions found in the data")
    
    # Calculate figure size
    n_celltypes = len(available_cell_types)
    print('Nr. cell types: ', n_celltypes)
    n_regions = len(available_regions)
    fig_width = n_celltypes * figsize_per_subplot[0]
    fig_height = n_regions * figsize_per_subplot[1]
    
    # Create the subplot grid
    fig, axes = plt.subplots(n_regions, n_celltypes, 
                            figsize=(fig_width, fig_height),
                            squeeze=False)
    
    # Initialize statistics storage
    stats_results = []
    
    # Set up color palette - modern colors similar to the example
    if f'{condition_col}_colors' in adata.uns:
        colors = adata.uns[f'{condition_col}_colors']
    else: 
        colors = ['#dc2626', '#2dd4bf']  # Red for AD, Teal for Control
    
    # Iterate through regions (rows) and cell types (columns)
    for row_idx, region in enumerate(available_regions):
        for col_idx, cell_type in enumerate(available_cell_types):
            ax = axes[row_idx, col_idx]
            
            # Get marker genes for this cell type
            marker_genes = marker_genes_dict[cell_type]
            
            # Filter available genes
            available_genes = [gene for gene in marker_genes if gene in adata.var_names]
            
            if not available_genes:
                ax.text(0.5, 0.5, f'No genes\navailable', 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=10, alpha=0.7)
                ax.set_title(f'{cell_type}\n{region}', fontsize=10, pad=20)
                continue
            
            # Filter data for this region and cell type
            mask = (adata.obs[region_col] == region) & (adata.obs[celltype_col] == cell_type)
            
            if mask.sum() == 0:
                ax.text(0.5, 0.5, f'No cells\navailable', 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=10, alpha=0.7)
                ax.set_title(f'{cell_type}\n{region}', fontsize=10, pad=20)
                continue
            
            # Get subset of data
            subset_adata = adata[mask]
            
            # Calculate mean expression across marker genes for each cell
            gene_indices = [i for i, gene in enumerate(adata.var_names) if gene in available_genes]
            
            if len(gene_indices) == 0:
                continue
                
            # Get expression data (handle both sparse and dense matrices)
            if hasattr(subset_adata.X, 'toarray'):
                expr_data = subset_adata.X.toarray()
            else:
                expr_data = subset_adata.X
                
            # Calculate mean expression across selected genes
            mean_expr = np.mean(expr_data[:, gene_indices], axis=1)
            
            # Create dataframe for plotting
            plot_df = pd.DataFrame({
                'expression': mean_expr,
                'condition': subset_adata.obs[condition_col].values,
                'cell_type': cell_type,
                'region': region
            })
            
            # Filter for AD and Control samples
            plot_df = plot_df[plot_df['condition'].isin([ad_label, ctrl_label])]
            
            if len(plot_df) == 0:
                ax.text(0.5, 0.5, f'No AD/Ctrl\ncells found', 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=10, alpha=0.7)
                ax.set_title(f'{cell_type}\n{region}', fontsize=10, pad=20)
                continue
            
            # Create box plot with modern styling
            try:
                # Create box plot
                box_plot = sns.boxplot(data=plot_df, x='condition', y='expression', 
                                     palette=colors, ax=ax, 
                                     width=0.6, linewidth=1.5,
                                     boxprops=dict(alpha=0.7),
                                     whiskerprops=dict(linewidth=1.5),
                                     capprops=dict(linewidth=1.5),
                                     medianprops=dict(linewidth=2, color='white'),
                                     flierprops=dict(marker='o', markersize=3, alpha=0.7))
                
                # Add individual points with strip plot
                sns.stripplot(data=plot_df, x='condition', y='expression', 
                             palette=colors, size=4, alpha=0.8, ax=ax,
                             dodge=True, jitter=0.3)
                
                # Statistical test
                ad_data = plot_df[plot_df['condition'] == ad_label]['expression']
                ctrl_data = plot_df[plot_df['condition'] == ctrl_label]['expression']
                
                if len(ad_data) > 0 and len(ctrl_data) > 0:
                    # Perform Mann-Whitney U test
                    statistic, p_value = stats.mannwhitneyu(ad_data, ctrl_data, 
                                                          alternative='two-sided')
                    
                    # Calculate effect size (Cohen's d)
                    pooled_std = np.sqrt(((len(ad_data) - 1) * np.var(ad_data, ddof=1) + 
                                         (len(ctrl_data) - 1) * np.var(ctrl_data, ddof=1)) / 
                                        (len(ad_data) + len(ctrl_data) - 2))
                    cohens_d = (np.mean(ad_data) - np.mean(ctrl_data)) / pooled_std if pooled_std > 0 else 0
                    
                    # Store statistics
                    stats_results.append({
                        'cell_type': cell_type,
                        'region': region,
                        'genes': ', '.join(available_genes),
                        'n_ad': len(ad_data),
                        'n_ctrl': len(ctrl_data),
                        'mean_ad': np.mean(ad_data),
                        'mean_ctrl': np.mean(ctrl_data),
                        'p_value': p_value,
                        'cohens_d': cohens_d,
                        'significant': p_value < 0.05
                    })
                    
                    # Add significance annotation
                    if p_value < 0.001:
                        sig_text = '***'
                    elif p_value < 0.01:
                        sig_text = '**'
                    elif p_value < 0.05:
                        sig_text = '*'
                    else:
                        sig_text = 'ns'
                    
                    # Add mean values to x-axis labels
                    ad_mean = np.mean(ad_data)
                    ctrl_mean = np.mean(ctrl_data)
                    
                    # Update x-axis labels with mean values
                    current_labels = [t.get_text() for t in ax.get_xticklabels()]
                    new_labels = []
                    for label in current_labels:
                        if label == ad_label:
                            new_labels.append(f'{ad_label}\n(μ={ad_mean:.2f})')
                        elif label == ctrl_label:
                            new_labels.append(f'{ctrl_label}\n(μ={ctrl_mean:.2f})')
                        else:
                            new_labels.append(label)
                    
                    ax.set_xticklabels(new_labels, fontsize=8)
                
                    # Add significance text
                    ax.text(0.5, 0.95, sig_text, transform=ax.transAxes, 
                           ha='center', va='top', fontsize=12, fontweight='bold')

                # Apply modern styling similar to the example
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_linewidth(1.5)
                ax.spines['bottom'].set_linewidth(1.5)
                
                # Add subtle grid
                ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
                ax.set_axisbelow(True)
                
                # Customize subplot
                # Only show cell type in title for top row, region for all rows
                if row_idx == 0:
                    ax.set_title(f'{cell_type}', fontsize=11, fontweight='bold', pad=20)
                else:
                    ax.set_title('', fontsize=11, pad=20)
                
                # Add region label on the left side for each row
                if col_idx == 0:
                    ax.text(-0.15, 0.5, region, transform=ax.transAxes, 
                           rotation=90, ha='center', va='center', 
                           fontsize=11, fontweight='bold')
            
                
            except Exception as e:
                print(f"Error plotting {cell_type} in {region}: {str(e)}")
                ax.text(0.5, 0.5, f'Plotting\nerror', 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=10, alpha=0.7)
                
                # Apply styling even for error cases
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_linewidth(1.5)
                ax.spines['bottom'].set_linewidth(1.5)
            
            # Set axis labels and formatting
            ax.set_xlabel('')
            ax.set_ylabel('Mean Expression' if col_idx == 0 else '', fontsize=9, fontweight='bold')
            ax.tick_params(axis='both', which='major', labelsize=8)
            
            # Format y-axis to show appropriate decimal places
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))
            
            # Always show x-axis labels with mean values for all rows
            if len(ax.get_xticklabels()) > 0:
                ax.tick_params(axis='x', rotation=0)
    
    # Adjust layout
    plt.tight_layout()
    
    # Adjust spacing to accommodate region labels
    plt.subplots_adjust(left=0.1)

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"Figure saved to: {save_path}")
    
    # Create statistics dataframe
    stats_df = pd.DataFrame(stats_results)
    
    return fig, stats_df