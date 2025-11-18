import pandas as pd
import numpy as np
from scipy.stats import fisher_exact, chi2_contingency
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
import warnings

"""
Test cell type proportion differences across categories.

Fisher's exact test for 2 categories
Chi-square test for >2 categories (with pairwise Fisher's as option)
Auto-detection based on number of categories


Multiple test types:

fishers_exact_test_pairwise(): For pairwise comparisons
chi2_test_multiple_categories(): For overall differences across >2 categories
run_statistical_tests(): Main function that handles any covariate


Comprehensive analysis function: run_comprehensive_analysis() can test multiple covariates at once
"""


import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad
from scipy.stats import fisher_exact, chi2_contingency
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
import warnings

def extract_cell_type_counts(adata, cell_type_col='cell_type', covariate_cols=None, 
                           sample_col=None, aggregate_by_sample=True):
    """
    Extract cell type counts from AnnData object
    
    Parameters:
    adata: AnnData object
    cell_type_col: str, column name for cell types in adata.obs
    covariate_cols: list, covariate columns to include
    sample_col: str, column for sample ID (for aggregating by sample)
    aggregate_by_sample: bool, whether to aggregate by sample or use all cells
    
    Returns:
    DataFrame with cell type counts per covariate combination
    """
    if covariate_cols is None:
        covariate_cols = [col for col in adata.obs.columns if col != cell_type_col]
    
    # Create DataFrame from obs
    obs_df = adata.obs.copy()
    
    if sample_col and aggregate_by_sample:
        # Aggregate by sample
        groupby_cols = [sample_col, cell_type_col] + covariate_cols
        counts_df = obs_df.groupby(groupby_cols).size().reset_index(name='count')
        
        # Pivot to get counts per cell type per sample
        pivot_df = counts_df.pivot_table(
            index=[sample_col] + covariate_cols,
            columns=cell_type_col,
            values='count',
            fill_value=0
        ).reset_index()
        
        # Melt back to long format
        result_df = pivot_df.melt(
            id_vars=[sample_col] + covariate_cols,
            var_name=cell_type_col,
            value_name='count'
        )
        
    else:
        # Use all cells directly
        if sample_col is None:
            # Create pseudo-sample grouping by covariates
            groupby_cols = [cell_type_col] + covariate_cols
            result_df = obs_df.groupby(groupby_cols).size().reset_index(name='count')
        else:
            groupby_cols = [sample_col, cell_type_col] + covariate_cols
            result_df = obs_df.groupby(groupby_cols).size().reset_index(name='count')
    
    return result_df

def create_contingency_table(df, 
                             cell_type, 
                             cell_type_col,
                             covariate_col, categories=None, 
                           sample_col=None):
    """
    Create contingency table for a specific cell type across covariate categories
    
    Parameters:
    df: DataFrame with cell type counts
    cell_type: str, specific cell type to analyze
    covariate_col: str, column name for the covariate
    categories: list, specific categories to include (if None, use all)
    sample_col: str, sample column for proper aggregation
    
    Returns:
    contingency_table: numpy array
    category_labels: list of category labels
    """
    if categories is None:
        categories = sorted(df[covariate_col].unique())
    
    # Filter for specific categories
    df_filtered = df[df[covariate_col].isin(categories)]
    
    contingency_data = []
    
    for category in categories:
        category_data = df_filtered[df_filtered[covariate_col] == category]
        
        if sample_col:
            # Aggregate by sample first, then sum
            sample_totals = category_data.groupby([sample_col, cell_type_col])['count'].sum().reset_index()
            cell_type_count = sample_totals[sample_totals[cell_type_col] == cell_type]['count'].sum()
            total_count = sample_totals['count'].sum()
        else:
            cell_type_count = category_data[category_data[cell_type_col] == cell_type]['count'].sum()
            total_count = category_data['count'].sum()
        
        other_cells_count = total_count - cell_type_count
        contingency_data.append([cell_type_count, other_cells_count])
    
    return np.array(contingency_data), categories

def fishers_exact_test_pairwise(df, cell_type, cell_type_col, covariate_col, category1, category2, 
                               sample_col=None):
    """
    Run Fisher's exact test for a specific cell type between two categories
    
    Parameters:
    df: DataFrame with cell type counts
    cell_type: str, cell type to test
    covariate_col: str, covariate column name
    category1, category2: str, categories to compare
    sample_col: str, sample column for proper aggregation
    
    Returns:
    dict with test results
    """
    contingency_table, categories = create_contingency_table(
        df, cell_type, cell_type_col, covariate_col, [category1, category2], sample_col
    )
    
    # Check for valid contingency table
    if np.any(contingency_table.sum(axis=1) == 0):
        return {
            'cell_type': cell_type,
            'covariate': covariate_col,
            'categories': f'{category1}_vs_{category2}',
            'category1': category1,
            'category2': category2,
            'contingency_table': contingency_table,
            'odds_ratio': np.nan,
            'p_value': np.nan,
            'test_type': 'fishers_exact',
            'warning': 'Empty category detected'
        }
    
    # Run Fisher's exact test
    try:
        odds_ratio, p_value = fisher_exact(contingency_table)
    except Exception as e:
        return {
            'cell_type': cell_type,
            'covariate': covariate_col,
            'categories': f'{category1}_vs_{category2}',
            'category1': category1,
            'category2': category2,
            'contingency_table': contingency_table,
            'odds_ratio': np.nan,
            'p_value': np.nan,
            'test_type': 'fishers_exact',
            'warning': str(e)
        }
    
    return {
        'cell_type': cell_type,
        'covariate': covariate_col,
        'categories': f'{category1}_vs_{category2}',
        'category1': category1,
        'category2': category2,
        'contingency_table': contingency_table,
        'odds_ratio': odds_ratio,
        'p_value': p_value,
        'test_type': 'fishers_exact'
    }

def chi2_test_multiple_categories(df, cell_type, cell_type_col, covariate_col, categories=None, 
                                sample_col=None):
    """
    Run chi-square test for a specific cell type across multiple categories
    
    Parameters:
    df: DataFrame with cell type counts
    cell_type: str, cell type to test
    covariate_col: str, covariate column name
    categories: list, categories to include (if None, use all)
    sample_col: str, sample column for proper aggregation
    
    Returns:
    dict with test results
    """
    contingency_table, category_labels = create_contingency_table(
        df, cell_type, cell_type_col, covariate_col, categories, sample_col
    )
    
    # Check for valid contingency table
    if np.any(contingency_table.sum(axis=1) == 0) or contingency_table.shape[0] < 2:
        return {
            'cell_type': cell_type,
            'covariate': covariate_col,
            'categories': '_vs_'.join(category_labels),
            'category_labels': category_labels,
            'contingency_table': contingency_table,
            'chi2_statistic': np.nan,
            'p_value': np.nan,
            'degrees_of_freedom': np.nan,
            'test_type': 'chi2',
            'warning': 'Invalid contingency table'
        }
    
    # Run chi-square test
    try:
        chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
    except Exception as e:
        return {
            'cell_type': cell_type,
            'covariate': covariate_col,
            'categories': '_vs_'.join(category_labels),
            'category_labels': category_labels,
            'contingency_table': contingency_table,
            'chi2_statistic': np.nan,
            'p_value': np.nan,
            'degrees_of_freedom': np.nan,
            'test_type': 'chi2',
            'warning': str(e)
        }
    
    return {
        'cell_type': cell_type,
        'covariate': covariate_col,
        'categories': '_vs_'.join(category_labels),
        'category_labels': category_labels,
        'contingency_table': contingency_table,
        'chi2_statistic': chi2_stat,
        'p_value': p_value,
        'degrees_of_freedom': dof,
        'expected_frequencies': expected,
        'test_type': 'chi2'
    }

def run_statistical_tests_anndata(adata, 
                                  covariate_col, 
                                  cell_type_col='cell_type', 
                                 test_type='auto', categories=None, cell_types=None, 
                                 sample_col=None, pairwise_only=False):
    """
    Run statistical tests for cell type proportions from AnnData object
    
    Parameters:
    adata: AnnData object
    covariate_col: str, name of covariate column to test
    cell_type_col: str, name of cell type column
    test_type: str, 'fishers', 'chi2', or 'auto'
    categories: list, specific categories to test (if None, use all)
    cell_types: list, specific cell types to test (if None, use all)
    sample_col: str, sample column for proper aggregation
    pairwise_only: bool, if True, only do pairwise comparisons
    
    Returns:
    results_df: DataFrame with test results
    detailed_results: list of detailed result dictionaries
    """
    
    # Extract cell type counts
    covariate_cols = [covariate_col] if isinstance(covariate_col, str) else covariate_col
    counts_df = extract_cell_type_counts(
        adata, cell_type_col, covariate_cols, sample_col
    )
    
    if cell_types is None:
        cell_types = counts_df[cell_type_col].unique()
    
    if categories is None:
        categories = sorted(counts_df[covariate_col].unique())
    
    results = []
    n_categories = len(categories)
    
    print(f"Testing {len(cell_types)} cell types across {n_categories} categories in '{covariate_col}'")
    print(f"Total cells: {len(adata)}")
    if sample_col:
        print(f"Total samples: {adata.obs[sample_col].nunique()}")
    
    # Determine test strategy
    if test_type == 'auto':
        if n_categories == 2:
            test_strategy = 'fishers'
        elif n_categories > 2 and not pairwise_only:
            test_strategy = 'both'
        else:
            test_strategy = 'fishers_pairwise'
    elif test_type == 'fishers' and n_categories > 2:
        test_strategy = 'fishers_pairwise'
    else:
        test_strategy = test_type
    
    # Run tests based on strategy
    for cell_type in cell_types:
        if test_strategy in ['fishers', 'fishers_pairwise']:
            # Pairwise Fisher's exact tests
            for cat1, cat2 in combinations(categories, 2):
                result = fishers_exact_test_pairwise(
                    counts_df, cell_type, cell_type_col, covariate_col, cat1, cat2, sample_col
                )
                results.append(result)
        
        if test_strategy in ['chi2', 'both']:
            # Chi-square test for overall difference
            result = chi2_test_multiple_categories(
                counts_df, cell_type, cell_type_col, covariate_col, categories, sample_col
            )
            results.append(result)
    
    # Convert to DataFrame
    results_df = pd.DataFrame([
        {
            'cell_type': r['cell_type'],
            'covariate': r['covariate'],
            'categories': r['categories'],
            'test_type': r['test_type'],
            'p_value': r['p_value'],
            'odds_ratio': r.get('odds_ratio', np.nan),
            'chi2_statistic': r.get('chi2_statistic', np.nan),
            'degrees_of_freedom': r.get('degrees_of_freedom', np.nan),
            'warning': r.get('warning', '')
        }
        for r in results
    ])
    
    # Filter out failed tests
    valid_results = results_df[results_df['warning'] == '']
    if len(valid_results) < len(results_df):
        print(f"Warning: {len(results_df) - len(valid_results)} tests failed")
    
    return valid_results, results

def multiple_comparisons_correction(p_values, method='fdr'):
    """Apply multiple comparisons correction"""
    p_values = np.array(p_values)
    
    if method == 'bonferroni':
        return np.minimum(p_values * len(p_values), 1.0)
    elif method == 'fdr':
        # Benjamini-Hochberg FDR correction
        sorted_pvals = np.sort(p_values)
        sorted_indices = np.argsort(p_values)
        m = len(p_values)
        
        corrected_pvals = np.zeros(m)
        for i in range(m-1, -1, -1):
            if i == m-1:
                corrected_pvals[sorted_indices[i]] = sorted_pvals[i]
            else:
                corrected_pvals[sorted_indices[i]] = min(
                    sorted_pvals[i] * m / (i + 1),
                    corrected_pvals[sorted_indices[i+1]]
                )
        
        return corrected_pvals
    else:
        return p_values

def add_significance_testing(results_df, correction_method='fdr', alpha=0.05):
    """Add multiple comparisons correction and significance indicators"""
    results_df = results_df.copy()
    
    # Apply correction
    corrected_pvals = multiple_comparisons_correction(
        results_df['p_value'].values, correction_method
    )
    results_df[f'p_value_{correction_method}'] = corrected_pvals
    
    # Add significance indicators
    results_df['significant_raw'] = results_df['p_value'] < alpha
    results_df[f'significant_{correction_method}'] = corrected_pvals < alpha
    
    return results_df

def visualize_results(results_df, covariate_col, save_plots=False):
    """Create visualizations of the statistical test results"""
    test_types = results_df['test_type'].unique()
    n_test_types = len(test_types)
    
    fig, axes = plt.subplots(2, max(2, n_test_types), figsize=(5*max(2, n_test_types), 10))
    if n_test_types == 1:
        axes = axes.reshape(-1, 1)
    
    sig_col = f'significant_fdr' if f'significant_fdr' in results_df.columns else 'significant_raw'
    
    plot_idx = 0
    
    for test_type in test_types:
        test_data = results_df[results_df['test_type'] == test_type]
        
        if len(test_data) == 0:
            continue
        
        # Heatmap of p-values
        try:
            pivot_data = test_data.pivot_table(
                index='cell_type', 
                columns='categories', 
                values='p_value_fdr' if 'p_value_fdr' in test_data.columns else 'p_value'
            )
            
            sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdYlBu_r', 
                       center=0.05, ax=axes[0, plot_idx])
            axes[0, plot_idx].set_title(f'{test_type.replace("_", " ").title()} - {covariate_col}')
        except Exception as e:
            axes[0, plot_idx].text(0.5, 0.5, f'Cannot create heatmap:\n{str(e)}', 
                                  ha='center', va='center', transform=axes[0, plot_idx].transAxes)
        
        # Scatter plot of test statistics
        if 'fishers' in test_type:
            y_vals = np.log(test_data['odds_ratio'].replace([np.inf, -np.inf], np.nan))
            y_label = 'Log Odds Ratio'
        else:
            y_vals = test_data['chi2_statistic']
            y_label = 'Chi-square Statistic'
        
        scatter = axes[1, plot_idx].scatter(
            range(len(test_data)), y_vals,
            c=test_data[sig_col], cmap='RdYlBu', alpha=0.7, s=50
        )
        
        if 'fishers' in test_type:
            axes[1, plot_idx].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        axes[1, plot_idx].set_xlabel('Test Index')
        axes[1, plot_idx].set_ylabel(y_label)
        axes[1, plot_idx].set_title(f'{test_type.replace("_", " ").title()} Statistics')
        
        plot_idx += 1
    
    # Remove empty subplots
    for i in range(plot_idx, axes.shape[1]):
        fig.delaxes(axes[0, i])
        fig.delaxes(axes[1, i])
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig(f'statistical_tests_{covariate_col}.png', dpi=300, bbox_inches='tight')
    
    plt.show()

def run_comprehensive_analysis_anndata(adata, covariate_cols, cell_type_col='cell_type',
                                      sample_col=None, test_type='auto', 
                                      correction_method='fdr', alpha=0.05, 
                                      save_results=True, save_plots=False):
    """
    Run comprehensive statistical analysis for AnnData object
    
    Parameters:
    adata: AnnData object
    covariate_cols: list of str, covariate columns to test
    cell_type_col: str, cell type column name
    sample_col: str, sample column for proper aggregation
    test_type: str, type of test to run
    correction_method: str, multiple comparison correction method
    alpha: float, significance threshold
    save_results: bool, whether to save results to CSV
    save_plots: bool, whether to save plots
    
    Returns:
    dict with results for each covariate
    """
    all_results = {}
    
    print(f"Analyzing AnnData object with {len(adata)} cells and {len(adata.var)} genes")
    print(f"Available covariates: {list(adata.obs.columns)}")
    
    for covariate_col in covariate_cols:
        print(f"\n{'='*60}")
        print(f"ANALYZING COVARIATE: {covariate_col.upper()}")
        print(f"{'='*60}")
        
        # Check if covariate exists
        if covariate_col not in adata.obs.columns:
            print(f"Warning: Column '{covariate_col}' not found in adata.obs")
            continue
        
        # Show covariate distribution
        print(f"Covariate distribution:")
        print(adata.obs[covariate_col].value_counts())
        
        # Run statistical tests
        results_df, detailed_results = run_statistical_tests_anndata(
            adata, covariate_col, cell_type_col, test_type, sample_col=sample_col
        )
        
        if len(results_df) == 0:
            print(f"No results generated for {covariate_col}")
            continue
        
        # Add significance testing
        results_df = add_significance_testing(results_df, correction_method, alpha)
        
        # Display results
        print(f"\nResults for {covariate_col}:")
        print(f"Total tests performed: {len(results_df)}")
        
        sig_col = f'significant_{correction_method}'
        significant_results = results_df[results_df[sig_col]]
        
        print(f"Significant results ({correction_method} < {alpha}): {len(significant_results)}")
        
        if len(significant_results) > 0:
            print("\nTop significant results:")
            display_cols = ['cell_type', 'categories', 'test_type', 'p_value', 
                           f'p_value_{correction_method}']
            if 'odds_ratio' in significant_results.columns:
                display_cols.append('odds_ratio')
            print(significant_results[display_cols].head(10))
        
        # Visualize results
        visualize_results(results_df, covariate_col, save_plots)
        
        # Save results
        if save_results:
            filename = f'statistical_tests_{covariate_col}.csv'
            results_df.to_csv(filename, index=False)
            print(f"Results saved to '{filename}'")
        
        all_results[covariate_col] = {
            'results_df': results_df,
            'detailed_results': detailed_results
        }
    
    return all_results

# Main execution example
if __name__ == "__main__":
    # Create or load AnnData object
    # adata = sc.read_h5ad('your_data.h5ad')  # Load your actual data
    
    print("AnnData object overview:")
    print(adata)
    print(f"\nObservations (cells): {adata.n_obs}")
    print(f"Variables (genes): {adata.n_vars}")
    print(f"Cell types: {adata.obs['cell_type'].unique()}")
    
    # Define covariates to test
    covariate_cols = ['condition', 'area', 'age', 'sex']
    
    # Run comprehensive analysis
    results = run_comprehensive_analysis_anndata(
        adata, 
        covariate_cols,
        cell_type_col='cell_type',
        sample_col=None,  # Set to your sample column if you have one
        test_type='auto',
        correction_method='fdr',
        alpha=0.05,
        save_results=True,
        save_plots=False
    )
    
    # Summary across all covariates
    print(f"\n{'='*60}")
    print("SUMMARY ACROSS ALL COVARIATES")
    print(f"{'='*60}")
    
    for covariate, result_data in results.items():
        results_df = result_data['results_df']
        n_significant = sum(results_df['significant_fdr'])
        total_tests = len(results_df)
        print(f"{covariate}: {n_significant}/{total_tests} significant tests")