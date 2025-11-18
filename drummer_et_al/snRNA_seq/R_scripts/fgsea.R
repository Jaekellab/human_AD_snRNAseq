# select gene set from Molecular Signatures Database (MSigDB)
# https://cran.r-project.org/web/packages/msigdbr/vignettes/msigdbr-intro.html

#### INSTALLATION #### 

#install.packages("msigdbr")

if (!require("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

BiocManager::install("fgsea")

library(devtools)
install_github("ctlab/fgsea")

#### IMPORT #### 

library(msigdbr)
library(fgsea)
library(dplyr)
library(tidyr) # for saving csv
library(ggplot2) # save figures
library(data.table)

#### RUN GENE SET ENRICHMENT ANALYSIS #### 

all_gene_sets = msigdbr(species = "Homo sapiens")
head(all_gene_sets)

gene_set_category_list <- c("H","C2","C5")
#cluster_of_interest_list <- c("0", "1", "2", "3", "4", "5", "prefrontal_cortex", "temporal_cortex", "visual_cortex", "AD")
obs_variable <- "leiden_merged"  # Change this to your desired .obs variable
cluster_of_interest_list <- c("0", "1_2", "3", "4")

for (cluster_of_interest in cluster_of_interest_list) {
  
  for (gene_set_category in gene_set_category_list) {

    ## Step 1: Select MSigDB gene sets for humans
    
    #gene_set_category = "C5"
    gene_set_of_interest <- msigdbr(species = "Homo sapiens", category = gene_set_category)
    #go_gene_sets <- msigdbr(species = "Homo sapiens", category = "C5")
    #c2_cp_gene_sets <- msigdbr(species = "Homo sapiens", category = "C2", subcategory = "CP:KEGG")
    
    pathways_list <- split(x = gene_set_of_interest$gene_symbol, f = gene_set_of_interest$gs_name)
    
    ## Step 2: Load DEG 
    
    #cluster_of_interest <- 'prefrontal_cortex'
    
    dge_oligo_results_path <- '/Users/francesca.drummer/Documents/1_Projects/jaekel/results/DE_results_oligo'
    deg_results <- read.csv(sprintf("%s/DESeq2_%s_%s_vs_all.csv", dge_oligo_results_path, obs_variable, cluster_of_interest), header = TRUE)
    
    ## Step 3: Prepare ranked gene list
    
    ranked_genes <- deg_results$log2FoldChange
    names(ranked_genes) <- deg_results$X
    ranked_genes <- sort(ranked_genes, decreasing = TRUE)
    
    ## Step 4: Run fgsea
    
    fgsea_results <- fgsea(pathways = pathways_list,
                           stats = ranked_genes,
                           eps = 0.0,
                           minSize = 15,
                           maxSize = 500)
    
    result_name <- sprintf("/Users/francesca.drummer/Documents/1_Projects/jaekel/results/fgsea/fgsea_%s_DESeq2_%s_vs_all.txt", gene_set_category, cluster_of_interest)
    fwrite(fgsea_results, file=result_name, sep="\t", sep2=c("", " ", ""))
    
    # save results
    # Unnest list columns
    fgsea_results_flat <- fgsea_results %>%
      unnest(cols = everything())
    # Save the flattened results as CSV
    write.csv(fgsea_results_flat, file = sprintf("/Users/francesca.drummer/Documents/1_Projects/jaekel/results/fgsea/fgsea_%s_DESeq2_%s_vs_all.csv", gene_set_category, cluster_of_interest), row.names = FALSE)
    
    #### INTERPRETATION ####
    plotEnrichment(pathways_list[["GOBP_CELL_CYCLE"]], ranked_genes)
    
    pval <- 0.2
    
    topPathwaysUp <- fgsea_results[ES > 0][head(order(pval), n=10), pathway]
    topPathwaysDown <- fgsea_results[ES < 0][head(order(pval), n=10), pathway]
    topPathways <- c(topPathwaysUp, rev(topPathwaysDown))
    plotGseaTable(pathways_list[topPathways], ranked_genes, fgsea_results, 
                  gseaParam=0.5)
    figure_name <- sprintf("/Users/francesca.drummer/Documents/1_Projects/jaekel/results/fgsea/fgsea_%s_DESeq2_%s_vs_all.pdf", gene_set_category, cluster_of_interest)
    ggsave(figure_name)  # Save as a PDF
    
    # to reduce to independent pathways
    collapsedPathways <- collapsePathways(fgsea_results[order(pval)][padj < 0.01], 
                                          pathways_list, ranked_genes)
    mainPathways <- fgsea_results[pathway %in% collapsedPathways$mainPathways][
      order(-NES), pathway]
    plotGseaTable(pathways_list[mainPathways], ranked_genes, fgsea_results, 
                  gseaParam = 0.5)
    figure_name <- sprintf("/Users/francesca.drummer/Documents/1_Projects/jaekel/results/fgsea/fgsea_%s_collapsed_DESeq2_oligo_%s_vs_all.pdf", gene_set_category, cluster_of_interest)
    ggsave(figure_name)  # Save as a PDF
    
  }
}

# https://bioconductor.org/packages/release/bioc/vignettes/fgsea/inst/doc/geseca-tutorial.html