# Following tutorial: https://mictott.github.io/SpotSweeper/articles/getting_started.html
# Better tutorial: https://lmweber.org/OSTA/pages/seq-quality-control.html
# Tool can only be run on one sample at the time
# Install necessary packages ------------------------------------------------------------

if (!requireNamespace("zellkonverter", quietly = TRUE)) {
  BiocManager::install("zellkonverter")
}
if (!requireNamespace("SpatialExperiment", quietly = TRUE)) {
  BiocManager::install("SpatialExperiment")
}
if (!requireNamespace("ggspavis", quietly = TRUE)) {
  BiocManager::install("ggspavis")
}
if (!requireNamespace("scater", quietly = TRUE)) {
  BiocManager::install("scater")
}
if (!require("devtools")) install.packages("devtools")
remotes::install_github("MicTott/SpotSweeper")

if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
BiocManager::install("scuttle")

# Install dependencies ------------------------------------------------------------

library(ggspavis)
library(scater)
library(scuttle)
library(SpotSweeper)
library(patchwork)
library(zellkonverter)
library(SpatialExperiment)

# Convert h5ad to spatialexperiment ------------------------------------------------------------

# convert .h5ad to spatialexperiment
sce <- readH5AD('/Users/francesca.drummer/Documents/1_Projects/jaekel/data/Jaekel_human_visium/adata_spatial.h5ad')
output_dir <- "/Users/francesca.drummer/Documents/1_Projects/jaekel/Figures/Visium/"

spatial_coords <- reducedDim(sce, "spatial")  # or however they're stored

spe <- SpatialExperiment(
  assays = assays(sce),
  colData = colData(sce),
  rowData = rowData(sce),
  spatialCoords = spatial_coords,
  metadata = metadata(sce)
)

# Explicitly set the rownames
rownames(spe) <- rownames(sce)

# rename assay name "X" to "counts" (requirement for scuttle)
assayNames(spe)[1] <- "counts"

# Check spatial coordinates and remove NA ------------------------------------------------------------

# Check spatial coordinates
spatial_coords <- spatialCoords(spe)
print("Spatial coordinates summary:")
print(summary(spatial_coords))

# Check for NA values in coordinates
print(paste("NA values in spatial coordinates:", sum(is.na(spatial_coords))))

# Check which spots have NA coordinates
na_coord_spots <- rowSums(is.na(spatial_coords)) > 0
print(paste("Spots with NA coordinates:", sum(na_coord_spots)))

# Remove spots that have NA spatial coordinates
print(paste("Original number of spots:", ncol(spe)))
print(paste("Removing", sum(na_coord_spots), "spots with NA coordinates"))

spe_clean <- spe[, !na_coord_spots]

print(paste("New number of spots:", ncol(spe_clean)))

# Verify the coordinates are now clean
spatial_coords_clean <- spatialCoords(spe_clean)
print(paste("NA values in cleaned spatial coordinates:", sum(is.na(spatial_coords_clean))))

# Basic QC ------------------------------------------------------------

# subset to keep only spots over tissue
spe_clean <- spe_clean[, spe_clean$in_tissue == 1]
dim(spe_clean)

# identify mitochondrial genes
is_mito <- grepl("(^MT-)|(^mt-)", rownames(spe_clean))
table(is_mito)

rownames(spe_clean)[is_mito]

# Select sample ------------------------------------------------------------

# First, check what samples you have
unique_samples <- unique(colData(spe_clean)$library_id)  # adjust column name
print("Available samples:")
print(unique_samples)

# Global threshold variables ------------------------------------------------------------
# Set your thresholds here - easy to modify for different datasets
USE_TOTAL_COUNTS <- TRUE      # Set to TRUE to use total_counts, FALSE to use sum
THRESHOLD_SUM <- 10            # minimum UMI counts (sum)
THRESHOLD_TOTAL_COUNTS <- 500  # minimum total counts
THRESHOLD_DETECTED <- 400      # minimum detected genes
THRESHOLD_MITO_PERCENT <- 30   # maximum mitochondrial percentage
THRESHOLD_MITO_UPDATED <- 28   # updated mitochondrial percentage threshold

# Loop through all samples ------------------------------------------------------------
for(i in seq_along(unique_samples)) {
  sample_name <- unique_samples[i]
  cat("\n=== Processing Sample:", sample_name, "(", i, "of", length(unique_samples), ") ===\n")
  
  # Subset to current sample
  spe_sample <- spe_clean[, colData(spe_clean)$library_id == sample_name]
  
  print(paste("Sample", sample_name, "- Number of spots:", ncol(spe_sample)))

# Run QC ------------------------------------------------------------

# increase memory
memory.limit(size = 32000)  # 32GB in MB

spe_sample <- addPerCellQC(spe_sample, subsets = list(mito = is_mito))
head(colData(spe_sample))

# Determine which metric to use and set variables accordingly
if (USE_TOTAL_COUNTS) {
  count_metric <- "total_counts"
  count_threshold <- THRESHOLD_TOTAL_COUNTS
  count_label <- "Total counts"
} else {
  count_metric <- "sum"
  count_threshold <- THRESHOLD_SUM
  count_label <- "Sum (UMI)"
}

# Global outlier detection ------------------------------------------------------------

# Check if it contains any non-numeric values
summary(colData(spe_sample)$total_counts)

filename <- file.path(paste0(output_dir, sample_name), paste0(sample_name, "_QC_histograms.png"))
png(filename, width = 12, height = 3, units = "in", res = 300)

par(mfrow = c(1, 4))
if (USE_TOTAL_COUNTS) {
  hist(colData(spe_sample)$total_counts, xlab = "Total counts", main = "UMIs per spot (total_counts)")
} else {
  hist(colData(spe_sample)$sum, xlab = "Sum", main = "UMIs per spot (sum)")
}
hist(spe_sample$detected, xlab = "detected", main = "Genes per spot")
hist(spe_sample$subsets_mito_percent, xlab = "pct mito", main = "Percent mito UMIs")
hist(spe_sample$n_genes_by_counts, xlab = "no. cells", main = "No. cells per spot")
dev.off()

print(paste("Plot saved to:", filename))

# plot library size vs. number of cells per spot
p1 <- 
  plotObsQC(spe_sample, plot_type = "scatter", 
            x_metric = "n_genes_by_counts", y_metric = count_metric, 
            y_threshold = count_threshold) + 
  ggtitle(paste0(count_label, " vs. cells per spot"))

# plot mito proportion vs. number of cells per spot
p2 <- 
  plotObsQC(spe_sample, plot_type = "scatter", 
            x_metric = "n_genes_by_counts", y_metric = "subsets_mito_percent", 
            y_threshold = THRESHOLD_MITO_PERCENT) + 
  ggtitle("Mito proportion vs. cells per spot")

combined_plot <- p1 | p2
ggsave(filename = file.path(paste0(output_dir, sample_name), paste0(sample_name, "_qc_combined.png")), 
       plot = combined_plot,
       width = 12, height = 5, 
       units = "in", dpi = 300)

# select QC threshold for library size, add to colData
spe_sample$qc_lib_size <- colData(spe_sample)[[count_metric]] < count_threshold
table(spe_sample$qc_lib_size)
spe_sample$qc_detected <- spe_sample$detected < THRESHOLD_DETECTED
table(spe_sample$qc_detected)
spe_sample$qc_mito_prop <- spe_sample$subsets_mito_percent > THRESHOLD_MITO_PERCENT
table(spe_sample$qc_mito_prop)

# check spatial pattern of discarded spots

colnames(spatialCoords(spe_sample)) <- c("x", "y")

p1 <- 
  plotObsQC(spe_sample, plot_type = "spot", 
            annotate = "qc_lib_size") + 
  ggtitle(paste0(count_label, " (< ", count_threshold, " UMI)"))

p2 <- 
  plotObsQC(spe_sample, plot_type = "spot", 
            annotate = "qc_detected") + 
  ggtitle(paste0("Detected genes (< ", THRESHOLD_DETECTED, " genes)"))

p3 <- 
  plotObsQC(spe_sample, plot_type = "spot", 
            annotate = "qc_mito_prop") + 
  ggtitle(paste0("Mito proportion (> ", THRESHOLD_MITO_PERCENT, "%)"))

p1 | p2 | p3

combined_plot <- p1 | p2 | p3
ggsave(filename = file.path(paste0(output_dir, sample_name), paste0(sample_name, "_global_outliers_spatial.png")), 
       plot = combined_plot,
       width = 12, height = 5, 
       units = "in", dpi = 300)

# Violin plots for outliers ------------------------------------------------------------

# library size and outliers
p1 <- 
  plotObsQC(spe_sample, plot_type = "violin", x_metric = count_metric, 
            annotate = "qc_lib_size", point_size = 0.5) + 
  xlab(paste0(count_label, " (threshold: ", count_threshold, ")"))

# detected genes and outliers
p2 <- 
  plotObsQC(spe_sample, plot_type = "violin", x_metric = "detected", 
            annotate = "qc_detected", point_size = 0.5) + 
  xlab(paste0("Detected genes (threshold: ", THRESHOLD_DETECTED, ")"))

# mito proportion and outliers
p3 <- plotObsQC(spe_sample, plot_type = "violin", x_metric = "subsets_mito_percent", 
                annotate = "qc_mito_prop", point_size = 0.5) + 
  xlab(paste0("Mito proportion (threshold: ", THRESHOLD_MITO_PERCENT, "%)"))

combined_plot <- p1 | p2 | p3
ggsave(filename = file.path(paste0(output_dir, sample_name), paste0(sample_name, "_global_outliers_violin.png")), 
       plot = combined_plot,
       width = 12, height = 5, 
       units = "in", dpi = 300)

# Detect local outliers ------------------------------------------------------------

# library size
spe_sample <- localOutliers(spe_sample,
                            metric = count_metric,
                            direction = "lower",
                            log = TRUE
)

# unique genes
spe_sample <- localOutliers(spe_sample,
                            metric = "detected",
                            direction = "lower",
                            log = TRUE
)

# mitochondrial percent
spe_sample <- localOutliers(spe_sample,
                            metric = "subsets_mito_percent",
                            direction = "higher",
                            log = FALSE
)

# Visualize local outliers ------------------------------------------------------------

# spot plot of log-transformed library size
if (USE_TOTAL_COUNTS) {
  p1 <- plotCoords(spe_sample, annotate="total_counts_log") + 
    ggtitle("log2(Total Counts)")
  p2 <- plotObsQC(spe_sample, plot_type = "spot", in_tissue = "in_tissue", 
                  annotate = "total_counts_outliers", point_size = 0.2) + 
    ggtitle("Local Outliers (Total Counts)")
} else {
  p1 <- plotCoords(spe_sample, annotate="sum_log") + 
    ggtitle("log2(Library Size)")
  p2 <- plotObsQC(spe_sample, plot_type = "spot", in_tissue = "in_tissue", 
                  annotate = "sum_outliers", point_size = 0.2) + 
    ggtitle("Local Outliers (Library Size)")
}

# spot plot of log-transformed detected genes
p3 <- 
  plotCoords(spe_sample, annotate = "detected_log") + 
  ggtitle("log2(Detected)")

p4 <- 
  plotObsQC(spe_sample, plot_type = "spot", in_tissue = "in_tissue", 
            annotate = "detected_outliers", point_size = 0.2) + 
  ggtitle("Local Outliers (Detected)")

# spot plot of mitochondrial proportion
p5 <- 
  plotCoords(spe_sample, annotate = "subsets_mito_percent") + 
  ggtitle("Mito Proportion")

p6 <- 
  plotObsQC(spe_sample, plot_type = "spot", in_tissue = "in_tissue", 
            annotate = "subsets_mito_percent_outliers", point_size = 0.2) + 
  ggtitle("Local Outliers (Mito Prop)")

# plot using patchwork
(p1 / p2) | (p3 / p4) | (p5 / p6)

combined_plot <- (p1 / p2) | (p3 / p4) | (p5 / p6)
ggsave(filename = file.path(paste0(output_dir, sample_name), paste0(sample_name, "_local_outliers_spatial.png")), 
       plot = combined_plot,
       width = 12, height = 5, 
       units = "in", dpi = 300)

# z-transformed library size and outliers
if (USE_TOTAL_COUNTS) {
  p1 <- plotObsQC(spe_sample, plot_type = "violin", x_metric = "total_counts_z", 
                  annotate = "total_counts_outliers", point_size = 0.5) + 
    xlab("total_counts_outliers")
} else {
  p1 <- plotObsQC(spe_sample, plot_type = "violin", x_metric = "sum_z", 
                  annotate = "sum_outliers", point_size = 0.5) + 
    xlab("sum_outliers")
}

# z-transformed detected genes and outliers
p2 <- 
  plotObsQC(spe_sample, plot_type = "violin", x_metric = "detected_z", 
            annotate = "detected_outliers", point_size = 0.5) + 
  xlab("detected_outliers")

# z-transformed mito percent and outliers
p3 <- 
  plotObsQC(spe_sample, plot_type = "violin", x_metric = "subsets_mito_percent_z", 
            annotate = "subsets_mito_percent_outliers", point_size = 0.5) + 
  xlab("mito_outliers")

# plot using patchwork
p1 | p2 | p3

combined_plot <- p1 | p2 | p3
ggsave(filename = file.path(paste0(output_dir, sample_name), paste0(sample_name, "_local_outliers_violin.png")), 
       plot = combined_plot,
       width = 12, height = 5, 
       units = "in", dpi = 300)

# Local vs Global QC spots ------------------------------------------------------------

# select updated threshold for mito percent
spe_sample$qc_mito <- spe_sample$subsets_mito_percent > THRESHOLD_MITO_UPDATED
table(spe_sample$qc_mito)

# combine global outliers
spe_sample$global_outliers <- 
  spe_sample$qc_lib_size | spe_sample$qc_detected | spe_sample$qc_mito

# check number of global outliers
table(spe_sample$global_outliers)

# combine local outliers
if (USE_TOTAL_COUNTS) {
  spe_sample$local_outliers <- 
    spe_sample$total_counts_outliers | spe_sample$detected_outliers |
    spe_sample$subsets_mito_percent_outliers
} else {
  spe_sample$local_outliers <- 
    spe_sample$sum_outliers | spe_sample$detected_outliers |
    spe_sample$subsets_mito_percent_outliers
}

# check number of local outliers
table(spe_sample$local_outliers)

# check spatial pattern of combined set of discarded spots
p1 <- plotObsQC(spe_sample, plot_type = "spot", annotate = "global_outliers") +
  ggtitle(paste0("Global Outliers (", count_label, "<", count_threshold, 
                 ", detected<", THRESHOLD_DETECTED, 
                 ", mito>", THRESHOLD_MITO_UPDATED, "%)"))

p2 <- plotObsQC(spe_sample, plot_type = "spot", annotate = "local_outliers") +
  ggtitle("Local Outliers (Neighborhood-based)")

p1 + p2

combined_plot <- p1 + p2
ggsave(filename = file.path(paste0(output_dir, sample_name), paste0(sample_name, "_local_vs_global_spatial.png")), 
       plot = combined_plot,
       width = 12, height = 5, 
       units = "in", dpi = 300)

}


# Remove low QC spots ------------------------------------------------------------

# combine local and global outliers and store in 'discard' column
spe_sample$discard <- spe_sample$global_outliers | spe_sample$local_outliers

# remove combined set of low-quality spots
spe_sample <- spe_sample[, !spe_sample$discard]

# remove features with all 0 counts
spe_sample <- spe_sample[rowSums(counts(spe_sample))>0, ]

dim(spe_sample)