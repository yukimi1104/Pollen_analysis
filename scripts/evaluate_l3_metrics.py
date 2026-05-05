#!/usr/bin/env python3
'''
Pollen Level-3 Advanced Metrics Evaluation Pipeline:
This script computes comprehensive machine learning performance and ecological 
assemblage metrics using the localized validation outcomes of Level-3 subgroups.

Project Architecture:
Pollen_analysis/
├── data/                  # Root data: Initial modelFeatures_1.mat
├── scripts/               # Processing and training scripts
│   └── evaluate_l3_metrics.py
└── output/                # Multilevel output hierarchy
    └── Level3/            # Stage 3: Recursive Refinement
        ├── results/       # Updated_L3_Mapping.csv
        ├── training/      # Refined model outputs
        └── audit/         # Confusion heatmaps and advanced metric logs

Key Functions:
1. Machine Learning Metrics Extraction: Computes precision, recall, and F1-score
   for individual species and generates macro-averages to assess class-level performance.
2. Ecological Assemblage Diagnostics: Compares ground truth vs predicted populations
   to calculate Species Richness and Shannon-Wiener Diversity Index (H') changes.
3. High Confusion Validation: Evaluates fine-grained expert classifications (e.g., the 
   ROSA Malus-Pyrus complex) to ensure zero ecological reconstruction distortion.

Environment Setup:
1. Virtual Environment: activate venv (source venv/bin/activate)
2. Dependencies: numpy

Input Configuration:
- Confusion Matrix (N x N) derived from Level-3 expert model testing runs.
- SubGroup Species Name List aligned to matrix indices.

Usage:
1. Ensure current path is in the root directory Pollen_analysis
2. Run the metrics pipeline: python3 scripts/evaluate_l3_metrics.py
'''

import os
# Path management and interacting with operating system
import numpy as np
# Numerical library for advanced matrix and array operations

def compute_advanced_metrics(cm, species_list, subgroup_id):
    '''
    Evaluates both machine learning and community ecology metrics from a confusion matrix.
    
    Parameters:
    cm (list or ndarray): Confusion matrix where rows are true labels and columns are predictions.
    species_list (list): List of actual species names matching the matrix row/col indices.
    subgroup_id (str): SubGroup identifier for dynamic console logging.
    '''
    # Convert input matrix into a standard NumPy array for slicing operations
    cm = np.array(cm, dtype=int)
    num_classes = cm.shape[0]
    
    # Validation gate: Ensure the matrix matches the total number of species entries
    if len(species_list) != num_classes:
        raise ValueError(f"Matrix size ({num_classes}x{num_classes}) must match species count ({len(species_list)}).")

    print(f"==================================================")
    print(f" ADVANCED EVALUATION REPORT FOR SUBGROUP: {subgroup_id} ")
    print(f"==================================================\n")

    # =====================================================================
    # SECTION 1: MACHINE LEARNING METRICS (Precision, Recall, F1-Score)
    # =====================================================================
    
    # Extract the main diagonal for True Positives
    true_positives = np.diag(cm)
    
    # Calculate False Positives and False Negatives along axes
    false_positives = np.sum(cm, axis=0) - true_positives
    false_negatives = np.sum(cm, axis=1) - true_positives

    # Use epsilon to prevent division-by-zero errors in case of empty classes
    eps = 1e-9
    precision = true_positives / (true_positives + false_positives + eps)
    recall = true_positives / (true_positives + false_negatives + eps)
    f1_score = 2 * (precision * recall) / (precision + recall + eps)

    # Calculate balanced macro-averages across the target subgroup
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1_score)

    print("--- Machine Learning Performance ---")
    for i, name in enumerate(species_list):
        print(f"Species: {name:<25} | Precision: {precision[i]:.4f} | Recall: {recall[i]:.4f} | F1-Score: {f1_score[i]:.4f}")
    
    print("-" * 50)
    print(f"Macro-Averages | Precision: {macro_precision:.4f} | Recall: {macro_recall:.4f} | F1-Score: {macro_f1:.4f}\n")

    # =====================================================================
    # SECTION 2: ECOLOGICAL METRICS (Richness & Diversity Index)
    # =====================================================================
    
    # Derive absolute species counts for ground truth vs model predictions
    true_counts = np.sum(cm, axis=1)
    pred_counts = np.sum(cm, axis=0)
    total_samples = np.sum(cm)

    # Compute Species Richness: number of existing species with counts > 0
    true_richness = np.sum(true_counts > 0)
    pred_richness = np.sum(pred_counts > 0)

    # Convert absolute counts to relative abundance (p_i)
    p_true = true_counts / total_samples
    p_pred = pred_counts / total_samples

    # Compute Shannon-Wiener Diversity Index (H'), masking zero values
    h_true = -np.sum(p_true[p_true > 0] * np.log(p_true[p_true > 0]))
    h_pred = -np.sum(p_pred[p_pred > 0] * np.log(p_pred[p_pred > 0]))

    print("--- Ecological Assemblage Metrics ---")
    print(f"Species Richness (True vs Predicted):     {true_richness} vs {pred_richness}")
    print(f"Shannon Diversity Index H' (True):        {h_true:.4f}")
    print(f"Shannon Diversity Index H' (Predicted):   {h_pred:.4f}")
    print(f"Diversity Reconstruction Error (DeltaH'): {abs(h_true - h_pred):.4f}")
    print(f"==================================================\n")
# Main execution
if __name__ == "__main__":
    # Define exact real species names within the target failed Level-2 subgroup
    real_species = ["ROSA Malus domestica", "ROSA Pyrus communis"]
    
    # Input actual Level-3 confusion matrix (rows = ground truth, cols = predictions)
    real_cm = [
        [9, 1],  
        [1, 9]   
    ]
    # Execute dynamic advanced evaluation
    compute_advanced_metrics(real_cm, real_species, subgroup_id="1.4.5")