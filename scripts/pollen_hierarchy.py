#!/usr/bin/env python3
"""
Pollen Level-1 Hierarchy Optimization:
The script performs a cutoff on the CNN features to
simplify the classification of 322 pollen species representing true taxonomic species. By calculating
Euclidean distances between the 512-dimensional morphological vectors,
The script cuts the dendrogram at a specific distance threshold to divide
the 322 samples into 10–15 distinct morphological clusters based on their
CNN feature similarity.It also employs UMAP projection to visually validate 
the 512-dimensional morphological distribution. The workflow also generates inter-cluster 
heatmaps and sub-group dendrograms to audit the structural consistency of the resulting 15 clusters.
Project Architecture:
Pollen_analysis/
├── data/                  # Root data: Initial modelFeatures_1.mat
├── scripts/               # Processing and training scripts
└── output/                # Multi-level output hierarchy
    ├── Level1/            # Stage 1: Global clustering (15 groups)
    │   ├── results/       # Mapping CSV and Partitioned .mat prototypes
    │   └── audit/         # Global diagnostic dendrogram plots
    ├── Level2/            # Stage 2: Balanced sub-grouping and training
    │   ├── results/       # Final_Training_Mapping.csv 
    │   ├── training/      # ResNet weights and CSV performance logs
    │   └── audit/         # Cluster-specific dendrograms and accuracy report
    └── Level3/            # Stage 3: Recursive Refinement 
        ├── results/       # Updated_L3_Mapping.csv 
        ├── training/      # New model outputs after Level-3 splitting
        └── audit/         # Confusion heatmaps and bottleneck resolution report
Setup:
It is recommended to run this script within a virtual environment (venv)
The root project directory is Pollen_analysis
1. Create venv: python3 -m venv venv
2. Activate: source venv/bin/activate 
3. Install: pip install numpy pandas scipy matplotlib
Input:
data/modelFeatures_1.mat (322 species x 512 dimensions)
Output:
output/Level1/results/species_to_cluster_mapping_v1.csv (Species and cluster ids mapping)
output/Level1/audit/01_cutoff_optimization.png (Cutoff value scan results)
output/Level1/audit/02_umap_distribution.png (2D manifold visualization)
output/Level1/audit/03_cluster_distance_heatmap.png (Inter-cluster similarity)
output/Level1/audit/04_pollen_cluster_hierarchy.png (Global dendrogram)
output/Level1/audit/05_cluster_10_detail.png (Localized sub-group dendrogram analysis)
Usage:
python3 scripts/pollen_hierarchy.py
"""
import scipy.io
#load MATLAB .mat files
import numpy as np
#handle data analysis and feature vectors
import pandas as pd
# main data operation library
import matplotlib.pyplot as plt
# for directory and path handling
import os
# for dendrogram plots
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
# for hierarchical clustering
from scipy.spatial.distance import pdist, squareform
# for non-linear dimensionality reduction
import umap
# for statistical data visualization
import seaborn as sns
# Main execution pipeline for the hierarchical clustering analysis
# This script should be executed under the Pollen_analysis directory root
def main(): 
    # Set out directories, if directories not exist create them
    results_dir='output/Level1/results'  
    # Define the path for output audit results
    audit_dir='output/Level1/audit'  
    # Define the path for audit diagnostic results
    for d in [results_dir, audit_dir]:  
        # Iterate through the results and audit directories,create them if they do not exist
        os.makedirs(d,exist_ok=True)  
    # Define input data path
    data_path='data/modelFeatures_1.mat'  
    # Check if the input file exists on disk
    if not os.path.exists(data_path):  
        # Print error message if file is missing
        print(f"Error: {data_path} not found.")  
        # Exit the function if data cannot be loaded
        return  
     # Load the input .mat data
    data=scipy.io.loadmat(data_path)  
    # Extract the 322*512 feature matrix
    prototypes=data['prototypes']  
    # Extract species names from cell array to string list
    species_names=[str(name[0][0]) for name in data['species_list']]  
    # Hierarchical clustering calculation
    # Compute pair-wise Euclidean distances
    dist_matrix=pdist(prototypes, metric='euclidean')  
    # Perform linkage using Ward's variance minimization
    Z=linkage(dist_matrix, method='ward')  
    #Automatic cutoff parameter scan and optimization plotting
    print(f"{'Cutoff':<10} | {'Clusters':<10} | {'Max Size':<10} | {'Min Size':<10}")  
    print("-" * 50)  
    # Initialize list to store iteration results
    summary_data=[]  
    # Iterate through a range of cutoff distance thresholds
    for i in range(30, 105, 5): 
        # Form clusters based on threshold i
        labels=fcluster(Z, i, criterion='distance')  
        # Calculate the size of each cluster
        counts=pd.Series(labels).value_counts()  
        # Count the number of unique clusters found
        num_clusters=len(counts)  
        # Store the metrics for the current threshold
        summary_data.append({  
            'Cutoff': i,
            'Num_Clusters': num_clusters,
            'Max_Size': counts.max(),
            'Min_Size': counts.min()
        })
        # Print summary data
        print(f"{i:<10} | {num_clusters:<10} | {counts.max():<10} | {counts.min():<10}") 
    # Convert result list into dataframe
    summary_df=pd.DataFrame(summary_data)  
    # Automatically select cutoff value to divide into 10-15 clusters
    # Define target cluster count for organizational hierarchy
    target_clusters=15
    # Calculate absolute delta distance from the targeted 15 clusters
    summary_df['Distance_From_Target']=abs(summary_df['Num_Clusters']-target_clusters)
    # Filter constraints to ensure the cluster count sits within a realistic 10-15 scope
    candidates=summary_df[(summary_df['Num_Clusters']>=10)&(summary_df['Num_Clusters']<=15)]
    if not candidates.empty:
        # Prioritize matching the target cluster size, minimizing the max group size for cluster balance
        best_row=candidates.sort_values(by=['Distance_From_Target', 'Max_Size']).iloc[0]
        final_cutoff=int(best_row['Cutoff'])
    else:
        # Fallback to the absolute minimum distance marker if strict range bounds cannot be satisfied
        best_row=summary_df.sort_values(by=['Distance_From_Target']).iloc[0]
        final_cutoff=int(best_row['Cutoff'])
    print(f"\nDynamically optimized Level-1 Cutoff chosen: {final_cutoff} (with {int(best_row['Num_Clusters'])} clusters)")
    # Initialize figure for optimization curve
    plt.figure(figsize=(10, 6))  
    plt.plot(summary_df['Cutoff'], summary_df['Num_Clusters'], marker='o', color='royalblue')  
    # Add target line
    plt.axhline(y=15, color='darkred', linestyle='--', label='Target (15 Clusters)')   
    # Add selection line
    plt.axvline(x=final_cutoff, color='forestgreen', linestyle='--', label=f'Selected Cutoff ({final_cutoff})')  
    plt.title('Optimization of Hierarchical Cutoff Height', fontsize=14)  
    # Set x-axis label
    plt.xlabel('Distance Cutoff (Euclidean)')  
     # Set y-axis label
    plt.ylabel('Number of Clusters')  
    plt.legend()  # Show legend
    # Add light grid lines
    plt.grid(True, alpha=0.3)  
    # Save plot as PNG
    plt.savefig(os.path.join(audit_dir, '01_cutoff_optimization.png'), dpi=300)  
    # Close the plot to free memory
    plt.close() 
    # Generate final cluster IDs based on the dynamically determined cutoff
    final_labels=fcluster(Z, final_cutoff, criterion='distance') 
    # Create a DataFrame to map species names to clusters ids
    mapping_df=pd.DataFrame({  
        'Species_Name': species_names,
        'Cluster_ID': final_labels
    })
    # Define CSV output path
    mapping_path=os.path.join(results_dir, 'species_to_cluster_mapping_v1.csv')  
    # Save mapping results without index column
    mapping_df.to_csv(mapping_path, index=False)  
    # Dimensionality reduction visualization 
    # Initialize UMAP reducer
    reducer=umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)  
    # Project 512-D data into 2-D space
    embedding=reducer.fit_transform(prototypes)  
     # Initialize figure for UMAP scatter plot
    plt.figure(figsize=(12, 10))  
    # Create scatter plot colored by cluster IDs
    sns.scatterplot( 
        x=embedding[:, 0], y=embedding[:, 1], 
        hue=final_labels, palette='turbo', 
        s=60, alpha=0.7, legend='full', edgecolor='none'
    )
    plt.title('UMAP Projection of 512-D Pollen Features', fontsize=16)  
    plt.savefig(os.path.join(audit_dir, '02_umap_distribution.png'), dpi=300)  
    plt.close()  
    # Inter-cluster distance matrix 
    # Identify all unique cluster IDs
    cluster_ids=np.unique(final_labels)  
    # Calculate means
    centroids=np.array([prototypes[final_labels==i].mean(axis=0) for i in cluster_ids]) 
     # Compute distances between centroids
    centroid_dist = squareform(pdist(centroids, metric='euclidean'))  
    # Initialize figure for heatmap
    plt.figure(figsize=(10, 8)) 
    # Plot annotated heatmap
    sns.heatmap(centroid_dist, annot=True, cmap='YlGnBu', fmt=".1f",
                xticklabels=cluster_ids, yticklabels=cluster_ids)  
    plt.title('Inter-Cluster Centroid Distance Matrix', fontsize=14)  
    plt.savefig(os.path.join(audit_dir, '03_cluster_distance_heatmap.png'), dpi=300)  
    plt.close()  
    # Global dendrogram visualization
    # Initialize wide figure for large tree plot
    plt.figure(figsize=(25, 12))  
    # Plot the hierarchical tree structure
    dendrogram( 
        Z, labels=species_names, 
        leaf_rotation=90, leaf_font_size=5, 
        color_threshold=final_cutoff
    )
    plt.axhline(y=final_cutoff, color='red', linestyle='--', linewidth=2, label=f'Cutoff {final_cutoff}') 
    plt.title(f'Global Pollen Species Hierarchy (Clusters={len(cluster_ids)})', fontsize=20)
    plt.tight_layout()  
    plt.savefig(os.path.join(audit_dir, '04_pollen_cluster_hierarchy.png'), dpi=300)  
    plt.close()  # Close the plot
    # Local sub-cluster analysis 
    # Specify cluster ID for detailed zoom-in analysis
    target_id=10 
    # Check if the target cluster exists
    if target_id in final_labels: 
        # Find indices of species in target cluster
        indices=np.where(final_labels==target_id)[0]  
        # Extract features for sub-group
        sub_proto=prototypes[indices] 
        # Extract names for sub-group
        sub_names=[species_names[i] for i in indices]  
        # Re-calculate linkage for sub-group
        Z_sub=linkage(pdist(sub_proto), method='ward')  
        plt.figure(figsize=(12, 8))  
        # Plot sub-group dendrogram
        dendrogram(Z_sub, labels=sub_names, leaf_rotation=90)  
        plt.title(f'Detailed Hierarchy of Cluster {target_id}', fontsize=14)  
        plt.tight_layout() 
        plt.savefig(os.path.join(audit_dir, f'05_cluster_{target_id}_detail.png'), dpi=300)  
        plt.close()
# Main execution entry point check
if __name__ == "__main__":  
    # Call the main function to start the script
    main()  