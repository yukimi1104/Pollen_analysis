#!/usr/bin/env python3
"""
The script performs a cutoff on the CNN features to 
simplify the classification of 322 pollen species. By calculating
Euclidean distances between the 512-dimensional morphological vectors, 
The script cuts the dendrogram at a specific distance threshold to divide
the 322 species into 10–15 distinct morphological clusters based on their 
CNN feature similarity.
Setup:
It is recommended to run this script within a virtual environment (venv)
1. Create venv: python3 -m venv venv
2. Activate: source venv/bin/activate 
3. Install: pip install numpy pandas scipy matplotlib
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
import os
# for directory and path handling
# for dendrogram plots
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
# for hierarchical clustering
# Main execution pipeline for the hierarchical clustering analysis
# This script should be executed under the Pollen_analysis directory root
def main():
    # Define output directory
    output_dir='output/Level1'
    # If there is no output directory, it will be created
    os.makedirs(output_dir,exist_ok=True)
    # Load the .mat file into a dictionary using scipy
    print("Load model feature data")
    data=scipy.io.loadmat('data/modelFeatures_1.mat')
    # Extract prototypes from dict
    prototypes=data['prototypes']
    # Extract species names from Matlab cell format
    # name[0] extrats the cell content, and the second [0] extracts the actual string
    # str() ensures the final object is a standard string 
    species_names=[str(name[0][0]) for name in data['species_list']]
    # Calculate the Euclidean distance between prototype vectors P and Q 
    # d(P,Q)=sqrt(sum((pi-qi)^2)), i represents the demension of CNN features
    dist_matrix=pdist(prototypes, metric='euclidean')
    # Ward's linkage minimizes intra-cluster variance
    # D(A,B)=√[(2·nA·nB/nA+nB)]·||m_A-m_B||
    # nA, nB: Number of species in clusters A and B
    # mA, mB: Centroids (mean feature vectors) of clusters A and B
    Z=linkage(dist_matrix, method='ward')
    # Cutoff automatic iteration
    # Find the best height to meet the 7-15 clusters requirement
    print(f"{'Cutoff'} | {'Clusters'} | {'Max Group Size'} | {'Min Group Size'}")
    print("-"* 50)  
    # Initiate empty list for summary data about cutoff and corresponding clusters
    summary_data=[]
    cutoff_range=range(30,105,5)   
    for i in cutoff_range:
        # Cut the dendrogram horizontally at height i
        labels=fcluster(Z,i,criterion='distance')
        # Get a list of unique cluster IDs to count the total number of groups
        unique_clusters=np.unique(labels)
        # Calculate the frequency of each label (the size of each cluster)
        counts=pd.Series(labels).value_counts() 
        # Append results as dicts into the summary data list
        summary_data.append({
            'Cutoff': i,
            'Num_Clusters': len(unique_clusters),
            'Max_Size': counts.max(),
            'Min_Size': counts.min()
        })
        print(f"{h} | {len(unique_clusters)} | {counts.max()} | {counts.min()}")
    # Find the best cutoff and map species names
    # Based on scan results: Cutoff 45 leads to 15 clusters with max size 35
    final_cutoff=45 
    print(f"\nFinal cutoff: {final_cutoff}")
    # Cut the dendrogram at cutoff height 45
    final_labels=fcluster(Z,final_cutoff,criterion='distance')
    # Create mapping table
    # final_labels contains the cluster ID (1 to 15) for each species
    mapping_df=pd.DataFrame({
        'Species_Name': species_names,
        'Cluster_ID': final_labels
    })
    #Save the mapping result as csv form in the output directory
    csv_filename=os.path.join(output_dir,'species_to_cluster_mapping_v1.csv')
    mapping_df.to_csv(csv_filename, index=False)
    print(f"Map for {len(np.unique(final_labels))} clusters saved to {csv_filename}")
    #Visualization 
    print("Generate dendrogram plot")
    plt.figure(figsize=(25, 12)) 
    plt.title(f'Pollen Species Hierarchy (Cutoff={final_cutoff}, Clusters=15)', fontsize=20)
    plt.xlabel('Species (Taxonomic Prefix)', fontsize=12)
    plt.ylabel('Euclidean Distance', fontsize=12)
    # Plot the tree
    dendrogram(
        Z, 
        labels=species_names, 
        leaf_rotation=90, 
        leaf_font_size=5, 
        color_threshold=final_cutoff
    )
    # Add the decision cutoff line
    plt.axhline(y=final_cutoff, color='red', linestyle='--', linewidth=2, label=f'Cutoff={final_cutoff}')
    plt.legend(loc='upper right', fontsize=15)
    # Save the figure to the output directory
    plt.tight_layout()
    plot_filename=os.path.join(output_dir, 'pollen_cluster_hierarchy_45.png')
    plt.savefig(plot_filename, dpi=300)
    plt.close()
    print(f"Plot saved as {plot_filename}")
# Run the full hierarchical clustering workflow
if __name__=="__main__":
    main()