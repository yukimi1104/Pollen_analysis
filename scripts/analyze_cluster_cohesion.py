#!/usr/bin/env python3
"""
Pollen Level-2 Hierarchy Optimization:
This script splits the previous clustered pollen prototypes into smaller groups to train 
specific CNN sub-models. This helps the system better distinguish between highly similar 
species within the same cluster.
Process:
1. Data Integrity:
Compare the 15 Level-1 cluster prototypes against the 322-species baseline
to identify any missing taxa('Astragalus glycyphyllos herbarium' )
2. Secondary Clustering (Level-2):
Use Ward's linkage on 512 dimension species prototypes to determine optimal cutoffs,
ensuring each sub-group contains a balanced set of 3–7 species.
3. Output
Produce a comprehensive mapping CSV for all represented species and 
individual dendrograms for each of the 15 clusters to document the final partitions
Setup:
It is recommended to run this script within a virtual environment (venv)
1. Create venv: python3 -m venv venv
2. Activate: source venv/bin/activate 
3. Install: pip install numpy pandas scipy matplotlib
Usage:
    python3 scripts/analyze_cluster_cohesion.py
"""
import scipy.io
# Load Matlab (.mat) data files
import os
# Manage file paths and interact with the operating system
import numpy as np
# Handle large numeric arrays
import pandas as pd
# Organize dataframe
from scipy.cluster.hierarchy import linkage  
# Calculate the hierarchical links between species
from scipy.cluster.hierarchy import fcluster 
# Split the dendrogram into specific groups based on a cutoff
from scipy.cluster.hierarchy import dendrogram
# Generate the tree-like visualization of the clusters.
import matplotlib.pyplot as plt              
# Visualization
# Ensure the current working directory is the root of 'Pollen_analysis' directory
# The data directory has 15 level-1 matlab prototype files
data_directory="data"                           
# The output results' tables and plots in output/Level2
output_directory="output/Level2"  
# Folder where the optimized level 1 results and plots are saved
v1_mapping_path="data/species_to_cluster_mapping_v1.csv"
# Create the output folder if it's missing
os.makedirs(output_directory,exist_ok=True)    
if not os.path.exists(output_directory):
    os.makedirs(output_directory)
# The expected species count should be same as level 1(322)
expected_count=322
# Standard threshold that works for most clusters
# This cutoff value was chosen as it could often produces 3-7 sub-groups
default_cutoff=25.0
# Manual determine the cutoffs for specific clusters based on dendrogram observations
# These adjustments prevent over-splitting (e.g., Model 4) or 
# force splitting in highly cohesive groups (e.g., Models 12-14)
optimized_cutoffs={
    "modelPrototypes_4.mat":31.0,   
    "modelPrototypes_12.mat":29.0,  
    "modelPrototypes_13.mat":31.0,  
    "modelPrototypes_14.mat":33.0,  
}
# This function extracts species names from nested Matlab cell arrays
def extract_species(raw_species):
    # Initiate empty list for extracted species names
    extracted=[]
    # Flatten array into a 1D list to iterate through every entry 
    for i in raw_species.flatten():
        # Assign the current array to a temporary variable for processing
        val=i
        # Extracting the first element if it is an array/list 
        while isinstance(val,(np.ndarray,list)) and len(val) > 0:
            val = val[0]
        # Convert the final value to a string and remove whitespaces
        extracted.append(str(val).strip())
    return extracted
def level2_pipeline():
    print(f'Check data integrity')
    # Verify if the baseline mapping file exists at the specified path
    if os.path.exists(v1_mapping_path):
        df_v1=pd.read_csv(v1_mapping_path)
        # Extract the 'Species_Name' column and convert it into a set of strings without whitespaces
        # This creates the 322-species baseline for comparison
        initial_species=set(df_v1['Species_Name'].str.strip())
    else:
        # Print error message if the baseline file is missing
        print(f"Warning: Baseline file '{v1_mapping_path}' not found.")
        # Initialize an empty set to avoid variable reference errors in later steps
        initial_species=set()
    # This function aims to sort matlab files numerically
    def sort_cluster(filename):
        return int(filename.split('_')[1].split('.')[0])
    # Find and sort all prototype files numerically from 1 to 15.
    proto_files=sorted(
        [f for f in os.listdir(data_directory) if f.startswith('modelPrototypes_') and f.endswith('.mat')],
        key=sort_cluster
    )
    # Initiate an empty list for csv mapping results
    final_mapping_records=[]    
    # Initiate an empty set for species in .mat files            
    species_mats = set()                   
    print(f"{'Processing File'} | {'Count'} | {'Cutoff'} | {'Sub-groups'}")
    print("-" * 50)
    # The main file processing
    for file in proto_files:
        # Load the Matlab file and extract feature vectors and names
        data=scipy.io.loadmat(os.path.join(data_directory, file))
        protos=data['prototypes']
        species=extract_species(data['species_list'])        
        # Track species found to identify if any missing from the 322 baseline
        species_mats.update(species)       
        # Determine the cutoff: use a specific manual value if defined, otherwise use default
        cutoff=optimized_cutoffs.get(file, default_cutoff)     
        # Perform hierarchial cluster using Ward's method
        Z=linkage(protos, method='ward')
        # Assign species to sub-groups based on the distance cutoff
        labels=fcluster(Z, cutoff, criterion='distance')
        num_clusters=len(np.unique(labels))
        print(f"{file}| {len(species)} | {cutoff} | {num_clusters}")
        # Generate and save a dendrogram for visual verification of the split
        plt.figure(figsize=(10, 5))
        dendrogram(Z, labels=species, leaf_rotation=90)
        plt.axhline(y=cutoff, color='r',linestyle='--', label=f'Cutoff={cutoff}')
        plt.title(f"Optimized Level-2 Split: {file}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_directory, f"Dendrogram_{file.replace('.mat', '.png')}"))
        plt.close()
        # Generate unique IDs for each sub-group (e.g., "10.1", "10.2").
        file_idx=file.split('_')[1].split('.')[0]
        for name,cluster_id in zip(species, labels):
            final_mapping_records.append({
                "Source_MAT": file,
                "Species_Name": name,
                "SubGroup_ID": f"{file_idx}.{cluster_id}"
            })
    #Report species comparison
    print(f"Missing Species Report':")
    print(f"Original Baseline Count: {len(initial_species)}")
    print(f"Total Species in MAT Files: {len(species_mats)}")    
    # Find species that exist in the CSV baseline but not in the .mat files.
    missing=initial_species-species_mats
    if missing:
        print(f"Missing Species Found: {missing}")
        print("These missing species may have been merged or excluded during model training.")
    else:
        print("Success: All species from baseline are accounted for.")
    # Create the final comprehensive mapping CSV.
    df_final=pd.DataFrame(final_mapping_records)
    csv_path=os.path.join(output_directory, "Final_Training_Mapping.csv")
    df_final.to_csv(csv_path, index=False, encoding='utf_8_sig')   
    # Create a summary report showing the complexity of each Level-1 cluster.
    summary_path=os.path.join(output_directory, "Split_Complexity_Summary.csv")
    stats=df_final.groupby("Source_MAT").size().reset_index(name='Species_Count')
    stats['SubGroup_Count']=df_final.groupby("Source_MAT")["SubGroup_ID"].nunique().values
    stats.to_csv(summary_path, index=False)
    print(f"Export Completed':")
    print(f"Final Mapping Table: {csv_path}")
    print(f"Complexity Summary: {summary_path}")
if __name__ == "__main__":
    level2_pipeline()