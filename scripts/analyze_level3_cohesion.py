#!/usr/bin/env python3
'''
Pollen Level-3 Secondary Clustering Pipeline:
This script splits high-confusion species groups into smaller sub-groups to improve classification accuracy.
Project Architecture:
Pollen_analysis/
├── data/                  # Root data: Initial modelFeatures_1.mat
├── scripts/               # Processing and training scripts
└── output/                # Multi level output hierarchy
    ├── Level1/            # Stage 1: Global clustering
    │   ├── results/       # Input: species_to_cluster_mapping_v1.csv
    │   └── audit/         # Global diagnostic dendrogram plots
    ├── Level2/            # Stage 2: Balanced subgrouping (3-7 subtaxa per cluster)
    │   ├── results/       # Input: Final_Training_Mapping.csv
    │   ├── training/      # Output: ResNet weights (.pth) and raw performance logs (csvs)
    │   └── audit/         # Output: Cluster-specific accuracy reports and heatmaps
    └── Level3/            # Stage 3: Recursive Refinement
        ├── results/       # Updated_L3_Mapping.csv
        ├── training/      # Refined model outputs
        └── audit/         # Confusion heatmaps and low accuracy diagnosis
Key Functions:
1. Low accuracy filtering: Parses the Level-2 Global Accuracy summary and 
isolates subgroups with accuracy < 90% for tertiary clustering.
2. Prototypes extraction: Extracts 512-dimensional feature vectors from the 
specific Level-1 prototypes file(one of the 15 cluster .mat files) for the species within the failed subgroups.
3. Secondary HAC segmentation: Reclusters the Level-2 vectors with a lower distance cutoff 
(t=15.0), splitting highly similar species into separate Level-3 subgroups.
4. Updated L3 mapping generation: Exports the refined mapping file (Updated_L3_Mapping.csv) 
that links each species folder name to its corresponding Level-3 id (e.g., 1.1.1).
Environment Setup:
1. Virtual Environment: activate venv (source venv/bin/activate)
2. Dependencies: numpy, pandas, scipy
Input Files:
1. Level-2 Audit: ./output/Level2/audit/Global_Accuracy_Summary.csv
2. Level-2 Mapping: ./output/Level2/results/Final_Training_Mapping.csv
3. Level-1 Data: ./output/Level1/results/modelPrototypes_*.mat
Output Files:
1. Level-3 Mapping: ./output/Level3/results/Updated_L3_Mapping.csv
Usage:
1. Ensure current path in the root directory Pollen_analysis
2. Run the pipeline: python3 scripts/analyze_level3_cohesion.py
'''
import os
# Path management and interacting with operating system
import numpy as np
# Numerical library for advanced matrix and array operations
import pandas as pd
# Handling data manipulation of species lists and CSV outputs
import scipy.io
# Tool for reading Matlab (.mat) formatted prototype feature dictionaries
from scipy.cluster.hierarchy import linkage, fcluster
# Layered clustering modules to group morphologically similar species
# Global configuration paths
L2_AUDIT_CSV="./output/Level2/audit/Global_Accuracy_Summary.csv"
L2_MAPPING_CSV="./output/Level2/results/Final_Training_Mapping.csv"
L1_PROTO_DIR="./output/Level1/results"
L3_OUT_DIR="./output/Level3/results"
# Generate output directories if they do not exist
os.makedirs(L3_OUT_DIR, exist_ok=True)
def level3_split():
    print("Step 1: Reading Level-2 audit reports...")
    # Check if the baseline summary file exists before parsing
    if not os.path.exists(L2_AUDIT_CSV):
        print(f"Error: {L2_AUDIT_CSV} not found. ")
        return
    # Load Level-2 master accuracy results 
    df_audit=d.read_csv(L2_AUDIT_CSV)
    # Extract only subgroups where the status column shows "FAIL"
    failed_subgroups=df_audit[df_audit['Status'].str.contains("FAIL")]['SubGroup_ID'].astype(str).tolist()
    # If no subgroups require refinement, exit
    if not failed_subgroups:
        print("No subgroups failed the Level-2 accuracy threshold. Level-3 not required.")
        return
    # Print the subgroup ids that require level-3 splitting 
    print(f"Found {len(failed_subgroups)} subgroups requiring Level-3 splitting: {failed_subgroups}")
    # Load the comprehensive level-2 mapping csv
    df_l2_map=pd.read_csv(L2_MAPPING_CSV)
    # Initiate an empty list for level-3 splitting results (including Source_MAT,Species_Name,level-2 subgroup ids and level-3 subgroup ids)
    l3_records=[]
    # Iterate through high-confusion subgroup
    for sg_id in failed_subgroups:
        print(f"\nProcessing SubGroup {sg_id} for Level-3 splitting")
        # Extract all species entries that belong to the current failed Level-2 subgroup
        sub_df=df_l2_map[df_l2_map['SubGroup_ID'].astype(str)==sg_id]
        # Source files: the correct Level-1 prototype files
        source_mat=sub_df['Source_MAT'].iloc[0]
        # Load the raw array data via scipy reader
        mat_data=scipy.io.loadmat(os.path.join(L1_PROTO_DIR, source_mat))
        # Extract the 512-dimensional prototype arrays
        all_protos=mat_data['prototypes']
        # Normalize cell string sequences extracted from the original Matlab file
        all_species = [str(s[0][0]).strip() if isinstance(s[0], (np.ndarray, list)) else str(s[0]).strip() 
                       for s in mat_data['species_list']]
        
        # Isolate the exact species indices found in the current subgroup
        sg_species = sub_df['Species_Name'].str.strip().tolist()
        indices = [all_species.index(sp) for sp in sg_species if sp in all_species]
        
        # Ensure there are sufficient classes left to perform a valid cluster split
        if len(indices) < 2:
            print(f"Skipping {sg_id}: Not enough species for further clustering.")
            continue
            
        # Retrieve the relevant high-dimensional vectors
        sg_protos = all_protos[indices]
        
        # Compute linkages using Ward's minimal variance criteria
        Z = linkage(sg_protos, method='ward')
        # Apply a lower distance threshold to divide highly convergent taxa
        labels = fcluster(Z, t=15.0, criterion='distance')
        
        # Construct refined metadata records
        for sp, l3_id in zip(sg_species, labels):
            l3_records.append({
                "Source_MAT": source_mat,
                "Species_Name": sp,
                "L2_SubGroup": sg_id,
                "SubGroup_ID": f"{sg_id}.{l3_id}"
            })
            
    # Export refined mapping matrix to CSV format
    df_l3 = pd.DataFrame(l3_records)
    l3_mapping_path = os.path.join(L3_OUT_DIR, "Updated_L3_Mapping.csv")
    df_l3.to_csv(l3_mapping_path, index=False)
    print(f"\nLevel-3 Mapping Table successfully saved: {l3_mapping_path}")

if __name__ == "__main__":
    level3_split()