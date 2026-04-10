#!/usr/bin/env python3
'''
Pollen Level-3 Evaluation Pipeline:
This script evaluates the performance of Level-2 clustering results to identify "high-confusion" subgroups.
It acts as a diagnostic filter to assess if Level-2 results are sufficient for species discrimination or if 
the project must proceed to Level-3 clustering to separate morphologically indistinguishable taxa.
Key Functions:
1. Feasibility Assessment: Executes an initial baseline training pass to 
   measure how well the ResNet18 model handles current Level-2 clusters. This 
   identifies high-confusion subgroups that are candidates for Level-3 split.
2. Targeted Refinement Logic: Automatically triggers a refined training pass 
   for subgroups where Accuracy < 80% and Total Samples ≥ 100. This ensures that 
   advanced data augumentation (e.g., 180° rotation) is only applied to data-rich clusters. 
   It helps determine if confusion is due to insufficient training or if the species 
   are morphologically inseparable, requiring a deeper Level-3 dendrogram split.
3. Automated Data Cleansing: Dynamically detects and cleans taxonomic noise ('x') and 
   environmental artifacts ('zz'), ensuring that the performance metrics 
   reflect true biological similarity rather than dataset contamination.
4. Evidence-Based Reporting: Generates detailed audit reports and confusion 
   matrices. These serve as evidence to decide whether a subgroup 
   is finalized or requires a Level-3 clustering strategy to separate species.
Environment Setup:
1. Virtual Env: Ensure 'venv' is activated (source venv/bin/activate)
2. Dependencies: torch, torchvision, pandas, scikit-learn, pillow, numpy
Input Files:
1.Mapping CSV: ./output/Level2/Final_Training_Mapping.csv (Species to SubGroup IDs)
2.Image Data: ./data/Sorted_224/ (Pollen images organized by species folders)
Output Files:
1.Trained Models: ./output/Level3/refined_output/models/*.pth
2.Confusion Matrices: ./output/Level3/refined_output/results/baseline/ and /refined/
3.Final Audit Report: Console output summarizing species filtering and training status
Usage:
1. Ensure current path in the root directory Pollen_analysis
2. Run the full pipeline: python3 scripts/main_pipeline.py
'''
import os
# Path management and interacting with operating system            
import torch         
# Core PyTorch library for deep learning and tensor computations
import torch.nn as nn  
# Submodule for defining neural network layers and loss functions
import torch.optim as optim  
# Submodule containing optimization algorithms like Adam or SGD
from torch.utils.data import DataLoader, Dataset
# Dataset: Defines the logic for mapping individual raw pollen images to their verified species labels
# DataLoader: Automates the data flow by organizing images into structured batches
from torchvision import models, transforms 
# Pre-trained architectures and image processing tools
from torchvision.models import ResNet18_Weights 
# Specific class to manage modern pre-trained weight versions
import pandas as pd  
# Handling data manipulation of species lists and performance metrics calculation
from PIL import Image 
# Python image library for opening and verifying image file integrity
import glob          
# Unix-style pathname pattern expansion to find image files on disk
import numpy as np   
# Numerical library for matrix operations and shuffling indices
import random        
# Module for generating random numbers and sampling classes
# Global configuration
# Defines the path to the previous input Level-2 metadata csv 
CSV_PATH="./output/Level2/Final_Training_Mapping.csv"
# Root directory where the physically sorted pollen images are stored
DATA_ROOT="./data/Sorted_224"
# Base directory for training outputs to ensure experimental isolation
BASE_DIR="output/Level3/refined_output" 
# If baseline accuracy is below 80% percentage, the refined training stage is activated
REFINEMENT_ACC_THRESHOLD = 80.0
# Minimum image count required to trigger refinement; prevents overfitting on small data
MIN_SAMPLES_FOR_REFINEMENT = 100  
# A global dictionary to track data filtering statistics for the final summary report
global_audit = {
    'total_species_processed': 0,    
    # Total taxonomic initial entries in the CSV
    'total_lumped_filtered': 0,      
    # Count of species discarded due to taxonomic uncertainty
    'total_garbage_filtered': 0,     
    # Count of entries discarded as environmental noise (zz/garbage)
    'total_missing_folders': 0,      #
     Count of species in CSV that have no corresponding folder on disk
    'total_valid_trained': 0         
    # Final count of biological classes successfully sent to training
}
# Mapping of sub-directories for models and result CSVs
DIRS = {
    'models': f"{BASE_DIR}/models",
    'baseline_csv': f"{BASE_DIR}/results/baseline",
    'refined_csv': f"{BASE_DIR}/results/refined"
}
# Automatically generate the folder structure on the disk if it doesn't already exist
for path in DIRS.values():
    os.makedirs(path, exist_ok=True)
# This function matches csv species entries to corresponding physical image folder
def find_folder(csv_species_name, actual_folders_dict):
    # Normalize the input name(remove spaces, convert to lowercases and string)
    name=str(csv_species_name).strip().lower()
    # Return immediately if an exact match is found in the directory keys
    if name in actual_folders_dict:
        return actual_folders_dict[name]
    # If not found, split the name into words (e.g., 'SAPI Acer campestre' to ['SAPI', 'Acer', 'campestre'])
    parts = name.split()
    # Split the name into words (e.g., 'SAPI Acer campestre' -> ['SAPI', 'Acer', 'campestre'])
    for i in range(len(parts)):
        # Try the full name first, if not match gradually reduce the name string to find a partial match 
        sub_name=" ".join(parts[i:]).strip()
        if sub_name in actual_folders_dict:
            return actual_folders_dict[sub_name]
    return None 
    # Return None if there is no match
# This function is used as training engine
def execute_training(sg_id,paths,labels,species_list,mode="baseline"):
    # Check if the current run is the refined stage or the initial baseline stage
    is_refined_mode=(mode=="refined")
    num_classes=len(species_list)
    # Create and shuffle indices to ensure a random distribution of pollen samples
    indices = np.arange(len(paths))
    np.random.shuffle(indices)
    # Define the 90% training and 10% validation split point
    split = int(0.9*len(paths))
    # Initialize datasets with the appropriate transforms based on the training mode
    train_ds=SubGroupPollenDataset([paths[i] for i in indices[:split]], [labels[i] for i in indices[:split]], get_transforms(is_refined_mode))
    test_ds=SubGroupPollenDataset([paths[i] for i in indices[split:]], [labels[i] for i in indices[split:]], get_transforms(False))
    # Create dataLoaders, use a small batch size (min 8) to accommodate low-sample subgroups
    train_loader=DataLoader(train_ds, batch_size=min(len(train_ds), 8), shuffle=True)
    test_loader=DataLoader(test_ds, batch_size=1)
    # Load a ResNet18 model pre-trained on ImageNet to leverage transfer learning
    model=models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    # Customize the final layer to match the number of species in the current subgroup
    model.fc=nn.Linear(model.fc.in_features, num_classes)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=model.to(device)
    # Refined mode uses a 10x smaller Learning Rate (0.0001)
    # weight adjustments without overshooting the optimal solution
    lr = 0.0001 if is_refined_mode else 0.001
    # The number of epochs for refined mode is increased to allow the model more 
    # time to converge and learn fine-grained morphological features from augmented data.
    epochs = 15 if is_refined_mode else 8
    # Initialize the Adam optimizer with the selected learning rate.
    # Adam is chosen for its adaptive moment estimation, which helps in navigating 
    # complex loss landscapes often found in fine-grained classification
    optimizer=optim.Adam(model.parameters(), lr=lr)
    # Define Cross-Entropy Loss as the objective function. 
    # This is the standard choice for multi-class classification, measuring the 
    # performance of the model whose output is a probability value between 0 and 1
    criterion = nn.CrossEntropyLoss()
    # Print to console of real experiment tracking
    print(f"[Phase] {mode.upper()} | Epochs: {epochs} | LR: {lr}")
    # Standard PyTorch training loop
    # Iterate through the entire dataset multiple times
    for epoch in range(epochs):
        # Set the model to training mode
        model.train()
        # Load images in small batches to save memory and improve learning stability
        for inputs, targets in train_loader:
            # Move data to the GPU or CPU for processing
            inputs, targets=inputs.to(device), targets.to(device)
            # Clear previous gradients so they don't accumulate and mess up the new update
            optimizer.zero_grad()
            # Forward pass, loss calculation, and backpropagation
            # Forward pass: The model makes a guess on the images
            # Loss calculation: Compare the guess with the true species label
            # Backward pass: Calculate how to change the weights to fix errors
            criterion(model(inputs), targets).backward()
            # Update the model's internal weights based on the backward pass
            optimizer.step()
    # Model evaluation to generate the performance confusion matrix
    model.eval()
    # Initialize an empty square matrix to track species-to-species confusion
    cm=np.zeros((num_classes, num_classes), dtype=int)
    # Disable gradient tracking to speed up calculation and save memory during testing
    with torch.no_grad():
        for inputs, targets in test_loader:
            # Get the raw probability scores for each possible species
            outputs=model(inputs.to(device))
            # Pick the species ID with the highest probability score as the true label
            _, preds=torch.max(outputs, 1)
            # Pick the species ID with the highest probability score as the model's prediction
            # (The '_' captures the value, while 'preds' captures the index/class ID)
            # [targets.item()] is the real species, [preds.item()] is the model's guess
            cm[targets.item(), preds.item()]+= 1
    # Calculate overall accuracy for the refinement training
    total= p.sum(cm)
    acc=(np.trace(cm)/total*100) if total > 0 else 0
    # Save results into folders based on the training phase (baseline/refined)
    csv_dir=DIRS['refined_csv'] if is_refined_mode else DIRS['baseline_csv']
    pd.DataFrame(cm, index=species_list, columns=species_list).to_csv(f"{csv_dir}/{mode}_cm_{sg_id}.csv")
    # Store the model weights for future inference or verification
    torch.save(model.state_dict(), f"{DIRS['models']}/{mode}_model_{sg_id}.pth")
    return acc
# Main automatic loop
def run():
    # Load the species-to-subgroup mapping metadata from the CSV file
    df = pd.read_csv(CSV_PATH)
    # Create a lookup dictionary of folders
    # We map {lowercase_name: original_name} to ensure case-insensitive matching 
    actual_folders = {f.strip().lower(): f for f in os.listdir(DATA_ROOT) 
                      if os.path.isdir(os.path.join(DATA_ROOT, f))}
    # Extract unique subgroup IDs and sort them numerically (e.g., '1.1' comes before '1.10').
    # Iterate through each sorted SubGroup ID to perform cluster-specific analysis
    for sg_id in subgroup_ids:
        # Filter the main DataFrame to isolate species belonging only to the current SubGroup
        sub_df = df[df['SubGroup_ID'].astype(str) == sg_id]
        # Extract and sort unique species names within this subgroup for consistent indexing
        species_names=sorted(sub_df['Species_Name'].unique())
        # Initialize containers for image paths, numerical labels, and tracking metadata
        # cur_lab: local label index; valid_sp: names of species that passed all filters
        paths, labels, cur_lab, valid_sp=[], [], 0, []
        for name in species_names:
            # Increment the global audit counter for every species entry encountered
            global_audit['total_species_processed']+= 1
            # Skip species with 'taxonomic uncertainty' (lumped groups or hybrids)
            # This ensures the model learns from distinct biological individuals
            if any(m in name.lower() for m in [' x', ' x ', 'group', '-group']):
                global_audit['total_lumped_filtered'] += 1
                continue
            # Skip 'garbage' or contaminant entries that shouldn't be in the dataset
            if name.lower().strip().startswith('zz') or 'garbage' in name.lower():
                global_audit['total_garbage_filtered'] += 1
                continue
            # Attempt to link the CSV species name to a physical directory on the disk
            target = find_folder(name, actual_folders)
            if target:
                imgs = []
                # Search for all supported image formats within the matched folder
                for ext in ['*.jpg', '*.JPG', '*.png', '*.PNG']:
                    imgs.extend(glob.glob(os.path.join(DATA_ROOT, target, ext)))
                if imgs:
                    # Class Balancing: Cap images at 1000 to prevent bias toward common species
                    if len(imgs) > 1000: 
                        imgs = random.sample(imgs, 1000) 
                    # Accumulate data: add full image paths and corresponding label IDs
                    paths.extend(imgs)
                    labels.extend([cur_lab] * len(imgs))
                    valid_sp.append(name)
                    # Increment local label ID for the next species in this subgroup
                    cur_lab += 1
                else:
                    # Folder exists but contains no usable image files
                    global_audit['total_missing_folders'] += 1
            else:
                # No matching folder found on disk for this CSV species entry
                global_audit['total_missing_folders'] += 1
        # Validation: A classification task requires at least two distinct species
        if len(valid_sp) < 2:
            print(f"kipping SubGroup {sg_id}: Insufficient valid classes after cleaning.")
            continue
        num_samples = len(paths)
        print(f"\nProcessing SubGroup {sg_id} | Classes: {len(valid_sp)} | Samples: {num_samples}")
        # Log the number of high-confidence species ready for training
        global_audit['total_valid_trained']+= len(valid_sp)
        # Phase 1: Execute initial Baseline training to establish a performance floor
        baseline_acc = execute_training(sg_id, paths, labels, valid_sp, mode="baseline")
        # Decision Logic: Check if the subgroup meets the criteria for Refined Training
        # We only refine if accuracy is low AND we have enough data to avoid overfitting
        should_refine = (baseline_acc < REFINEMENT_ACC_THRESHOLD) and (num_samples >= MIN_SAMPLES_FOR_REFINEMENT)
        if should_refine:
            print(f"Refinement triggered: Acc {baseline_acc:.1f}% < {REFINEMENT_ACC_THRESHOLD}% and Samples {num_samples} >= {MIN_SAMPLES_FOR_REFINEMENT}")
            # Phase 2: Execute Refined training with specialized parameters and augmentation
            execute_training(sg_id, paths, labels, valid_sp, mode="refined")
        elif baseline_acc < REFINEMENT_ACC_THRESHOLD:
            # Report cases where accuracy is low but data is too scarce for safe refinement
            print(f"Low accuracy ({baseline_acc:.1f}%) but SKIPPING refinement due to small sample size ({num_samples} < {MIN_SAMPLES_FOR_REFINEMENT}).")
    # Print the summarized statistics of the entire Level-3 preprocessing and training run
    print("\n" + "="*60)
    print("📜 FINAL LEVEL-3 DATA PIPELINE AUDIT REPORT")
    print("="*60)
    print(f"Total Species processed from CSV:   {global_audit['total_species_processed']}")
    print(f"Filtered (Uncertain/x/Group):       {global_audit['total_lumped_filtered']}")
    print(f"Filtered (Garbage/zz/Contaminant): {global_audit['total_garbage_filtered']}")
    print(f"Missing (No images on disk):        {global_audit['total_missing_folders']}")
    print(f"Final valid species trained:        {global_audit['total_valid_trained']}")
    print("="*60)
    # Output the exact locations of the generated confusion matrices and models
    print(f"Baseline Results: {DIRS['baseline_csv']}")
    print(f"Refined Results:  {DIRS['refined_csv']}")
    print("="*60 + "\n")
if __name__ == "__main__":
    # Standard Python entry point to execute the run function
    run()