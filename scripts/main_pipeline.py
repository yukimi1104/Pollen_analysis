#!/usr/bin/env python3
'''
Pollen Level-3 Evaluation Pipeline:
This script evaluates the performance of Level-2 clustering results to identify high-confusion subgroups.
It is used to diagnosis if Level-2 results are sufficient for species discrimination or if 
the project must proceed to Level-3 clustering to separate morphologically indistinguishable taxa with highly similarity.
Project Architecture:
Pollen_analysis/
├── data/                  # Root data: Initial modelFeatures_1.mat
├── scripts/               # Processing and training scripts
└── output/                # Multi level output hierarchy
    ├── Level1/            # Stage 1: Global clustering
    │   ├── results/       # Input: species_to_cluster_mapping_v1.csv
    │   └── audit/         # Global diagnostic dendrogram plots
    ├── Level2/            # Stage 2: Balanced subgrouping(3-7 subtaxa per cluster)
    │   ├── results/       # Input: Final_Training_Mapping.csv
    │   ├── training/      # Output: ResNet weights (.pth) and raw performance logs(csvs)
    │   └── audit/         # Output: Cluster-specific accuracy reports & heatmaps
    └── Level3/            # Stage 3: Recursive Refinement 
        ├── results/       # Updated_L3_Mapping.csv
        ├── training/      # Refined model outputs
        └── audit/         # Confusion heatmaps and low accuracy diagnosis
Key Functions:
1. Feasibility Assessment: Run an initial baseline training pass to 
   evaluate how well the ResNet18 model handles current Level-2 clusters. This 
   identifies high-confusion subgroups that are candidates for Level-3 split.
2. Targeted Refinement Logic: Automatically triggers a refined training pass 
   for subgroups where Accuracy < 80% and Total Samples ≥ 100. This ensures that 
   advanced data augumentation (e.g., 180° rotation) is only applied to clusters with rich data. 
   It diagnoses if confusion is due to insufficient training or if the species 
   are morphologically inseparable, requiring a deeper Level-3 dendrogram split.
3. Automated Data Cleaning: Dynamically detects and cleans taxonomic noise ('x') and 
   environmental artifacts ('zz'), ensuring that the performance metrics 
   reflect true biological similarity rather than dataset contamination.
4. Evidence-Based Reporting: Generates detailed audit reports and confusion 
   matrices. These serve as evidence to decide whether a subgroup 
   is finalized or requires a Level-3 clustering strategy to separate species.
Environment Setup:
1. Virtual Environment: activate venv (source venv/bin/activate)
2. Dependencies: torch, torchvision, pandas, scikit-learn, pillow, numpy
Input Files:
1.Mapping CSV: ./output/Level2/results/Final_Training_Mapping.csv (Species to subgroup IDs)
2.Image Data: ./data/Sorted_224/ (Pollen images organized by species folders)
Output Files:
1.Trained Models: ./output/Level2/training/models/*.pth
2.Confusion Matrices: ./output/Level2/training/results/baseline/ and /refined/
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
CSV_PATH="./output/Level2/results/Final_Training_Mapping.csv"
# Root directory where the physically sorted pollen images are stored
DATA_ROOT="./data/Sorted_224"
# Base directory for Level-2 training outputs (models and raw logs)
BASE_DIR="output/Level2/training" 
# Base directory for Level-2 audit reports in human readable format
AUDIT_DIR="output/Level2/audit"
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
    'total_missing_folders': 0,      
    # Count of species in CSV that have no corresponding folder on disk
    'total_valid_trained': 0         
    # Final count of biological classes successfully sent to training
}
# Mapping of sub-directories for models and result CSVs
DIRS = {
    'models': f"{BASE_DIR}/models",
    'baseline_csv': f"{BASE_DIR}/results/baseline",
    'refined_csv': f"{BASE_DIR}/results/refined"
}
# Automatically create the folder structure it doesn't already exist
for path in DIRS.values():
    os.makedirs(path,exist_ok=True)
os.makedirs(AUDIT_DIR,exist_ok=True)
# Dataset class defining the logic for mapping individual raw pollen images to labels
class SubGroupPollenDataset(Dataset):
    #Custom Dataset class for handling pollen imagery within specific subgroups
    #Inherits from torch.utils.data.Dataset to integrate with PyTorch DataLoaders
    def __init__(self, paths, labels, transform=None):
        """
        paths (list):A list of absolute or relative file paths to the physical images (e.g., 'data/Sorted_224/Species_A/img_01.jpg').
        labels (list):A list of numerical category indices corresponding to 
        the species identity of each image(local indices (0, 1, 2...) unique to the current subgroup).
        transform (callable, optional): A function/transform (like torchvision.transforms) 
        that takes in a PIL image and returns a transformed version (e.g., Resizing, Tensor conversion, or Data Augmentation).
        """
        self.paths = paths
        self.labels = labels
        self.transform = transform
    def __len__(self):
        Returns the total number of image samples in this subgroup.
        return len(self.paths)
    def __getitem__(self, idx):
        """
        Fetches a single data-label pair from the dataset.
        Steps:
        1. Retrieve the image path and label at the specified index.
        2. Open the image using PIL and ensure it is in RGB format.
        3. Apply the 'transform' pipeline (e.g., convert to Tensor).
        4. Return the processed image and its corresponding label.
        """
        # Open image and convert to RGB (to handle grayscale or CMYK)
        img = Image.open(self.paths[idx]).convert('RGB')
        label = self.labels[idx]
        # Using PIL to open the image (Standard for PyTorch vision models)
        if self.transform:
            img = self.transform(img)
        return img, label
# Pre-trained architectures and image processing tools
def get_transforms(is_refined=False):
    # Standard normalization values for pre-trained ResNet on ImageNet
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if is_refined:
        # Refined Mode: Adding 180-degree rotation and horizontal flips
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=(180, 180)), 
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        # Baseline Mode: Standard resize only
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
# This function matches csv species entries to corresponding physical image folder
def find_folder(csv_species_name, actual_folders_dict):
    # Normalize the input name(remove spaces, convert to lowercases and string)
    name=str(csv_species_name).strip().lower()
    # Return immediately if an exact match is found in the directory keys
    if name in actual_folders_dict:
        return actual_folders_dict[name]
    # If not found, split the name into words
    parts=name.split()
    for i in range(len(parts)):
        sub_name=" ".join(parts[i:]).strip()
        if sub_name in actual_folders_dict:
            return actual_folders_dict[sub_name]
    return None 
# This function is used as training engine
def execute_training(sg_id, paths, labels, species_list, mode="baseline"):
    # Check if the current run is the refined stage or the initial baseline stage
    is_refined_mode=(mode=="refined")
    num_classes=len(species_list)
    # Create and shuffle indices to ensure a random distribution of pollen samples
    indices=np.arange(len(paths))
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
    lr=0.0001 if is_refined_mode else 0.001
    epochs= 15 if is_refined_mode else 8
    # Initialize the Adam optimizer with the selected learning rate.
    optimizer=optim.Adam(model.parameters(), lr=lr)
    # Define Cross-Entropy Loss as the objective function. 
    criterion=nn.CrossEntropyLoss()
    # Print to console of real experiment tracking
    print(f"[Phase] {mode.upper()} | Epochs: {epochs} | LR: {lr}")
    # Standard PyTorch training loop
    for epoch in range(epochs):
        model.train()
        running_loss=0.0
        for inputs, targets in train_loader:
            inputs,targets=inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs=model(inputs)
            loss=criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()
    # Model evaluation to generate the performance confusion matrix
    model.eval()
    cm=np.zeros((num_classes, num_classes), dtype=int)
    # Disable gradient tracking to speed up calculation
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs=model(inputs.to(device))
            _, preds=torch.max(outputs, 1)
            cm[targets.item(), preds.item()]+= 1
    # Calculate overall accuracy for the refinement training
    total=np.sum(cm)
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
    # Extract unique subgroup IDs and sort them numerically
    subgroup_ids = sorted(df['SubGroup_ID'].astype(str).unique(), key=lambda x: [int(i) for i in x.split('.')])
    # Create a lookup dictionary of folders
    actual_folders = {f.strip().lower(): f for f in os.listdir(DATA_ROOT) 
                      if os.path.isdir(os.path.join(DATA_ROOT, f))}
    # Iterate through each sorted SubGroup ID to perform cluster-specific analysis
    for sg_id in subgroup_ids:
        # Filter the main DataFrame to isolate species belonging only to the current subgroup
        sub_df=df[df['SubGroup_ID'].astype(str)==sg_id]
        # Extract and sort unique species names within this subgroup for consistent indexing
        species_names=sorted(sub_df['Species_Name'].unique())
        paths, abels,cur_lab,valid_sp=[], [], 0, []
        for name in species_names:
            # Increment the global audit counter for every species entry encountered
            global_audit['total_species_processed']+= 1
            # Skip species with taxonomic uncertainty
            if any(m in name.lower() for m in ['x', 'x','group','-group']):
                global_audit['total_lumped_filtered'] += 1
                continue
            # Skip contaminant entries 
            if name.lower().strip().startswith('zz') or 'garbage' in name.lower():
                global_audit['total_garbage_filtered']+= 1
                continue
            # Attempt to link the CSV species name to a physical directory
            target=find_folder(name, actual_folders)
            if target:
                imgs=[]
                for ext in ['*.jpg', '*.JPG', '*.png', '*.PNG']:
                    imgs.extend(glob.glob(os.path.join(DATA_ROOT, target, ext)))
                if imgs:
                    # Cap images at 1000
                    if len(imgs) > 1000: 
                        imgs = random.sample(imgs, 1000) 
                    paths.extend(imgs)
                    labels.extend([cur_lab] * len(imgs))
                    valid_sp.append(name)
                    cur_lab+= 1
                else:
                    global_audit['total_missing_folders']+= 1
            else:
                global_audit['total_missing_folders']+= 1
        # Validation: A classification task requires at least two distinct species
        if len(valid_sp) < 2:
            print(f"Skipping SubGroup {sg_id}: Insufficient valid classes after cleaning.")
            continue
        num_samples = len(paths)
        print(f"\nProcessing SubGroup {sg_id} | Classes: {len(valid_sp)} | Samples: {num_samples}")
        global_audit['total_valid_trained']+= len(valid_sp)
        # Phase 1: Execute initial Baseline training to establish a performance floor
        baseline_acc = execute_training(sg_id, paths, labels, valid_sp, mode="baseline")
        # Decision Logic: Check if the subgroup meets the criteria for Refined Training
        should_refine = (baseline_acc<REFINEMENT_ACC_THRESHOLD) and (num_samples>=MIN_SAMPLES_FOR_REFINEMENT)
        if should_refine:
            print(f"Refinement triggered: Acc {baseline_acc:.1f}% < {REFINEMENT_ACC_THRESHOLD}% and Samples {num_samples} >= {MIN_SAMPLES_FOR_REFINEMENT}")
            # Phase 2: Execute Refined training with specialized parameters and augmentation
            execute_training(sg_id, paths, labels, valid_sp, mode="refined")
        elif baseline_acc < REFINEMENT_ACC_THRESHOLD:
            print(f"Low accuracy ({baseline_acc:.1f}%) but SKIPPING refinement due to small sample size ({num_samples} < {MIN_SAMPLES_FOR_REFINEMENT}).")
    # Print the summarized statistics of the entire Level-3 preprocessing and training run
    print("\n" + "="*60)
    print("Evalution of level-3 cutoff feasibility report")
    print("="*60)
    print(f"Total Species processed from CSV: {global_audit['total_species_processed']}")
    print(f"Filtered (Uncertain/x/Group): {global_audit['total_lumped_filtered']}")
    print(f"Filtered (Garbage/zz/Contaminant): {global_audit['total_garbage_filtered']}")
    print(f"Missing (No images on disk):  {global_audit['total_missing_folders']}")
    print(f"Final valid species trained:  {global_audit['total_valid_trained']}")
    print("="*60)
    print(f"Baseline Results: {DIRS['baseline_csv']}")
    print(f"Refined Results: {DIRS['refined_csv']}")
    print("="*60 + "\n")
if __name__ == "__main__":
    run()