#!/usr/bin/env python3
'''
Pollen Level-3 Training Pipeline:
This script executes targeted Level-3 fine-tuning for highly confused subgroups. 
It applies intense data augmentation and micro learning rates to enable the 
pre-trained ResNet-18 to distinguish between highly similar morphological features.
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
        └── audit/         # Confusion heatmaps and bottleneck diagnosis

Key Functions:
1. Intensive data augmentation: Applies strong random transformations (e.g., 180° rotations, 
vertical flips, and color jitter) to expose subtle local diagnostic nuances.
2. Micro learning adjustments: Uses a highly specialized learning rate (0.00005) 
and extended epochs to prevent catastrophic forgetting and stabilize fine boundary shifts.
3. Confusion evaluation: Calculates sub-group specific matrices and exports them as CSV logs.
4. Model weight archiving: Exports trained ResNet parameters to specific Level-3 folders.
Environment Setup:
1. Virtual Environment: activate venv (source venv/bin/activate)
2. Dependencies: torch, torchvision, pandas, pillow, numpy
Input Files:
1. Refined Mapping CSV: ./output/Level3/results/Updated_L3_Mapping.csv
2. Raw Image Folders: ./data/Sorted_224/ (Pollen images organized by species folders)
Output Files:
1. Level-3 Models: ./output/Level3/training/models/*.pth
2. Level-3 Results: ./output/Level3/training/results/*.csv
Usage:
1. Ensure current path in the root directory Pollen_analysis
2. Execute the processing pipeline: python3 scripts/level3_pipeline.py
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
# Global Configuration parameters
L3_CSV = "./output/Level3/results/Updated_L3_Mapping.csv"
DATA_ROOT = "./data/Sorted_224"
L3_OUT_DIR = "output/Level3/training"
# Define isolated storage structures for weights and evaluation metrics
DIRS = {
    'models': f"{L3_OUT_DIR}/models",
    'results': f"{L3_OUT_DIR}/results"
}
# Automatically generate local folder directory structures if they do not exist
for path in DIRS.values():
    os.makedirs(path, exist_ok=True)
class L3PollenDataset(Dataset):
    # Custom Dataset class optimized for handling secondary refined pollen imagery 
    # Inherits from standard torch dataset class for DataLoader support
    def __init__(self, paths, labels, transform=None):
        # List of absolute/relative file paths to the images on disk
        self.paths=paths
        # List of continuous integer labels (0 to N-1) assigned to the species for training
        self.labels=labels
        # Image transformations (Data augmentation, normalization)
        self.transform=transform
    def __len__(self):
        # Returns the total number of images across all species contained in this subgroup
        return len(self.paths)
    def __getitem__(self, idx):
        # Retrieve image path, load as RGB, and apply data augmentation/normalization
        img=Image.open(self.paths[idx]).convert('RGB')
        label=self.labels[idx]
        if self.transform:
            img=self.transform(img)
        return img, label
def get_l3_transforms():
    # Advanced augmentation pipeline to focus on local features
    return transforms.Compose([
        #Standardize image dimensions to the input size expected by ResNet-18
        transforms.Resize((224, 224)),
        #Mirror images to teach the model orientation invariance
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        #180-degree freedom allows the network to learn isotropic
        transforms.RandomRotation(180),
        #Slight shifts in brightness/contrast to prevent the model from overfitting to specific illumination or staining conditions
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        #Convert the PIL image (0-255) to a PyTorch tensor (0.0-1.0)
        transforms.ToTensor(),
        # Rescale tensor values using ImageNet's channel means and std devs for stable gradient descent during fine-tuning
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
def train_l3_subgroup(l3_id, paths, labels, species_list):
    # Number of target species within this specific Level-3 subgroup
    num_classes=len(species_list)
    # Create an array of indices corresponding to all available images in this subgroup
    indices=np.arange(len(paths))
    #Shuffle indices randomly to mix up species samples for an unbiased evaluation
    np.random.shuffle(indices)
    #Split as 90/10 for training and validation sets
    split=int(0.9*len(paths))
    # Load training and validation datasets
    train_ds=L3PollenDataset([paths[i] for i in indices[:split]], [labels[i] for i in indices[:split]], get_l3_transforms())
    test_ds=L3PollenDataset([paths[i] for i in indices[split:]], [labels[i] for i in indices[split:]], get_l3_transforms())
    # Establish batching configurations (small size to fit low-sample subsets)
    # Set training batch size to 4, or the maximum available samples if < 4
    train_loader=DataLoader(train_ds, batch_size=min(len(train_ds),4), shuffle=True)
    test_loader=DataLoader(test_ds, batch_size=1)
    # Import a ResNet18 model pre-trained on ImageNet to leverage transfer learning
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    # Customize the final layer to match the number of species in the current Level-3 subgroup
    model.fc=nn.Linear(model.fc.in_features, num_classes)
    #Dynamically select GPU (CUDA) if available, otherwise fallback to CPU
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #Transfer the entire model architecture and parameters to the selected device
    model=model.to(device)
    # Use micro-learning rates as 0.00005 for precise parameter tuning in high confusion groups
    optimizer=optim.Adam(model.parameters(), lr=0.00005)
    # Standard cross-entropy loss function for multi-class classification
    criterion=nn.CrossEntropyLoss()
    print(f"Level-3 Training | SubGroup: {l3_id} | Epochs: 20")
    # Standard PyTorch training loop across extended epochs
    #Iterate 20 times over the entire training subset
    for epoch in range(20):
        #Set the model to training mode
        model.train()
        #Iterate through the training set, loading images (inputs) and labels (targets) in batches
        for inputs,targets in train_loader:
            #Move inputs and ground-truth labels to the selected hardware (GPU via CUDA or CPU)
            inputs,targets=inputs.to(device), targets.to(device)
            #Clear previously computed gradients from the optimizer so they don't accumulate
            optimizer.zero_grad()
            #Forward pass: Pass images through the ResNet-18 architecture to predict raw class scores (logits)
            outputs=model(inputs)
            #Compute the loss (Cross-Entropy) by comparing model predictions with the true labels
            loss=criterion(outputs, targets)
            #Backward pass: Compute the gradients of the loss with respect to all trainable model parameters
            loss.backward()
            #Parameter update: Adjust model weights in the direction that minimizes the loss using the Adam optimizer
            optimizer.step()
    # Model evaluation pass without gradient updates
    model.eval()
    #Initialize an empty N x N confusion matrix with zeros, where N is the number of species
    cm = np.zeros((num_classes, num_classes), dtype=int)
    #Disable gradient tracking to save memory and speed up inference
    with torch.no_grad():
        #Iterate through the testing set (batch_size=1) to evaluate the model on unseen images
        for inputs, targets in test_loader:
            #Move input images to GPU/CPU and pass them through the model to get prediction scores (logits)
            outputs=model(inputs.to(device))
            #Extract the class index with the highest prediction score along dimension 1
            _, preds=torch.max(outputs, 1)
            #Update the confusion matrix: row index is the true label, column index is the prediction
            cm[targets.item(), preds.item()]+=1    
    # Calculate performance accuracy
    total=np.sum(cm)
    acc=(np.trace(cm) / total*100) if total > 0 else 0
    # Save the local confusion matrix to CSV
    pd.DataFrame(cm, index=species_list, columns=species_list).to_csv(f"{DIRS['results']}/l3_cm_{l3_id}.csv")
    # Store weights for reproduction and inference
    torch.save(model.state_dict(), f"{DIRS['models']}/l3_model_{l3_id}.pth")
    print(f"SubGroup {l3_id} Completed. Validation Accuracy: {acc:.2f}%")
    # Free up memory resources
    del model
    torch.cuda.empty_cache()
def run_l3_pipeline():
    # Freeze seed values to ensure deterministic iterations
    torch.manual_seed(42)
    np.random.seed(42)
    # Abort if the splitting step has not been executed previously
    if not os.path.exists(L3_CSV):
        print("Error: Run scripts/analyze_level3_cohesion.py first.")
        return
    df=pd.read_csv(L3_CSV)
    # Extract unique SubGroup strings from mapping files
    l3_ids=df['SubGroup_ID'].unique()
    # Map raw folder locations from disk for physical lookup
    actual_folders={f.strip().lower(): f for f in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, f))}
    # Process each SubGroup individually
    #Loop through each highly confused Level-3 SubGroup ID
    for l3_id in l3_ids:
        #Filter the mapping CSV to get only the rows belonging to the current subgroup
        sub_df=df[df['SubGroup_ID']==l3_id]\
        #Extract the unique species names in this subgroup and sort them alphabetically
        species_names=sorted(sub_df['Species_Name'].unique())
        #Initialize empty containers for paths, continuous labels, the current label ID, and matched species
        paths, labels, cur_lab, valid_sp=[], [], 0, []
        # Link CSV entries to physical disk image files
        for name in species_names:
            target=None
            norm_name=name.strip().lower()
            # Look for an exact match between CSV species name and actual directory names
            if norm_name in actual_folders:
                target=actual_folders[norm_name]
            else:
                #If exact match fails, use a fallback bidirectional substring search
                for k in actual_folders.keys():
                    if norm_name in k or k in norm_name:
                        target=actual_folders[k]
                        break
            if target:
                imgs=[]
                # Scan the folder for all possible image file extensions
                for ext in ['*.jpg', '*.JPG', '*.png', '*.PNG']:
                    imgs.extend(glob.glob(os.path.join(DATA_ROOT, target, ext)))
                # Downsample to exactly 1000 images if the species folder contains more.
                # This prevents memory overflow and maintains data balance across species.
                if imgs:
                    if len(imgs) > 1000:
                        imgs=random.sample(imgs, 1000)
                    #Collect image file paths and assign current integer label (0, 1, 2...) for training
                    paths.extend(imgs)
                    labels.extend([cur_lab]*len(imgs))
                    #Add this species to valid list and increment class ID for next species
                    valid_sp.append(name)
                    cur_lab+=1  
        # Proceed with training if at least 2 distinct species are available
        if len(valid_sp)>=2:
            train_l3_subgroup(l3_id, paths, labels, valid_sp)
#Main execution          
if __name__ == "__main__":
    run_l3_pipeline()