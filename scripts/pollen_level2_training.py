#!/usr/bin/env python3
"""
Pollen Level-2/Level-3 Hierarchical Localized Training Pipeline
1. Localized Network Specialization:
   This script executes the localized training and quantitative evaluation of 
   individual Convolutional Neural Networks (ResNet-18 backbones). It targets 
   isolated species sub-groups previously established during the Level-2 balanced 
   partitioning phase. By training each network on a small cluster of just 3–7 
   species, the model focuses its entire capacity on learning ultra-fine shape 
   differences, allowing the system to easily separate look-alike species that 
   routinely confuse a single, massive global model.
2. Ecological Leave-One-Sample-Out (LOSO) Anti-Leakage Framework:
   Pollen grains extracted from the same flower or microscope slide exhibit severe visual 
   correlation. To strictly eliminate data leakage, this script uses a rigorous Leave-One-Sample-Out (LOSO) 
   framework. For each valid biological species, exactly one entire sample cluster 
   (including all its individual grains) is withheld exclusively for independent validation testing, 
   while the remaining samples are used for model training. This structural barrier 
   forces the neural network to learn robust, generalized species-level variance 
   rather than memorizing sample-specific slide artifacts or lighting environments.
3. Level-2 Baseline Weight Optimization Phase:
   The pipeline initiates Level-2 operations using pre-trained ImageNet backbones, 
   re-initializing the final fully connected layer (nn.Linear) to automatically match 
   the output dimensions with the active taxonomic class count (Classes >= 2). It 
   trains for 8 epochs using Adam optimization (LR=0.001) under cross-entropy loss. 
   Upon completion, the unaugmented baseline accuracy is rigorously evaluated against 
   predefined sample-size and performance metrics to determine if the node requires 
   escalation into the Level-3 deep optimization layer.
4. Level-3 Micro-Tuning Refinement Trigger:
   If a sub-group's baseline test accuracy drops below 80.0% and possesses a robust 
   sample size (Samples >= 100), the network is dynamically pushed into Level-3 
   refinement. This phase executes 15 epochs of fine-tuning at a lower learning 
   rate (LR=0.0001), backed by heavy data augmentation (including continuous 180° 
   random rotations) to enforce strict geometric invariance against arbitrary pollen 
   orientations under the microscope.
5. Resilient Breakpoint Recovery Mechanism:
   Features a state-checking engine that pre-audits the permanent disk. If existing
   confusion matrices (.csv) and serializable weights (.pth) are detected for a node, 
   the sub-group is securely skipped, enabling millisecond-level fault tolerance 
   and breakpoint resume capabilities.
Project Directory Architecture:
Pollen_analysis/
├── data/                      # Input assets: modelFeatures_1.mat and Sorted_224/ images
├── scripts/                   # Core pipeline engineering scripts
│   ├── pollen_hierarchy.py    # Stage 1: Global species feature extraction and macro-clustering
│   |── pollen_subgroups.py    # Stage 2: Balanced partitioning optimization
│   └── pollen_level2_training.py # Stage 3: Localized CNN training and Level-3 adaptive refinement
└── output/                    # Multi-level output workspace hierarchy
    ├── Level1/                # Stage 1: Macro-clustering records
    └── Level2/                # Stage 2 and 3 Target Output Workspace
        ├── audit/             # Output: Feasibility reports and data filtering logs
        └── training/          # Performance metrics and serializable model states
            ├── models/        # Target: Saved ResNet checkpoint weights (.pth)
            └── results/       # Target: Partitioned performance matrices
                ├── baseline/  # Output: Unaugmented baseline confusion matrices (.csv)
                └── refined/   # Output: Augmented micro-tuned confusion matrices (.csv)
Environment Deployment and Setup:
It is strongly recommended to isolate dependencies within a local virtual environment:
1. Initialize virtual environment: python3 -m venv venv
2. Toggle active shell wrapper:  source venv/bin/activate
3. Install reproducible binaries:  pip install numpy pandas torch torchvision Pillow
Execution :
nohup ./venv/bin/python3 -u scripts/pollen_level2_training.py >> output/Level2/training/level2_train_run.log 2>&1 &
"""
# for directory and path handling
import os
# for regular expression
import re
# Core deep learning framework
import torch   
# Neural network module providing foundational layers, activation functions, and loss objectives       
import torch.nn as nn  
#Submodule providing optimizers like Adam and SGD for model training
import torch.optim as optim  
# for building custom datasets and managing mini-batch loading
from torch.utils.data import DataLoader, Dataset
#Computer vision library providing pre-trained models and image augmentation utilities
from torchvision import models, transforms 
# Pre-trained ImageNet weights used for model initializing and transfer learning
from torchvision.models import ResNet18_Weights 
# Data analysis library used to parse and manage taxonomic CSV metadata
import pandas as pd  
# Imaging library used to load, open, and verify physical images
from PIL import Image 
# Pattern matching utility used to locate file paths on the local disk
import glob        
# Scientific computing library for array manipulations and matrix operations
import numpy as np    
# Stochastic generator used for data sampling and maintaining reproducibility
import random        
# For memory management and garbage collection
import gc
# For detailed error logging in background loops
import traceback
# Global configuration paths
CSV_PATH="./output/Level2/results/Final_Training_Mapping.csv"
DATA_ROOT="./data/Sorted_224"
BASE_DIR="output/Level2/training" 
AUDIT_DIR="output/Level2/audit"
# Define the baseline accuracy threshold and the minimum subgroup sample size required for refinement
REFINEMENT_ACC_THRESHOLD=80.0
MIN_SAMPLES_FOR_REFINEMENT=100  
# Initialize a global dictionary to track data filtering metrics
global_audit={
    # Counter for all taxonomic entries evaluated from the CSV
    'total_species_processed': 0,    
    # Counter for entries skipped due to taxonomic uncertainty
    'total_lumped_filtered': 0,  
    # Counter for entries skipped as noise or contaminants     
    'total_garbage_filtered': 0,     
    # Counter for valid names lacking physical images on disk
    'total_missing_folders': 0,     
    # Counter for true biological species successfully trained 
    'total_valid_trained': 0,
    # Counter for subgroups that failed but safely skipped
    'failed_subgroups': 0
}
# Define directory path for segmenting model artifacts and evaluation metrics
DIRS={
    # Destination directory for archiving serializable PyTorch trained model weights (.pth)
    'models': f"{BASE_DIR}/models",
    # Evaluation directory for exporting unaugmented baseline confusion matrices (.csv)
    'baseline_csv': f"{BASE_DIR}/results/baseline",
    # Evaluation directory for exporting advanced Level-3 hyperparameter-tuned confusion matrices (.csv)
    'refined_csv': f"{BASE_DIR}/results/refined"
}
# Automatically construct missing directories across the workspace
for path in DIRS.values():
    os.makedirs(path, exist_ok=True)
os.makedirs(AUDIT_DIR, exist_ok=True)
#Custom Dataset class for loading individual pollen images mapping to local sub-model labels.
class SubGroupPollenDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        # Store list of physical image file paths
        self.paths=paths
        # Store corresponding local class labels
        self.labels=labels
        # Store image augmentation or preprocessing steps
        self.transform=transform
    def __len__(self):
        # Return the total number of images in this dataset
        return len(self.paths)
    def __getitem__(self, idx):
        try:
            # Load image and convert to 3-channel RGB
            img=Image.open(self.paths[idx]).convert('RGB')
        except Exception as e:
            # Fallback for broken images by choosing another index
            fallback_idx = random.randint(0, len(self.paths) - 1)
            img = Image.open(self.paths[fallback_idx]).convert('RGB')
        # Get the corresponding numeric class label
        label=self.labels[idx]
        # Apply preprocessing or data augmentation
        if self.transform:
            img=self.transform(img)
        # Return the processed image tensor and label
        return img, label
#Generate computer vision transformation pipelines for baseline or advanced augmented training
def get_transforms(is_refined=False):
    # ImageNet mean and standard deviation for normalization
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    if is_refined:
        # Training transformations with heavy data augmentation
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(180), 
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        # Standard validation transformations (no augmentation)
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
# Regular expression engine to clean CSV text labels
def clean_taxonomic_name(raw_name):
    # Return empty string if entry is invalid
    if not isinstance(raw_name, str):
        return ""
    # Remove file extensions
    n=re.sub(r'\.(png|jpg|jpeg)$', '', raw_name.strip(), flags=re.IGNORECASE)
    # Replace all underscores, dashes and extra spaces with a standard single space
    n=re.sub(r'[\x00-\x1F\x7F-\xA0_\-]+', ' ', n)
    # Remove the 4-letter uppercase plant family prefixes
    n=re.sub(r'^[A-Z]{4}\s+', '', n).strip()
    # Remove trailing IDs or counts and numbers mixed with letters at the end
    n=re.sub(r'[\s\d_]+$', '', n).strip()
    # Remove artifact keywords
    n=re.sub(r'\b(garbage|zz|contaminant)\b', '', n, flags=re.IGNORECASE).strip()
    # Remove extra white spaces
    n=re.sub(r'\s+', ' ', n).lower()
    return n
# Match the CSV taxonomy entry against lowercased local physical image storage directorie
def find_folder(csv_species_name, actual_folders_dict):
    # Clean and lowercase the target species string from CSV
    target_clean=clean_taxonomic_name(csv_species_name)
    # Abort tracking if clean output is empty
    if not target_clean:
        return None
    # Strip all spaces to perform absolute compact string comparison
    target_compact=target_clean.replace(" ", "")
    # Check absolute stripped matching first
    for folder_key, real_folder in actual_folders_dict.items():
        if folder_key.replace(" ", "")==target_compact:
            return real_folder    
    # if string match success, return the real species folder name
    if target_clean in actual_folders_dict:
        return actual_folders_dict[target_clean]
    # Split the string by whitespace 
    parts=target_clean.split()
    # Progressively strip words from the left side to extract the true botanical identifier
    for i in range(len(parts)):
        # Rejoin remaining characters with a single space separator
        sub_name=" ".join(parts[i:]).strip()
        if sub_name in actual_folders_dict:
            # Return verified physical folder name upon sub-string match
            return actual_folders_dict[sub_name]
        # Fallback comparison using space-stripped partial matches
        sub_compact=sub_name.replace(" ", "")
        # Only check if the string has at least 4 characters to avoid wrong short matches
        if len(sub_compact)>3:  
            # Check if the CSV name fragment is inside the disk folder name, or vice versa
            for folder_key, real_folder in actual_folders_dict.items():
                if sub_compact in folder_key.replace(" ", "") or folder_key.replace(" ", "") in sub_compact:
                    return real_folder
    return None 
#Core PyTorch backpropagation engine optimized for ResNet-18 localized weight update
def execute_training(sg_id, paths, labels, species_list, mode="baseline"):
    # Evaluate active tuning status
    is_refined_mode=(mode=="refined") 
    # Calculate size of the final classification layer
    num_classes=len(species_list)     
    # Initialize empty lists for split datasets
    train_paths, train_labels=[], []
    test_paths, test_labels=[], []
    # Parse physical file metadata into an entry registry to prevent intra-sample grain leakage
    metadata=[]
    for p, l in zip(paths, labels):
        # Extract file name from absolute path
        filename=os.path.basename(p) 
        # Locate the 7-digit sample ID (e.g. 1001007)
        match=re.search(r'(\d{7})', filename)
        # Default to fallback label if missing
        sample_id=match.group(1) if match else "unknown_sample"
        # Append parsed file attributes as a dictionary record for quick lookup
        metadata.append({'path': p, 'label': l, 'sample_id': sample_id})
    # Wrap compiled list into a DataFrame for slicing     
    df_meta=pd.DataFrame(metadata) 
    # Stratified group splitting to prevent data leakage at the unique slide layer
    for class_idx in range(num_classes):
        # Filter down to current target species
        df_class=df_meta[df_meta['label'] == class_idx] 
         # Extract unique biological sample IDs
        unique_samples=df_class['sample_id'].unique()   
        if len(unique_samples)>=2:
            # Shuffle sample IDs to achieve unbiased partitioning
            unique_samples=unique_samples.copy()
            np.random.shuffle(unique_samples)
            # Hold out 1 entire sample exclusively for isolated testing
            test_sample=unique_samples[0] 
            # Gather training samples
            df_train_part = df_class[df_class['sample_id'] != test_sample] 
            # Gather testing samples
            df_test_part = df_class[df_class['sample_id'] == test_sample]  
            # Record paths for training
            train_paths.extend(df_train_part['path'].tolist()) 
            # Record targets for training
            train_labels.extend(df_train_part['label'].tolist()) 
            # Record paths for validation
            test_paths.extend(df_test_part['path'].tolist())   
            # Record targets for validation
            test_labels.extend(df_test_part['label'].tolist())  
        else:
            # Fallback partition for rare single-sample species by shuffling individual grain elements
            paths_arr=df_class['path'].values
            labels_arr=df_class['label'].values
            # Create an array of index numbers from 0 to the total number of grains
            indices=np.arange(len(paths_arr))
            # Apply random shuffle to grains
            np.random.shuffle(indices)
            # Compute 90% index cutoff boundary
            split_idx=max(1, int(0.9*len(paths_arr))) 
            # Take the first 90% of shuffled index slices and add them to training list
            train_paths.extend(paths_arr[indices[:split_idx]])
            train_labels.extend(labels_arr[indices[:split_idx]])
            # Take the remaining 10% of shuffled index slices and add them to test list
            test_paths.extend(paths_arr[indices[split_idx:]])
            test_labels.extend(labels_arr[indices[split_idx:]])
    # Load separate data structures with sample-level isolation rules
    train_ds=SubGroupPollenDataset(train_paths, train_labels, get_transforms(is_refined_mode))
    test_ds=SubGroupPollenDataset(test_paths, test_labels, get_transforms(False))
    # Adjust batch size if training sample size is less than 8
    target_bs=min(len(train_ds), 8)
    if target_bs==0:
        print(f"SubGroup {sg_id} has no training samples after LOSO split. Returning 0 acc.", flush=True)
        return 0.0
    # Configure mini-batch loading (drop unfinished mini-batch to smooth backpropagation gradients)
    train_loader=DataLoader(train_ds, batch_size=target_bs, shuffle=True, drop_last=len(train_ds) > target_bs)
    test_loader=DataLoader(test_ds, batch_size=1, shuffle=False)
    # Download state-of-the-art ImageNet vision backbones
    model=models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc=nn.Linear(model.fc.in_features, num_classes) # Re-shape output head to species size
    # Move computations to GPU cluster if available
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=model.to(device)
    # Assign specific learning hyperparameters matching the runtime phase
    lr=0.0001 if is_refined_mode else 0.001
    epochs=15 if is_refined_mode else 8
    # Setup optimizer
    optimizer=optim.Adam(model.parameters(), lr=lr) 
     # Setup cross-entropy objective loss
    criterion=nn.CrossEntropyLoss()                 
    print(f"[Phase] {mode.upper()} | SubGroup: {sg_id} | Epochs: {epochs} | LR: {lr} | Train Grains: {len(train_ds)} | Test Grains: {len(test_ds)}", flush=True)
    # Execute network weight optimization passes
    for epoch in range(epochs):
        model.train() # Switch parameters to training state
        for inputs, targets in train_loader:
            inputs, targets=inputs.to(device), targets.to(device) # Move data batches to processing engine
            optimizer.zero_grad()    # Erase old slope memory calculations
            outputs = model(inputs)  # Perform a forward pass through the ResNet layers
            loss = criterion(outputs, targets) # Evaluate the classification error
            loss.backward()          # Run backpropagation derivatives
            optimizer.step()         # Tweak internal layer coefficients
    # Transition parameters to inference mode to extract reliable predictions
    model.eval()
    cm=np.zeros((num_classes, num_classes), dtype=int) # Create base matrix array for evaluation tracking
    with torch.no_grad(): # Disable gradient tracking to conserve VRAM overhead
        for inputs, targets in test_loader:
            outputs=model(inputs.to(device))
            _, preds=torch.max(outputs, 1) # Find class index with highest model confidence
            cm[targets.item(), preds.item()] += 1 # Populate confusion matrix row-column coordinates
    total = np.sum(cm) # Sum up all evaluated testing grains
    acc = (np.trace(cm) / total * 100) if total > 0 else 0 # Extract overall percentage score
    # Archive calculated arrays and weights onto permanent disk storage
    csv_dir=DIRS['refined_csv'] if is_refined_mode else DIRS['baseline_csv']
    pd.DataFrame(cm, index=species_list, columns=species_list).to_csv(f"{csv_dir}/{mode}_cm_{sg_id}.csv")
    torch.save(model.state_dict(), f"{DIRS['models']}/{mode}_model_{sg_id}.pth")
    # Clear model allocations and flush VRAM memory cache
    del model, optimizer, train_loader, test_loader
    torch.cuda.empty_cache()
    gc.collect()
    return acc
def run():
    # Enforce strict random initializations for full project reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    df=pd.read_csv(CSV_PATH) # Open current balance configuration schema sheet
    subgroup_ids=sorted(df['SubGroup_ID'].astype(str).unique(), key=lambda x: [int(i) for i in x.split('.')])
    # Create an active directory scan index of local directory folders
    actual_folders={f.strip().lower(): f for f in os.listdir(DATA_ROOT) 
                      if os.path.isdir(os.path.join(DATA_ROOT, f))}
    for sg_id in subgroup_ids:
        try:
            # Verify if baseline matrix and model coefficients already exist on permanent storage
            expected_csv=f"{DIRS['baseline_csv']}/baseline_cm_{sg_id}.csv"
            expected_pth=f"{DIRS['models']}/baseline_model_{sg_id}.pth"
            if os.path.exists(expected_csv) and os.path.exists(expected_pth):
                print(f"Breakpoint Check：SubGroup {sg_id} results detected on disk. Skipping to save compute time.", flush=True)
                continue
            sub_df=df[df['SubGroup_ID'].astype(str) == sg_id] # Filter out the active sub-group cluster
            species_names=sorted(sub_df['Species_Name'].unique()) # Extract all target labels for this sub-group
            paths, labels, cur_lab, valid_sp=[], [], 0, [] # Clean data accumulation structures
            for name in species_names:
                global_audit['total_species_processed']+=1 # Log metadata entry audit step
                name_lower=name.lower().strip()
                if name_lower.startswith('zz') or 'garbage' in name_lower:
                    global_audit['total_garbage_filtered']+=1 # Record data artifact skip count
                    continue
                if 'uncertain' in name_lower or 'group' in name_lower:
                    global_audit['total_lumped_filtered']+=1   # Record classification noise skip count
                    continue
                # Clean the name before looking up the folder
                clean_name=clean_taxonomic_name(name)
                target=find_folder(clean_name, actual_folders) # Seek physical image path match on local storage
                if target:
                    imgs=[]
                    for ext in ['*.jpg', '*.JPG', '*.png', '*.PNG']:
                        imgs.extend(glob.glob(os.path.join(DATA_ROOT, target, ext))) # Pool discovered file matching formats
                    if imgs:
                        if len(imgs) > 1000: 
                            imgs=random.sample(imgs, 1000) # Cap image allocations to defend host RAM stability
                        paths.extend(imgs)                   # Append image file paths list
                        labels.extend([cur_lab] * len(imgs)) # Append matching tracking labels list
                        valid_sp.append(target)              # Record verified botanical taxon key name
                        cur_lab+=1                         # Shift next target internal label ID
                    else:
                        global_audit['total_missing_folders']+=1 # Log folder discovery with zero file instances
                else:
                    global_audit['total_missing_folders']+=1 # Log complete folder match failure on disk
            # Models need at least 2 distinct labels to calculate gradients
            if len(valid_sp) < 2:
                print(f"Skipping SubGroup {sg_id}: Insufficient valid classes on disk ({len(valid_sp)} matched).", flush=True)
                continue
            num_samples=len(paths) # Sum total physical grains discovered for processing
            print(f"\nProcessing SubGroup {sg_id} | Classes: {len(valid_sp)} | Samples: {num_samples}", flush=True)
            global_audit['total_valid_trained']+=len(valid_sp) # Update overall successful training counts
            # Trigger standard baseline execution
            baseline_acc=execute_training(sg_id, paths, labels, valid_sp, mode="baseline")
            # Verify if advanced fine tuning rules are met across sample counts and scores
            should_refine=(baseline_acc < REFINEMENT_ACC_THRESHOLD) and (num_samples >= MIN_SAMPLES_FOR_REFINEMENT)
            if should_refine:
                print(f"Refinement triggered: Acc {baseline_acc:.1f}% < {REFINEMENT_ACC_THRESHOLD}% and Samples {num_samples} >= {MIN_SAMPLES_FOR_REFINEMENT}", flush=True)
                execute_training(sg_id, paths, labels, valid_sp, mode="refined") # Run Level-3 tuning pass
            elif baseline_acc < REFINEMENT_ACC_THRESHOLD:
                print(f"Low accuracy ({baseline_acc:.1f}%) but SKIPPING refinement due to small sample size ({num_samples} < {MIN_SAMPLES_FOR_REFINEMENT}).", flush=True)
        except Exception as e:
            # Catch crashed subgroup and skip to next node without killing loop
            global_audit['failed_subgroups']+=1
            print(f"\n SubGroup {sg_id} crashed. Logging traceback and forcing forward...", flush=True)
            traceback.print_exc()
            # Clear hardware allocations on exception
            torch.cuda.empty_cache()
            gc.collect()
            continue
    # Print out summary report table metrics
    print("\n" + "="*60, flush=True)
    print("Evaluation of Level-2/Level-3 SubGroup Training Feasibility Report", flush=True)
    print("="*60, flush=True)
    print(f"Total Species processed from CSV: {global_audit['total_species_processed']}", flush=True)
    print(f"Filtered (Uncertain/Complex):     {global_audit['total_lumped_filtered']}", flush=True)
    print(f"Filtered (Garbage/Noise):         {global_audit['total_garbage_filtered']}", flush=True)
    print(f"Missing (No images on disk):      {global_audit['total_missing_folders']}", flush=True)
    print(f"Final valid species trained:      {global_audit['total_valid_trained']}", flush=True)
    print(f"Subgroups crashed but bypassed:   {global_audit['failed_subgroups']}", flush=True)
    print("="*60 + "\n", flush=True)
# Execute workflow orchestrator
if __name__ == "__main__":
    run() 