```markdown
# Pollen Analysis: Recursive Hierarchical Classification
**Author:** Yiran Chen  
**Project:** BINP37 Research project, Master's program in bioinformatics  
---
## 1. Project overview
Pollen grains and spores are important environmental indicators across diverse ecological and evolutionary contexts. However, palynological analysis remains a time-consuming task primarily limited by manual microscopy identification, which is both labor-intensive and constrained by the analytical capacity of available taxonomic experts. While deep learning could serve as an automated alternative with reported accuracies of 70–98%, these models frequently reach accuracy plateaus when applied to large-scale, high-diversity datasets. This discriminative limit is particularly critical within large datasets, where numerous taxonomically dense groups exhibit profound morphological convergence, resulting in significant feature overlap within standard monolithic label spaces.
To resolve these fine-grained ambiguities, we used a Recursive Hierarchical Classification Strategy. By using Hierarchical Agglomerative Clustering (HAC) with Ward’s linkage, the 322-species spectrum is partitioned into localized morphological subgroups based on their 512-dimensional CNN feature similarity. This hierarchical decomposition allows specialized ResNet-18 sub-models to prioritize highly similar morphological signals, such as intricate exine sculpturing or nearly imperceptible micro-puncta.
---
## 2. Software, tools, and environment
### 2.1 Feature extraction and image processing
* **PyTorch (v2.2.1):** Core deep learning framework for tensor computations and ResNet-18 implementation.
* **Torchvision (v0.17.1):** Used for pre-trained ImageNet-1K weights and advanced data transformations.
* **PIL (Pillow v10.2.0):** Python Imaging Library for handling image file integrity and RGB conversion.
### 2.2 Hierarchical optimization and metrics evaluation
* **Scipy (cluster.hierarchy):** Used for performing HAC, calculating linkages, and generating diagnostic plots.
* **Ward’s Linkage Method:** Minimize intra-cluster variance during recursive taxonomic partitioning.
* **Scikit-learn:** Used for cluster validation and auxiliary classification analytics.
* **Numpy (v1.26.4) & Pandas (v2.2.1):** Core libraries for advanced matrix calculations, Matlab prototype parsing, and metadata file manipulation.
---
## 3. Analysis pipeline and execution flow
Pollen_analysis/
├── data/                      # Root data: Initial modelFeatures_1.mat and Sorted_224/ images
├── scripts/                   # Processing, training, and evaluation scripts
└── output/                    # Multilevel output workspace hierarchy
    ├── Level1/                # Stage 1: Global clustering (15 parent clusters)
    │   ├── results/           # Output: species_to_cluster_mapping_v1.csv
    │   └── audit/             # Output: pollen_global_dendrogram.png
    └── Level2/                # Stage 2 (Subgrouping) and Stage 3 (Refinement) Workspace
        ├── results/           # Output: Final_Training_Mapping.csv
        ├── audit/             # Output: Feasibility reports and data filtering logs
        └── training/          # Performance metrics and serializable model states
            ├── models/        # Target: Saved ResNet checkpoint weights (.pth)
            └── results/       # Target: Partitioned performance matrices
                ├── baseline/  # Output: Unaugmented baseline confusion matrices (.csv)
                └── refined/   # Output: Augmented micro-tuned confusion matrices (.csv)
### 3.1 Preliminary data analysis and global hierarchy (Level-1)
1. Load 322 species prototypes (512-dimensional morphological vectors) from the Matlab input file.
2. Calculate Euclidean distances between CNN features to simplify the 322-species spectrum.
3. Cut the dendrogram at an optimized distance threshold to generate 10-15 distinct parent clusters.
### 3.2 Balanced sub-grouping (Level-2)
1. Split the Stage-1 clustered prototypes into smaller subgroups to train specific CNN sub-models.
2. Use Ward's linkage to ensure each subgroup contains a balanced set of 3–7 species.
3. Evaluate baseline validation performance using standard cross-entropy objectives.
### 3.3 Recursive hierarchical training and expert models (Level-3 Refinement Layer)
For subgroups that fail to meet the target baseline accuracy threshold of 80%, a recursive Level-3 refinement pass is dynamically triggered:
1. Dynamic data cap: Species datasets are capped at a maximum of 1,000 images to ensure host memory stability.
2. Continuous data augmentation: Apply continuous $180^\circ$ random rotations and horizontal flips to break orientation bias and slide preparation artifacts.
3. Micro-learning rates: An optimized lower learning rate of 0.0001 over 15 epochs is applied for slow, localized fine-tuning.
4. Breakpoint recovery mechanism: Automate disk state checking; if previous confusion matrices are present, the node safely skips to save compute time.
---
## 4. Mathematical methodology
###  4.1 Feature extraction (global average pooling)
This layer reduces the spatial dimensions of feature maps into a vector:
<!-- workaround -->

```math
A \in \mathbb{R}^{C \times H \times W}
```

```math
v_c = \frac{1}{H \times W} \sum_{i=1}^{H} \sum_{j=1}^{W} A_{c,i,j}
```

### 4.2 Centroid (Prototype) calculation

For each taxon $k$ with $m$ samples, the prototype $P_k$ is computed as the arithmetic mean of its latent feature vectors $v_i$:

```math
P_k = \frac{1}{m} \sum_{i=1}^{m} v_i
```

### 4.3 Ward’s linkage objective

Clustering is performed by minimizing the increase in total within-cluster variance. The distance between clusters $u$ and $v$ is defined as:

```math
d(u, v) = \frac{|u| \cdot |v|}{|u| + |v|} \| P_u - P_v \|_2^2
```

### 4.4 Loss function (Cross-entropy)

The model training is optimized using the Cross-Entropy Loss function, where $y_c$ is the ground truth and $\hat{y}_c$ is the predicted probability:

```math
\mathcal{L} = -\sum_{c=1}^{C} y_c \log(\hat{y}_c)
```

## 5. Advanced evaluation metrics

### 5.1 Machine learning performance (Individual image level)

Precision: The proportion of correctly predicted positive observations out of all predictions made for that specific class.

```math
\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}
```

Recall Rate: The proportion of correctly predicted positive observations out of all actual members of that class.

```math
\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}
```

F1-Score: The harmonic mean of Precision and Recall, serving as the balanced metric for fine-tuning performance.

```math
F_1\text{-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
```

Macro-Averages: The arithmetic mean of the metric calculated independently for each species class.

```math
\text{Macro } F_1 = \frac{1}{N} \sum_{i=1}^{N} F_{1}\text{-Score}_i
```

### 5.2 Ecological assemblage metrics (Macro-ecological context)

Species Richness ($S$): The absolute count of distinct taxa (species) detected within the pollen assemblage.

```math
S = \sum_{i=1}^{N} \mathbb{I}(\text{count}_i > 0)
```

Relative Abundance ($p_i$): The numerical proportion of a single species relative to the total number of individual pollen grains.

```math
p_i = \frac{n_i}{\sum_{j=1}^{S} n_j}
```

Shannon-Wiener Diversity Index ($H'$): Quantifies biodiversity, accounting for both richness and evenness within the predicted community.

```math
H' = -\sum_{i=1}^{S} (p_i \ln p_i)
```

Diversity Reconstruction Error ($\Delta H'$): The absolute error between true biological Shannon diversity and the diversity computed from predictions.

```math
\Delta H' = |H'_{\text{True}} - H'_{\text{Predicted}}|
```

---
## 6. Methods
### 6.1 Pre-processing and quality control filtering
* **Quality filtering:** Eradicate data leakage between training passes and guarantees that identical images do not skew cross-validation metrics.
* **LOSO:** Replaces traditional grain splits with slide-level group data splitting. This step blocks microscope slide background leakage and ensures validation is 
completed on independent botanical individuals.
* **Fallback rare species safeguard:** For rare species with only 1 available slide, the system automatically triggers a localized 90/10 random grain split backed by max(1, ...) bounding rules to keep small sample structures stable.
### 6.2 Data refinement and augmentation (Level-3 Expert Models)
To resolve close inter-species similarities (e.g., the *ROSA Malus-Pyrus* complex), specialized data transformations are applied:
* **$180^\circ$ Random rotation:** Eradicates orientation bias, forcing the neural network to identify isotropic exine and pore textures.
* **Color jittering:** Improves invariant feature extraction across variable slide preparations and staining conditions.
* **Stratified sample splitting:** A localized 90/10 random stratified split is dynamically computed for the subgroup test set to ensure minority or rare species are fairly evaluated.
---
## 7. Execution and Reproducibility Workflow
To fully reproduce the multi-level classification and advanced metric evaluation pipeline, activate Python3 environment and execute the scripts in the following order:
```bash
# 0. Navigate to project root and activate Python 3 environment
cd ~/Pollen_analysis
source venv/bin/activate
# Level 1 and Level 2 classification stage
# 1. Establish the global taxonomic and morphological hierarchy
# Partition the 322 species into 10–15 broad Level-1 clusters based on CNN features
# Outputs: species_to_cluster_mapping_v1.csv and dendrogram plot
python3 scripts/pollen_hierarchy.py
# 2. Analyze cluster complexity and class cohesion
# Use secondary Ward's linkage clustering on Level-1 prototypes to split broad groups
# Outputs: Final_Training_Mapping.csv and Split_Complexity_Summary.csv
python3 scripts/pollen_subgroups.py
# 3. Train baseline expert models and trigger automated Level-3 micro-tuning
# This master pipeline processes subgroups, filters noisy data, applies LOSO splits,
# and escalates low-accuracy groups (<80%) into deep augmented refinement passes.
# Outputs: ResNet weights (.pth) and confusion matrix logs (.csv)
nohup python3 -u scripts/pollen_level2_training.py >> output/Level2/training/level2_train_run.log 2>&1 &
# 4. Monitor real-time cross-entropy iterations and refinement logs
tail -f output/Level2/training/level2_train_run.log
```