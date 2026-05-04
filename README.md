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
* **PyTorch (v2.0.1):** Core deep learning framework for tensor computations and ResNet-18 implementation.
* **Torchvision (v0.15.2):** Used for pre-trained ImageNet-1K weights and advanced data transformations.
* **PIL (Pillow):** Python Imaging Library for handling image file integrity and RGB conversion.
### 2.2 Hierarchical optimization and metrics evaluation
* **Scipy (cluster.hierarchy):** Used for performing HAC, calculating linkages, and generating diagnostic plots.
* **Ward’s Linkage Method:** Minimize intra-cluster variance during recursive taxonomic partitioning.
* **Scikit-learn:** Used for cluster validation and auxiliary classification analytics.
* **Numpy (v1.24.3) & Pandas (v2.0.3):** Core libraries for advanced matrix calculations, Matlab prototype parsing, and metadata file manipulation.
---
## 3. Analysis pipeline and execution flow
Pollen_analysis/
├── data/                  # Root data: Initial modelFeatures_1.mat
├── scripts/               # Processing, training, and evaluation scripts
└── output/                # Multilevel output hierarchy
    ├── Level1/            # Stage 1: Global clustering (15 parent clusters)
    │   ├── results/       # Output: species_to_cluster_mapping_v1.csv
    │   └── audit/         # Output: pollen_global_dendrogram.png
    ├── Level2/            # Stage 2: Balanced subgrouping
    │   ├── results/       # Output: Final_Training_Mapping.csv and Split_Complexity_Summary.csv
    │   ├── training/      # Output: ResNet weights (.pth) and training logs (.csv)
    │   └── audit/         # Output: Global_Accuracy_Summary.csv and Level-2 heatmaps
    └── Level3/            # Stage 3: Recursive refinement for confused subgroups
        ├── results/       # Output: Updated_L3_Mapping.csv (Specific for L3 failures)
        ├── training/      # Output: Refined L3 model weights and specific logs
        └── audit/         # Output: L3 advanced metrics and ecological evaluation logs
### 3.1 Preliminary data analysis and global hierarchy (Level-1)
1. Load 322 species prototypes (512-dimensional morphological vectors) from the Matlab input file.
2. Calculate Euclidean distances between CNN features to simplify the 322-species spectrum.
3. Cut the dendrogram at an optimized distance threshold to generate 10-15 distinct parent clusters.
### 3.2 Balanced sub-grouping (Level-2)
1. Split the Stage-1 clustered prototypes into smaller subgroups to train specific CNN sub-models.
2. Use Ward's linkage to ensure each subgroup contains a balanced set of 3–7 species.
3. Evaluate baseline validation performance using standard cross-entropy objectives.
### 3.3 Recursive hierarchical training and expert models (Level-3)
For subgroups that fail to meet the target baseline accuracy of $90\%$, a recursive Level-3 refinement pass is triggered:
1. **Dynamic data cap:** Species datasets are capped at a maximum of 1,000 images via random sampling to maintain balanced representation.
2. **Micro-learning rates:** An optimized learning rate of $5 \times 10^{-5}$ over 20 epochs is applied using the Adam optimizer to prevent gradient collapse.
3. **Hardware acceleration:** Automatic hardware selection (`cuda` vs. `cpu`) transfers active computational graphs to GPU VRAM for rapid convergence.
---
## 4. Mathematical methodology
### 4.1 Feature extraction (global average pooling)
This layer reduces the spatial dimensions of feature maps $A \in \mathbb{R}^{C \times H \times W}$ into a 512-dimensional vector $v_c$:
$$v_c = \frac{1}{H \times W} \sum_{i=1}^{H} \sum_{j=1}^{W} A_{c,i,j}$$
### 4.2 Centroid (Prototype) calculation
For each taxon $k$ with $m$ samples, the prototype $P_k$ is computed as the arithmetic mean of its latent feature vectors $v_i$:
$$P_k = \frac{1}{m} \sum_{i=1}^{m} v_i$$
### 4.3 Ward’s linkage objective
Clustering is performed by minimizing the increase in total within-cluster variance. The distance between clusters $u$ and $v$ is defined as:
$$d(u, v) = \sqrt{\frac{|u||v|}{|u|+|v|}} \|P_u - P_v\|_2$$
### 4.4 Loss function (Cross-entropy)
The model training is optimized using the Cross-Entropy Loss function, where $y_c$ is the ground truth and $\hat{y}_c$ is the predicted probability:
$$\mathcal{L} = -\sum_{c=1}^{C} y_c \log(\hat{y}_c)$$
---
## 5. Advanced evaluation metrics
To evaluate both high-resolution image classification and the pipeline's real-world utility in community ecology, advanced evaluation metrics are extracted from testing confusion matrices ($cm$).
### 5.1 Machine learning performance (Individual image level)
* **Precision:** The proportion of correctly predicted positive observations out of all predictions made for that specific class. This metric evaluate the ability to avoid false positives.
$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$
* **Recall Rate:** The proportion of correctly predicted positive observations out of all actual members of that class. Measures the ability to avoid false negatives.
$$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
* **$F_1$-Score:** The harmonic mean of Precision and Recall, serving as the balanced metric for fine-tuning performance.
$$F_1\text{-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
* **Macro-Averages:** The arithmetic mean of the metric calculated independently for each species class, treating all classes equally regardless of sample volume.
$$\text{Macro } F_1 = \frac{1}{N} \sum_{i=1}^{N} F_{1}\text{-Score}_i$$
### 5.2 Ecological assemblage metrics (Macro-ecological context)
* **Species Richness ($S$):** The absolute count of distinct taxa (species) detected within the pollen assemblage.
$$S = \sum_{i=1}^{N} \mathbb{I}(\text{count}_i > 0)$$
* **Relative Abundance ($p_i$):** The numerical proportion of a single species relative to the total number of individual pollen grains sampled in the cluster.
$$p_i = \frac{n_i}{\sum_{j=1}^{S} n_j}$$
* **Shannon-Wiener Diversity Index ($H'$):** Quantifies biodiversity, accounting for both richness and evenness within the predicted community.
$$H' = -\sum_{i=1}^{S} (p_i \ln p_i)$$
* **Diversity Reconstruction Error ($\Delta H'$):** The absolute error between true biological Shannon diversity and the diversity computed from predictions. A value of zero ($\Delta H' = 0.0000$) denotes zero algorithmic bias.
$$\Delta H' = |H'_{\text{True}} - H'_{\text{Predicted}}|$$
---
## 6. Methods
### 6.1 Pre-processing and quality control filtering
* **Quality filtering:** Eradicate data leakage between training passes and guarantees that identical images do not skew cross-validation metrics.
* **Dynamic subgroup Partition:** Isolate species by both visual and taxonomic proximity, which prevents severe class imbalance and accelerates training.
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
python3 scripts/analyze_cluster_cohesion.py
# 3. Train and evaluate the Level-2 baseline expert models
# Evaluate classification feasibility and trigger local ResNet-18 fine-tuning
# Outputs: ResNet weights (.pth) and training logs (csvs)
python3 scripts/main_pipeline.py
# Level-3 subgroup classification
# 4. Run automated post-Level 2 training audit and performance report
# Summarize scattered logs and identify high-confusion species subgroups
# Outputs: Global_Accuracy_Summary.csv & diagnostic confusion heatmaps
python3 scripts/analysis_report_post_lv2.py
# 5. Extract prototypes from Level-1 files and execute Level-3 secondary clustering
# Re-cluster failed subgroups (accuracy < 90%) into smaller target label sets
# Output: Updated_L3_Mapping.csv
python3 scripts/analyze_level3_cohesion.py
# 6. Execute expert model training for high-confusion Level-3 subgroups
# Apply intense data augmentation and micro learning rates on ResNet-18
# Outputs: Refined Level-3 model weights and performance outputs
python3 scripts/level3_pipeline.py
# 7. Run post-training evaluation to extract machine learning and ecological metrics
# Parses localized validation outcomes to calculate precision, recall, F1-score,
# true vs predicted Species Richness, and Shannon-Wiener Diversity Index (H')
python3 scripts/evaluate_l3_metrics.py