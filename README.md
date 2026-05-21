```markdown
# HEDR: Hierarchical Expert Decoupled Refinement for Fine-Grained Pollen Classification
**Author:** Yiran Chen  
**Project:** BINP37 Research project, Master's program in bioinformatics  
---
## 1. Project overview
Pollen grains and spores are important environmental indicators across diverse ecological and evolutionary contexts. However, palynological analysis remains a time-consuming task primarily limited by manual microscopy identification, which is both labor-intensive and constrained by the analytical capacity of available taxonomic experts. While deep learning framework candidates could be used as promising alternatives for automated palynology, these models often reach accuracy plateaus when applied to large-scale, high-diversity datasets. This discriminative limit is particularly critical within large datasets, where numerous taxonomically dense groups exhibit profound morphological convergence, resulting in significant feature overlap within standard monolithic label spaces.
To resolve these fine-grained ambiguities, a framework based on Hierarchical Expert Decoupled Refinement (HEDR) is used in this study. Following standard practices in fine-grained visual recognition [1], the structural topology of this framework is established via Hierarchical Agglomerative Clustering (HAC). By applying pairwise Euclidean distances to our 512-dimensional CNN feature centroids, we quantify the morphological divergence across the 322-species spectrum. The tree topology is optimized using Ward’s linkage criterion, which progressively merges sub-clusters to minimize the total within-cluster variance, thereby ensuring compact cluster boundaries and highly cohesive local subgroups.
In our pipeline, strict limits are set on our tree structure to keep the computational load balanced, dividing the 322-species dataset into manageable, parallel processing paths [2, 3]. Instead of letting the hierarchical tree branch out randomly, we choose these structural boundaries based on a trade-off between our total data scale and the training capacity of our hardware. At Level-1, the system divides the 322 species into 10–15 broad morphological clusters. This initial range ensures the massive species pool is distributed evenly across our computing pipelines, preventing individual large branches from running out of memory or creating processing bottlenecks [3].
The Level-2 split forces each final subgroup to contain only 3–7 closely related species by tuning the distance thresholds on the tree. This restriction addresses how small neural networks struggle when trying to separate species with nearly identical shapes. While a single monolithic model hits a performance ceiling when trying to learn all 322 classes at once due to massive feature overlap, a standard ResNet-18 achieves high sensitivity to subtle, micro-scale morphological differences when its search space is restricted to fewer than 7 look-alike candidates [2]. This partitioning strategy results in exactly 44 independent, specialized ResNet-18 sub-models, which isolates misclassifications to local subgroups and shields the rest of the architecture from unnecessary parameter noise.
This hierarchical decomposition allows the specialized sub-models to prioritize localized, group-specific geometric variances instead of suffering from individual expert over-tuning or computational skew [3]. Given that our strict slide-level Leave-One-Sample-Out (LOSO) partitioning completely cross-validates against slide-specific staining and environmental artifacts [4], the robust sub-model performance implicitly indicates that the network successfully captures stable, fine-grained morphological features to achieve high-fidelity classification within highly ambiguous evolutionary clades.
---
## 2. Software, tools, and environment
### 2.1 Feature extraction and image processing
* **PyTorch (v2.11.0)::** Core deep learning framework for tensor computations and model training.
* **Torchvision (v0.26.0):** Used to load pre-trained ResNet-18 weights (ImageNet-1K) and implement standard image preprocessing transformations, including resizing and tensor normalization.
* **PIL (Pillow v10.2.0):** Standard Python Imaging Library for handling input image verification and RGB conversion.
### 2.2 Hierarchical optimization and metrics evaluation
* **Scipy (v1.15.3):** Core scientific computing library. The cluster.hierarchy module is utilized to execute Hierarchical Agglomerative Clustering (HAC) using Ward’s variance minimization linkage criterion, while the io module is applied to parse and extract raw data from Matlab (.mat) prototype files.
* **Scikit-learn (v1.4.1):** Used for managing cross-validation tracking, slide-level sample partitioning, and generating performance classification metrics.
* **Numpy (v1.26.4), Pandas (v2.2.1):** Core data analysis tools for multi-dimensional array operations, metadata dataframe indexing, and downstream evaluation log management.
---
## 3. Analysis pipeline and execution flow
```text
Pollen_analysis/
├── data/                             # Input data: raw modelFeatures_1.mat and Sorted_224/ physical images
├── scripts/                         
│   ├── pollen_hierarchy.py           # Stage 1: Global 512-Dimension centroid distance macro-clustering
│   ├── pollen_subgroups.py           # Stage 2: Recursive sub-group balanced capacity slicing
│   ├── pollen_level2_training.py     # Stage 3: Multi-expert ResNet-18 optimization and Level-3 refinement
│   └── generate_lv2_plots.py         # Stage 4: Automated metrics extraction and results visualization
└── output/                           # Managed results workspace mapping to pipeline execution stages
    ├── Level1/                       # Global macroscopic partitioning workspace
    │   ├── results/                  # Registry: species_to_cluster_mapping_v1.csv
    │   └── audit/                    # Diagnostics: Cutoff optimization curves, UMAP scatterplots, and global dendrograms
    └── Level2/                       # Localized expert network fine-grained specialized workspace
        ├── results/                  # Registry: Final_Training_Mapping.csv and Split_Complexity_Summary.csv
        ├── audit/                    # Diagnostics: 15 cluster dendrograms, data-filtering tables, and Figure 4 performance plots
        └── training/                 # Parameter weights and quantitative confusion statistics
            ├── models/               # Checkpoints: Saved localized baseline and refined model states (.pth)
            └── results/              # Metrics: Row-column arrays matching true-vs-predicted botanical labels
                ├── baseline/         # Matrix: Unaugmented local baseline confusion metrics (.csv)
                └── refined/          # Matrix: Data-augmented Level-3 hyperparameter-tuned matrices (.csv)
```
### 3.1 Global macroscopic clustering (Level-1 Pipeline)
```text
1. Parse the raw Matlab file to extract the 322 species names and their corresponding 512-dimensional feature prototypes.
2. Compute pairwise Euclidean distances across the feature space and execute Hierarchical Agglomerative Clustering (HAC) using Ward’s variance minimization linkage.
3. Run an automated cutoff height scan to dynamically determine the optimal distance threshold, partitioning the 322-species spectrum into 10–15 distinct macro-morphological cohorts to balance initial parallel computation tracks.
```
### 3.2 Balanced sub-grouping and baseline training (Level-2 Pipeline)
```text
1.Load the Level-1 mapping registry and apply tailored distance cutoffs to split each macro-cluster into tight, local subgroups containing 3–7 proximal species.
2. Implement a slide-level Leave-One-Sample-Out (LOSO) validation framework, withholding exactly one unique sample cluster per species for independent testing to strictly eliminate intra-flower data leakage.
3. Initialize a standard ResNet-18 backbone pre-trained on ImageNet-1K, modify the linear head to match the local subclass count, and execute 8 epochs of Adam optimization using a learning rate of 0.001 under cross-entropy loss.
4.Evaluate the hold-out test split to generate baseline confusion matrices, automatically auditing group-specific accuracies against performance metrics.
```

### 3.3 Adaptive micro-tuning refinement (Level-3 Pipeline)
```text
1. Activate a state-checking checkpoint mechanism that pre-audits the permanent disk to skip previously completed nodes, providing fault tolerance and breakpoint recovery during long training runs.
2. Trigger conditional Level-3 refinement for any sub-group that exhibits a baseline test accuracy below 80.0%, provided the local data pool contains at least 100 images.
3. Cap the training input at a maximum of 1,000 randomly sampled images per single species to maintain host RAM stability during large-scale metadata clustering and listing operations.
4.Execute 15 epochs of local fine-tuning at a reduced learning rate of 0.0001, applying data augmentation via random horizontal flips and continuous 360-degree rotations to enforce geometric invariance.
```
## 4. Mathematical methodology
###  4.1 Feature extraction (global average pooling)
This layer compresses the spatial topography of the convolutional feature maps into a dense multi-dimensional vector. Given an intermediate feature tensor A with dimensions C x H x W (where C represents the channel dimensionality, which is 512 for the baseline feature extractor, while H and W denote the spatial height and width), the global average pooling operation computes the vector element v_c for each channel by averaging all pixel activations across the spatial matrix:
<!-- workaround -->

```math
$$v_c = \frac{1}{H \times W} \sum_{i=1}^{H} \sum_{j=1}^{W} A_{c,i,j}$$
```

### 4.2 Centroid (Prototype) calculation

For each individual botanical taxon $k$ containing $m$ distinct samples within the latent space, the reference morphological prototype $P_k$ is defined as the arithmetic mean of its pooled feature vectors $v_i$:

```math
$$P_k = \frac{1}{m} \sum_{i=1}^{m} v_i$$
```

### 4.3 Ward’s linkage objective

Hierarchical clustering proceeds by recursively merging sub-clusters to minimize the total increase in intra-cluster variance. The distance between two clusters $u$ and $v$ is determined by the discrete scaling coefficient and the squared Euclidean distance between their respective centroids:

```math
$$d(u, v) = \frac{|u| \cdot |v|}{|u| + |v|} \| P_u - P_v \|_2^2$$
```

### 4.4 Loss function (Mini-batch Cross-entropy)

To optimize the active weights of each independent ResNet-18 expert model within its localized class space ($3 \le C_{\text{local}} \le 7$), the pipeline minimizes the mean cross-entropy loss over a synchronized mini-batch of size $N$. For a single input instance $n$, $y_{n,c}$ represents the binary ground-truth indicator, and $\hat{y}_{n,c}$ denotes the Softmax probability assigned to local class $c$:

```math
$$\mathcal{L} = -\frac{1}{N} \sum_{n=1}^{N} \sum_{c=1}^{C_{\text{local}}} y_{n,c} \log(\hat{y}_{n,c})$$
```

## 5. Advanced evaluation metrics

### 5.1 Machine learning performance (Individual image level)

Precision quantifies the true positive rate relative to all positive assignments generated by a specific local expert sub-model:

```math
\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}
```

Recall measures the capability of the local sub-model to correctly retrieve all true members belonging to a given botanical class:

```math
\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}
```

F1-Score represents the harmonic mean of precision and recall, tracking individual class cohesion within the local classification space:

```math
$$F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
```

Macro-Averages are applied to evaluate the overall unweighted performance across the local subgroup array, where $M$ denotes the active class count assigned to that specific sub-model:

```math
$$\text{Macro } F_1 = \frac{1}{M} \sum_{i=1}^{M} F_{1,i}$$
```

### 5.2 Ecological assemblage metrics (Macro-ecological context)

Species Richness ($S$) represents the absolute count of distinct botanical taxa successfully detected within an independent test assemblage:

```math
$$S = \sum_{i=1}^{M} \mathbb{I}(\text{count}_i > 0)$$
```

Relative Abundance ($p_i$) defines the numerical frequency of a single species relative to the cumulative count of individual grains recorded within the test partition:

```math
$$p_i = \frac{n_i}{\sum_{j=1}^{S} n_j}$$
```

Shannon-Wiener Diversity Index ($H'$) quantifies the structural biodiversity of the reconstructed plant community, combining both richness and evenness components:

```math
$$H' = -\sum_{i=1}^{S} p_i \ln p_i$$
```

Diversity Reconstruction Error ($\Delta H'$) measures the absolute variation between the true biological diversity of the hold-out sample slide $j$ and the corresponding diversity derived from model predictions:

```math
$$\Delta H'_j = |H'_{\text{True}, j} - H'_{\text{Predicted}, j}|$$
```

---
## 6. Methods
### 6.1 Data integrity and quality control filtering
```text
Slide-Level Leave-One-Sample-Out (LOSO) Protocol
Standard randomized cross-validation on individual image matrices causes catastrophic accuracy inflation due to severe visual correlation among pollen grains extracted from the same physical flower or slide. Following the validation principles established by Olsson et al. (2021) [4], the methodology extracts the 7-digit unique identifier from each filename. Splitting is strictly restricted at the level of individual glass slides, ensuring that entire sample cohorts are withheld together for independent testing. This structural barrier blocks the deep neural network from memorizing slide-specific staining intensities or microscope artifacts, enforcing true generalization across completely unseen biological individuals.
### 6.2 Localized classification space and augmentation
Multi-Level Structural Capacity Balancing
To resolve deep feature overlap among dense taxonomic groups exhibiting profound morphological convergence, the global 322-species label space is broken down into localized subgroups. Implementing hierarchical agglomerative clustering paired with Ward's variance minimization constructs an explicit topological constraint. By capping local classification capacity to strict cohorts of 3 to 7 proximal species per subgroup, individual ResNet-18 expert models allocate their entire parameters to learning subtle geometric variances within a narrow feature domain, mitigating the representation saturation common in monolithic classifiers.
Isotropic Geometric Invariance
To neutralize arbitrary sample placement and orientation variation under the microscope lens, the optimization pipeline introduces continuous random rotations ranging from 0 to 360 degrees paired with discrete horizontal flips. This spatial randomization forces the model's receptive fields to rely on stable, rotationally invariant phenotypic signatures rather than directional biases. All training images are consistently scaled to 224 x 224 pixels and standardized using ImageNet-1K channel means and standard deviations, aligning the transfer learning weights with the local feature extraction space.
```
## 7. Execution and Reproducibility Workflow
To fully reproduce the multi-level classification hierarchy, localized expert training, and downstream ecology metric evaluation, execute the scripts sequentially within the active shell framework:
```bash
# 0. Navigate to the verified project root directory
cd ~/Pollen_analysis
# 1. Initialize and activate the isolated virtual environment
source venv/bin/activate
# 2. Stage 1: Macro-clustering and dimensional validation
# Processes the 322-species spectrum into 10-15 broad parent clusters based on 512-D centroids.
# Outputs: output/Level1/results/species_to_cluster_mapping_v1.csv
# Audits: 5 diagnostic plots (cutoff optimizations, UMAP manifolds, and global hierarchy trees)
python3 scripts/pollen_hierarchy.py
# 3. Stage 2: Secondary balanced taxonomy partitioning
# Computes secondary Ward's links to subdivide macro-clusters into tight 3-7 species cohorts.
# Outputs: output/Level2/results/Final_Training_Mapping.csv and Split_Complexity_Summary.csv
# Audits: 15 independent cluster-specific subgroup dendrograms (.png)
python3 scripts/pollen_subgroups.py
# 4. Stage 3: Multi-expert model optimization and adaptive refinement
# Executes parallel training of 44 ResNet-18 subnetworks under LOSO isolation.
# Automatically triggers Level-3 data-augmented fine-tuning for low-accuracy nodes (<80.0%).
# Outputs: Serializable model states (.pth) and confusion matrix logs (.csv)
nohup ./venv/bin/python3 -u scripts/pollen_level2_training.py >> output/Level2/training/level2_train_run.log 2>&1 &
# 5. Live process auditing (Optional monitoring track)
# Streams standard runtime outputs, data filtering tallies, and cross-entropy step counts.
tail -f output/Level2/training/level2_train_run.log
# 6. Stage 4: Downstream metrics compilation and figure synthesis
# Parses all independent refined matrices, generates ecological summaries, and compiles LaTeX tables.
# Outputs: output/Level2/audit/figure4_performance_leap.png and production LaTeX code
python3 scripts/generate_lv2_plots.py
```
## 8. References
```text
[1] Shi, G., Liang, X., Li, W., & Lin, X. (2025). Learning separable fine-grained representation via dendrogram construction from coarse labels for fine-grained visual recognition. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV).
[2] Pavlitskaya, S., Struppek, L., Hubschneider, C., & Zöllner, J. M. (2022). Balancing expert utilization in mixture-of-experts layers embedded in CNNs. arXiv preprint arXiv:2204.10598. https://doi.org/10.48550/arXiv.2204.10598
[3] Yemets, K., Lukashchuk, M., & Izonin, I. (2025). Load-balancing strategies for forecasting with mixture-of-experts architecture. Procedia Computer Science, 272, 155-162. https://doi.org/10.1016/j.procs.2025.03.5380
[4] Olsson, O., Karlsson, M., Persson, A. S., & Smith, H. G. (2021). Efficient, automated and robust pollen analysis using deep learning. Methods in Ecology and Evolution, 12(5), 850–862. https://doi.org/10.1111/2041-210X.13575
```