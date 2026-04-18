```markdown
# Pollen analysis: recursive hierarchical classification
**Author:** Yiran Chen 
**Project:** BINP37 Research project, Master's program in bioinformatics  
---
## 1. Project Overview
Pollen grains and spores are critical environmental indicators across diverse ecological and evolutionary contexts. However, palynological analysis remains a time-consuming task primarily limited by manual microscopy, which is both labor-intensive and constrained by the analytical capacity of available taxonomic experts. While deep learning offers an automated alternative with reported accuracies of 70–98%, these models frequently reach accuracy plateaus when applied to large-scale, high-diversity datasets. This discriminative limit is particularly evident within large dataset, where numerous taxonomically dense groups exhibit profound morphological convergence, resulting in significant feature overlap within standard monolithic label spaces.
To resolve these fine-grained ambiguities, we implemented a recursive hierarchical classification strategy. By utilizing Hierarchical Agglomerative Clustering (HAC) with Ward’s linkage, the 322-species spectrum is partitioned into localized morphological SubGroups based on their 512-dimensional CNN feature similarity. This hierarchical decomposition allows specialized ResNet-18 sub-models to prioritize highly similar morphological signals, such as intricate exine sculpturing or nearly-imperceptible micro-puncta,which are typically statistically diluted during standard global training passes. By re-weighting the model’s attention toward these localized nuances, this pipeline provides a scalable solution for automated pollen identification, bridging the persistent accuracy gap in the most taxonomically dense sectors of the pollen library.
---
## 2. Software and tools
### 2.1 Feature extraction and image processing
PyTorch (v2.1.0): Core deep learning framework for tensor computations and ResNet-18 implementation.
Torchvision (v0.16.0): Used for pre-trained ImageNet-1K weights and data augmentation.
PIL (Pillow): Python Imaging Library for handling image file integrity and RGB conversion.
## 2.2 Hierarchical Optimization
Scipy (cluster.hierarchy):Used for performing HAC and generating diagnostic dendrogram plots.
Ward’s Linkage Method:Used to minimize intra-cluster variance during taxonomic partitioning.
Scikit-learn:Used for secondary clustering logic and performance metric calculations.
## 2.3 Evaluation Tools
Matplotlib and Seaborn:Statistical visualization tools for generating confusion matrices and conflict diagnosis heatmaps.
Numpy and Pandas:Core libraries for matrix operations, species-level prototype calculation, and metadata management.
---
## 3. Analysis Pipeline
### 3.1 Preliminary Data Analysis and Global Hierarchy
1. Load 322 species prototypes (512-dimensional morphological vectors) from the Olsson Lab dataset.
2. Calculate Euclidean distances between CNN features to simplify the 322-species spectrum.
3. Cut the dendrogram at a distance threshold to generate 15 distinct clusters.
### 3.2 Balanced Sub-grouping (Level-2 Optimization)
1. Split the 15 clustered prototypes into smaller groups to train specific CNN sub-models.
2. Use Ward's linkage to ensure each SubGroup contains a balanced set of 3–7 species.
### 3.3 Recursive Hierarchical Training
1.Baseline Phase:Execute initial ResNet-18 training for each SubGroup with a learning rate of 0.001.
2.Refined Phase:Triggered if Baseline Accuracy < 80%; employs 180° rotation and a reduced learning rate (0.0001) to amplify subtle discriminative features.
### 3.4 Performance Audit and Level-3 Evaluation
1.Aggregate metrics across all subgroups to build a high-level summary.
2.Generate high-resolution confusion heatmaps for high-confusion species pairs to identify morphologically inseparable taxa.
---
## 4. Mathematical Methodology
**Feature Extraction (Global Average Pooling)**
This layer reduces the spatial dimensions of feature maps $A \in \mathbb{R}^{C \times H \times W}$ into a 512-dimensional vector $v_c$:
$$v_c = \frac{1}{H \times W} \sum_{i=1}^{H} \sum_{j=1}^{W} A_{c,i,j}$$
**Centroid (Prototype) Calculation**
For each taxon $k$ with $m$ samples, the prototype $P_k$ is computed as the arithmetic mean of its latent feature vectors $v_i$:
$$P_k = \frac{1}{m} \sum_{i=1}^{m} v_i$$
**Ward’s Linkage Objective**
Clustering is performed by minimizing the increase in total within-cluster variance. The distance between clusters $u$ and $v$ is defined as:
$$d(u, v) = \sqrt{\frac{|u||v|}{|u|+|v|}} \|P_u - P_v\|_2$$
**Loss Function (Cross-Entropy)**
The model training is optimized using the Cross-Entropy Loss function, where $y_c$ is the ground truth and $\hat{y}_c$ is the predicted probability:
$$\mathcal{L} = -\sum_{c=1}^{C} y_c \log(\hat{y}_c)$$
