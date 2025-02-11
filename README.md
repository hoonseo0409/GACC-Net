# GACC: Generative Adversarial-Cooperative Clustering Network

GACC is a novel semi-supervised learning framework that combines generative adversarial networks (GANs) with cooperative learning to perform robust clustering. It's particularly effective for 3D data clustering tasks, such as brain region segmentation, while being generalizable to other clustering applications.

## Overview

GACC leverages three key components:

1. **Generator**: Outputs key and query matrices using Transformer architecture. The product of these matrices produces cluster membership assignments.

2. **Discriminator**: Distinguishes between ground truth and generated cluster membership matrices.

3. **Cooperative Learning Module**: Consists of two predictors:
   - Entity-to-Cluster Predictor: Minimizes mutual information between clusters
   - Cluster-to-Entity Predictor: Maximizes intra-cluster mutual information

## Key Features

- Semi-supervised learning approach
- Transformer-based architecture for capturing complex relationships
- Dual learning objectives through adversarial and cooperative components
- Support for overlapping clusters
- Designed for 3D spatial data with temporal components
- Scalable to different numbers of clusters

## Dependencies

- TensorFlow 2.x
- NumPy
- Nilearn (for brain imaging examples)
- tqdm
- keras

## Usage

### Basic Example

```python
import gacc
import numpy as np

# Initialize GACC model
cluster_obj = gacc.GACC(
    num_clusters=50,
    run_eagerly=True,
    min_max_scale_range=[0., 1.]
)

# Fit the model
cluster_obj.fit(
    data_xyzt=data_xyzt,          # 4D input data array (x, y, z, time)
    cluster_xyzp=cluster_xyzp,     # Ground truth clusters
    num_batches=10,
    epochs=40
)

# Generate predictions
cluster_xyzp_pred = cluster_obj.predict(data_xyzt=data_xyzt)
```

### Brain Region Clustering Example

```python
import nilearn
from nilearn import datasets, plotting

# Load Allen RSN networks
allen = datasets.fetch_atlas_allen_2011()
data_xyzt = nilearn.image.load_img(allen['maps']).get_fdata()
cluster_xyzp = nilearn.image.load_img(allen['rsn28']).get_fdata()

# Initialize and train GACC
cluster_obj = gacc.GACC(num_clusters=50)
cluster_obj.fit(data_xyzt=data_xyzt, cluster_xyzp=cluster_xyzp)

# Generate predictions
predictions = cluster_obj.predict(data_xyzt=data_xyzt)
```

## Input Format

- `data_xyzt`: 4D numpy array (x, y, z, time) containing the input data
- `cluster_xyzp`: 4D numpy array (x, y, z, num_clusters) containing ground truth cluster assignments
- Last dimension of `cluster_xyzp` represents cluster membership (1 for inside cluster, 0 for outside)

## Architecture Details

### Generator
- Uses Transformer architecture to learn spatial and temporal relationships
- Produces key and query matrices that determine cluster assignments
- Incorporates positional encoding for spatial awareness

### Discriminator
- Evaluates the authenticity of generated cluster assignments
- Provides feedback for generator optimization
- Helps maintain consistency with ground truth patterns

### Cooperative Learning
- Entity-to-Cluster (Cx→c): Minimizes inter-cluster mutual information
- Cluster-to-Entity (Cc→x): Maximizes intra-cluster mutual information
- Balances cluster separation and cohesion

## Model Parameters

- `num_clusters`: Number of clusters to generate (default: 50)
- `min_max_scale_range`: Range for input data scaling (default: [0, 1])
- `run_eagerly`: Enable eager execution for debugging (default: True)
- Additional parameters can be configured through the model constructor

## License

This project is licensed under the MIT License.