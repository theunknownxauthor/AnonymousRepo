# GeoKRL Model

This directory contains the complete implementation of **GeoKRL**, a multimodal multi-task geospatial  model designed to learn transferable representations from heterogeneous Earth observation data. It includes all components required for backbone pretraining, downstream transfer learning, dataset indexing, dynamic raster loading, balanced multimodal sampling, and experiment management.

The implementation is designed around an **on-demand data pipeline**, allowing large raster datasets to be processed without loading them entirely into memory.

---

# Directory Structure

| File | Description |
|------|-------------|
| `GeoKRL.py` | Implementation of the GeoKRL backbone and downstream prediction models. |
| `train.py` | multi-task backbone pretraining. |
| `train_task.py` | Downstream transfer learning for population estimation, biomass estimation, and building prediction. |
| `dataset_index.py` | Builds the hierarchical dataset index. |
| `raster_manager.py` | Dynamic raster loading and patch extraction. |
| `sample_generator.py` | On-demand sample generation for multi-task learning. |
| `sample_generator_task.py` | Sample generation for downstream prediction tasks. |
| `modality_sampler.py` | Balanced multimodal sampling strategy. |
| `batch_generator.py` | Mini-batch generator for backbone pretraining. |
| `batch_generator_task.py` | Mini-batch generator for downstream learning. |
| `config.py` | Dataset paths and global configuration. |
| `hyperparameters.py` | Backbone hyperparameter search space. |
| `task_hyperparameters.py` | Downstream hyperparameter search space. |

---

# Overall Training Pipeline

GeoKRL follows a four-stage pipeline.

```
GeoKG Dataset
      │
      ▼
Dataset Index Construction
      │
      ▼
Dynamic Raster Loading
      │
      ▼
On-demand Sample Generation
      │
      ▼
Mini-batch Generation
      │
      ▼
GeoKRL Backbone Pretraining
      │
      ▼
Transfer Learning
      │
      ▼
Task-specific Prediction
```

---

# 1. Dataset Index Construction

Before training, the complete GeoKG dataset is indexed.

Rather than loading raster data into memory, the index stores only metadata required during training, including

- dataset split
- country
- ROI
- raster file locations
- normalization statistics

The hierarchical index enables efficient random access to every sample during training.

Implemented in

```
dataset_index.py
```

---

# 2. Dynamic Raster Loading

GeoKRL never loads the complete dataset into memory.

Instead, raster files remain on disk throughout training.

Whenever a sample is requested, the framework

1. locates the corresponding raster,
2. extracts only the requested pixel or patch,
3. applies normalization,
4. immediately returns the sample.

This mechanism enables training on datasets much larger than the available system memory.

Implemented in

```
raster_manager.py
```

---

# 3. On-demand Sample Generation

Training samples are generated dynamically.

For every sampled pixel, GeoKRL extracts

- pixel-level features,
- local spatial context,
- global spatial context,
- GPS coordinates,
- modality availability mask.

No intermediate datasets are generated.

Implemented in

```
sample_generator.py
sample_generator_task.py
```

---

# 4. Balanced Multimodal Sampling

During backbone pretraining, GeoKRL constructs balanced multimodal samples by

- uniformly sampling countries,
- uniformly sampling ROIs,
- uniformly sampling tasks,
- randomly selecting modality combinations,
- randomly selecting pixels.

This strategy avoids over-representation of any particular modality or geographic region.

Implemented in

```
modality_sampler.py
```

---

# 5. Backbone Pretraining

The GeoKRL backbone is trained using multi-task multimodal learning.

The complete training pipeline is

```
CSV Sample Index
        │
        ▼
Batch Generator
        │
        ▼
Sample Generator
        │
        ▼
Raster Manager
        │
        ▼
GeoKRL Backbone
```

Backbone pretraining is launched using

```bash
python train.py
```

During training,

- multimodal samples are generated dynamically,
- raster patches are extracted on demand,
- validation loss is monitored,
- the best model checkpoint is automatically saved.

---

# 6. Downstream Transfer Learning

After backbone pretraining, the learned representations are transferred to downstream prediction tasks.

Currently supported tasks are

- Population estimation
- Biomass estimation
- Building prediction

The pretrained backbone may either

- remain frozen, or
- be fine-tuned,

depending on the selected experiment.

Downstream training is performed using

```bash
python train_task.py --task population
```

```bash
python train_task.py --task biomass
```

```bash
python train_task.py --task building
```

---

# Hyperparameter Optimization

GeoKRL performs exhaustive grid search for both backbone pretraining and downstream transfer learning.

Backbone hyperparameters include

- patch size
- global patch size
- latent dimension
- learning rate
- reconstruction loss weight
- classification loss weight
- KL-divergence weight
- batch size

The search space is defined in

```
hyperparameters.py
```

Downstream prediction hyperparameters include

- number of residual prediction blocks
- bottleneck ratio
- dropout
- learning rate
- weight decay
- backbone freezing strategy

The search space is defined in

```
task_hyperparameters.py
```

The configuration achieving the lowest validation loss is automatically retained.

---

# Automatic Model Checkpoints

The best backbone model is automatically saved in

```
model/
└── auto_train_models/
```

Each checkpoint corresponds to the configuration achieving the lowest validation loss during multi-task pretraining.

---

# Downstream Models

Task-specific models are automatically stored in

```
model/
└── task_auto_train_models/
```

Separate models are generated for each downstream task and hyperparameter configuration.

---

# Training Logs

GeoKRL automatically records

- experiment configuration,
- hyperparameters,
- training history,
- validation history,
- test performance,
- model checkpoints.

These files are generated automatically during training and can be used to reproduce every experiment reported in the accompanying publication.

---

# Reproducibility

GeoKRL fixes the random seed for

- Python
- NumPy
- TensorFlow

to ensure deterministic and reproducible experiments whenever supported by the execution environment.

---

# Requirements

The implementation requires

- Python 3.10+
- TensorFlow
- NumPy
- Rasterio
- GDAL
- ArcPy (only for data preprocessing; not required during model training)

---

# Citation

If you use GeoKRL in your research, please cite the accompanying publication once available.

```
@article{GeoKRL2026,
  title   = {GeoKRL: A Multimodal multi-task  Model for Geospatial Representation Learning},
  author  = {...},
  journal = {...},
  year    = {2026}
}
```
