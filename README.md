# An End-to-End Multi-View Unsupervised Contrastive Learning Framework for Seismic Facies Analysis

## 📖 Overview

Unsupervised seismic facies analysis efficiently characterizes lateral reservoir distributions without the need for extensive annotations. However, existing methods often lack physically-informed architectures, resulting in inadequate reservoir representation and struggling to finely delineate complex sedimentary boundaries.

To address this, we propose an end-to-end unsupervised Multi-View Contrastive Clustering (MVCC) framework. Driven by the massive data volumes typical of seismic exploration, our designed feature extractor synergizes **lightweight StarBlocks** with a **Global-Local State Space module (GLSS2D)** to model long-range stratigraphic dependencies. Ultimately, by integrating spatial translation contrastive learning and Deep Embedded Clustering (DEC), the framework achieves the progressive alignment of the feature space and classification boundaries.



## 🗂️ Repository Structure

```
├── model.py      # Core network architectures (SeismicStarViT, GLSS2D, CAM, etc.)
├── train.py      # Data loading, Contrastive + DEC joint training engine
├── plot.py       # Full-map inference and facies overlay visualization
└── README.md     # Project documentation
```



## ⚙️ Installation & Dependencies

Ensure you have Python 3.8+ and PyTorch 1.12+ installed.

```
# Core dependencies
pip install torch torchvision numpy scipy matplotlib tqdm

# (Recommended) Install Mamba for fast State Space Modeling
# If not installed, the code will automatically fallback to a standard PyTorch implementation (slower).
pip install causal-conv1d>=1.2.0
pip install mamba-ssm
```

🚀 Usage

```
python train.py
python plot.py
```
