# EpiReasoner

[![License:
MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python
3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![PyTorch
2.7](https://img.shields.io/badge/PyTorch-2.7-red.svg)](https://pytorch.org/)

------------------------------------------------------------------------

## 📘 Introduction 

**EpiReasoner: An Integrated Artificial Intelligence Framework for
Phenotype-to-Genotype Reasoning in Plant Epidermal Development**

EpiReasoner integrates visual phenotyping and knowledge-driven reasoning
to support research on stomatal complexes and pavement cells.

------------------------------------------------------------------------

## 🧩 Core Modules

### 1️⃣ EpiVision

-   Cross-species & cross-modality segmentatio
-   Supports bright-field, SEM, DIC imaging
-   Outputs 23 quantitative phenotypic indices

### 2️⃣ EpiBrain

-   LLM-based reasoning engine
-   Phenotype prediction
-   Genotype inference
-   Mechanistic hypothesis generation

------------------------------------------------------------------------

## 🚀 Quick Start 

### Option 1: Windows EXE (No Python required)

1.  Download `EpiReasoner_v1.0.0.zip`
2.  Unzip the package
3.  Double-click `EpiReasoner.exe`

------------------------------------------------------------------------

### Option 2: Run from Source

#### Requirements

-   Python 3.10
-   CUDA 12.x (GPU recommended)
-   16GB+ RAM

#### Installation
``` bash
conda create -n epireasoner python=3.10 -y
conda activate epireasoner

# Install PyTorch (CUDA example)
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu126

pip install -r requirements.txt
```

------------------------------------------------------------------------

## 📊 Example Usage

### Segmentation

``` python
from epivision import EpiVision

model = EpiVision()
result = model.segment("leaf.jpg", output_dir="results/")
```

### Phenotype Extraction

``` python
from phenotyping import PhenotypeExtractor

extractor = PhenotypeExtractor()
metrics = extractor.compute_all(...)
```

### Reasoning

``` python
from epibrain import EpiBrain

brain = EpiBrain()
response = brain.query("Explain stomatal clustering mechanism.")
print(response)
```

------------------------------------------------------------------------

## 📈 Quantitative Indices

### Morphology

-   Stomatal area
-   Stomatal perimeter
-   Pore area

### Spatial Distribution

-   SD (Stomatal Density)
-   SI (Stomatal Index)
-   SCVR (Cluster Violation Rate)

------------------------------------------------------------------------

## 📦 Dependencies

-   Python 3.10
-   PyTorch 2.7.0
-   Transformers 4.x
-   OpenCV
-   NumPy
-   Pandas

------------------------------------------------------------------------

## 📝 Citation

``` bibtex
@article{epireasoner2026,
  title={EpiReasoner: AI Framework for Phenotype-to-Genotype Reasoning},
  year={2026}
}
```

------------------------------------------------------------------------

## 📄 License

MIT License

------------------------------------------------------------------------

## 🤝 Contribution

Pull requests and issues are welcome.
