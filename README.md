# Radiomics Feature Processing Pipeline (NPC)

This project provides a robust, end-to-end pipeline for Medical Image (MRI/CT) Preprocessing and High-Dimensional Radiomics Feature Extraction. It is specifically optimized for complex mask scenarios such as unified multi-class masks, separate dual masks (Primary Tumor + Lymph Nodes), and instance-level sub-masks.

## 🚀 Key Features
1. **Dynamic Dataset Conversion**: Tools to seamlessly ingest raw data and translate them into standardized formats.
2. **Unified Data Preprocessing**: 
   - Non-zero Bounding Box Auto-Cropping.
   - N4 Bias Field Correction.
   - B-Spline Image Resampling & Nearest-Neighbor Mask Resampling.
   - High-precision Z-Score Intensity Normalization.
   - Native support for **Multi-Mask Preprocessing** (e.g., maintaining temporal and spatial alignment between `segmentation_primary.nii.gz` and `segmentation_lymph.nii.gz`).
3. **Smart Experiment Planning**: Automatically scans your dataset properties (spacing, sizes, intensity statistics) to generate strategic PyRadiomics configuration files (`radiomics_config.yaml`), intelligently avoiding double-normalization conflicts.
4. **Robust OOP Feature Extraction**: Built on a solid `BaseRadiomicsExtractor`, supporting:
   - `SingleMaskExtractor`: Standard single label extraction.
   - `MultiInstanceExtractor`: Automatic scanning and multi-label extraction for instance-segmentation masks (`1, 2, 3..`).
   - `SeparateMasksExtractor`: Simultaneous double label extraction for spatially separated masks.

## 📁 Project Structure
```text
RadiomicsProject/
├── dataset_conversion/    # Scripts to convert/clean raw masks (Dataset001_NPC.py, etc.)
├── experiment_planning/   # Generates data fingerprints and `radiomics_config.yaml`
├── feature_extraction/    # PyRadiomics core classes (Base, Single, Multi, Separate)
├── preprocessing/         # Z-Score, N4 Correction, Resampling, Cropping
├── scripts/               # Individual launch scripts (preprocessing, extraction)
├── config/                # PyRadiomics template yaml files
└── run_pipeline.py        # 🌟 Unified CLI Entrypoint
```

## 🛠️ Unified CLI Usage (`run_pipeline.py`)

The entire pipeline is wrapped inside a single command-line interface `run_pipeline.py`. 

### 1. Data Conversion
Before using the pipeline, organize your raw data using the dataset conversion scripts (e.g., `dataset_conversion/Dataset001_NPC.py`). These scripts will build standardized structures inside your designated `Radiomics_raw` folder.

### 2. Preprocessing & Config Generation
Run preprocessing on a raw dataset. This step will automatically generate a tailored `radiomics_config.yaml` and preprocess all images and labels into the `Radiomics_preprocessed` folder.

**For standard unified masks (`segmentation.nii.gz`):**
```bash
python run_pipeline.py npc_preprocess Dataset001_NPC_Combined
```

**For separate dual-masks (e.g., Primary and Lymph separated):**
```bash
python run_pipeline.py npc_preprocess Dataset004_NPC_Separate --label_files segmentation_primary.nii.gz segmentation_lymph.nii.gz
```

### 3. Feature Extraction
Extracts radiomics features using PyRadiomics. Outputs are saved as `.csv` and/or `.parquet` inside the `Radiomics_results` folder.

**Mode: Single (One specific label, e.g., 1)**
```bash
python run_pipeline.py extract Dataset001_NPC_Combined --mode single --label 1
```

**Mode: Multi (Extracts all found instances `1, 2, 3...` in the mask)**
```bash
python run_pipeline.py extract Dataset003_NPC_Instance_Combined --mode multi
```

**Mode: Separate (Extracts from two separate mask geometries simultaneously)**
```bash
python run_pipeline.py extract Dataset004_NPC_Separate --mode separate --label_files segmentation_primary.nii.gz segmentation_lymph.nii.gz
```

*Optional Arguments:*
- `--cores <INT>`: Specify multi-processing threads (default: 8).
- `--force`: Force overwrite existing extraction results.

---

## ⚙️ Radiomics Configurations & Strategy

By default, the pipeline uses the `Z-Score Normalization Strategy`. 
- **Z-Score in Preprocessing**: The preprocessing engine explicitly clips the background and performs rigorous Z-Score normalization on the foreground (Mean=0, Std=1).
- **PyRadiomics Config (`radiomics_config.yaml`)**:
  - `normalize`: **Disabled** (`False`). Because the image is already explicitly standardized, enabling internal PyRadiomics normalization would severely distort the texture matrices (GLCM, GLRLM).
  - `binWidth`: `0.5`. Perfectly scaled for Z-Score normalized pixels (which typically span from -3 to +3). 
  - `imageType`: Supports `Original`, `LoG` (Laplacian of Gaussian, Sigma 1.0/3.0/5.0), and `Wavelet` transformations out of the box.

## 📌 Environment
Require a Python 3.x environment with `SimpleITK`, `pyradiomics`, `pandas`, `pyarrow`, and `numpy` installed.
```bash
conda activate python312_venv
```
