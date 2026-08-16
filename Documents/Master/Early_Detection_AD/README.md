# Early Detection of Alzheimer’s Disease (Early_Detection_AD)

This repository focuses on the early detection of Alzheimer’s Disease (AD) using structural MRI (sMRI) preprocessing, handcrafted features (volumes, shape, texture), and both classical and quantum machine learning classifiers.

## Project Structure

- **01_data/**: Contains raw (reference), processed (symlinked), and external data (atlases/templates).
- **02_preprocessing/**: Bash scripts for sMRI processing, logs, and registration configurations.
- **03_code/**:
    - `03a_data_investigation_visualization/`: EDA and Quality Control scripts/notebooks.
    - `03b_classical_models/`: Training and prediction for classical ML models.
    - `03c_quantum_models/`: Quantum circuits and hybrid quantum-classical classifiers.
    - `03d_baselines/`: Comparison models (RF, SVM, XGBoost, CNN).
- **04_results/**: Figures, metrics (accuracy, sensitivity, specificity, AUC), and saved models.
- **05_requirements/**: Environment setup and dependencies.

## Data Setup

The processed data is stored externally to the repository. To link your local processed data to this project, use one of the following methods:

### On Windows (Command Prompt as Administrator)
```cmd
mklink /J "01_data\processed" "C:\path\to\your\data\processed_without_progressed_subjects\AD"
```

### On Linux / WSL
```bash
ln -s /home/enooo/Documents/Master/data/processed_without_progressed_subjects/AD ./01_data/processed
```

## Preprocessing Pipeline

The preprocessing pipeline follows standard neuroimaging practices optimized for early detection:

1.  **Skull Stripping (Brain Extraction)**: Removal of non-brain tissues. While the skull is not "unchangeable" and can serve as a spatial reference, modern pipelines often strip it to focus analysis on brain parenchyma and reduce noise (Fischl, 2012).
2.  **N4 Bias Field Correction**: Corrects for B1 field inhomogeneities in sMRI data, which is crucial for intensity-based feature extraction (Tustison et al., 2014).
3.  **Registration**: Spatial normalization to a standard template (e.g., MNI152).
    - **Justification**: For early detection, high-dimensional non-linear registration (DARTEL or SyN) is often preferred over simple affine transformations to better capture subtle hippocampal and cortical atrophy (Ashburner, 2007).

### Order of Operations
The order (Skull Strip -> Bias Correction -> Registration) is critical because bias field correction improves skull stripping accuracy, and both are required for high-quality spatial normalization.

## Technical Considerations

- **sMRI Protocol**: This pipeline is optimized for **MPRAGE** (Magnetization Prepared Rapid Gradient Echo) T1-weighted sequences.
- **Domain Adaptation**: Protocol mismatches (e.g., different scanner manufacturers or field strengths) can introduce site effects. Users should be aware of domain adaptation techniques when generalizing models across different cohorts.

## References
- Ashburner, J. (2007). A fast diffeomorphic image registration algorithm. *NeuroImage*.
- Fischl, B. (2012). FreeSurfer. *NeuroImage*.
- Tustison, N. J., et al. (2014). Large-scale reproducible white matter microstructural imaging. *Nature Communications*.
