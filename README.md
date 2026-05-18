# SR_ISS_Water_Extraction

Super-resolution water body extraction from Sentinel-2 imagery.

## Introduction

This repository provides the implementation of the SR_ISS framework proposed for super-resolution reconstruction and water body extraction from Sentinel-2 imagery.

The proposed method reconstructs Sentinel-2 multispectral imagery from 10 m / 20 m spatial resolution to 2.5 m spatial resolution based on image self-similarity. The reconstructed SWIR1 band is subsequently used for high-resolution water body extraction.

---

## Repository Structure

```text
SR_ISS/
│
├── training/
│   ├── create_patches.py
│   ├── create_random.py
│   └── test_train.ipynb
│
├── testing/
│   └── experiment.ipynb
│
├── utils/
│   └── model files
│
└── README.md
```

---

## Requirements

Recommended environment:

* Python 3.9+
* PyTorch
* NumPy
* OpenCV
* Rasterio
* GDAL
* Matplotlib
* Jupyter Notebook

Install dependencies using:

```bash
pip install -r requirements.txt
```

---

## Training Workflow

### 1. Create Training Patches

Generate image patches for model training:

```bash
python SR_ISS/training/create_patches.py
```

---

### 2. Generate Random Training Samples

Create randomized training datasets:

```bash
python SR_ISS/training/create_random.py
```

---

### 3. Train the ISS Model

Open and run:

```text
SR_ISS/training/test_train.ipynb
```

The notebook contains the complete training workflow for the ISS reconstruction model.

Required model files are provided in:

```text
SR_ISS/utils/
```

---

## Testing and Reconstruction

Example reconstruction workflow:

```text
SR_ISS/testing/experiment.ipynb
```

The reconstruction process performs a single ×2 super-resolution operation at each stage.

To obtain the final 2.5 m reconstruction result, the reconstruction procedure must be executed three times sequentially:

1. 20 m → 10 m
2. 10 m → 5 m
3. 5 m → 2.5 m

After each reconstruction stage, replace the input file path with the generated output from the previous stage before running the next reconstruction step.

The output from the third reconstruction stage is the final super-resolved result.

---

## Water Body Extraction

After super-resolution reconstruction, water body extraction can be performed using:

* MNDWI
* Otsu thresholding

The reconstructed SWIR1 band is used to improve water boundary delineation and reduce mixed-pixel effects.

---

## Notes

* `experiment.ipynb` is provided as an example reconstruction workflow.
* Sentinel-2 imagery used in this study includes:

  * Red
  * Green
  * Blue
  * NIR
  * SWIR1 bands
* The repository focuses on reconstruction and water extraction workflows.

---

## Citation

If you use this repository in your research, please cite the corresponding paper.

```text
Super-resolution Water Body Extraction from Sentinel-2 Imagery
```

---

## License

This project is released for academic research purposes.
