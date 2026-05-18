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
│   ├── supres.py
│   └── experiment.ipynb
│
├── utils/
│   └── model files
│
├── data/
│   ├── network/
│   ├── test_data/
│   │   ├── E98N34_10bands_10m_clip4.tif
│   │   ├── E98N34_10bands_5m_clip4.tif
│   │   ├── E98N34_10bands_2.5m_clip4.tif
│   │   ├── E98N34_4328bands_5m_clip4.tif
│   │   └── E98N34_4328bands_2.5m_clip4.tif
│   └── train_data/
│
└── README.md
```

---

## Requirements

Recommended environment:

* Python 3.9+
* NumPy
* OpenCV
* Rasterio
* GDAL
* Matplotlib
* Jupyter Notebook
* tensorflow
* keras

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
SR_ISS/training/train.ipynb
```

The notebook contains the complete training workflow for the ISS reconstruction model.

Required model files are provided in:

```text
SR_ISS/utils/
```

Due to storage limitations, the Sentinel-2 training datasets are provided via Google Drive:

```text
Google Drive link
```

---

## Testing and Reconstruction

Example reconstruction workflow:

```text
SR_ISS/testing/experiment.ipynb
```

The reconstruction process performs a single ×2 super-resolution operation at each stage.


In `experiment.ipynb`, the variable `dir_im_high` corresponds to the auxiliary high-resolution imagery (`4328bands`), while `dir_im_low` corresponds to the low-resolution imagery (`10bands`) to be super-resolved at each reconstruction stage.

To obtain the final 2.5 m reconstruction result, the reconstruction procedure must be executed three times sequentially:

1. 20 m → 10 m
   (`E98N34_4328bands_10m.tif` + `E98N34_6bands_20m.tif`
   → `E98N34_10bands_10m.tif`)

2. 10 m → 5 m
   (`E98N34_4328bands_5m_clip4.tif` + `E98N34_10bands_10m_clip4.tif`
   → `E98N34_10bands_5m_clip4.tif`)

3. 5 m → 2.5 m
   (`E98N34_4328bands_2.5m_clip4.tif` + `E98N34_10bands_5m_clip4.tif`
   → `E98N34_10bands_2.5m_clip4.tif`)

The `10bands` data and `4328bands` data should be used together during reconstruction.The first reconstruction stage is performed on the full Sentinel-2 image to generate the initial 10 m super-resolved result. The subsequent 10 m → 5 m and 5 m → 2.5 m reconstruction stages are conducted on local clipped regions to facilitate detailed visual comparison, reduce computational cost, and improve the efficiency of high-resolution reconstruction experiments.

At each stage:

* `4328bands` data are used as auxiliary high-resolution imagery
* `10bands` data are progressively super-resolved during reconstruction

After each reconstruction stage, replace the input file path with the generated output from the previous stage before running the next reconstruction step.

Final reconstruction result:

```text
E98N34_10bands_2.5m_clip4.tif
```

Due to storage limitations, the complete 10 m and 20 m Sentinel-2 testing datasets are provided via Google Drive, while only partial 10 m, 5 m, and 2.5 m sample data are included in `data/test_data` for demonstration and reconstruction examples.

```text
Google Drive link
```

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

