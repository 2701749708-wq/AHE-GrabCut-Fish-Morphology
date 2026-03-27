# Enhanced Underwater Fish Morphology Quantification via AHE-GrabCut Fusion and Binocular Stereo Vision

This is the official implementation for the manuscript submitted to **The Visual Computer**.

> Important Notice
> This code is directly associated with the manuscript submitted to *The Visual Computer*. If you use this code or dataset in your research, please cite our corresponding paper.

---

## 1. Project Introduction
This project implements a non-destructive machine vision system for underwater fish morphology quantification.
It integrates:
- Adaptive Histogram Equalization (CLAHE/AHE)
- GrabCut foreground segmentation
- Binocular stereo vision measurement
- Contour extraction and physical size calculation

The system achieves sub-millimeter accuracy and 64.0% faster speed than state-of-the-art deep learning models.

---

## 2. Environment & Dependencies
- Python 3.8 or higher
- opencv-python >= 4.5
- numpy >= 1.21
- pillow >= 8.0
- tkinter (built-in)

### Install Dependencies
```bash
pip install opencv-python numpy pillow
```



---

## 3. How to Run
1. Prepare fish images (single or stereo side-by-side)
2. Run the main GUI program
```bash
python code.py
```
3. Select image type: single/stereo
4. Load image and click **Image processing** to process
5. Click **Start measurement** to obtain real physical distance (mm)

---

## 4. Core Algorithms
- CLAHE/AHE Enhancement: `adaptive_histogram_equalization()`
- GrabCut Segmentation: `grab_cut_segmentation()`
- Largest Contour Extraction: `outline_largest_contour()`
- Stereo Image Split: `split_stereo_image()`
- Physical Distance Calculation: `calculate_and_display_distance()`
- GUI System: `ImageProcessorUI`

---

## 5. Camera Parameters Used in This Study
```bash
K = np.array([[643.1389, 0, 0],
              [0, 642.0301, 0],
              [320.6402, 227.6466, 1]],
              dtype=np.float32)dist = np.array([0.1836, -0.3672, 0, 0, 0], dtype=np.float32)
```

---

## 6. Dataset
The dataset used in this study includes underwater fish images in various scenarios:
- Clear/turbid water
- Single/multiple fish
- Binocular stereo image pairs

Dataset is available upon publication.

Public Benchmark Dataset
The paper uses the public underwater fish dataset published by Luo et al. (2025) in Sensors for model validation (the dataset is the benchmark of MasYOLOv11).
The public dataset used in the experiment is cited as follows:
@Article{s25113433,
AUTHOR = {Luo, Yang and Wu, Aiping and Fu, Qingqing},
TITLE = {MAS-YOLOv11: An Improved Underwater Object Detection Algorithm Based on YOLOv11},
JOURNAL = {Sensors},
VOLUME = {25},
YEAR = {2025},
NUMBER = {11},
ARTICLE-NUMBER = {3433},
URL = {https://www.mdpi.com/1424-8220/25/11/3433},
PubMedID = {40968954},
ISSN = {1424-8220},
}



---

## 7. Code & Data Availability
- GitHub: https://github.com/2701749708-wq/AHE-GrabCut-Fish-Morphology/tree/main
- Code DOI: https://doi.org/10.5281/zenodo.19253733
- Dataset DOI: https://doi.org/10.5281/zenodo.19255131

---

## 8. Experimental Results
- Right camera MAE: 0.525 ± 0.150 mm
- Left camera MAE: 1.140 ± 0.230 mm
- Average measurement error < 1.74%
- Speed improvement: 64.0% compared with SOTA deep learning

---

## 9. Citation
If you use this code, please cite our paper:
```bash
@article{qin2026enhanced,
title={Enhanced Underwater Fish Morphology Quantification via AHE-GrabCut Fusion and Binocular Stereo Vision},
author={Yongqi Qin, Yixiang Xu, Mengran Liu, Bo Zhang, Feng Sun},
journal={The Visual Computer},
year={2026},
note={Submitted}}
```
---

## 10. License
This project is released under the MIT License for academic use.
