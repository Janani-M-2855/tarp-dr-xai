# Explainable AI for Diabetic Retinopathy Screening 

A lightweight diabetic retinopathy (DR) screening prototype. Given a retinal fundus photo, it checks whether the image is usable, grades it on the 0–4 ICDR scale, shows a Grad-CAM heatmap of where the model looked, and suggests a referral action.

Course project for CSI3901 (TARP) by Janani M and Vaishnavi P.V.

> **Research prototype, not a medical device.** It has not been clinically validated and must not be used to make decisions about patients.

## What the demo does

1. **Quality gate.** Checks basic image quality (for example blur and brightness) against limits learned from the training images, and warns before grading.
2. **Grading.** MobileNetV3-Large predicts DR grade 0–4 and a referable-DR probability (grade ≥ 2). The referral cut-off can be changed in the sidebar.
3. **Explanation.** Grad-CAM overlay masked to the retina, with a text description of the main hotspot region. The "typical findings" text is a per-grade lookup, not lesion detection.
4. **Report.** Downloadable text summary of the result and the recommended action.

## Results

### APTOS 2019, held-out test split (550 images, single seed)

Stratified 70/15/15 split (2,563 / 549 / 550), seed 42.

| Run | Model / change | QWK | Accuracy | Referable AUC | Sensitivity | Specificity | Border heat ↓ | Size (MB) | CPU ms |
|---|---|---|---|---|---|---|---|---|---|
| A0 | EfficientNet-B0, cross-entropy | 0.889 | 0.822 | 0.979 | 0.919 | 0.933 | 52.8% | 16.4 | 77 |
| A1 | EfficientNet-B0, class-weighted CE | 0.887 | 0.807 | 0.980 | 0.951 | 0.917 | 49.9% | 16.4 | 93 |
| A2 | EfficientNet-B0, ordinal regression | 0.887 | 0.716 | 0.975 | 0.942 | 0.902 | 47.8% | 16.3 | 87 |
| B1 | EfficientNet-B0, circular mask | 0.875 | 0.802 | 0.976 | 0.933 | 0.921 | 54.1% | 16.4 | 86 |
| C1 | ResNet50 | 0.917 | 0.849 | 0.988 | 0.933 | 0.945 | 12.4% | 94.4 | 206 |
| **C2** | **MobileNetV3-Large (used in the demo)** | 0.898 | 0.844 | 0.984 | 0.951 | 0.939 | 30.8% | 17.0 | 40 |

- QWK = quadratic weighted kappa on the 5 grades. Referable = grade ≥ 2 vs. grade 0–1.
- Border heat = share of Grad-CAM heat that falls outside the inner retina (lower is better). Different backbones use different Grad-CAM layers, so compare it with care.
- CPU ms = single-image latency measured on a Kaggle notebook CPU, not on a laptop, phone or field device.
- With 550 test images the 95% interval on QWK is about ±0.03, so most of the gaps between runs are within noise. The loss functions and the circular mask made no clear difference.
- ResNet50 scores slightly higher on QWK, AUC and border heat, but the gap to MobileNetV3 is not significant and ResNet50 is about 5× larger and 5× slower. MobileNetV3 was chosen as the best size/speed/accuracy trade-off, not as the most accurate model. That choice was made after seeing test results.

### EyePACS (external, resized Kaggle mirror, 4,000 evaluation images)

MobileNetV3-Large, referable DR:

| Setting | AUC | QWK | Sensitivity | Specificity |
|---|---|---|---|---|
| Zero-shot (trained on APTOS only) | 0.780 | 0.422 | 0.405 | 0.929 |
| Threshold recalibrated (Youden) | – | – | 0.603 | 0.824 |
| Threshold recalibrated (sensitivity ≥ 0.90) | – | – | 0.899 | 0.368 |
| Fine-tuned on 2,000 EyePACS images | 0.854 | 0.533 | 0.507 | 0.961 |

Models that score well on APTOS lose a lot of discrimination on EyePACS (AUC about 0.98 → 0.78). Recalibrating the threshold restores sensitivity only at a large false-referral cost, and small-scale fine-tuning recovers part of the gap. The quality gate rejected 22.7% of EyePACS images but did not improve agreement on the images that passed.

## Repository layout

```
tarp_demo/        Streamlit demo (app.py, dr_utils.py, model files, requirements.txt)
notebooks/        Kaggle notebooks: M0_setup, Day1_train, M5_improvements, M7_eyepacs
results/          results tables and figures
```

Datasets are not included. Download them from Kaggle under their own terms:
APTOS 2019 Blindness Detection and the EyePACS / Diabetic Retinopathy Detection data.

## Run the demo

Python 3.11 is recommended.

```bash
cd tarp_demo
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m streamlit run app.py
```

`tarp_demo/` needs `app.py`, `dr_utils.py`, `best_model.pth`, `model_config.json`, `metrics.json` and `quality_limits.json`. `best_model.pth` is the MobileNetV3-Large (run C2) checkpoint.

## Reproduce the experiments

Run the notebooks on Kaggle (GPU T4 ×2, Internet on) in this order:

1. `M0_setup.ipynb` – environment check and smoke test
2. `Day1_train.ipynb` – split, preprocessing, EfficientNet-B0 baseline
3. `M5_improvements.ipynb` – loss, mask and backbone comparison, border-heat metric
4. `M7_eyepacs.ipynb` – EyePACS zero-shot evaluation, recalibration, fine-tuning, quality gate

Preprocessing: crop black border → pad to square → resize to 384 → green-channel CLAHE → optional circular mask.
Training: ImageNet-pretrained backbones from `timm`, 20 epochs, AdamW (lr 3e-4), OneCycle, batch 32, mixed precision, flip test-time augmentation, best epoch picked by validation QWK.

## Limitations

- One training seed per run, two datasets, no external clinical validation.
- APTOS has no patient IDs, so duplicate or same-patient images across splits can't be ruled out.
- The EyePACS images come from a resized mirror, so camera, resolution and compression effects are mixed with dataset shift.
- Grad-CAM is a model-attention view, not proof that the model found lesions. Heatmaps were not checked against lesion annotations.
- Latency was measured on a Kaggle CPU. Smartphone or field-camera performance has not been tested.
- The quality-gate thresholds were computed on all APTOS images, including the test split (no labels used).

## Built with

PyTorch, timm, pytorch-grad-cam, OpenCV, albumentations, scikit-learn, Streamlit.
