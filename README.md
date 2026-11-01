# Explainable AI for Diabetic Retinopathy Screening 

A lightweight diabetic retinopathy (DR) screening prototype. Given a retinal fundus photo, it checks whether the image is usable, grades it on the 0–4 ICDR scale, shows a Grad-CAM heatmap of where the model looked, and suggests a referral action.

Course project for CSI3901 (TARP) by Janani M and Vaishnavi P.V.

> **Research prototype, not a medical device.** It has not been clinically validated and must not be used to make decisions about patients.

## What the demo does

1. **Quality gate.** Checks basic image quality (for example blur and brightness) against limits learned from the training images, and warns before grading.
2. **Grading.** MobileNetV3-Large predicts DR grade 0–4 and a referable-DR probability (grade ≥ 2). The referral cut-off can be changed in the sidebar.
3. **Explanation.** Grad-CAM overlay masked to the retina, with a text description of the main hotspot region. The "typical findings" text is a per-grade lookup, not lesion detection.
4. **Report.** Downloadable text summary of the result and the recommended action.


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

## Built with

PyTorch, timm, pytorch-grad-cam, OpenCV, albumentations, scikit-learn, Streamlit.
