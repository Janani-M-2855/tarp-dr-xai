"""TARP — Explainable DR Screening demo (Streamlit).
Run:  streamlit run app.py
Needs in the same folder: dr_utils.py, best_model.pth, quality_limits.json
Optional: model_config.json (which backbone / head / mask the model uses), metrics.json (sidebar numbers)"""
import json
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

import dr_utils as U

HERE = Path(__file__).parent
st.set_page_config(page_title="DR Screening — XAI", page_icon="👁️", layout="wide")


# ------------------------------------------------------------------ loading (cached once)
def load_json(name, default):
    p = HERE / name
    return json.load(open(p)) if p.exists() else default


MC = {**U.DEFAULT_MODEL_CONFIG, **load_json("model_config.json", {})}   # Day-1 model if no config file
IS_REG = MC["head"] == "reg"
THR = MC["thresholds"]


@st.cache_resource
def load_model(backbone, n_out):
    model = U.build_model(backbone, n_out, pretrained=False)
    model.load_state_dict(torch.load(HERE / "best_model.pth", map_location="cpu"))
    return model.eval()


model = load_model(MC["backbone"], 1 if IS_REG else 5)
limits = load_json("quality_limits.json", U.QUALITY_LIMITS)
metrics_all = load_json("metrics.json", {})
metrics = metrics_all.get("test", {})

# Grad-CAM layers: last spatial layer (12x12, coarse) vs stride-16 layer (24x24, finer)
CAM_LAYERS = {"Last layer (12×12, stable)": lambda m: U.cam_layer(m, MC["backbone"]),
              "Mid layer (24×24, finer detail)": lambda m: U.fine_cam_layer(m, MC["backbone"])}


# ------------------------------------------------------------------ helpers
def predict(x):
    """Flip-TTA. Classification -> 5 probabilities; regression -> one severity score (0-4)."""
    with torch.no_grad():
        a, b = model(x), model(torch.flip(x, dims=[3]))
    if IS_REG:
        return float((a + b)[0, 0]) / 2
    return ((a.softmax(1) + b.softmax(1)) / 2)[0].numpy()


def gradcam(x, grade, layer_name):
    target = 0 if IS_REG else grade          # regression: explain what raises the severity score
    with GradCAM(model=model, target_layers=[CAM_LAYERS[layer_name](model)]) as cam:
        return cam(input_tensor=x, targets=[ClassifierOutputTarget(target)])[0]


def overlay(img_rgb, heat, alpha):
    """Heatmap masked to the retina (no heat on the black border), blended on the image."""
    mask = (cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY) > 10).astype(np.float32)
    mask = cv2.erode(mask, np.ones((9, 9), np.uint8))          # drop the rim itself
    heat = cv2.GaussianBlur(heat, (0, 0), 3) * mask
    heat = heat / (heat.max() + 1e-8)
    color = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * heat), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    a = (alpha * heat)[..., None]                                # transparent where heat is low
    return np.uint8(img_rgb * (1 - a) + color * a), heat


def hotspot_regions(heat, thr=0.6):
    """Describe where the strongest evidence is (quadrant of the image)."""
    ys, xs = np.where(heat >= thr)
    if len(xs) == 0:
        return "no strong focal region"
    h, w = heat.shape
    names = set()
    for y, x in zip(ys[::25], xs[::25]):
        names.add(("upper" if y < h / 2 else "lower") + "-" + ("left" if x < w / 2 else "right"))
    return ", ".join(sorted(names))


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    cam_layer = st.radio("Heatmap resolution", list(CAM_LAYERS), index=0)
    alpha = st.slider("Heatmap opacity", 0.2, 0.9, 0.55, 0.05)
    st.divider()
    st.subheader("Referral cut-off")
    ref_thr = st.slider("Refer when P(grade ≥ 2) ≥", 0.05, 0.90, 0.50, 0.01,
                        help="Lower = catch more disease, more false referrals. On a new dataset this "
                             "cut-off must be re-tuned: on EyePACS, 0.50 caught only 40% of referable "
                             "cases, while 0.06 caught 90%.")
    st.divider()
    st.subheader("Demo tools")
    simulate_blur = st.checkbox("Simulate a blurry capture", help="Blurs the uploaded image to show the quality gate rejecting it.")
    bypass = st.checkbox("Analyse even if quality check fails")
    st.divider()
    st.subheader("Model")
    st.caption(f"{MC['backbone']} · {'ordinal regression' if IS_REG else '5-class'} head · "
               f"{'border mask · ' if MC['use_mask'] else ''}ImageNet → APTOS 2019 · 384×384 · green-channel CLAHE")
    if metrics:
        c1, c2 = st.columns(2)
        c1.metric("Test QWK", f"{metrics['qwk']:.3f}")
        c2.metric("Referable AUC", f"{metrics['ref_auc']:.3f}")
        c1.metric("Sensitivity", f"{metrics['sensitivity']:.1%}")
        c2.metric("Specificity", f"{metrics['specificity']:.1%}")
        if "border_heat" in metrics_all:
            c1.metric("Border heat", f"{metrics_all['border_heat']:.1%}", help="Share of Grad-CAM heat outside the retina (lower = better)")
        if "cpu_ms" in metrics_all:
            c2.metric("CPU time", f"{metrics_all['cpu_ms']:.0f} ms")

# ------------------------------------------------------------------ main page
st.title("👁️ Explainable Diabetic Retinopathy Screening")
st.caption("Screening aid for primary health centres · Quality check → DR grade (0–4) → Grad-CAM explanation → referral advice")

up = st.file_uploader("Upload a fundus photograph", type=["png", "jpg", "jpeg"])
if up is None:
    st.info("Upload a retinal fundus image to begin.")
    st.stop()

raw = cv2.imdecode(np.frombuffer(up.read(), np.uint8), cv2.IMREAD_COLOR)
if raw is None:
    st.error("Could not read this file as an image.")
    st.stop()
raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
if simulate_blur:
    raw = cv2.GaussianBlur(raw, (0, 0), max(raw.shape[:2]) / 150)

# ---- Step 1: quality gate
st.subheader("Step 1 · Image quality check")
ok, reasons, scores = U.quality_check(raw, limits)
q1, q2, q3, q4 = st.columns(4)
q1.metric("Sharpness", f"{scores['sharpness']:.0f}", help=f"reject below {limits['sharpness']:.0f}")
q2.metric("Brightness", f"{scores['brightness']:.0f}", help=f"accepted {limits['brightness'][0]:.0f}–{limits['brightness'][1]:.0f}")
q3.metric("Contrast", f"{scores['contrast']:.0f}", help=f"reject below {limits['contrast']:.0f}")
q4.metric("Retina area", f"{scores['fundus_area']:.0%}", help=f"reject below {limits['fundus_area']:.0%}")
if ok:
    st.success("✅ Image quality acceptable for grading")
else:
    st.error("❌ Image rejected — ungradable:\n\n" + "\n".join(f"- {r}" for r in reasons)
             + "\n\n**Action:** retake the photograph (clean lens, steady camera, dilate pupil if needed).")
    if not bypass:
        st.image(cv2.resize(raw, (384, int(384 * raw.shape[0] / raw.shape[1]))), caption="Uploaded image")
        st.stop()
    st.warning("Quality check bypassed — result below is unreliable.")

# ---- Step 2: grading
img = U.preprocess(raw, use_mask=MC["use_mask"])
x = torch.from_numpy(U.to_tensor_array(img))[None]
out = predict(x)
if IS_REG:
    score = out
    grade = U.score_to_grade(score, THR)
    margin = float(min(abs(score - t) for t in THR))        # distance to nearest cut-point
    refer = score >= THR[1]   # regression head uses its fitted cut-point, not the slider
    headline = f"Severity score {score:.2f} on a 0–4 scale · Referral cut-point {THR[1]:.2f}"
    uncertain = margin < 0.25
else:
    probs = out
    grade = int(probs.argmax())
    conf = float(probs[grade])
    p_ref = float(probs[2:].sum())
    refer = p_ref >= ref_thr
    headline = (f"Confidence {conf:.0%} · Probability of referable DR (grade ≥ 2): {p_ref:.0%} "
                f"(cut-off {ref_thr:.2f})")
    uncertain = conf < 0.5
action = U.ACTIONS[grade] if grade >= 2 or not refer else U.ACTIONS[2]

st.subheader("Step 2 · DR grading")
colour = ["#2e7d32", "#9e9d24", "#ef6c00", "#d84315", "#b71c1c"][grade]
st.markdown(
    f"<div style='padding:14px 18px;border-radius:10px;background:{colour};color:white'>"
    f"<div style='font-size:14px;opacity:.85'>Predicted stage</div>"
    f"<div style='font-size:28px;font-weight:700'>Grade {grade} — {U.GRADES[grade]}</div>"
    f"<div style='font-size:15px'>{headline}</div></div>",
    unsafe_allow_html=True)
st.write("")
left, right = st.columns([1, 1])
with left:
    if IS_REG:
        st.markdown("**Where the score falls on the grading scale**")
        edges = [-np.inf] + list(THR) + [np.inf]
        for g in range(5):
            lo, hi = max(edges[g], -0.5), min(edges[g + 1], 4.5)
            mark = "  ◀ **this image**" if g == grade else ""
            st.markdown(f"`{lo:5.2f} – {hi:5.2f}`  Grade {g} · {U.GRADES[g]}{mark}")
        st.progress(float(np.clip(score / 4, 0, 1)), text=f"Severity {score:.2f} / 4")
    else:
        st.markdown("**Probability per grade**")
        for g in range(5):
            st.progress(float(probs[g]), text=f"{g} · {U.GRADES[g]} — {probs[g]:.1%}")
with right:
    st.markdown("**Clinical summary**")
    st.markdown(f"- **Predicted stage:** Grade {grade} ({U.GRADES[grade]})\n"
                f"- **Typical findings at this stage:** {U.FINDINGS[grade]}\n"
                f"- **Referral:** {'🔴 YES' if refer else '🟢 NO'}\n"
                f"- **Recommended action:** {action}")
    if uncertain:
        st.warning("⚠️ Borderline case — the model is unsure between neighbouring grades. Recommend specialist review.")

# ---- Step 3: explanation
st.subheader("Step 3 · Why? (Grad-CAM explanation)")
with st.spinner("Computing heatmap…"):
    heat = cv2.resize(gradcam(x, grade, cam_layer), (img.shape[1], img.shape[0]))
    vis, heat_m = overlay(img, heat, alpha)
c1, c2, c3 = st.columns(3)
c1.image(cv2.resize(U.crop_fundus(raw), (384, 384)), caption="Original (cropped)")
c2.image(img, caption="Preprocessed (green-channel CLAHE" + (" + border mask" if MC["use_mask"] else "") + ") — model input")
c3.image(vis, caption=f"Grad-CAM for '{U.GRADES[grade]}'")
st.caption(f"Strongest evidence located in the **{hotspot_regions(heat_m)}** region(s). "
           "Red = areas that most increased the predicted grade. Grad-CAM shows *where* the model looked; "
           "the findings list describes *what* is typical at this grade.")

# ---- report download
report = "\n".join([
    "DIABETIC RETINOPATHY SCREENING REPORT (AI-assisted)", f"File: {up.name}", "",
    f"Image quality: {'Acceptable' if ok else 'REJECTED - ' + '; '.join(reasons)}",
    f"Predicted stage: Grade {grade} - {U.GRADES[grade]}",
    headline.replace("·", "|"),
    f"Referral: {'YES' if refer else 'NO'}",
    f"Typical findings: {U.FINDINGS[grade]}",
    f"Recommended action: {action}",
    f"Grad-CAM focus: {hotspot_regions(heat_m)}", "",
    "Screening aid only - not a diagnosis. Confirm with an ophthalmologist."])
st.download_button("📄 Download report", report, file_name=f"dr_report_{Path(up.name).stem}.txt")
st.divider()
st.caption("⚕️ Research prototype for screening support only — not a medical diagnosis. "
           "All referable or uncertain cases must be confirmed by an ophthalmologist.")
