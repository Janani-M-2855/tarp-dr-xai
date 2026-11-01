"""dr_utils.py — shared code for the TARP DR-XAI project.
Used by BOTH the Kaggle training notebook and the Streamlit demo, so an image
is preprocessed the exact same way at training time and at demo time."""
import cv2
import numpy as np

IMG_SIZE = 384
GRADES = {0: "No DR", 1: "Mild NPDR", 2: "Moderate NPDR", 3: "Severe NPDR", 4: "Proliferative DR"}
# Typical findings per grade (ICDR scale). Grad-CAM shows WHERE the model looked;
# these describe WHAT is usually present at that grade.
FINDINGS = {
    0: "No visible abnormalities",
    1: "Microaneurysms only",
    2: "Microaneurysms, dot/blot haemorrhages, hard exudates",
    3: "Extensive haemorrhages in 4 quadrants, venous beading, IRMA",
    4: "Neovascularisation and/or vitreous/pre-retinal haemorrhage",
}
ACTIONS = {
    0: "Routine re-screening in 12 months",
    1: "Re-screen in 6–12 months; optimise blood-sugar control",
    2: "REFER to ophthalmologist within 3 months",
    3: "URGENT referral to ophthalmologist within 4 weeks",
    4: "URGENT referral — same week (risk of vision loss)",
}
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)  # ImageNet stats (pretrained backbone)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------- preprocessing
def crop_fundus(img_rgb, thresh=10):
    """Remove the black border around the circular fundus."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    mask = gray > thresh
    if mask.sum() < 0.05 * mask.size:          # nearly all black -> leave as is
        return img_rgb
    rows, cols = np.where(mask.any(1))[0], np.where(mask.any(0))[0]
    return img_rgb[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]


def pad_to_square(img):
    """Pad with black so resizing doesn't squash the round fundus into an oval."""
    h, w = img.shape[:2]
    s = max(h, w)
    top, left = (s - h) // 2, (s - w) // 2
    return cv2.copyMakeBorder(img, top, s - h - top, left, s - w - left,
                              cv2.BORDER_CONSTANT, value=(0, 0, 0))


def green_clahe(img_rgb, clip=2.0, tiles=8):
    """CLAHE on the green channel only: green has the best contrast for
    red lesions (microaneurysms, haemorrhages) and yellow exudates."""
    out = img_rgb.copy()
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles))
    out[:, :, 1] = clahe.apply(out[:, :, 1])
    return out


def circle_mask(img, r_frac=0.95):
    """Black out everything outside a centred circle. Every image then has the SAME
    round boundary, so the border/notch shape can no longer be used as a shortcut."""
    h, w = img.shape[:2]
    m = np.zeros((h, w), np.uint8)
    cv2.circle(m, (w // 2, h // 2), int(r_frac * min(h, w) / 2), 1, -1)
    return img * m[..., None]


def preprocess(img_rgb, size=IMG_SIZE, use_clahe=True, use_mask=False):
    """Full pipeline: crop border -> pad square -> resize -> (green CLAHE) -> (circle mask). uint8 RGB."""
    img = pad_to_square(crop_fundus(img_rgb))
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    if use_clahe:
        img = green_clahe(img)
    return circle_mask(img) if use_mask else img


def retina_region(img_rgb, erode_px=12):
    """Boolean mask of the retina, shrunk by erode_px so the rim itself counts as 'border'."""
    m = (cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY) > 10).astype(np.uint8)
    return cv2.erode(m, np.ones((erode_px, erode_px), np.uint8)).astype(bool)


def to_tensor_array(img_rgb_uint8):
    """uint8 HxWx3 -> normalised float32 3xHxW (what the model expects)."""
    x = (img_rgb_uint8.astype(np.float32) / 255.0 - MEAN) / STD
    return x.transpose(2, 0, 1)


# ---------------------------------------------------------------- image quality
def quality_scores(img_rgb):
    """Cheap image-quality measures, computed on the resized green channel."""
    img = cv2.resize(crop_fundus(img_rgb), (512, 512), interpolation=cv2.INTER_AREA)
    g = img[:, :, 1]
    mask = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) > 10
    fundus = g[mask] if mask.any() else g.ravel()
    return {
        "sharpness": float(cv2.Laplacian(g, cv2.CV_64F)[mask].var()) if mask.any() else 0.0,
        "brightness": float(fundus.mean()),
        "contrast": float(fundus.std()),
        "fundus_area": float(mask.mean()),   # fraction of the frame that is retina
    }


# Thresholds are placeholders; the Day-1 notebook prints dataset percentiles to tune them.
QUALITY_LIMITS = {"sharpness": 15.0, "brightness": (25.0, 220.0), "contrast": 10.0, "fundus_area": 0.35}


def quality_check(img_rgb, limits=QUALITY_LIMITS):
    """Returns (ok: bool, reasons: list[str], scores: dict)."""
    s = quality_scores(img_rgb)
    reasons = []
    if s["fundus_area"] < limits["fundus_area"]:
        reasons.append("Retina not detected / badly framed")
    if s["sharpness"] < limits["sharpness"]:
        reasons.append("Image is blurry (out of focus or motion)")
    lo, hi = limits["brightness"]
    if s["brightness"] < lo:
        reasons.append("Image too dark (under-exposed)")
    if s["brightness"] > hi:
        reasons.append("Image too bright (over-exposed / glare)")
    if s["contrast"] < limits["contrast"]:
        reasons.append("Very low contrast")
    return len(reasons) == 0, reasons, s


# ---------------------------------------------------------------- model
def build_model(backbone="efficientnet_b0", num_classes=5, pretrained=True):
    """num_classes=5 -> classification head; num_classes=1 -> regression (ordinal) head."""
    import timm
    return timm.create_model(backbone, pretrained=pretrained, num_classes=num_classes, drop_rate=0.3)


def cam_layer(model, backbone):
    """Last SPATIAL conv layer for Grad-CAM. (MobileNetV3's conv_head runs after pooling,
    so it has no spatial map - use its last block instead.)"""
    if backbone.startswith("resnet"):
        return model.layer4[-1]
    if backbone.startswith("mobilenetv3"):
        return model.blocks[-1]
    return model.conv_head                      # EfficientNet


def fine_cam_layer(model, backbone):
    """A stride-16 layer (24x24 map at 384px) for finer heatmaps."""
    if backbone.startswith("resnet"):
        return model.layer3[-1]
    return model.blocks[4]                      # EfficientNet & MobileNetV3: stride-16 stage


DEFAULT_THRESHOLDS = [0.5, 1.5, 2.5, 3.5]


def score_to_grade(score, thresholds=DEFAULT_THRESHOLDS):
    """Regression score (continuous 0-4) -> integer grade via cut-points."""
    return int(np.digitize(score, np.asarray(thresholds)))   # same rule used when fitting thresholds


DEFAULT_MODEL_CONFIG = {"backbone": "efficientnet_b0", "head": "cls", "use_mask": False,
                        "thresholds": DEFAULT_THRESHOLDS, "img_size": IMG_SIZE}
