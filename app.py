import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from cnn_model import CNN
from model import get_model

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

IMG_SIZE = 224

CLASS_LABELS = {
    0: "No DR",
    1: "Mild DR",
    2: "Moderate DR",
    3: "Severe DR",
    4: "Proliferative DR",
}

CLASS_COLORS = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c", "#8e44ad"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────

inference_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# ─────────────────────────────────────────────
# MODEL LOADING (cached)
# ─────────────────────────────────────────────

@st.cache_resource
def load_cnn():
    model = CNN()
    ckpt = torch.load("best_cnn_model.pth", map_location=DEVICE)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


@st.cache_resource
def load_effnet():
    model = get_model()
    ckpt = torch.load("best_effnet_model.pth", map_location=DEVICE)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model

# ─────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────

def predict(model, tensor):
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
    pred_class = int(np.argmax(probs))
    return pred_class, probs

# ─────────────────────────────────────────────
# PAGE SETUP
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Diabetic Retinopathy Grader",
    page_icon="👁️",
    layout="wide",
)

# ── Custom CSS ──
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main { background: #0f1117; }

    .hero {
        background: linear-gradient(135deg, #1a1f35 0%, #0d1b2a 100%);
        border: 1px solid #2a3550;
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
        text-align: center;
    }
    .hero h1 {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(90deg, #60a5fa, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .hero p { color: #8b9ab3; font-size: 1rem; margin: 0; }

    .card {
        background: #1a1f35;
        border: 1px solid #2a3550;
        border-radius: 12px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1.2rem;
    }
    .card h3 { color: #c9d4f0; font-size: 1rem; font-weight: 600; margin: 0 0 0.8rem 0; }

    .badge {
        display: inline-block;
        padding: 0.35rem 0.9rem;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.03em;
    }
    .grade-label {
        font-size: 1.5rem;
        font-weight: 700;
        color: #e2e8f0;
    }
    .conf-val {
        font-size: 1.1rem;
        font-weight: 600;
        color: #60a5fa;
    }
    .separator { border-color: #2a3550; margin: 0.5rem 0; }

    /* Progress bar colors */
    .stProgress > div > div { background-color: #60a5fa !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────

st.markdown("""
<div class="hero">
    <h1>👁️ Diabetic Retinopathy Grader</h1>
    <p>Upload a retinal fundus image — both CNN and EfficientNet-B0 models will grade it simultaneously.</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────

with st.spinner("Loading models…"):
    cnn_model   = load_cnn()
    effnet_model = load_effnet()

st.success(f"✅  Both models loaded  |  Device: `{DEVICE}`")

# ─────────────────────────────────────────────
# UPLOAD
# ─────────────────────────────────────────────

uploaded = st.file_uploader(
    "Upload a retinal fundus image (PNG / JPG / JPEG)",
    type=["png", "jpg", "jpeg"],
)

if uploaded is not None:

    image = Image.open(uploaded).convert("RGB")
    tensor = inference_tf(image).unsqueeze(0).to(DEVICE)

    # ── Run both models ──
    cnn_class,    cnn_probs    = predict(cnn_model,   tensor)
    effnet_class, effnet_probs = predict(effnet_model, tensor)

    # ─────────────────────────────────────────────
    # LAYOUT
    # ─────────────────────────────────────────────

    col_img, col_cnn, col_eff = st.columns([1.2, 1, 1], gap="large")

    # ── Image preview ──
    with col_img:
        st.markdown('<div class="card"><h3>📷 Uploaded Image</h3>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── CNN results ──
    def render_model_card(col, model_name, icon, pred_class, probs):
        with col:
            color = CLASS_COLORS[pred_class]
            label = CLASS_LABELS[pred_class]
            conf  = probs[pred_class] * 100

            st.markdown(
                f'<div class="card"><h3>{icon} {model_name}</h3>'
                f'<p class="grade-label">{label}</p>'
                f'<span class="badge" style="background:{color}22;color:{color};border:1px solid {color}66;">'
                f'Grade {pred_class}</span><br><br>'
                f'<span class="conf-val">Confidence: {conf:.1f}%</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

            st.markdown('<div class="card"><h3>Class Probabilities</h3>', unsafe_allow_html=True)
            for i, (lbl, p) in enumerate(zip(CLASS_LABELS.values(), probs)):
                st.markdown(
                    f'<span style="color:{CLASS_COLORS[i]};font-size:0.82rem;">'
                    f'Grade {i} — {lbl}</span>',
                    unsafe_allow_html=True,
                )
                st.progress(float(p))
                st.markdown(
                    f'<p style="text-align:right;color:#8b9ab3;font-size:0.78rem;margin-top:-0.6rem;">'
                    f'{p*100:.1f}%</p>',
                    unsafe_allow_html=True,
                )
            st.markdown('</div>', unsafe_allow_html=True)

    render_model_card(col_cnn, "Custom CNN",       "🧠", cnn_class,    cnn_probs)
    render_model_card(col_eff, "EfficientNet-B0",  "⚡", effnet_class, effnet_probs)

    # ─────────────────────────────────────────────
    # COMPARISON CHART
    # ─────────────────────────────────────────────

    st.markdown("---")
    st.markdown("### 📊 Side-by-Side Confidence Comparison")

    fig, ax = plt.subplots(figsize=(10, 3.5))
    fig.patch.set_facecolor("#1a1f35")
    ax.set_facecolor("#1a1f35")

    x       = np.arange(5)
    width   = 0.35
    bars1   = ax.bar(x - width/2, cnn_probs * 100,    width, label="Custom CNN",      color="#60a5fa", alpha=0.85)
    bars2   = ax.bar(x + width/2, effnet_probs * 100, width, label="EfficientNet-B0", color="#818cf8", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Grade {i}\n{CLASS_LABELS[i]}" for i in range(5)], color="#c9d4f0", fontsize=8)
    ax.set_ylabel("Confidence (%)", color="#8b9ab3")
    ax.set_ylim(0, 105)
    ax.tick_params(colors="#8b9ab3")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a3550")
    ax.yaxis.label.set_color("#8b9ab3")
    ax.legend(facecolor="#0d1b2a", edgecolor="#2a3550", labelcolor="#c9d4f0")

    # value labels on bars
    for bar in bars1:
        h = bar.get_height()
        if h > 2:
            ax.text(bar.get_x() + bar.get_width()/2, h + 1, f"{h:.1f}%",
                    ha="center", va="bottom", color="#c9d4f0", fontsize=7)
    for bar in bars2:
        h = bar.get_height()
        if h > 2:
            ax.text(bar.get_x() + bar.get_width()/2, h + 1, f"{h:.1f}%",
                    ha="center", va="bottom", color="#c9d4f0", fontsize=7)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # ─────────────────────────────────────────────
    # AGREEMENT / DISAGREEMENT BADGE
    # ─────────────────────────────────────────────

    if cnn_class == effnet_class:
        st.success(
            f"✅ **Both models agree:** Grade {cnn_class} — {CLASS_LABELS[cnn_class]}"
        )
    else:
        st.warning(
            f"⚠️ **Models disagree** — "
            f"CNN predicts **{CLASS_LABELS[cnn_class]}** (Grade {cnn_class}), "
            f"EfficientNet predicts **{CLASS_LABELS[effnet_class]}** (Grade {effnet_class})."
        )

else:
    st.info("👆  Please upload a retinal fundus image to get started.")
