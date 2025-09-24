import os
import json
import io
from functools import lru_cache
import requests
import re

import streamlit as st
from PIL import Image
import torch
import timm
import torchvision.transforms as transforms

try:
    import google.generativeai as genai  # optional
    GENAI_API_KEY = os.getenv("GENAI_API_KEY") or st.secrets.get("GENAI_API_KEY", None)
    GENAI_MODEL = os.getenv("GENAI_MODEL", "gemini-1.5-flash")
    if GENAI_API_KEY:
        genai.configure(api_key=GENAI_API_KEY)
except Exception:
    GENAI_API_KEY = None
    GENAI_MODEL = None

MODEL_PATH = os.path.join('backend', 'plant_disease_model.pth')
CLASS_INDICES_PATH = os.path.join('backend', 'class_indices.json')


def resolve_model_url():
    """Return the model URL from env or secrets (or None)."""
    env_url = os.getenv('MODEL_URL')
    if env_url:
        return env_url.strip()
    if 'MODEL_URL' in st.secrets:
        return str(st.secrets['MODEL_URL']).strip()
    return None

st.set_page_config(page_title="Plant Disease Detector", page_icon="🌿", layout="centered")
st.title("🌿 Plant Disease Detection")
st.write("Upload a clear leaf image to classify one of the supported disease categories and optionally generate structured treatment information.")


@st.cache_resource(show_spinner=True)
def load_model():
    # Attempt to ensure model file exists; optionally download from MODEL_URL if provided
    if not os.path.exists(MODEL_PATH):
        dl_url = resolve_model_url()
        if dl_url:
            st.warning(f"Model file not found – attempting download from MODEL_URL: {dl_url}")
            try:
                os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
                # Detect Google Drive sharing link patterns
                gdrive_match = re.search(r"drive\.google\.com/file/d/([^/]+)/", dl_url)
                if gdrive_match:
                    file_id = gdrive_match.group(1)
                    st.info("Detected Google Drive link – using gdown to fetch the file.")
                    import gdown  # lazy import
                    gdown.download(f"https://drive.google.com/uc?id={file_id}", MODEL_PATH, quiet=False)
                else:
                    resp = requests.get(dl_url, stream=True, timeout=300)
                    resp.raise_for_status()
                    total = int(resp.headers.get('Content-Length', 0))
                    bytes_so_far = 0
                    CHUNK = 8192
                    progress = st.progress(0.0)
                    with open(MODEL_PATH, 'wb') as f:
                        for chunk in resp.iter_content(CHUNK):
                            if chunk:
                                f.write(chunk)
                                bytes_so_far += len(chunk)
                                if total:
                                    progress.progress(min(1.0, bytes_so_far / total))
                    if total and bytes_so_far < total:
                        raise RuntimeError("Download incomplete.")
                st.success("Model downloaded successfully.")
            except Exception as e:
                st.error(f"Automatic download failed: {e}")
                raise FileNotFoundError(f"Model missing and download failed. Provide 'backend/plant_disease_model.pth' or set MODEL_URL.")
        else:
            raise FileNotFoundError(
                "Model file 'backend/plant_disease_model.pth' not found. Commit it to the repo or set a MODEL_URL secret/env variable pointing to the weights.")
    model = timm.create_model('convmixer_1024_20_ks9_p14.in1k', pretrained=True, num_classes=38)
    state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    return model


@st.cache_resource
def load_class_indices():
    with open(CLASS_INDICES_PATH, 'r') as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def preprocess_image(pil_img, target_size=(256, 256)):
    pil_img = pil_img.convert('RGB').resize(target_size)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    tensor_img = transform(pil_img).unsqueeze(0)
    return tensor_img


@lru_cache(maxsize=256)
def fetch_disease_info(disease_name: str):
    if not GENAI_API_KEY:
        return "(No GENAI_API_KEY provided – skipping AI enrichment.)"
    try:
        model = genai.GenerativeModel(GENAI_MODEL)
        prompt = f"""
        You are a plant pathology assistant. Write a concise, structured note for the disease: {disease_name}.
        Use EXACT section headings with colons:
        What it is:
        Causes:
        Symptoms:
        Effects on the plant:
        Treatment Plan:
        Prevention Methods:
        Keep sentences simple.
        """
        resp = model.generate_content(prompt)
        return resp.text
    except Exception as e:
        return f"Error fetching AI details: {e}"


def parse_sections(text: str):
    import re
    sections = {}
    pattern = r"(What it is:|Causes:|Symptoms:|Effects on the plant:|Treatment Plan:|Prevention Methods:)([\s\S]*?)(?=(What it is:|Causes:|Symptoms:|Effects on the plant:|Treatment Plan:|Prevention Methods:|$))"
    for match in re.finditer(pattern, text):
        heading = match.group(1).strip(':')
        body = match.group(2).strip().replace('*', '').strip()
        if body:
            sections[heading] = body
    return sections or {"Details": text.strip() or "No details."}


with st.sidebar:
    st.header("⚙️ Options")
    show_ai = st.checkbox("Generate AI disease info", value=bool(GENAI_API_KEY))
    st.markdown("**Model file:** `plant_disease_model.pth`")
    st.markdown("**Architecture:** convmixer_1024_20_ks9_p14.in1k (timm)")
    st.markdown("**Mode:** Pure Streamlit (no Flask server)")
    if not GENAI_API_KEY:
        st.info("Add GENAI_API_KEY in Streamlit secrets to enable AI enrichment.")
    debug = st.checkbox("Show debug info")
    if debug:
        st.write("**Resolved MODEL_URL:**", resolve_model_url())
        st.write("**Model file exists?**", os.path.exists(MODEL_PATH))
        if not os.path.exists(MODEL_PATH):
            st.caption("If MODEL_URL is None: Go to Streamlit Cloud → Settings → Secrets and add a line: MODEL_URL=your_link")
            st.caption("Google Drive link format accepted: https://drive.google.com/file/d/<FILE_ID>/view?usp=...  (automatic conversion)")

uploaded = st.file_uploader("Upload leaf image", type=["jpg", "jpeg", "png"])

if uploaded:
    image_bytes = uploaded.read()
    pil_img = Image.open(io.BytesIO(image_bytes))
    st.image(pil_img, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Loading model & running inference..."):
        try:
            model = load_model()
        except FileNotFoundError as e:
            st.error("Model weights are missing on the server.")
            st.info(
                "Fix steps: 1) Commit 'backend/plant_disease_model.pth' (ensure <100MB) OR 2) Add a direct download URL as secret 'MODEL_URL' in app settings. "
                "Then rerun the app."
            )
            st.caption(str(e))
            st.stop()
        class_labels = load_class_indices()
        tensor = preprocess_image(pil_img)
        with torch.no_grad():
            outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_idx = int(probs.argmax(dim=1).item())
        pred_label = class_labels.get(pred_idx, f"Class {pred_idx}")

    st.success(f"Prediction: **{pred_label}**")

    # Show top-3 probabilities for transparency
    try:
        inv_map = {int(k): v for k, v in class_labels.items()}
        topk = torch.topk(probs, k=min(3, probs.shape[1]), dim=1)
        rows = []
        for rank, (idx, p) in enumerate(zip(topk.indices[0], topk.values[0]), start=1):
            rows.append({"Rank": rank, "Class": inv_map.get(int(idx), str(int(idx))), "Probability": f"{float(p)*100:.2f}%"})
        st.subheader("Top Predictions")
        st.table(rows)
    except Exception as e:
        st.caption(f"Could not compute top-k table: {e}")

    if show_ai:
        with st.spinner("Generating disease info (Gemini)..."):
            info_text = fetch_disease_info(pred_label)
        sections = parse_sections(info_text)
        st.subheader("Disease Information")
        for heading, body in sections.items():
            st.markdown(f"### {heading}")
            st.write(body)
else:
    st.info("Upload a leaf image (jpg / png) to begin. Clear, centered leaves perform best.")

with st.expander("ℹ️ How it works"):
    st.markdown(
        """
        **Pipeline**
        1. Image is resized to 256×256 and normalized (mean=0.5, std=0.5).
        2. ConvMixer (timm) model performs inference on CPU.
        3. Highest softmax probability is shown along with top-3 classes.
        4. (Optional) Gemini generates structured disease information (cached in memory).

        **Tips**
        - Use single leaf images on plain backgrounds for best accuracy.
        - Lighting and focus matters.
        - AI text is advisory; always validate agronomic recommendations.

        **Performance**
        - First prediction loads the model (cached afterwards).
        - Repeated disease info queries are cached (LRU cache size 256).
        """
    )

st.markdown("---")
st.caption("© 2025 Plant Disease Detector – Streamlit Edition")
