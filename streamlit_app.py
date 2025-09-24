import os
import json
import io
from functools import lru_cache

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

st.set_page_config(page_title="Plant Disease Detector", page_icon="🌿", layout="centered")
st.title("🌿 Plant Disease Detection")
st.write("Upload a leaf image to classify the disease and optionally generate treatment info.")


@st.cache_resource(show_spinner=True)
def load_model():
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
    st.markdown("**Backend removed:** This is a pure Streamlit deployment.")
    if not GENAI_API_KEY:
        st.info("Add GENAI_API_KEY in Streamlit secrets to enable AI enrichment.")

uploaded = st.file_uploader("Upload leaf image", type=["jpg", "jpeg", "png"])

if uploaded:
    image_bytes = uploaded.read()
    pil_img = Image.open(io.BytesIO(image_bytes))
    st.image(pil_img, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Loading model & predicting..."):
        model = load_model()
        class_labels = load_class_indices()
        tensor = preprocess_image(pil_img)
        with torch.no_grad():
            outputs = model(tensor)
        pred_idx = outputs.argmax(dim=1).item()
        pred_label = class_labels.get(pred_idx, f"Class {pred_idx}")

    st.success(f"Prediction: **{pred_label}**")

    if show_ai:
        with st.spinner("Generating disease info (Gemini)..."):
            info_text = fetch_disease_info(pred_label)
        sections = parse_sections(info_text)
        st.subheader("Disease Information")
        for heading, body in sections.items():
            st.markdown(f"### {heading}")
            st.write(body)
else:
    st.info("Upload a leaf image to begin.")

st.markdown("---")
st.caption("© 2025 Plant Disease Detector – Streamlit Edition")
