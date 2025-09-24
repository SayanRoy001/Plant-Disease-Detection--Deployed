# 🌿 Plant Disease Detection – Streamlit Edition

This project started as a **Flask API + React frontend** and is now simplified into a single **Streamlit app** that bundles the model inference and (optionally) AI‑generated disease treatment information. The legacy Flask + React stack is still in the repo but you only need `streamlit_app.py` for a live deployment.

---

## 🌱 Why This Project Exists

* ✨ **Instant Feedback**: Upload a picture of a leaf, get a diagnosis in seconds.
* ⚖️ **Hands-On Learning**: Combines a Flask API with a React UI.
* 📊 **Open to Contributions**: Add new plant types, improve the UI, or optimize the model.

---

## 🖼️ What You'll See

### 1. **Home Page (React)**

A clean, modern interface where you can **drag & drop** or browse for a plant leaf image.

### 2. **Backend Endpoint (Flask)**

Receives the image and returns a JSON response like:

```json
{
  "disease": "Tomato Early Blight",
}
```

### 3. **Results**

Displays the predicted disease name and confidence after clicking “Submit.”

> ⚡ Tip: Use a clear, well-lit image of a single leaf for best results.

---

## 🔨 How to Run Locally (Streamlit Mode)

### 1. ✨ Clone the Repository

```bash
git clone https://github.com/SayanRoy001/Plant-Disease-Detection.git
cd Plant-Disease-Detection
```

### 1. Create & Activate Virtual Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Place Model Weights

Ensure your trained model file is at `backend/plant_disease_model.pth`.

If you cannot commit it (size > 100MB), set an environment variable or Streamlit secret:

```
MODEL_URL=https://direct-download-link/plant_disease_model.pth
```

### 4. (Optional) Gemini API Key

Add `GENAI_API_KEY` to a local `.env` (not committed) or Streamlit secrets to enable AI disease info generation.

### 5. Run Streamlit

```powershell
streamlit run streamlit_app.py
```

Browse: http://localhost:8501

### 6. (Legacy Mode – Flask + React)
If you want the old architecture, see the previous revision of this README or the `backend/` and `frontend/` folders.

---

## 🚀 Deploying a Live Streamlit App (Persistent Link)

You can host this app on **Streamlit Community Cloud** for a free always-on (lightly sleeping) link:

1. Push this repository to GitHub (public).
2. Go to https://share.streamlit.io
3. Click "Deploy an app".
4. Select the repo + branch `main` + main file: `streamlit_app.py`.
5. (Secrets) Add in the UI under Settings → Secrets:
  ```
  GENAI_API_KEY = "your_real_key"
  # Only if not committing weights:
  # MODEL_URL = "https://direct-download-link/plant_disease_model.pth"
  ```
6. Deploy – first run may take longer (model download / install).

### Custom Domain (Optional)
Use a redirect or a small HTML page elsewhere linking to the Streamlit URL.

### Alternatives
| Platform | Notes |
|----------|-------|
| Streamlit Cloud | Easiest. Paste secrets. |
| Render (Web Service) | Use `streamlit run streamlit_app.py` start command. Add persistent disk if downloading model each boot. |
| Hugging Face Spaces | Use `docker` or `streamlit` space; good for ML demos. |

## 🔐 Secrets & Environment

Local development: create `.env` (NOT committed):
```
GENAI_API_KEY=your_key_here
```

Streamlit Cloud: use **Secrets** UI (or copy `.streamlit/secrets.example.toml` → `secrets.toml` locally for testing).

In production the app auto-detects:
* `GENAI_API_KEY` – enables Gemini disease info
* `MODEL_URL` – optional model auto-download if weights not committed

---

## 🎓 Technologies Used

* Streamlit (Unified UI + inference delivery)
* PyTorch + timm (ConvMixer model)
* Pillow / torchvision (Preprocessing)
* Optional: Google Gemini (`google-generativeai`)
* (Legacy) Flask + React stack retained but not required

---

## 💼 License

Open-source for educational and research use. Contributions are welcome!
