# 🌿 Plant Disease Detection

Welcome! This is a pet project that combines **deep learning** and **web development** to detect plant diseases from leaf images.

A **Flask-powered API** hosts a PyTorch model, and a **React frontend** lets you upload a photo of a leaf and instantly see a prediction.

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

## 🔨 How to Run Locally

### 1. ✨ Clone the Repository

```bash
git clone https://github.com/SayanRoy001/Plant-Disease-Detection.git
cd Plant-Disease-Detection
```

### 2. 💻 Backend Setup (Flask + PyTorch)

```bash
cd backend
```

#### a. Create a Virtual Environment

**PowerShell (Windows)**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

#### b. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### c. Start Flask Server

```bash
python main.py
```

Output:

```
* Running on http://127.0.0.1:5000/
```

### 3. 🔎 Frontend Setup (React)

```bash
cd ../frontend
```

#### a. Install Node.js & npm

If not installed, download from [https://nodejs.org](https://nodejs.org)

#### b. Check Installation

```bash
node --version
npm --version
```

#### c. Install React Dependencies

```bash
npm install
```

#### d. Start React Dev Server

```bash
npm start
```

This opens: [http://localhost:3000](http://localhost:3000)

---

## 🧪 Testing the App

1. Go to **[http://localhost:3000/](http://localhost:3000/)**
2. You'll see the **"Predict the Disease"** interface.
3. Upload a diseased plant leaf image.
4. Click **Submit**.
5. See the predicted disease name and confidence score.

Example:

```
Disease: Apple Early Blight
Confidence: 0.97
```

---

## 🎓 Technologies Used

* React.js (Frontend)
* Flask (Backend)
* PyTorch (Model Inference)
* Axios (API calls)
* CSS/Bootstrap (Styling)

---

## 💼 License

Open-source for educational and research use. Contributions are welcome!
