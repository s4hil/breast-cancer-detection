
# 🩺 BreastcareAI

**BreastcareAI** is a Flask-based medical diagnostic web application designed to predict breast cancer malignancy using either:

- 🖼️ **Image-based input** (breast scan)
- 📊 **Feature-based JSON input** (30 clinical features)

It supports intelligent diagnostics powered by a DeepSeek inference engine and provides both visual and API interfaces.

---

## 🚀 Features

- 🧠 Deep learning (`.h5`) model for breast scan image analysis.
- 🔍 SVM-based model (`.pkl`) for 30-feature JSON data prediction.
- 📈 Confidence score and model accuracy shown.
- 🧾 JSON API endpoint for programmatic access.
- 🧬 DeepSeek Diagnostics for severity and suggestions.
- 🌓 Light/Dark theme toggle.
- 📂 Upload system for scan images.

---

## 🛠️ Tech Stack

- **Backend:** Flask, Python
- **ML Models:** Keras (CNN), Scikit-learn (SVM)
- **Frontend:** HTML, CSS (Custom + FontAwesome), Bootstrap
- **Utilities:** `deepseek`, `tensorflow`, `pickle`, `json`

---

