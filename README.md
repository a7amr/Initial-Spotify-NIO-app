# Initial-Spotify-NIO-app

[![Hugging Face – Live Demo](https://img.shields.io/badge/🤗%20Hugging%20Face-Live%20Demo-yellow.svg)](https://huggingface.co/spaces/a7madAmro/spotify-nio-demo)

Interactive app that predicts a Spotify track’s popularity from audio features and uses
**Neural Input Optimization (NIO)** to suggest how those features should change to make the
song *look* more popular to the model.

👉 **Live demo:** https://huggingface.co/spaces/a7madAmro/spotify-nio-demo  

---

## 🚀 Project overview

The goal of this project is to explore how far we can push a regression model by
optimizing its **inputs** instead of its weights.

1. **Popularity prediction model**
   - Train a regression model to predict a track’s popularity score (0–100) from
     Spotify audio features such as danceability, energy, loudness, tempo, etc.
   - Compare several models:
     - Baseline MLPs (PyTorch)
     - XGBoost (tree-based model)
     - A **hybrid neural net** trained on both the true labels and XGBoost’s
       predictions (distillation).
   - XGBoost gives the best raw accuracy, but the **hybrid neural net** is chosen
     as the final model because it is **fully differentiable** – which is
     required for NIO.

2. **Neural Input Optimization (NIO)**
   - Once the neural network is trained and frozen, we treat it as a function  

     \[
     \text{popularity} = f(\text{audio\_features})
     \]
   - For a given song (a vector of audio features), NIO optimizes the **inputs**
     themselves to increase the predicted popularity.
   - Only a subset of interpretable, continuous features is optimized:
     **danceability, energy, loudness, speechiness, acousticness,
     instrumentalness, liveness, valence, tempo**.
   - Other features remain fixed.
   - Optimization is gradient-based, with:
     - an L2 penalty to keep changes close to the original track
     - clamping to realistic ranges (0–1 for most features, −60–0 dB for
       loudness, 60–220 BPM for tempo).

The result is a system that can take a song’s audio profile and say:

> “If the world behaved like this model, moving your song’s features in these
> directions should increase its predicted popularity.”

It does **not** rewrite audio or guarantee real-world success – it’s a way to
interrogate the model and visualize its preferences.

---

## 🧱 Tech stack

- **Python**
- **PyTorch** – hybrid MLP popularity model
- **XGBoost** – teacher model for distillation
- **scikit-learn** – preprocessing & metrics
- **Streamlit** – interactive UI
- **Hugging Face Spaces** – deployment (Docker + Streamlit)

---

## 📊 Data (high-level)

The model is trained on a public Spotify tracks dataset containing:

- numeric **audio features** (danceability, energy, loudness, speechiness,
  acousticness, instrumentalness, liveness, valence, tempo, duration_ms, etc.)
- a **popularity** score (0–100) per track

Preprocessing:

- categorical columns one-hot encoded (e.g. genres / playlists)
- all numeric inputs standardized with `StandardScaler`
- train / validation / test split
- scaler, feature order, and model weights are saved as:
  - `models/nio_scaler.pkl`
  - `models/nio_feature_names.pkl`
  - `models/nio_model_hybrid.pth`

These artifacts are loaded by the app so it can run inference and NIO without
retraining.

---

## 🖥️ App features

The Streamlit app exposes the model and NIO in an interactive way:

- **Sidebar sliders** for the key audio features (with tooltips).
- **Presets** (Custom, Chill, High-energy EDM, Sad ballad) to quickly generate
  realistic starting tracks.
- **Popularity prediction cards**:
  - original predicted popularity
  - after NIO (clamped)
  - Δ popularity
- **NIO suggestions summary** – human-readable list of the top features the
  model wants to increase/decrease.
- **Detailed view**:
  - table comparing *original* vs *NIO-clamped* values for each feature
  - bar chart to visualize how each feature moves.

---

## 📁 Repository structure

```text
.
├─ app.py                     # Streamlit UI + NIO interface (entry point)
├─ model_nio.py               # Model loading + NIO optimization logic
├─ requirements.txt
├─ spotify-nio-clean.ipynb    # Notebook used for exploration / training
├─ spotify_xgb_best.json      # Saved XGBoost model (teacher)
├─ models/
│  ├─ nio_model_hybrid.pth    # Trained hybrid MLP weights
│  ├─ nio_scaler.pkl          # StandardScaler for input features
│  └─ nio_feature_names.pkl   # Ordered list of feature names used by the model
└─ README.md
