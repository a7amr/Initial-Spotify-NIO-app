# app.py
import streamlit as st
import pandas as pd

from model_nio import (
    load_nio_artifacts,
    run_nio_for_features,
    OPT_FEATURE_NAMES,
    NIO_RANGES,
)

st.set_page_config(page_title="Spotify NIO Demo", layout="centered")

st.title("🎧 Spotify Popularity – Neural Input Optimization (NIO)")
st.write(
    "This app uses a neural network trained on Spotify audio features. "
    "We apply **Neural Input Optimization (NIO)** to suggest changes in audio features "
    "that increase the model's predicted popularity."
)

# ---- Load model/scaler only once ----
@st.cache_resource
def get_model_bundle():
    return load_nio_artifacts(model_dir="models")

model, scaler, feature_names, device = get_model_bundle()

st.sidebar.header("Input audio features")

# Default slider values
defaults = {
    "danceability": 0.5,
    "energy": 0.5,
    "loudness": -8.0,
    "speechiness": 0.05,
    "acousticness": 0.1,
    "instrumentalness": 0.0,
    "liveness": 0.15,
    "valence": 0.3,
    "tempo": 120.0,
}

# 1) Collect user inputs for the optimizable features
user_features = {}

for feat in OPT_FEATURE_NAMES:
    vmin, vmax = NIO_RANGES[feat]
    default_val = defaults.get(feat, (vmin + vmax) / 2)
    step = 0.01 if vmax <= 1.0 else 1.0

    user_features[feat] = st.sidebar.slider(
        feat, float(vmin), float(vmax), float(default_val), step=step
    )

# 2) Fill the rest of the features with 0.0
for f in feature_names:
    if f not in user_features:
        user_features[f] = 0.0

if st.button("Run NIO"):
    with st.spinner("Optimizing features..."):
        res = run_nio_for_features(
            feature_dict=user_features,
            model=model,
            scaler=scaler,
            feature_names=feature_names,
            device=device,
        )

    y_orig = res["y_pred_orig"]
    y_nio  = res["y_pred_nio_clamped"]

    st.subheader("Popularity prediction")
    st.write(f"**Original input prediction:** {y_orig:.2f}")
    st.write(f"**After NIO (clamped):** {y_nio:.2f}")
    st.write(f"Δ popularity: **{(y_nio - y_orig):+.2f}**")

    base = res["base_dict"]
    optc = res["opt_clamped_dict"]

    st.subheader("Feature changes (original vs NIO)")
    rows = []
    for f in OPT_FEATURE_NAMES:
        rows.append({
            "feature": f,
            "original": base[f],
            "nio_clamped": optc[f],
            "delta": optc[f] - base[f],
        })
    df = pd.DataFrame(rows)
    st.dataframe(
        df.style.format(
            {"original": "{:.4f}", "nio_clamped": "{:.4f}", "delta": "{:+.4f}"}
        )
    )
else:
    st.info("Set the sliders on the left and click **Run NIO** to see suggestions.")
