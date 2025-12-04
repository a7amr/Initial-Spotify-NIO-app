# model_nio.py

import torch
import torch.nn as nn
import numpy as np
import joblib

# ---------------- Model definition ----------------

class SpotifyNioMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 768),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_nio_artifacts(model_dir="models"):
    scaler = joblib.load(f"{model_dir}/nio_scaler.pkl")
    feature_names = joblib.load(f"{model_dir}/nio_feature_names.pkl")

    input_dim = len(feature_names)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SpotifyNioMLP(input_dim).to(device)
    state = torch.load(f"{model_dir}/nio_model_hybrid.pth", map_location=device)
    model.load_state_dict(state)
    model.eval()

    return model, scaler, feature_names, device

# ---------------- NIO logic ----------------

# Features we allow to change
OPT_FEATURE_NAMES = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
]

# Realistic ranges in ORIGINAL feature space
NIO_RANGES = {
    "danceability":      (0.0, 1.0),
    "energy":            (0.0, 1.0),
    "speechiness":       (0.0, 1.0),
    "acousticness":      (0.0, 1.0),
    "instrumentalness":  (0.0, 1.0),
    "liveness":          (0.0, 1.0),
    "valence":           (0.0, 1.0),
    "loudness":         (-60.0, 0.0),
    "tempo":            (60.0, 220.0),
}


def run_nio_for_features(
    feature_dict,
    model,
    scaler,
    feature_names,
    device,
    num_steps=300,
    lr=0.05,
    lambda_l2=0.1,
    clamp_min=-3.0,
    clamp_max=3.0,
):
    """
    feature_dict: {feature_name: value in ORIGINAL space}
                  should at least contain all OPT_FEATURE_NAMES.
    Returns:
      {
        "y_pred_orig",
        "y_pred_nio_clamped",
        "base_dict",
        "opt_clamped_dict",
      }
    """

    # 1) Build original feature vector in correct order
    x0_unscaled = np.array(
        [feature_dict.get(f, 0.0) for f in feature_names],
        dtype=np.float32
    ).reshape(1, -1)

    # Scale
    x0_scaled = scaler.transform(x0_unscaled).astype(np.float32)
    x0_scaled_np = x0_scaled.copy()

    # Original prediction
    x0_scaled_t = torch.from_numpy(x0_scaled).float().to(device)
    with torch.no_grad():
        y0_pred = float(model(x0_scaled_t).item())

    # 2) Setup optimizable parameters in SCALED space
    opt_indices = [feature_names.index(f) for f in OPT_FEATURE_NAMES if f in feature_names]
    if len(opt_indices) == 0:
        raise ValueError("No optimizable features found in feature_names.")

    x_params_init = x0_scaled_np[:, opt_indices]
    x_params = torch.tensor(
        x_params_init,
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    x_params_init_t = torch.tensor(
        x_params_init,
        dtype=torch.float32,
        device=device,
    )

    optimizer_nio = torch.optim.Adam([x_params], lr=lr)

    # 3) NIO loop (gradient descent on inputs)
    for _ in range(num_steps):
        optimizer_nio.zero_grad()

        # full scaled input
        x_full = torch.from_numpy(x0_scaled_np).float().to(device)
        x_full = x_full.clone()
        x_full[:, opt_indices] = x_params

        y_pred = model(x_full)
        loss_pop = -y_pred.mean()  # maximize popularity

        l2_term = torch.mean((x_params - x_params_init_t) ** 2)
        loss = loss_pop + lambda_l2 * l2_term

        loss.backward()
        optimizer_nio.step()

        with torch.no_grad():
            x_params.clamp_(clamp_min, clamp_max)

    # 4) Decode NIO result (scaled → original)
    x_opt_scaled = x0_scaled_np.copy()
    x_opt_scaled[:, opt_indices] = x_params.detach().cpu().numpy()

    # Back to original feature space
    x0_unscaled_back = scaler.inverse_transform(x0_scaled_np)[0]
    x_opt_unscaled_back = scaler.inverse_transform(x_opt_scaled)[0]

    base_dict = dict(zip(feature_names, x0_unscaled_back))
    opt_dict  = dict(zip(feature_names, x_opt_unscaled_back))

    # 5) Clamp in ORIGINAL space
    opt_clamped_dict = opt_dict.copy()
    for feat, (vmin, vmax) in NIO_RANGES.items():
        if feat in opt_clamped_dict:
            v = opt_clamped_dict[feat]
            opt_clamped_dict[feat] = max(vmin, min(vmax, v))

    # Rebuild vector, scale, predict again
    x_opt_clamped_unscaled = np.array(
        [opt_clamped_dict[f] for f in feature_names],
        dtype=np.float32
    ).reshape(1, -1)
    x_opt_clamped_scaled = scaler.transform(x_opt_clamped_unscaled).astype(np.float32)

    x_opt_clamped_t = torch.from_numpy(x_opt_clamped_scaled).float().to(device)
    with torch.no_grad():
        y_opt_clamped = float(model(x_opt_clamped_t).item())

    return {
        "y_pred_orig": y0_pred,
        "y_pred_nio_clamped": y_opt_clamped,
        "base_dict": base_dict,
        "opt_clamped_dict": opt_clamped_dict,
    }
