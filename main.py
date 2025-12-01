print(">>> main.py loaded from:", __file__)

# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import json, numpy as np, torch
import joblib

app = FastAPI(title="PAM50-50feature-predictor")

# load meta
with open("feature_json_v2.json") as f:
    meta = json.load(f)
feature_methyl = meta["feature_methyl"]
feature_rna = meta["feature_rna"]
with open("mean_methyl_json.json") as f:
    methyl_mean_meta = json.load(f)
methyl_mean = [methyl_mean_meta[i] for i in feature_methyl]
with open("mean_rna_json.json") as f:
    rna_mean_meta = json.load(f)
rna_mean = [rna_mean_meta[i] for i in feature_rna]

# Save to file
rna_raw_scaler = joblib.load("rna_scaler_np1.pkl")
rna_scaler = joblib.load( "scaler_rna_lw.pkl")
methyl_scaler = joblib.load("scaler_methyl_lw.pkl")
print('this is right!!!')
label_map = {'LumA': 0, 'LumB': 1, 'Her2': 2, 'Basal': 3}
# load TorchScript
model_t = torch.jit.load("mamba_ts5.pt", map_location="cpu")
model_t.eval()


# --- Pydantic models ---
class Sample(BaseModel):
    sample_id: str = None
    feature_methyl: dict
    feature_rna: dict

class PredictRequest(BaseModel):
    samples: List[Sample]

# --- Helper function ---
def preprocess_sample(sample: Sample):
    # Validate keys
    if set(sample.feature_methyl.keys()) != set(feature_methyl) or set(sample.feature_rna.keys()) != set(feature_rna):
        raise HTTPException(status_code=400, detail="Feature keys mismatch")
    print(sample.feature_methyl[feature_methyl[0]])
    x_methyl = np.array([sample.feature_methyl[k] for k in feature_methyl], dtype=np.float32)
    x_rna = np.array([sample.feature_rna[k] for k in feature_rna], dtype=np.float32)

    # Impute missing values
    methyl_nan_mask = np.isnan(x_methyl)
    if methyl_nan_mask.any():
        x_methyl[methyl_nan_mask] = methyl_mean[methyl_nan_mask]
    rna_nan_mask = np.isnan(x_rna)
    if rna_nan_mask.any():
        x_rna[rna_nan_mask] = rna_mean[rna_nan_mask]

    # Scale RNA
    x_rna_scaled = rna_raw_scaler.transform(x_rna.reshape(1, -1))
    x_rna_scaled = rna_scaler.transform(x_rna_scaled)
    x_methyl_scaled = methyl_scaler.transform(x_methyl.reshape(1,-1))

    # Convert to tensors
    x_methyl_tensor = torch.from_numpy(x_methyl_scaled)  # shape (1, feature_dim)
    x_rna_tensor = torch.from_numpy(x_rna_scaled)  # already (1, feature_dim)

    return x_methyl_tensor.float(), x_rna_tensor.float()
label_map = {'LumA': 0, 'LumB': 1, 'Her2': 2, 'Basal': 3}
label_map_n = {v:k for k,v in label_map.items()}
# --- Predict endpoint ---
@app.post("/predict")
def predict(req: PredictRequest):
    results = []

    for sample in req.samples:
        x_methyl_tensor, x_rna_tensor = preprocess_sample(sample)

        with torch.no_grad():
            # TorchScript traced with batch=1 -> process each sample individually
            logits = model_t(x_methyl_tensor, x_rna_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_idx = int(np.argmax(probs))
            pred_label = label_map_n[pred_idx]

        results.append({
            "sample_id": sample.sample_id,
            "pred_label": pred_label,
            "probs": {label_map_n[i]: float(probs[i]) for i in range(len(probs))}
        })

    return {"results": results}