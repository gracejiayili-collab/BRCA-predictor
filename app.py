import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from transformers import MambaConfig, MambaModel
import joblib
import json
import io

# =========================================
# Page config + simple page routing
# =========================================
st.set_page_config(page_title="BRCA Subtype Predictor", layout="wide")

if "page" not in st.session_state:
    st.session_state.page = "overview"   # "overview" or "predict"

def go_to_predict():
    st.session_state.page = "predict"

def go_to_overview():
    st.session_state.page = "overview"

# =========================================
# Model definition
# =========================================
class MultiModalMambaClassifier2(nn.Module):
    def __init__(self, methyl_dim=500, rna_dim=47, hidden_size=256, num_classes=5, seq_len=2):
        super().__init__()
        self.methyl_embed = nn.Linear(methyl_dim, hidden_size)
        self.rna_embed = nn.Linear(rna_dim, hidden_size)

        config = MambaConfig(
            hidden_size=hidden_size,
            intermediate_size=hidden_size * 2,
            num_attention_heads=4,
            num_hidden_layers=2,
            seq_len=seq_len,
        )
        self.mamba = MambaModel(config)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, methyl_features, rna_features):
        methyl_emb = self.methyl_embed(methyl_features)
        rna_emb = self.rna_embed(rna_features)
        methyl_x = methyl_emb.unsqueeze(1)
        rna_x = rna_emb.unsqueeze(1)
        outputs1 = self.mamba(inputs_embeds=methyl_x)
        outputs2 = self.mamba(inputs_embeds=rna_x)
        last_hidden1 = outputs1.last_hidden_state
        last_hidden2 = outputs2.last_hidden_state
        last_hidden = torch.cat([last_hidden1, last_hidden2], dim=1)
        pooled = last_hidden.mean(dim=1)
        logits = self.classifier(pooled)
        return logits

# =========================================
# Load model & metadata 
# =========================================
model_t = torch.jit.load("mamba_ts5.pt", map_location="cpu")
model_t.eval()

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

rna_raw_scaler = joblib.load("rna_scaler_np1.pkl")
rna_scaler = joblib.load("scaler_rna_lw.pkl")
methyl_scaler = joblib.load("scaler_methyl_lw.pkl")

label_map_n = {0: "LumA", 1: "LumB", 2: "Her2", 3: "Basal"}
label_display = {
    "LumA": "Luminal A",
    "LumB": "Luminal B",
    "Her2": "HER2-enriched",
    "Basal": "Basal-like"
}

shap_explainer = joblib.load("shap_explainer_lw_final1.joblib")

# =========================================
# Helpers
# =========================================
def preprocess_sample(x_methyl_dict, x_rna_dict):
    x_methyl = np.array([x_methyl_dict[k] for k in feature_methyl], dtype=np.float32)
    x_rna = np.array([x_rna_dict[k] for k in feature_rna], dtype=np.float32)

    x_methyl = np.where(np.isnan(x_methyl), np.array(methyl_mean, dtype=np.float32), x_methyl)
    x_rna = np.where(np.isnan(x_rna), np.array(rna_mean, dtype=np.float32), x_rna)

    x_rna_scaled = rna_raw_scaler.transform(x_rna.reshape(1, -1))
    x_rna_scaled = rna_scaler.transform(x_rna_scaled)
    x_methyl_scaled = methyl_scaler.transform(x_methyl.reshape(1, -1))

    x_methyl_tensor = torch.from_numpy(x_methyl_scaled).float()
    x_rna_tensor = torch.from_numpy(x_rna_scaled).float()
    return (
        x_methyl_tensor,
        x_rna_tensor,
        x_methyl_scaled.ravel(),
        x_rna_scaled.ravel(),
    )


def predict_sample(x_methyl_tensor, x_rna_tensor):
    with torch.no_grad():
        logits = model_t(x_methyl_tensor, x_rna_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))
        pred_label = label_map_n[pred_idx]
    return pred_label, probs


def _fig_to_png_bytes(fig, dpi=240):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


def plot_local_shap_auto(
    feature_type,
    shap_value,
    feature_values,
    feature_names,
    class_prob,
    class_names=None,
    top_n=15,
    sample_idx=0,
    fig_width=7.0,
    row_height=0.55,
    use_gradient=True,
    cmap_colors=("#3B82F6", "#EF4444"),
):
    import matplotlib as mpl
    from matplotlib.colors import LinearSegmentedColormap

    shap_value = np.asarray(shap_value)
    feature_values = np.asarray(feature_values).ravel()
    class_prob = np.asarray(class_prob).ravel()

    if class_names is None:
        class_names = [f"class_{i}" for i in range(shap_value.shape[2])]

    cls_idx = int(np.argmax(class_prob))
    cls_name = class_names[cls_idx]
    shap_vals = shap_value[sample_idx, :, cls_idx]

    df = pd.DataFrame(
        {"feature": feature_names, "value": feature_values, "shap": shap_vals}
    )
    df["abs_shap"] = np.abs(df["shap"])
    df_top = df.sort_values("abs_shap", ascending=False).head(top_n).copy()
    df_top = df_top.iloc[::-1]

    fig_h = max(3.0, row_height * len(df_top))
    fig, ax = plt.subplots(figsize=(fig_width, fig_h))

    colors = None
    if use_gradient:
        cmap = LinearSegmentedColormap.from_list("blue_red_custom", list(cmap_colors), N=256)
        norm = mpl.colors.Normalize(vmin=0, vmax=float(df_top["abs_shap"].max() + 1e-12))
        colors = [cmap(norm(v)) for v in df_top["abs_shap"].values]

    bars = ax.barh(df_top["feature"], df_top["shap"], color=colors)

    ax.set_title(
        f"Local SHAP (Top {top_n}) {feature_type} — predicted: {cls_name} (p={class_prob[cls_idx]:.3f})",
        fontsize=11,
    )
    ax.set_xlabel("SHAP value (signed contribution)", fontsize=9)
    ax.grid(axis="x", linestyle=":", linewidth=0.8, alpha=0.6)
    plt.subplots_adjust(left=0.40, right=0.97)

    xmax = float(np.max(np.abs(df_top["shap"])) + 1e-9)
    for bar, (v, s) in zip(bars, zip(df_top["value"], df_top["shap"])):
        ax.text(
            bar.get_width() + 0.02 * xmax,
            bar.get_y() + bar.get_height() / 2,
            f"value={v:.3f} | shap={s:.3f}",
            va="center",
            ha="left",
            fontsize=8,
        )

    if use_gradient:
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label("|SHAP| magnitude", fontsize=8)

    png_bytes = _fig_to_png_bytes(fig, dpi=230)
    plt.close(fig)
    return png_bytes


def get_therapy_recommendation(subtype: str):
    """
    Map predicted PAM50-like subtype to a high-level, research-only therapy suggestion.

    NOTE: This is NOT a treatment guideline and must NOT be used for real clinical decisions.
    It is only meant to illustrate how molecular subtype links to typical systemic strategies.
    """
    display = label_display.get(subtype, subtype)

    # Luminal A / Luminal B – ER+/PR+ disease
    if subtype in ["LumA", "LumB"]:
        return {
            "heading": "Endocrine (Hormone) Therapy ± Targeted Agents",
            "subheading": (
                "Luminal tumors are usually ER/PR-positive. Endocrine therapy blocks the "
                "estrogen signaling that fuels their growth."
            ),
            "bullets": [
                "Selective estrogen receptor modulators (SERMs), e.g. Tamoxifen.",
                "Aromatase inhibitors, e.g. Letrozole, Anastrozole, Exemestane.",
                "In higher-risk or metastatic ER+ disease, endocrine therapy is often combined with CDK4/6 inhibitors or other targeted agents.",
            ],
            "note": (
                f"{display} cancers typically have strong hormone-receptor signaling. "
                "Blocking estrogen production or receptor binding can slow or stop tumor cell proliferation."
            ),
        }

    # HER2-enriched disease
    elif subtype == "Her2":
        return {
            "heading": "HER2-Targeted Therapy (often with Chemotherapy)",
            "subheading": (
                "HER2-enriched tumors overexpress the HER2 receptor. Anti-HER2 drugs "
                "bind this protein and shut down its growth signals."
            ),
            "bullets": [
                "Anti-HER2 monoclonal antibodies, e.g. Trastuzumab (Herceptin), Pertuzumab.",
                "Other HER2-targeted agents (antibody–drug conjugates or TKIs) in selected settings.",
                "Frequently combined with chemotherapy in early-stage and metastatic disease.",
            ],
            "note": (
                "Adding HER2-targeted therapy to standard chemotherapy markedly improves outcomes "
                "for HER2-driven tumors compared with chemotherapy alone."
            ),
        }

    # Basal-like / Triple-negative
    elif subtype == "Basal":
        return {
            "heading": "Cytotoxic Chemotherapy ± Immunotherapy",
            "subheading": (
                "Basal-like tumors are usually triple-negative (ER-, PR-, HER2-), "
                "so there is no hormone or HER2 target. Treatment relies on systemic chemo and, "
                "in some patients, immunotherapy."
            ),
            "bullets": [
                "Anthracycline- and taxane-based chemotherapy regimens.",
                "Platinum-based chemotherapy in selected high-risk or BRCA-mutated cases.",
                "Immune checkpoint inhibitors (e.g. anti–PD-1/PD-L1) for eligible patients.",
            ],
            "note": (
                "Basal-like cancers often have high proliferation and genomic instability. "
                "They are less likely to benefit from endocrine or HER2-directed therapy, "
                "so aggressive systemic chemotherapy and immunotherapy are key strategies."
            ),
        }

    # Fallback (should rarely be used)
    else:
        return {
            "heading": "Multidisciplinary Evaluation",
            "subheading": (
                "Subtype not clearly mapped to a single standard strategy. Treatment decisions "
                "normally integrate receptor status, stage, comorbidities, and patient preferences."
            ),
            "bullets": [
                "Combine ER, PR, HER2, Ki-67, stage, and genomic risk tools where available.",
                "Discuss in a multidisciplinary tumor board when possible.",
            ],
            "note": (
                "This model is for research only and cannot replace a full clinical workup "
                "or guideline-based treatment planning."
            ),
        }

# =========================================
# Session init
# =========================================
for key in ["results_df", "shap_map", "probs_map", "feat_methyl_map", "feat_rna_map", "current_sid"]:
    if key not in st.session_state:
        st.session_state[key] = None

# =========================================
# Overview page
# =========================================
def render_overview():
    st.title("BRCA Subtype Predictor — Overview")

    st.markdown("""
### About
The BRCA Subtype Predictor is a **multimodal, explainable deep learning software system** designed for research and educational purposes. It performs **multi-class classification** to predict the molecular PAM50 subtypes (**Luminal A, Luminal B, HER2-enriched, Basal-like**) from complex genomic data.

The core innovation is the integration of a custom **MAMBA-based deep learning model** with **SHAP (SHapley Additive exPlanations)** to provide mathematical interpretability, addressing the "black box" challenge of AI in medicine.

### Technical Stack & Data (Programming Documentation)
| Component | Technology | Programming Focus |
| :--- | :--- | :--- |
| **Model Core** | **PyTorch (MAMBA)** | Custom neural network architecture for **multimodal feature fusion** (DNA Methylation + RNA Expression). |
| **Interface** | **Streamlit** | Python-native web framework chosen for rapid development and clean data visualization. |
| **XAI** | **SHAP, Matplotlib** | Implementation of **local feature attribution** to explain individual predictions. |
| **Data Source** | **TCGA/BRCA** | Model trained on data from **[UCSC Xena The Cancer Genome Atlas Breast Cancer (TCGA/BRCA)](https://xenabrowser.net/datapages/?cohort=TCGA%20Breast%20Cancer%20(BRCA)&removeHub=https%3A%2F%2Fxena.treehouse.gi.ucsc.edu%3A443)**. |            
                
### Inputs
**DNA Methylation (25 features)**  
• Source: Illumina **HumanMethylation450** BeadChip (`HumanMethylation450.gz`)  
• **Scale:** β-values in **[0, 1]** (0 = unmethylated, 1 = fully methylated)  
• Columns: 25 pre-selected CpGs (the app knows their names via `feature_methyl`)

**RNA Expression (25 features)**  
• Source: TCGA **HiSeqV2** (`HiSeqV2.gz`)  
• **Scale:** log2-transformed RSEM-normalized counts (log2(RSEM)); typical range ~ **[0, 15]**  
• Columns: 25 pre-selected genes (the app knows their names via `feature_rna`)

### How to use
**Manual Input**: enter 25 methyl β-values and 25 RNA log2(RSEM) values → Predict → view local SHAP.  
**CSV Upload**: upload a file with all 50 feature columns (+ optional `sample_id`) → Start Prediction → inspect predictions and select a sample on the right to view local SHAP.

### Outputs
• Predicted subtype + class probabilities  
• Local SHAP bar charts (Methylation & RNA) showing the top-N features, with a **blue→red** gradient by |SHAP| magnitude

### Notes
• Inputs must match expected feature names and scales  
• Methylation should be **β-values**, RNA should be **log2(RSEM)**  
• **Research use only**; not for clinical decisions
""")

    st.divider()
    st.button("Start prediction interface →", type="primary", use_container_width=True, on_click=go_to_predict)

# =========================================
# Main prediction interface 
# =========================================
def render_predict():
    # Title
    st.markdown(
        """
        <style>
        .card {
            padding: 0 18px 16px 18px;
            border-radius: 18px;
            background-color: #ffffff;
            box-shadow: 0 4px 18px rgba(15, 23, 42, 0.08);
            border: 1px solid rgba(148, 163, 184, 0.25);
            margin-bottom: 14px;
        }
        .card-header {
            font-size: 0.90rem;
            font-weight: 600;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 8px;
        }
        .pill {
            padding: 6px 10px;
            border-radius: 999px;
            background: #e0f2fe;
            color: #0369a1;
            font-size: 0.75rem;
            font-weight: 600;
            display: inline-block;
        }
        .subtype-text {
            font-size: 2.2rem;
            font-weight: 800;
            color: #16a34a;
            margin: 4px 0 6px 0;
        }
        .confidence-bar {
            height: 8px;
            border-radius: 999px;
            background: #e5e7eb;
            position: relative;
            margin-top: 6px;
        }
        .confidence-fill {
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, #22c55e, #16a34a);
        }
        .small-label {
            font-size: 0.78rem;
            color: #6b7280;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    top_left, top_right = st.columns([3, 1])
    with top_left:
        st.markdown("## BRCA Subtype Predictor")
        st.markdown(
            '<span class="small-label">Multimodal Breast Cancer Subtype Prediction System (Research-only)</span>',
            unsafe_allow_html=True,
        )
    with top_right:
        st.button("Back to Overview", use_container_width=True, on_click=go_to_overview)

    st.markdown("---")

    col_left, col_mid, col_right = st.columns([1.1, 1.2, 1.3], gap="small")

    # ---------- 1) LEFT: input ----------
    with col_left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">Prediction Input</div>', unsafe_allow_html=True)

        input_mode = st.radio("Input method", ["Batch CSV Upload", "Single Sample Manual"], horizontal=False)

        if input_mode == "Batch CSV Upload":
            uploaded_file = st.file_uploader("Upload CSV (one row = one sample)", type=["csv"])
            st.caption(
                "CSV should contain all methylation & RNA feature columns. "
                "Optional `sample_id` column will be used as ID; otherwise row index is used."
            )
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file).dropna(how="all")
                    st.caption("Preview")
                    st.dataframe(df.head(), use_container_width=True)

                    if st.button("🔄 Run Batch Prediction", type="primary", use_container_width=True):
                        results_list = []
                        shap_map = {}
                        probs_map = {}
                        feat_methyl_map = {}
                        feat_rna_map = {}

                        id_col = "sample_id" if "sample_id" in df.columns else None
                        if id_col is None:
                            df["_row_id_"] = df.index.astype(str)
                            id_col = "_row_id_"
                        df[id_col] = df[id_col].astype(str)

                        for _, row in df.iterrows():
                            sid = row[id_col]
                            x_methyl_dict = {k: row.get(k, np.nan) for k in feature_methyl}
                            x_rna_dict = {k: row.get(k, np.nan) for k in feature_rna}

                            x_methyl_tensor, x_rna_tensor, x_m_vec, x_r_vec = preprocess_sample(
                                x_methyl_dict, x_rna_dict
                            )
                            pred_label, probs = predict_sample(x_methyl_tensor, x_rna_tensor)

                            results_list.append(
                                {
                                    "sample_id": sid,
                                    "pred_label": pred_label,
                                    **{label_map_n[i]: float(probs[i]) for i in range(len(probs))},
                                }
                            )

                            shap_values = shap_explainer.shap_values([x_methyl_tensor, x_rna_tensor])
                            shap_map[sid] = shap_values
                            probs_map[sid] = probs
                            feat_methyl_map[sid] = x_m_vec
                            feat_rna_map[sid] = x_r_vec

                        st.session_state["results_df"] = pd.DataFrame(results_list)
                        st.session_state["shap_map"] = shap_map
                        st.session_state["probs_map"] = probs_map
                        st.session_state["feat_methyl_map"] = feat_methyl_map
                        st.session_state["feat_rna_map"] = feat_rna_map

                        if results_list:
                            st.session_state["current_sid"] = results_list[0]["sample_id"]

                except Exception as e:
                    st.error(f"CSV Read Fail: {e}")

        else:  # Single Sample Manual
            sample_id = st.text_input("Sample ID", value="P001")
            with st.expander("Methylation (β / standardized)", expanded=False):
                methyl_values = []
                cols = st.columns(2)
                for i, f in enumerate(feature_methyl):
                    with cols[i % 2]:
                        v = st.number_input(f, value=0.0, format="%.6f")
                        methyl_values.append(v)

            with st.expander("RNA (log2(RSEM) / standardized)", expanded=False):
                rna_values = []
                cols2 = st.columns(2)
                for i, f in enumerate(feature_rna):
                    with cols2[i % 2]:
                        v = st.number_input(f, value=0.0, format="%.6f")
                        rna_values.append(v)

            if st.button("✨ Predict (Manual)", type="primary", use_container_width=True):
                x_methyl_tensor, x_rna_tensor, x_m_vec, x_r_vec = preprocess_sample(
                    dict(zip(feature_methyl, methyl_values)),
                    dict(zip(feature_rna, rna_values)),
                )
                pred_label, probs = predict_sample(x_methyl_tensor, x_rna_tensor)

                results_df = pd.DataFrame(
                    [
                        {
                            "sample_id": sample_id,
                            "pred_label": pred_label,
                            **{label_map_n[i]: float(probs[i]) for i in range(len(probs))},
                        }
                    ]
                )
                shap_values = shap_explainer.shap_values([x_methyl_tensor, x_rna_tensor])

                st.session_state["results_df"] = results_df
                st.session_state["shap_map"] = {sample_id: shap_values}
                st.session_state["probs_map"] = {sample_id: probs}
                st.session_state["feat_methyl_map"] = {sample_id: x_m_vec}
                st.session_state["feat_rna_map"] = {sample_id: x_r_vec}
                st.session_state["current_sid"] = sample_id

        st.markdown("</div>", unsafe_allow_html=True)

    # ---------- 2) Mid：Batch distribution + Download ----------
    with col_mid:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">Batch Prediction Subtype Distribution</div>', unsafe_allow_html=True)

        if st.session_state["results_df"] is not None:
            results_df = st.session_state["results_df"]
            subtype_counts = results_df["pred_label"].value_counts().reindex(
                ["LumA", "LumB", "Her2", "Basal"], fill_value=0
            )
            total_n = int(subtype_counts.sum())

            # <<< Reduce Donut Size >>>
            fig, ax = plt.subplots(figsize=(1.4,1.4))
            labels = [label_display[k] for k in subtype_counts.index]
            values = subtype_counts.values
            colors = ["#22c55e", "#0ea5e9", "#f97316", "#a855f7"]

            if total_n > 0:
                wedges, _ = ax.pie(values, labels=None, startangle=90, colors=colors)
                centre_circle = plt.Circle((0, 0), 0.6, fc="white")
                fig.gca().add_artist(centre_circle)
                ax.text(
                    0,
                    0.02,
                    f"N={total_n}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                )
                ax.axis("equal")
                st.pyplot(fig)
            else:
                st.info("No batch predictions yet.")

            if total_n > 0:
                st.markdown("**Distribution**")
                for label, count, color in zip(labels, values, colors):
                    pct = 100.0 * count / total_n if total_n > 0 else 0
                    st.markdown(
                        f"<span style='display:inline-block;width:10px;height:10px;border-radius:999px;background:{color};margin-right:6px;'></span>"
                        f"<span class='small-label'>{label}</span> — **{pct:.1f}%** ({int(count)} cases)",
                        unsafe_allow_html=True,
                    )

            csv_bytes = results_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇ Download all prediction results (CSV)",
                data=csv_bytes,
                file_name="mamba_b_predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.caption("Download per-sample predicted subtype and class probabilities.")
        else:
            st.info("Upload data or run a single prediction to see subtype distribution.")

        st.markdown("</div>", unsafe_allow_html=True)

    # ---------- 3) Right：Sample dropdown, update current_sid ----------
    with col_right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">Current Sample Selection</div>', unsafe_allow_html=True)

        if st.session_state["results_df"] is not None:
            results_df = st.session_state["results_df"]
            ids = results_df["sample_id"].astype(str).tolist()

            id_to_rows = {}
            for i, sid in enumerate(ids):
                id_to_rows.setdefault(sid, []).append(i)

            display_items, display_to_sid = [], {}
            for sid, rows in id_to_rows.items():
                if len(rows) == 1:
                    txt = f"{sid}  [row {rows[0]}]"
                    display_items.append(txt)
                    display_to_sid[txt] = sid
                else:
                    for k, r in enumerate(rows, start=1):
                        txt = f"{sid}  (#{k})  [row {r}]"
                        display_items.append(txt)
                        display_to_sid[txt] = sid

            default_index = 0
            if st.session_state["current_sid"] is not None:
                for i, txt in enumerate(display_items):
                    if display_to_sid[txt] == st.session_state["current_sid"]:
                        default_index = i
                        break

            # First initialize the current selection（only set up once when there's no session_state）
            if display_items:
                if "sample_select" not in st.session_state:
                    # default to one corresponding to current_sid, if not select first one as current_sid
                    init_idx = 0
                    if st.session_state.get("current_sid") is not None:
                        for i, txt in enumerate(display_items):
                            if display_to_sid[txt] == st.session_state["current_sid"]:
                                init_idx = i
                                break
                    st.session_state["sample_select"] = display_items[init_idx]

            # use key, not depend on index
            chosen_display = st.selectbox(
                "Select sample for prediction card, therapy, raw features, and SHAP plots:",
                options=display_items,
                key="sample_select",
            )

            # map back to sample id based on the selection
            sid_selected = display_to_sid[chosen_display]
            st.session_state["current_sid"] = sid_selected
            st.caption(f"Currently viewing: **{sid_selected}**")
        else:
            st.info("After running any prediction, you can choose a sample here.")
        st.markdown("</div>", unsafe_allow_html=True)

    # ---------- 4) Mid：Current Therapy and Prediction ----------
    with col_mid:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="card-header">Predicted Subtype Result (Current Sample)</div>',
            unsafe_allow_html=True,
        )

        if st.session_state["current_sid"] and st.session_state["probs_map"]:
            sid = st.session_state["current_sid"]
            probs = st.session_state["probs_map"][sid]
            pred_idx = int(np.argmax(probs))
            subtype = label_map_n[pred_idx]
            display_name = label_display.get(subtype, subtype)
            conf = float(probs[pred_idx])

            st.markdown(
                f"<div class='pill'>Current Sample ID: {sid}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(f"<div class='subtype-text'>{display_name}</div>", unsafe_allow_html=True)
            st.markdown(
                f"<span class='small-label'>**Model confidence:** <b>{conf*100:.1f}%</b></span>",
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width:{conf*100:.1f}%;"></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            prob_df = pd.DataFrame(
                {
                    "Subtype": [label_display[v] for v in label_map_n.values()],
                    "Probability": [float(p) for p in probs],
                }
            )
            st.table(prob_df.style.format({"Probability": "{:.3f}"}))
        else:
            st.info("No current sample selected yet.")

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">Therapy Recommendation (Research-only)</div>', unsafe_allow_html=True)

        if st.session_state["current_sid"] and st.session_state["probs_map"]:
            sid = st.session_state["current_sid"]
            probs = st.session_state["probs_map"][sid]
            subtype = label_map_n[int(np.argmax(probs))]
            rec = get_therapy_recommendation(subtype)
            display_name = label_display.get(subtype, subtype)

            st.markdown(f"**Predicted subtype:** {display_name}")
            st.markdown(f"**Recommended focus:** {rec['heading']}")
            st.caption(rec["subheading"])
            st.markdown("- " + "\n- ".join(rec["bullets"]))
            st.info(rec["note"])
        else:
            st.info("Therapy suggestions will appear once a sample has been predicted.")

        st.warning(
            "⚠️ This interface is for **research and education only** and must **not** be used for real clinical decisions.",
            icon="⚠️",
        )

        st.markdown("</div>", unsafe_allow_html=True)

    # ---------- 5) Left：Raw features ----------
    with col_left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">Raw Feature View (Current Sample)</div>', unsafe_allow_html=True)

        if st.session_state["current_sid"] and st.session_state["feat_methyl_map"]:
            sid = st.session_state["current_sid"]
            x_m_vec = st.session_state["feat_methyl_map"][sid]
            x_r_vec = st.session_state["feat_rna_map"][sid]

            st.caption(f"Viewing standardized inputs for: **{sid}**")
            methyl_df = pd.DataFrame(
                {"Feature": feature_methyl, "Std. value": x_m_vec.round(4), "Type": "Methylation"}
            )
            rna_df = pd.DataFrame({"Feature": feature_rna, "Std. value": x_r_vec.round(4), "Type": "RNA"})
            feat_df = pd.concat([methyl_df, rna_df], ignore_index=True)

            st.dataframe(feat_df, height=360, use_container_width=True)
        else:
            st.info("Run a prediction to view the 50 standardized features here.")

        st.markdown("</div>", unsafe_allow_html=True)

    # ---------- 6) Right：SHAP two images ----------
    with col_right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-header">SHAP Feature Contribution (Local Explainability)</div>', unsafe_allow_html=True)

        if st.session_state["current_sid"] and st.session_state["shap_map"]:
            sid = st.session_state["current_sid"]
            shap_values = st.session_state["shap_map"][sid]
            probs = st.session_state["probs_map"][sid]
            x_m_vec = st.session_state["feat_methyl_map"][sid]
            x_r_vec = st.session_state["feat_rna_map"][sid]

            st.markdown("**Methylation (Top 15 features)**")
            png_m = plot_local_shap_auto(
                "Methylation",
                shap_values[0],
                x_m_vec,
                feature_methyl,
                probs,
                class_names=list(label_map_n.values()),
                top_n=15,
            )
            st.image(png_m, use_container_width=True)

            st.markdown("---")

            st.markdown("**RNA (Top 15 features)**")
            png_r = plot_local_shap_auto(
                "RNA",
                shap_values[1],
                x_r_vec,
                feature_rna,
                probs,
                class_names=list(label_map_n.values()),
                top_n=15,
            )
            st.image(png_r, use_container_width=True)
        else:
            st.info("Run prediction and select a sample to view local SHAP explanations.")

        st.markdown("</div>", unsafe_allow_html=True)

# =========================================
# Router
# =========================================
if st.session_state.page == "overview":
    render_overview()
else:
    render_predict()
