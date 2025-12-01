# Multimodal AI System for Breast Cancer Subtype Prediction

This repository contains a **deployed software system** for predicting breast cancer molecular subtypes using multimodal genomic data, with built-in explainability.

The system integrates **DNA methylation** and **RNA expression** data using a modern deep learning architecture and provides **feature-level explanations** for every prediction through an interactive web interface.

---

## Project Overview

Breast cancer subtypes (Luminal A, Luminal B, HER2, Basal) play a critical role in guiding treatment decisions.  
However, genomic data is high-dimensional, complex, and difficult to interpret.

This project addresses these challenges by:
- Integrating multiple genomic modalities
- Using an efficient deep learning architecture (MAMBA)
- Providing transparent, interpretable predictions through SHAP

This is a **full software pipeline**, not just a standalone model.

---

## Live Web Application

A live, deployed version of the system is available at:

**Live App:**  
https://brca-predictor-shxl7owfhp6faxf8wi5f6u.streamlit.app

The web app allows users to:
- Upload or input genomic features
- View predicted cancer subtype and probabilities
- Explore feature-level explanations for both RNA and methylation inputs

Judges and reviewers can interact with the system directly through the browser.  
No local setup or code execution is required.

---

## System Architecture

### Inputs
- DNA Methylation (CpG β-values)
- RNA Expression (gene expression levels)

### Core Ideas
- Treat methylation and RNA as **two complementary data streams**
- Project both modalities into the same hidden space
- Encode each modality separately
- Fuse representations before classification

### Outputs
- Predicted breast cancer subtype
- Class probabilities
- Feature-level explanations (SHAP)

---

## Deep Learning Model

- Both modalities are projected into a 256-dimensional latent space
- Each modality is encoded using **MAMBA**, a state-space neural architecture
- Encoded representations are concatenated and pooled
- Final classification is performed using a linear layer

Unlike Transformers, MAMBA does not rely on attention mechanisms.  
Instead, it uses a state-space model with linear scaling, allowing efficient capture of long-range patterns while remaining lightweight and deployable.

---

## Model Evaluation

- Evaluated using **5-fold cross validation**
- Performance measured using **macro-average F1 score**
- Mamba-based models outperform Transformer and XGBoost baselines
- Lightweight feature sets retain strong performance
- HER2 remains the most challenging subtype, but deep learning maintains an advantage

