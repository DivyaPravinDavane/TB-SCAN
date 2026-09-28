# TB-SCAN: Clinical Tuberculosis Detection & Triage Web Platform

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://python.org)
[![TensorFlow 2.16](https://img.shields.io/badge/TensorFlow-2.16-FF6F00?logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Clinical Triage](https://img.shields.io/badge/Clinical_Triage-WHO_DOTS_Compliant-0d9488)](https://who.int)
[![Surveillance](https://img.shields.io/badge/Surveillance-NTEP_Ready-0284c7)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An enterprise-grade, clinical computer-aided detection (CADx) and triage web platform for digital Chest Radiographs (CXR). Designed for radiologists, pulmonologists, and public health screening teams, it combines multi-task deep neural networks, pixel-level Grad-CAM++ explainability, longitudinal treatment response analytics, and automated reporting.

---

## Complete Project Documentation

Full technical documentation, architectural diagrams, deep learning module breakdowns, API telemetry specifications, and local setup instructions are available in the application directory:

👉 **[TuberculosisAI/README.md](TuberculosisAI/README.md)**

---

## High-Level Architecture & Modules Summary

```
[ CXR Ingestion ] ──► [ Client-Side PHI Scrub ]
                               │
                               ▼
  ┌────────────────────────────────────────────────────────┐
  │         DEEP LEARNING CLINICAL PIPELINE (6 MODULES)    │
  ├────────────────────────────┬───────────────────────────┤
  │ 01. Input QA & Artifacts   │ 02. U-Net Lung & Rib Seg  │
  │ 03. Differential 6-Class   │ 04. Grad-CAM++ & YOLO XAI │
  │ 05. DOTS Siamese Delta Map │ 06. MC Dropout Triage     │
  └────────────────────────────┴───────────────────────────┘
                               │
                               ▼
  ┌────────────────────────────────────────────────────────┐
  │       PULMONARY LUNG IMPROVEMENTS & RECOVERY           │
  │ • Opacity Clearance (+74.9%) • Vital Capacity (+25.6%) │
  │ • Cavity Contraction (75.3%) • Sputum Conversion (W6)  │
  └────────────────────────────────────────────────────────┘
                               │
                               ▼
  [ PACS Viewer & Opacity Blend ] ──► [ Doctor E-Sign-Off & NTEP CSV ]
```

### Dedicated Clinical Pages & Routes

1. **Diagnostic Workspace** (`/`): Primary CXR classifier, dual-pane PACS viewer, Grad-CAM++ opacity slider, and doctor sign-off.
2. **Module 1: Input QA & Artifacts** (`/input-qa`): MobileNetV3 CXR verification, IQA exposure/blur scoring, and ECG lead segmentation.
3. **Module 2: Anatomical Segmentation** (`/segmentation`): U-Net lung field isolation and dual-energy simulated bone shadow suppression.
4. **Module 3: Differential Diagnostics** (`/differential`): 4 TB hallmarks (cavitary, infiltrates, miliary, effusion) + 6-disease differential panel.
5. **Module 4: Explainable AI (XAI)** (`/explainability`): Grad-CAM++ heatmaps and YOLOv9 lesion bounding boxes.
6. **Module 5: DOTS Treatment Tracking** (`/longitudinal`): Pre-treatment vs follow-up Siamese registration and Healing Delta Map.
7. **Module 6: Uncertainty Estimation** (`/uncertainty`): 15-pass Monte Carlo Dropout epistemic variance ($\sigma^2$) and automated triage routing.
8. **Pulmonary Lung Improvements & Treatment Analytics** (`/treatment-analytics`): 24-week recovery trajectory curve, zonal clearance progress bars, and WHO DOTS outcomes.
9. **Surveillance & Registry** (`/analytics`): Epidemiological KPIs (Sensitivity, Specificity, AUROC), search audit trail, and WHO/NTEP CSV export.

---

## Quick Start

```bash
cd TuberculosisAI
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```
Open **`http://127.0.0.1:5000`** in your browser.
