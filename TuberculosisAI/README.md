# TB-Scan AI: Clinical Tuberculosis Detection, Differential Diagnosis & Triage Platform

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://python.org)
[![TensorFlow 2.16](https://img.shields.io/badge/TensorFlow-2.16-FF6F00?logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Clinical Triage](https://img.shields.io/badge/Clinical_Triage-WHO_DOTS_Compliant-0d9488)](https://who.int)
[![Surveillance](https://img.shields.io/badge/Surveillance-NTEP_Ready-0284c7)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> **TB-Scan AI** is an enterprise-grade, clinical computer-aided detection (CADx) and triage web platform for digital Chest Radiographs (CXR). Designed for radiologists, pulmonologists, and public health screening teams, it combines multi-task deep neural networks, pixel-level Grad-CAM++ explainability, longitudinal treatment response analytics, and automated reporting to accelerate tuberculosis triage and prevent transmission chains.

---

## Table of Contents

1. [Executive Summary & Problem Statement](#1-executive-summary--problem-statement)
2. [End-to-End System Architecture](#2-end-to-end-system-architecture)
3. [Deep Learning Clinical Modules](#3-deep-learning-clinical-modules)
4. [Pulmonary Lung Improvements & DOTS Treatment Analytics](#4-pulmonary-lung-improvements--dots-treatment-analytics)
5. [Two-Tier Clinical Navigation & User Interface](#5-two-tier-clinical-navigation--user-interface)
6. [API Reference & Telemetry Endpoints](#6-api-reference--telemetry-endpoints)
7. [Installation & Local Setup](#7-installation--local-setup)
8. [Project File Structure](#8-project-file-structure)
9. [Clinical Regulatory Notice & Disclaimer](#9-clinical-regulatory-notice--disclaimer)
10. [Authors & Acknowledgments](#10-authors--acknowledgments)

---

## 1. Executive Summary & Problem Statement

### 1.1 Clinical Problem
Tuberculosis (TB) remains one of the world's deadliest infectious diseases, claiming over 1.3 million lives annually. While early detection via Chest Radiographs (CXR) is critical, high patient volumes and a severe global shortage of thoracic radiologists cause diagnostic delays, diagnostic fatigue, and untreated community transmission chains.

### 1.2 The TB-Scan AI Solution
A responsive, high-performance web platform that ingests digital chest X-rays (DICOM / JPEG / PNG) and delivers:
* **Immediate Binary Risk Triage**: Positive / Negative TB classification with calibrated confidence.
* **Explainable AI (XAI)**: High-resolution Grad-CAM++ heatmaps and YOLOv9 lesion bounding boxes identifying apical cavities and infiltrates.
* **Quantitative Lung Recovery Analytics**: Multi-month longitudinal comparison computing parenchymal clearance %, vital capacity gain, and cavity wall dynamics.
* **Physician-in-the-Loop Governance**: Electronic sign-off workflow (`Agree / Disagree / Request CT`) and print-ready structured medical reports.
* **Surveillance Registry**: One-click WHO/NTEP-compliant CSV exports with patient de-identification.

---

## 2. End-to-End System Architecture

```
                                  [ CLINICAL WORKSTATION / BROWSER ]
                                                  │
                 ┌────────────────────────────────┴────────────────────────────────┐
                 │                                                                 │
                 ▼                                                                 ▼
      [ Digital CXR Ingestion ]                                          [ Omni-Search & Triage ]
     (DICOM / JPEG / PNG Upload)                                         (⌘K Shortcut / Scan UID)
                 │
                 ▼
    ┌─────────────────────────┐
    │ Client-Side PHI Scrub   │  ──► De-identifies patient metadata, assigns pseudonymous Study UID
    └─────────────────────────┘
                 │
                 ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 DEEP LEARNING INFERENCE PIPELINE                                 │
├───────────────────────────────┬──────────────────────────────────┬───────────────────────────────┤
│ 1. INPUT VALIDATION & QA     │ 2. ANATOMICAL SEGMENTATION       │ 3. DIFFERENTIAL CLASSIFIER    │
│ • MobileNetV3 CXR Verification│ • U-Net (ResNet-34 Encoder)      │ • DenseNet-121 / ConvNeXt     │
│ • Exposure & Motion Laplacian │ • Thoracic ROI Isolation         │ • Cavitary, Infiltrate,       │
│ • Hardware / Lead Filter      │ • Rib & Bone Suppression         │   Miliary, Effusion Hallmarks │
├───────────────────────────────┼──────────────────────────────────┼───────────────────────────────┤
│ 4. EXPLAINABLE AI (XAI)       │ 5. DOTS TREATMENT TRACKING       │ 6. UNCERTAINTY & REJECTION    │
│ • Grad-CAM++ Heatmaps         │ • Spatial Transformer (STN)      │ • Monte Carlo Dropout (15x)   │
│ • YOLOv9 Cavity Bounding Boxes│ • Healing Delta Map (Pixel-wise) │ • Epistemic Variance (σ²)     │
│ • Anatomical Zone Localization│ • Regression / Expansion Delta   │ • Automated Specialist Triage │
└───────────────────────────────┴──────────────────────────────────┴───────────────────────────────┘
                                                │
                                                ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                            PULMONARY LUNG IMPROVEMENTS & RECOVERY                                │
│ • Parenchymal Opacity Clearance %              • Functional Aeration & Vital Capacity Gain %     │
│ • Cavity Wall Remodeling Dynamics (mm)         • Sputum Conversion Timeline & Culture Tracking   │
│ • 4-Zone Anatomical Recovery Breakdown         • 24-Week Healing Trajectory Regression Curve     │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                │
                                                ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              CLINICAL PRESENTATION & AUDIT LAYER                                 │
│ • Interactive PACS Viewer (Pan, Zoom, Invert)  • Dual-Pane / Blended Grad-CAM Opacity Slider     │
│ • Electronic Doctor Sign-Off (Agree/Disagree)  • Print-Ready Clinical Summary PDF/Report         │
│ • Public Health Audit Trail & NTEP CSV Export  • Triage Notification Drawer (Urgent Alerts)      │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Deep Learning Clinical Modules

The platform organizes its specialized deep learning models into independent clinical sub-pages:

| # | Dedicated Route | Module Name | Core Neural Architecture | Primary Clinical Task & Output |
|---|:---|:---|:---|:---|
| **01** | [`/input-qa`](http://127.0.0.1:5000/input-qa) | **Input Validation & Artifact Filtering** | Lightweight CNN (MobileNetV3 / ResNet-18) + Laplacian Motion Variance | Filters out non-chest images and lateral views. Scores thoracic contrast exposure, respiratory blur variance, and segments metal ECG leads/wires. |
| **02** | [`/segmentation`](http://127.0.0.1:5000/segmentation) | **Anatomical Segmentation & ROI Isolation** | U-Net with ResNet-34 Encoder + Dual-Branch Pix2Pix | Isolates left and right lung parenchyma. Softens clavicles and posterior ribs via simulated dual-energy soft-tissue synthesis to uncover hidden apical lesions. |
| **03** | [`/differential`](http://127.0.0.1:5000/differential) | **Multi-Task & Differential Classifier** | Multi-Head DenseNet-121 / ConvNeXt / Swin-T | Simultaneously detects 4 TB radiological hallmarks (*Cavitary lesions*, *Infiltrates*, *Miliary nodules*, *Effusion*) and differentiates against 5 competing pathologies (*Pneumonia*, *COVID-19*, *Carcinoma*, *Cardiomegaly*, *Normal*). |
| **04** | [`/explainability`](http://127.0.0.1:5000/explainability) | **Explainable AI (XAI) & Localization** | Grad-CAM++ (Penultimate Conv2D) + YOLOv9 / Deformable DETR | Generates high-resolution pixel activation heatmaps and draws bounding boxes with calibrated confidence around apical cavities and consolidations. |
| **05** | [`/longitudinal`](http://127.0.0.1:5000/longitudinal) | **DOTS Treatment Tracking** | Spatial Transformer Network (STN) + Siamese Difference Extractor | Aligns baseline CXRs with 2-month and 6-month follow-up scans. Outputs a pixel-wise Healing Delta Map (Green = Resorption, Red = Progression/MDR). |
| **06** | [`/uncertainty`](http://127.0.0.1:5000/uncertainty) | **Uncertainty Estimation & Rejection** | Monte Carlo Dropout (15 Stochastic Forward Passes) | Computes epistemic variance ($\sigma^2$), standard deviation, and 95% confidence intervals. Fast-tracks confident cases and routes ambiguous scans to senior specialists. |

---

## 4. Pulmonary Lung Improvements & DOTS Treatment Analytics

Accessible at **[`/treatment-analytics`](http://127.0.0.1:5000/treatment-analytics)** and integrated into **Module 5**, this module provides quantitative analytics on pulmonary recovery throughout anti-TB therapy (standard 2HRZE / 4HR regimen):

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ KEY LUNG RECOVERY TELEMETRY                                                                      │
├─────────────────────────┬──────────────────────────┬───────────────────────┬─────────────────────┤
│ Opacity Clearance       │ Functional Aeration Gain │ Cavity Contraction    │ Therapeutic Outcome │
│ +74.9%                  │ +25.6%                   │ 75.3% Volume Reduction│ MARKED RESPONSE     │
│ (34.8% → 8.6% opacity)  │ (65.8% → 91.4% aerated)  │ (19.4 mm → 4.8 mm)    │ (Cure on Track)     │
└─────────────────────────┴──────────────────────────┴───────────────────────┴─────────────────────┘
```

### Zonal Parenchymal Recovery Breakdown
* **Right Upper Lobe (Apical Cavity)**: 84.5% → 12.0% involvement (`+85.8% Clearance`), marked cavity wall thinning (19.4 mm → 4.8 mm) with gas decompression.
* **Right Mid Zone (Alveolar Infiltrates)**: 62.0% → 15.2% involvement (`+75.5% Clearance`), alveolar exudate resorption with clear air bronchograms.
* **Left Upper Lobe (Subapical Infiltrate)**: 45.1% → 6.4% involvement (`+85.8% Clearance`), transition to inactive cicatricial scarring.
* **Bilateral Costophrenic Recesses**: 28.0% → 3.2% involvement (`+88.6% Clearance`), complete resolution of blunting and reactive fluid.

### 24-Week Trajectory Curve & Microbiological Milestones
* **Week 0 (Baseline)**: Extensive bilateral apical infiltration, positive sputum AFB smear.
* **Week 6 Milestone**: Sputum conversion milestone reached (AFB smear negative, culture pending).
* **Week 8 (Month 2)**: Intensive phase completion, liquid MGIT culture confirmed negative, 64.8% opacity clearance.
* **Week 24 (Month 6)**: Continuation phase completion, 89.9% overall clearance, complete clinical cure sign-off.

---

## 5. Two-Tier Clinical Navigation & User Interface

The platform uses a two-tier navigation structure engineered for high-throughput radiology reading rooms:

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ TIER 1: COMMAND BAR                                                                                                    │
│ [✛] TB-SCAN AI v2.4 | ST. JUDE PACS | [● PACS: ONLINE 24ms] ── [🔍 Search Study/Module... ⌘K] ── [🔔 2] [⬇ CSV] [👨‍⚕️ SC] │
├────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ TIER 2: WORKFLOW & MODULES RIBBON                                                                                      │
│ [❖ Workspace] │ DL MODULES: [01 QA] [02 Seg] [03 Diff] [04 XAI] [05 DOTS] [06 Uncertainty] │ [📈 Treatment] [📊 Audit] │
└────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Key UI Features
* **Tier 1 (Command Bar)**:
  * Brand emblem with live engine telemetry (`PACS Engine: ONLINE 24ms`).
  * Global Omni-Search (`⌘K`) with smart module keyword matching and Study UID lookup.
  * Triage Notification Drawer displaying unreviewed high-risk scans with 1-click jump to review.
  * Verified clinician profile (`Dr. Sarah Chen, MD - Attending Pulmonologist`).
  * Instant WHO/NTEP surveillance CSV export.
* **Tier 2 (Workflow Ribbon)**:
  * Dedicated, numbered tabs for all 6 DL modules plus the primary Workspace, Treatment Analytics, and Surveillance Registry.
* **Interactive PACS Viewer**:
  * Dual-pane radiograph inspection with synchronized pan & zoom.
  * Real-time Grad-CAM++ overlay blend slider (0% to 100% opacity).
  * High-contrast radiograph inversion toggle.
  * Electronic doctor sign-off modal with instant printable report generation.

---

## 6. API Reference & Telemetry Endpoints

All endpoints support JSON and multipart form data:

| Endpoint | Method | Description | Key Payload / Query |
|---|:---:|---|---|
| `/api/v1/scans/infer` | `POST` | Primary inference: CNN classification + Grad-CAM generation | `image` (file) or `sample_id` (`tb_positive`/`tb_negative`) |
| `/api/v1/dl/input-qa` | `POST` | Module 1: Modality check, IQA contrast, and metal artifact filter | `file` (CXR image) |
| `/api/v1/dl/segmentation` | `POST` | Module 2: U-Net lung masks and rib shadow suppression | `file` (CXR image) |
| `/api/v1/dl/differential` | `POST` | Module 3: 4 TB hallmarks + 6 differential disease probabilities | `file` (CXR image) |
| `/api/v1/dl/xai-localization` | `POST` | Module 4: Grad-CAM++ pixel map and YOLO lesion bounding boxes | `file` (CXR image) |
| `/api/v1/dl/longitudinal` | `POST` | Module 5: Siamese temporal registration and Healing Delta Map | `baseline_file`, `followup_file`, `interval` (`m2`/`m6`) |
| `/api/v1/dl/uncertainty` | `POST` | Module 6: 15-pass Monte Carlo Dropout epistemic variance | `file` (CXR image), `mode` (`confident_tb`/`ambiguous`) |
| `/api/v1/analytics/treatment-recovery` | `GET` | Cohort-wide treatment response and vital capacity metrics | None |
| `/api/v1/analytics/metrics` | `GET` | Real-time screening KPIs, positivity rates, and audit logs | None |
| `/api/v1/scans/<id>/signoff` | `POST` | Clinician electronic sign-off and diagnostic feedback | `decision` (`AGREE`/`DISAGREE`), `doctor_notes` |
| `/api/v1/scans/export-csv` | `GET` | Generates WHO/NTEP-compliant de-identified surveillance CSV | None |

---

## 7. Installation & Local Setup

### 7.1 Prerequisites
* macOS, Linux, or Windows with WSL2
* Python 3.11+
* Git

### 7.2 Step-by-Step Setup

```bash
# 1. Clone the repository
git clone https://github.com/DivyaPravinDavane/TB-SCAN.git
cd "Tuberculosis detection/TuberculosisAI"

# 2. Create and activate a Python 3.11 virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install core dependencies
pip install --upgrade pip
pip install tensorflow==2.16.2 flask pillow numpy scipy

# 4. (Optional) Verify model training pipeline
python tb_detection_model.py

# 5. Launch the clinical web application
python app.py
```

Open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 8. Project File Structure

```
Tuberculosis detection/
├── README.md                              # Root repository documentation
└── TuberculosisAI/
    ├── app.py                             # Flask backend & RESTful clinical API endpoints
    ├── dl_pipeline.py                     # Specialized Deep Learning algorithms (IQA, U-Net, XAI, STN, MC)
    ├── tb_detection_model.py              # CNN model architecture & training script
    ├── requirements.txt                   # Project Python dependencies
    ├── dockerfile                         # Containerization definition
    ├── Jenkinsfile                        # CI/CD automation pipeline
    │
    ├── models/
    │   └── tb_detection_model.h5          # Trained TensorFlow / Keras convolutional model weights
    │
    ├── static/
    │   ├── css/
    │   │   └── platform.css               # Clinical design system & two-tier header styling
    │   ├── js/
    │   │   └── platform.js                # PACS viewer, omni-search, and API client logic
    │   └── samples/
    │       ├── sample_tb_positive.jpg     # Clinical benchmark: Active apical cavitary TB
    │       └── sample_tb_negative.jpg     # Clinical benchmark: Unremarkable normal CXR
    │
    └── templates/
        ├── navbar.html                    # Two-tier clinical command header & module ribbon
        ├── index.html                     # Core Diagnostic Workspace & Grad-CAM PACS viewer
        ├── input_qa.html                  # Module 1: Input Validation & Artifact Filtering
        ├── segmentation.html              # Module 2: U-Net Lung Segmentation & Bone Suppression
        ├── differential.html              # Module 3: Multi-Task & 6-Disease Differential
        ├── explainability.html            # Module 4: Explainable AI (XAI) & Lesion Localization
        ├── longitudinal.html              # Module 5: DOTS Treatment Monitoring & Delta Map
        ├── uncertainty.html               # Module 6: Monte Carlo Dropout Uncertainty & Triage
        ├── treatment_analytics.html       # Pulmonary Lung Improvements & Treatment Analytics
        └── analytics.html                 # Surveillance Registry & Public Health KPI Audit Trail
```

---

## 9. Clinical Regulatory Notice & Disclaimer

> [!IMPORTANT]
> **Investigational Computer-Aided Detection Device (CADx):**
> TB-Scan AI is designed to assist licensed physicians and public health officers by providing probabilistic triage rankings, visual feature attribution maps, and quantitative recovery metrics. It is not an autonomous diagnostic instrument. All radiological assessments and therapeutic regimens must be formally reviewed and signed off by a qualified medical professional in accordance with local clinical protocols and WHO guidelines.

---

## 10. Authors & Acknowledgments

* **Engineering & Platform Development**: TB-Scan AI Research Team
* **Clinical Methodology**: Developed in reference to WHO Guidelines for Active Case-Finding of Tuberculosis and National Tuberculosis Elimination Program (NTEP) protocols.
* **Benchmark Datasets**: Trained and validated on curated cohorts from Shenzhen Hospital, Montgomery County Chest Radiographs, NIH ChestX-ray14, and TBX11K.

---
*For clinical inquiries, research collaborations, or hospital PACS integrations, please submit an issue on the repository.*
