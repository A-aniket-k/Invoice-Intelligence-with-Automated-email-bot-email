# 📊 Invoice Intelligence Portal (with Automated Email Bot Engine)

An end-to-end Machine Learning and Data Analytics pipeline architecture designed to automate logistics invoice verification, audit supply chain transaction risks, and manage administrative alerts. This repository features production-grade modular components alongside an integrated automated email bot engine that handles alert escalation completely autonomously.

🚀 **[Live Streamlit Web Application](https://invoice-intelligence-portal.streamlit.app)**

---

## 📁 Project Structure

```text
Invoice Intelligence - Pipeline-Configuration_Bot/
│
├── .gitignore                          # Excludes heavy database binaries and caches
├── README.md                           # Documentation
├── app.py                              # Entrypoint Streamlit dashboard script
├── requirements.txt                    # Pinpoint python package dependencies
│
├── Freight_Cost_Prediction/            # Pipelines tracking distribution expense baselines
│   ├── data_preprocessing.py
│   ├── train.py
│   └── model_evaluation.py
│
├── invoice_flagging/                   # Anomaly validation track
│   ├── data_preprocessing.py
│   ├── train.py
│   └── modeling_evaluation.py
│
├── inference/                          # Production runtime parsing layers
│   ├── ai_extractor.py                 # AI Vision invoice data extraction track
│   ├── predict_freight.py
│   └── predict_invoice_flag.py
│
└── notebooks/                          # Development notebooks & sandbox prototyping
    ├── Invoice_Flagging.ipynb
    └── Predicting Freight Cost.ipynb
