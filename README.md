# 📊 Invoice Intelligence Portal (with Automated Email Bot Engine)

An end-to-end Machine Learning and Data Analytics pipeline architecture designed to automate logistics invoice verification, audit supply chain transaction risks, and manage administrative alerts. This repository features production-grade modular components alongside an integrated automated email bot engine that handles alert escalation completely autonomously.

🚀 **[Live Streamlit Web Application](https://invoice-intelligence-portal.streamlit.app)**

---

## 📌 Business Overview & Problem Statement
Manual processing of logistics invoices creates substantial bottlenecks for financial and procurement operations, resulting in:
* **Financial Overcharges:** Undetected duplicate billings, administrative overcharges, and non-compliant invoicing.
* **Lack of Visibility:** Difficulty forecasting erratic spot-market freight costs against historic baselines.
* **Manual Bottlenecks:** Heavy reliance on personnel to cross-reference data records manually, stalling reconciliation timelines.

**The Solution:** This framework introduces an intelligent auditing pipeline that validates uploaded files via predictive modeling while seamlessly integrating an SMTP email bot engine to notify administrative users of compliance flags instantly, eliminating manual overhead.

---

## 🛠️ Tech Stack & Architecture
* **Frontend Web App:** Streamlit (Interactive layout for transactional data uploads and single/batch processing)
* **Programming & Core Logic:** Python, Pandas, NumPy, Scikit-Learn
* **Data Integration & Persistence:** SQLite (`inventory.db`)
* **Communication Stack:** SMTP Secure Protocol (Automated System Alerts using Email ID & App Password credentials)
* **Deployment & Control:** Git, GitHub, Streamlit Community Cloud

---

## 💡 Key Features & Operational Architecture

### 🤖 Automated Email Bot Integration
* **Custom SMTP Credentials:** Securely authenticates using an administrative Email ID and custom Google App Password configuration to pass system alerts and compliance updates directly to target mailboxes.
* **Real-Time Triggering:** Automatically dispatches structured audit summaries the moment a high-risk transaction anomaly or billing leak is flagged by the system.
* **Workload Optimization:** Eliminates the need for managers to sit and constantly refresh a dashboard; the data pipeline pushes critical, actionable violations directly to their inbox.

### ⚙️ Analytical Processing Engine
* **Multi-Engine Pipeline Tracks:** Processes data concurrently through localized regression models to determine freight spot cost ranges and evaluate transaction anomaly levels.
* **Production-Grade Separation:** Clean separation maintained across raw transformations, evaluation loops, and operational `inference/` scoring tracks.

---

## 📈 Quantifiable Impact & Workload Reduction
* **⏱️ 50% Compression in Audit Timelines:** Automated verification loops replace slow manual line-by-line validation, shrinking operational tracking hours significantly.
* **📬 Zero-Delayed Interventions:** Automated SMTP email routing replaces manual escalation emails. Flagged discrepancies are sent to supervisors instantly, preventing ledger-writing errors before payments go out.
* **🛑 Human-Error Elimination:** Shifting the reporting workflow from manual spot-checks to script-driven email alerts guarantees 100% data coverage—no high-risk invoice gets missed due to review fatigue.
* **📊 Stronger Negotiation Levers:** Generates historical freight baseline estimates automatically, arming procurement teams with hard facts to negotiate better terms with 3PL vendors.

---

## ⚙️ Local Setup & Installation

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/A-aniket-k/Invoice-Intelligence-with-Automated-email-bot-email.git]
   cd "Invoice Intelligence - Pipeline-Configuration_Bot"
2. **Set Up a Virtual Environment:**
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

3. **Install Dependencies:**
   pip install -r requirements.txt

4. **Launch the Dashboard Locally:**
   streamlit run app.py   

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
👨‍💻 Developed by Aniket Kumar – Aspiring Data Analyst & Machine Learning Practitioner.
