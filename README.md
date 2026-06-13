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
│   ├── predict_freight.py
│   └── predict_invoice_flag.py
│
└── notebooks/                          # Development notebooks & sandbox prototyping
    ├── Invoice_Flagging.ipynb
    └── Predicting Freight Cost.ipynb
