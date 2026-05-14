# Diabetes Risk — 30-Day Hospital Readmission Predictor

A fully deployable machine learning web application that predicts 30-day hospital readmission risk for Type 2 diabetes patients using Electronic Health Record (EHR) data.

Built as part of a Bachelor's thesis: **"Predicting 30-Day Hospital Readmissions in Type 2 Diabetes Patients Using Machine Learning"**

\---

## Features

* 3 ML models: Logistic Regression, Random Forest, XGBoost
* Trained on 101,766 real patient records (UCI Diabetes 130-US Hospitals Dataset)
* Interactive web interface for real-time predictions
* Model comparison showing all 3 predictions simultaneously
* SHAP-based feature importance explanations
* Fully deployable Flask web application

\---

## Model Performance

|Model|Accuracy|AUC Score|F1 Score|
|-|-|-|-|
|Logistic Regression|71.4%|0.649|0.363|
|Random Forest|65.3%|0.651|0.423|
|**XGBoost**|**68.8%**|**0.667**|**0.428**|

\---

## Setup Instructions



Recommended Python version: Python 3.11



### 1\. Clone the repository

```bash
git clone https://github.com/murtaza65/diabetes-readmission-prediction
cd diabetes-readmission-prediction
```

If using the submitted ZIP file, extract the ZIP and open Command Prompt inside the project folder instead.



### 2\. Create and activate a virtual environment

```bash
py -3.11 -m venv venv
venv\Scripts\activate
```

### 3\. Install dependencies

```bash
pip install -r requirements.txt
```

### 4\. Add the dataset

Download the dataset from UCI:
https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008

Place `diabetic_data.csv` in the project root folder.

Note: The dataset file `diabetic_data.csv` is already included in this GitHub repository, so downloading it separately is not required.



### 5\. Train the models

```bash
python train_model.py
```

This generates a `models.pkl` file with all trained models saved.

Note: `models.pkl` is generated locally after running train_model.py and is not included in the repository due to file size limitations.



### 6\. Run the web application

```bash
python app.py
```

Open your browser and go to: **http://localhost:5000**

\---



## Project Structure

```
diabetes-readmission-prediction/
│
├── app.py                  # Flask web application
├── train_model.py          # Model training script
├── models.pkl              # Saved trained models (generated after training)
├── requirements.txt        # Python dependencies
├── ML.ipynb                # Jupyter Notebook (exploratory analysis)
├── ML.html                 # Exported notebook with results
│
├── templates/
│   └── index.html          # Web application frontend
│
└── README.md
```

\---

## Dataset

* **Source:** UCI Machine Learning Repository
* **Name:** Diabetes 130-US Hospitals for Years 1999-2008
* **Records:** 101,766 patient encounters
* **Features:** 50 original (43 after cleaning)
* **Target:** Readmission within 30 days (binary)

\---

## Tech Stack

* Python 3
* Flask (web framework)
* Scikit-learn (ML models)
* XGBoost
* SHAP (explainability)
* Pandas, NumPy
* HTML/CSS/JavaScript (frontend)

\---

## Disclaimer

This tool is for research and educational purposes only. It does not replace clinical judgment. Any real-world deployment requires ethical approval and clinical validation.

