# Medical Insurance Claim Predictor

[![Hugging Face Spaces](https://img.shields.io/badge/Live%20Demo-Hugging%20Face-yellow)](https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end machine learning application that predicts medical insurance claim amounts based on patient demographics, health metrics, and lifestyle factors. The project covers data preprocessing, exploratory data analysis (EDA), feature engineering, model training, evaluation, and deployment using Gradio on Hugging Face Spaces.

---

## Overview

Medical insurance claim prediction is a regression-based machine learning problem in the healthcare industry. This project uses multiple machine learning algorithms to estimate annual medical expenses for individuals based on health and demographic data.

The final application provides:

- Instant insurance claim predictions
- Risk categorization
- Personalized health insights
- Interactive web interface using Gradio

---

## Dataset

The dataset (`insurance.csv`) contains approximately 1,340 records with the following features:

| Column            | Description                                      | Type |
|-------------------|--------------------------------------------------|------|
| `age`             | Beneficiary age (18–60)                          | Numeric |
| `gender`          | Male / Female                                    | Categorical |
| `bmi`             | Body Mass Index                                  | Numeric |
| `bloodpressure`   | Systolic blood pressure                          | Numeric |
| `diabetic`        | Diabetes status (Yes / No)                       | Categorical |
| `children`        | Number of dependents                             | Numeric |
| `smoker`          | Smoking status (Yes / No)                        | Categorical |
| `region`          | Residential region                               | Categorical |
| `claim`           | Annual medical insurance claim amount (Target)   | Numeric |

> Note: Some features such as `region` may be removed during feature engineering depending on model performance.

---

## Features

- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Feature encoding and scaling
- Multiple regression model training
- Hyperparameter tuning and evaluation
- Model persistence using Joblib
- Interactive Gradio web application
- Insurance risk categorization
- Personalized health recommendations

---

## Tech Stack

| Category              | Technologies |
|-----------------------|--------------|
| Programming Language  | Python |
| Data Processing       | Pandas, NumPy |
| Visualization         | Matplotlib, Seaborn |
| Machine Learning      | Scikit-learn |
| Gradient Boosting     | XGBoost |
| Model Persistence     | Joblib |
| Web Framework         | Gradio |
| Deployment            | Hugging Face Spaces |

---

## Project Structure

```bash
medical-insurance-claim-predictor/
│
├── data/
│   └── insurance.csv
│
├── notebooks/
│   └── Medical_Insurance_Analysis.ipynb
│
├── models/
│   ├── best_model.pkl
│   ├── scaler.pkl
│   ├── label_encoder_gender.pkl
│   ├── label_encoder_diabetic.pkl
│   └── label_encoder_smoker.pkl
│
├── app.py
├── requirements.txt
├── README.md
└── LICENSE
````

---

## Model Performance

After training and evaluation, the Random Forest Regressor achieved the best performance.

| Metric   | Score |
| -------- | ----- |
| R² Score | 0.83  |
| MAE      | 3,739 |
| RMSE     | 4,946 |

### Models Compared

* Linear Regression
* Polynomial Regression
* Random Forest Regressor
* Support Vector Regressor (SVR)
* XGBoost Regressor

---

## Installation and Local Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/medical-insurance-claim-predictor.git
cd medical-insurance-claim-predictor
```

### 2. Create a Virtual Environment (Optional)

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### Linux / macOS

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
python app.py
```

Open the local URL displayed in the terminal, typically:

```bash
http://127.0.0.1:7860
```

---

## Deployment on Hugging Face Spaces

This project is deployed using Gradio on Hugging Face Spaces.

### Deployment Steps

1. Create a new Space on Hugging Face
2. Select Gradio as the SDK
3. Upload the following files:

   * `app.py`
   * `requirements.txt`
   * Model `.pkl` files
4. Wait for the build process to complete

The application will automatically deploy and become publicly accessible.

---

## Live Demo

Access the deployed application here:

```text
https://ajilatosin-mediclaimai.hf.space/

```

> Replace the placeholder URL with your actual Hugging Face Space link.

---

## Future Improvements

* Add deep learning models using TensorFlow or PyTorch
* Deploy a REST API using FastAPI
* Containerize the project using Docker
* Add user authentication
* Integrate real-time healthcare data
* Improve feature engineering and model explainability

---

## Screenshots

### Application Dashboard

```text
Add dashboard screenshot here
```

### Prediction Interface

```text
Add prediction interface screenshot here
```

### Data Visualization

```text
Add EDA or visualization screenshot here
```

---

## License

This project is licensed under the MIT License.

---

## Acknowledgements

* Scikit-learn
* Pandas
* NumPy
* Gradio
* XGBoost
* Hugging Face Spaces

---

## Author

Developed by Your Name

Contributions, suggestions, and feedback are welcome.

```
```
