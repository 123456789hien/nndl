**AI-Based Diabetes Risk Prediction Web Application
1. Project Overview**

_This project is a browser-based dashboard for **early diabetes risk prediction** built for the course **Neural Network and Deep Learning**.
_
The app allows users to:

- Load a diabetes dataset (either from GitHub or from a local CSV file)

- Train two models directly in the browser

  +) Logistic regression (implemented as a single dense layer)

  +) A small feed-forward neural network

- Explore basic EDA and model behaviour

- Enter personal health indicators to get an estimated diabetes risk

- See “what-if” scenarios that simulate improvement of key risk factors

Everything runs **client-side** using **TensorFlow.js** so no Python backend is required.

**2. Team**

**Do Thi Hien** – Project Manager, Data Lead and ML Application Architect

- Coordinated project tasks

- Designed data pipeline, EDA, model integration

- Designed and refined the web UI, UX and interactive model logic

**Bhuian Md Waliulla** – Model Lead and Backend Logic Design

- Designed the neural network architecture

- Helped define training strategy and evaluation metrics

- Specified how model components should be exposed to the web app

**Ahmed Md Zunayed** – Frontend and UI/UX Lead

- Built the original HTML & CSS layout

- Integrated dashboard workflow across steps

- Tuned the visual style for a dark “plasma” premium look

**3. Technology Stack**

**Frontend**

HTML5

CSS3

Vanilla JavaScript

**Libraries**

TensorFlow.js – in-browser training and inference

Chart.js – interactive charts

PapaParse – CSV parsing in the browser

**Hosting**

Designed to run on **GitHub Pages** or any static file server

Everything is contained in four files so the structure stays simple and easy to deploy.

**4. Repository Structure**
diabetes-risk-webapp/
├── index.html                  # Main dashboard layout and app structure
├── style.css                   # Dark plasma UI theme and responsive layout
├── script.js                   # EDA, model training, prediction and what-if logic
└── diabetes_raw_cleaned_25k.csv   # Default Kaggle-based diabetes dataset


If you move the CSV into a _data/_ folder you must update the path in _script.js_ inside the _autoLoadDataset()_ function.

**5. Dataset**

Source: Diabetes Prediction Dataset (Kaggle)
_https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset_

**Columns used in this project**

_gender_ – Male or Female

_age_ – continuous age in years

_hypertension_ – 0 (no) or 1 (yes)

_heart_disease_ – 0 (no) or 1 (yes)

_smoking_history_ – categorical
  - Never, No Info, Current, Former, Ever, Not Current

_bmi_ – body mass index

_HbA1c_level_ – glycated hemoglobin

_blood_glucose_level_ – blood glucose (mg/dL)

_diabetes_ – binary target 0 or 1

The dataset in this repo is a cleaned subset with at most **25 000 rows**

_diabetes_raw_cleaned_25k.csv_

**6. Core Features**

The dashboard is organised into **three main steps** on the UI (left column for data and models, right column for prediction).

**Step 1 – Upload Dataset**

Users can choose between two options:

_**1. Choose CSV File**_

  - Upload any compatible diabetes CSV from local disk

  - The app parses it with PapaParse, filters empty rows and computes class counts

_**2. Auto-Load Dataset**_

  - Automatically loads the default dataset

  - Path is defined in _script.js_ in _autoLoadDataset()_

  - Designed for quick demo and reproducible experiments

After loading the dataset, the app shows:

  - Total number of rows

  - Number of diabetes cases

  - Percentage of positive vs negative classes

It also enables the **Train Models** button and reveals the EDA charts.

**Step 2 – Train AI Models**

From the same left column, Step 2 lets the user train two models **in the browser**:

A **logistic regression model**

  Implemented as Dense(1, sigmoid) in TensorFlow.js

A **neural network model**

  _Dense(16, relu)_ + dropout

  _Dense(8, relu)_ + dropout

  _Dense(1, sigmoid)_

Implementation details:

  Uses up to **5 000 samples** from the loaded dataset to prevent freezing the browser

  Splits data into **80% train** and **20% test**

  Trains both models for **20 epochs** with batch size **32**

  Input vector uses this order of features

  ['gender', 'age', 'hypertension', 'heart_disease', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']


After training, the dashboard displays:

  Logistic regression accuracy and loss

  Neural network accuracy

  A **feature importance chart** based on logistic regression weights

The “Feature Importance” chart shows which features push risk up or down according to the linear model.

**Step 2 – EDA (Left Column)**

After dataset loading the app computes lightweight EDA:

  **1. Class Distribution**

    Bar chart of _diabetes = 0_ vs _diabetes = 1_

  **2. Age Distribution (grouped)**

    Bins: _<30, 30–39, 40–49, 50–59, 60+_

 **3.  BMI Distribution (categorised)**

    <18.5

    18.5–24.9

    25–29.9

    30–34.9

    35+

These charts help users visually understand the sample composition before or after training.

**Step 3 – Check Your Risk (Right Column)**

The right column is an interactive form that lets a user enter health indicators and obtain an estimated diabetes risk from the **neural network** model.

Inputs:

  Gender

  Age

  Hypertension (Yes/No)

  Heart disease (Yes/No)

  Smoking history

  BMI

  HbA1c (optional)

  Blood glucose

_**HbA1c is optional**_

Many people do not know their **HbA1c level**. The app therefore:

  Treats the HbA1c input as **optional**

  If HbA1c is missing, the model still runs using the other features

  In the risk factors list the app explains that HbA1c was not provided

If HbA1c is provided, the app also interprets it as:

  Normal (< 5.7)

  Pre-diabetes (5.7–6.4)

  Diabetes range (≥ 6.5)

_**BMI can be entered or calculated**_

BMI is often unfamiliar despite being easy to compute. The UI supports both:

  _**1. Direct BMI input**_

    User can type BMI directly if they already know it

  _**2. Built-in BMI calculator**_

    Additional inputs:

      Weight (kg)

      Height (cm) or (m)

    The app uses:

      BMI = weight in kg / (height in metres)^2

    If height is entered in centimetres the app converts to metres automatically

    On clicking “Calculate BMI” the BMI field is filled for the user

If BMI is still not provided, the model falls back to other features and the risk factor section explains that BMI was missing.

**Risk output**

After pressing **Predict Risk** the app:

  Builds a feature vector from the form

  Passes it through the trained neural network

  Displays:

    - Estimated risk probability (0–100%)

    - Risk level badge: low, moderate or high

    - Colored progress bar

**Risk Factors Analysis**

Below the probability, the app shows a **Risk Factors Analysis** list.

For each feature the app shows:

  - The factor name

  - The numeric value or message (for example “Not provided”)

  - A short comment with a human-readable interpretation

  - A color coding

    +) Red for risk factors

    +) Green for protective factors

Examples:

  “High HbA1c Level – High (≥ 6.5, diabetes range)”

  “Normal Blood Glucose – Normal (< 100 mg/dL)”

  “High BMI – Obese (30–34.9)”

  “Age – Older age group (≥ 60, higher risk)”

  “BMI – Not provided, you can calculate it using weight and height”

**What-if Scenarios**

To connect the model with actionable insight, the app includes a **What-if** block below the risk factors.

Two scenarios are computed:

**1. If BMI were 25**

  Keeps all other features fixed

  Sets BMI = 25

  Computes new risk with the same neural network

  Shows new risk percentage and change vs current risk

**2. If HbA1c were 6.0**

  Keeps other features fixed

  Sets HbA1c = 6.0

  Computes new risk and shows the change

This gives an intuitive sense of how improving weight or glycemic control might affect predicted risk according to the learned model.
The app clearly labels these as **hypothetical scenarios** not medical advice.

**7. Implementation Details
7.1 Preprocessing**

Categorical features are numeric-encoded in JS

  _gender_: Female → 0, Male → 1

  _smoking_history_: mapping Never → 0, No Info → 1, Current → 2, Former → 3, Ever → 4, Not Current → 5

Missing numeric values in the dataset are replaced with 0 during tensor creation

For user inputs, BMI and HbA1c can be left empty

  They are treated as NaN then replaced by 0 in the model vector while still being explained in the risk analysis section

No explicit scaling or normalization is applied in this version since the focus is on demonstrating **model training with TensorFlow.js** and **interactive behaviour** rather than achieving optimal benchmark scores.

**7.2 Models**

_**Logistic Regression**_

  TensorFlow.js sequential model with a single Dense layer

  Binary cross-entropy loss and Adam optimizer

_**Neural Network**_

  Two hidden layers with ReLU

  Dropout to reduce overfitting

  Outputs probability via sigmoid

Both models are trained on the same training split and evaluated on the same test split.

**8. Evaluation Metrics**

The dashboard reports:

  **Accuracy** for logistic regression

  **Loss** (binary cross-entropy) for logistic regression

  **Accuracy** for the neural network

The intention for the course is to:

  Compare logistic regression vs neural network

  Observe the trade-off between model complexity and performance

  Show how feature weights (from logistic regression) relate to model explanations

**9. Limitations**

- Predictions are **not medical diagnoses**

- The dataset is limited and may not generalise to all populations

- No cross-validation or hyperparameter search is implemented

- No calibration check for the predicted probabilities

- HbA1c and BMI imputation is very simple (missing → 0) on the model side

  The UI tries to compensate by clearly informing the user when values are not provided

**10. Possible Future Work**

- Add more robust preprocessing and feature scaling

- Add ROC and AUC visualisations

- Implement model comparison plots over epochs

- Add saving and loading of trained models (TensorFlow.js _model.save()_ and _model.load_)

- Extend the what-if module with more scenarios (for example age groups or blood glucose improvement)

- Integrate more detailed educational content explaining each biomarker

**11. Disclaimer**

This application is developed for **educational purposes** within a Neural Network and Deep Learning course.

The predictions are:

  - Based on historical data

  - Simplified

  - Not validated for clinical use

Users should **never** use this tool as a substitute for professional medical advice, diagnosis or treatment.
