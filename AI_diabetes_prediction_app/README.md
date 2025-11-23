# AI-Based Diabetes Risk Prediction Web Application

## 1. Project Overview

This project implements an **AI-based web application** for predicting diabetes risk from basic health indicators. The system is designed as an **interactive client-side dashboard** that:

- Allows users to **upload a cleaned diabetes dataset**
- Performs **Exploratory Data Analysis (EDA)** directly in the browser
- Trains both a **Logistic Regression baseline (1-layer NN)** and a **deeper Neural Network** using **TensorFlow.js**
- Provides an **interactive risk prediction form** with **visual feedback** and **risk factor explanation**

The entire pipeline runs **in the browser (client-side only)**, making the application easy to deploy via GitHub Pages and suitable for teaching and experimentation in Neural Network & Deep Learning courses.

---

## 2. Team

- **Do Thi Hien** – Project Manager & Data Lead  & Web Integration
- **Bhuian Md Waliulla** – Model Lead & Backend logic (ported to client-side TF.js)  
- **Ahmed Md Zunayed** – Frontend & UI/UX Lead  

---

## 3. Problem Definition

Early detection of type 2 diabetes is critical for reducing long-term complications. Traditional screening often requires in-person clinical workflows and manual interpretation of risk factors.

This project addresses the following problem:

> Given a set of routinely available health indicators, can we build an AI-based tool that estimates a person’s diabetes risk and explains which factors contribute to that risk?

Target users include:

- Health-conscious individuals with access to their basic lab results
- Healthcare students / practitioners who want to understand ML-based risk scoring
- Instructors and students in **Neural Network & Deep Learning** courses

---

## 4. Dataset

- **Source:** Diabetes Prediction Dataset (originally from Kaggle)  
- **Local file used in this project:** `diabetes_raw_cleaned_25k.csv`  
- **Columns:**

  - `gender`
  - `age`
  - `hypertension`
  - `heart_disease`
  - `smoking_history`
  - `bmi`
  - `HbA1c_level`
  - `blood_glucose_level`
  - `diabetes` (target: 0/1)

For this web app, we assume that:

- Categorical features (`gender`, `smoking_history`) are **already numerically encoded** in the CSV in a way that is consistent with the encoding used in the prediction form:
  - `gender`: `0 = Female`, `1 = Male`
  - `smoking_history`: mapped via  
    `Never → 0, No Info → 1, Current → 2, Former → 3, Ever → 4, Not Current → 5`
- There are **no missing values** in the cleaned dataset (or they were imputed beforehand).

---

## 5. Methodology

### 5.1 Client-Side Architecture

The application is entirely client-side:

- **Data ingestion:** CSV uploaded via `<input type="file">` and parsed by **PapaParse**
- **EDA & Visualization:**  
  - Class distribution and feature distributions plotted with **Chart.js**
- **Model training:**  
  - Implemented using **TensorFlow.js** running in the browser
- **Prediction:**  
  - User form inputs → converted into feature vector → passed to trained neural network

This satisfies the course requirement of **Neural Networks & Deep Learning** while also demonstrating a practical, fully browser-based deployment.

### 5.2 Exploratory Data Analysis (EDA)

After CSV upload (before training):

1. **Class distribution**  
   - Bar chart of `diabetes = 0` vs `diabetes = 1`
   - Helps quickly assess class imbalance

2. **Numeric feature distributions (Age, BMI)**  
   - Age grouped into: `<30`, `30–39`, `40–49`, `50–59`, `60+`  
   - BMI grouped into: `<18.5`, `18.5–24.9`, `25–29.9`, `30–34.9`, `35+`  
   - Both are visualized with Chart.js bar charts on the left-hand side

These EDA plots are computed and rendered dynamically from the uploaded dataset to connect data understanding with model training.

---

## 6. Model Design

### 6.1 Input Features

The model uses the following features:

- `gender`
- `age`
- `hypertension`
- `heart_disease`
- `smoking_history`
- `bmi`
- `HbA1c_level`
- `blood_glucose_level`

The target label is `diabetes` (0 = no diabetes, 1 = diabetes).

### 6.2 Logistic Regression Baseline

Implemented as a single-layer neural network:

```text
Input (8 features) → Dense(1, activation='sigmoid')

_**Link page:**_ https://123456789hien.github.io/nndl/AI_diabetes_prediction_app/
