# Adult Census Income Prediction

An end-to-end Machine Learning project that predicts whether an individual's annual income is <=50K or >50K using the UCI Adult Census Income Dataset.

## Project Overview

This project demonstrates the complete Machine Learning workflow:

Data Ingestion → Data Preprocessing → Model Training → Model Evaluation → Model Integration → Streamlit Application → Deployment

## Dataset

- Dataset: UCI Adult / Census Income Dataset
- Source: UCI Machine Learning Repository
- Records: 32,561
- Problem Type: Binary Classification

Dataset:
https://archive.ics.uci.edu/dataset/2/adult

## Machine Learning Model

Random Forest Classifier

Test Accuracy: ~86%

## Technologies Used

- Python
- Snowflake
- Snowpark
- Pandas
- Scikit-learn
- Random Forest
- Streamlit
- Joblib

## Features

The application accepts user information such as:

- Age
- Work Class
- Education
- Education Level
- Marital Status
- Occupation
- Relationship
- Race
- Sex
- Hours per Week
- Capital Gain
- Capital Loss
- Native Country

The trained Random Forest model predicts:

- <=50K
- >50K

## Snowflake Implementation

The dataset was loaded into Snowflake and processed for Machine Learning.

The trained model was stored in a Snowflake stage and integrated with an interactive Streamlit application.

## Application

The Streamlit application provides an interactive interface where users can enter individual details and receive an income prediction with model confidence.

## Project Structure

```text
snowflake-adult-income-prediction/
│
├── income_prediction_ml.ipynb
├── streamlit_app.py
├── requirements.txt
└── README.md
