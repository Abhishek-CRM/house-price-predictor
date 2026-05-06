# 🏠 California House Price Predictor

## Overview
An end-to-end Machine Learning web application that predicts California 
house prices based on location, income, and property features. Built with 
Python and deployed as an interactive web app using Streamlit.

## Live Demo
[Click here to try the app](https://house-price-predictor-hpp.streamlit.app/)

## Tech Stack
- **Python** (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn)
- **Machine Learning** — Linear Regression
- **Deployment** — Streamlit
- **Model Persistence** — Joblib

## Model Performance
| Metric | Score |
|--------|-------|
| R² Score | 0.582 |
| RMSE | $73,983 |
| MAE | $54,257 |

## Key Findings
- Median Income is the #1 price driver — every $10k income increase = ~$37k price increase
- Location (Latitude/Longitude) has a stronger impact than property features
- House age has minimal effect on price
- Model trained on 20,640 California housing records

## Project Structure
- `app.py` — Streamlit web application
- `house_price_prediction.ipynb` — Full ML analysis notebook
- `house_price_model.pkl` — Saved trained model
- `eda_analysis.png` — EDA visualizations
- `model_results.png` — Model performance charts
