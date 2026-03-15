# Implementation Plan: AI-Based Crop Recommendation System

## Goal Description
Build an AI-based web application to recommend the most suitable crop for farming based on soil nutrients and environmental conditions. We will use Python/Flask for the backend, Scikit-learn for the ML model, and vanilla HTML/CSS/JS for an interactive frontend.

## User Review Required
> [!IMPORTANT]
> Since we do not currently have a trained model or dataset in the workspace, this plan includes a script to **generate a synthetic dataset** with realistic ranges for `N, P, K, temperature, humidity, pH, rainfall` across 22 common crops, and then trains the machine learning model on it. 
> Alternatively, if you have a specific `Crop_recommendation.csv` file you want to use, let me know, and I can use that instead.

## Proposed Changes

### Machine Learning and Data
#### [NEW] `dataset_generator.py`
Creates a synthetic dataset `crop_data.csv` containing samples for 22 different crops with randomized but realistic parameters (N, P, K, temperature, humidity, pH, rainfall).
#### [NEW] `train.py`
Reads `crop_data.csv`, trains a Scikit-Learn `RandomForestClassifier`, and saves the trained model to `model.pkl`.

### Backend (Python/Flask)
#### [NEW] `app.py`
A Flask web application with two routes:
- `/` - Serves the main HTML page.
- `/predict` (POST) - Receives JSON with the 7 environmental parameters, loads `model.pkl`, and returns the predicted crop.
#### [NEW] `requirements.txt`
Dependencies: `Flask`, `scikit-learn`, `pandas`, `numpy`.

### Frontend
#### [NEW] `templates/index.html`
An interactive web interface with a responsive form to collect user inputs. 
#### [NEW] `static/style.css`
A premium, modern design aesthetic featuring a glassmorphic form, dynamic hover effects, and a farming/nature-inspired color palette (greens/earth tones).
#### [NEW] `static/script.js`
JavaScript to intercept form submission, send asynchronous POST requests to the `/predict` endpoint, and display the recommended crop dynamically with animations.

## Verification Plan

### Automated Tests
- Run `python dataset_generator.py` and verify `crop_data.csv` is created.
- Run `python train.py` and ensure `model.pkl` is saved successfully.

### Manual Verification
- Start the Flask app via `python app.py`.
- Open the application in a local browser via `http://127.0.0.1:5000`.
- Input sample data (e.g., N=90, P=42, K=43, Temp=20.8, Humidity=82.0, pH=6.5, Rainfall=202.9) and verify that the UI returns a valid crop prediction (like Rice).
- Inspect the visual aesthetics and ensure it feels premium and responsive.
