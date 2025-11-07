# Used Car Price Prediction — README.md

> **Used Car Price Prediction** — A Flask web application that predicts the resale price of used cars using a pre-trained machine learning model. Includes user authentication (register/login), model artifacts (encoder, scaler), and a simple HTML front-end.

---

## Table of Contents
1. [Project Overview](#project-overview)  
2. [Features](#features)  
3. [Repository Structure](#repository-structure)  
4. [Prerequisites](#prerequisites)  
5. [Installation & Setup](#installation--setup)  
6. [Environment Variables (.env)](#environment-variables-env)  
7. [Run the Application](#run-the-application)  
8. [How It Works](#how-it-works)  
9. [Files of Interest](#files-of-interest)  
10. [Testing](#testing)  
11. [Troubleshooting](#troubleshooting)  
12. [Future Improvements](#future-improvements)  
13. [Author & Contact](#author--contact)  
14. [License](#license)

---

## Project Overview
This repository contains a Flask application that serves a web interface for predicting used car prices. Users can register and log in, submit car details, and receive a predicted resale price calculated by a saved ML model. The project bundles model artifacts so the app can make predictions without retraining.

---

## Features
- User registration and login (SQLite-backed).
- Car price prediction via pre-trained regression model.
- Preprocessing artifacts included (encoder, scaler, features metadata).
- Clean frontend templates (HTML) and static assets (images).
- Single-file startup (`app.py`) for ease of deployment.

---

## Repository Structure
```
Used_car_prediction-main/
├── app.py
├── test.py
├── realistic_car_data.csv
├── car_price_model.pkl
├── encoder.pkl
├── scaler.pkl
├── features.pkl
├── users.db
├── .env
├── Templates/
│   ├── index.html
│   ├── car.html
│   ├── admin.html
│   ├── Login.html
│   ├── Register.html
│   ├── forgot.html
│   ├── reset.html
│   └── aboutus.html
└── static/
    ├── Elite_i20.webp
    ├── Honda_city.avif
    └── aboutus1.jpg
```

---

## Prerequisites
- Python 3.8+ recommended
- pip
- (Optional) virtualenv or venv

---

## Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/your-username/Used_car_prediction.git
cd Used_car_prediction-main
```

### 2. Install dependencies

```bash
pip install flask pandas scikit-learn numpy joblib sqlalchemy
```

## Run the Application

### Local run (development)
```bash
python app.py
```
By default Flask will serve on `http://127.0.0.1:5000/` (or check terminal output). Open that address in your browser.

### If using `flask` CLI
```bash
export FLASK_APP=app.py        # macOS / Linux
set FLASK_APP=app.py           # Windows (cmd)
flask run
```

---

## How It Works
1. **Frontend**: The HTML form collects car attributes (brand/model, year, km driven, fuel type, transmission, etc.).  
2. **Backend**: `app.py` loads model artifacts at startup:
   - `car_price_model.pkl` — trained regression model
   - `encoder.pkl` — categorical encoder (e.g., OneHot/LabelEncoder)
   - `scaler.pkl` — scaler for numeric features
   - `features.pkl` — expected feature order and metadata  
   The server preprocesses incoming form data to match the model's expected input, applies encoding & scaling, and returns the predicted price on the results page.
3. **Auth**: Simple SQLite-based user authentication stored in `users.db`.

---

## Files of Interest
- **`app.py`** — main Flask application (routing, auth, model inference).
- **`car_price_model.pkl`** — serialized ML model (do not re-train here).
- **`encoder.pkl`, `scaler.pkl`, `features.pkl`** — preprocessing artifacts; required for correct inference.
- **`realistic_car_data.csv`** — dataset used for model training/analysis.
- **`Templates/`** & **`static/`** — front-end assets and templates.
- **`users.db`** — SQLite DB used for user credentials (contains default/test users if included).

---


## Troubleshooting
- **Model load errors**: Ensure `.pkl` files are present and created with a compatible sklearn/pickle version. Re-train or re-pickle with your current environment if needed.
- **Dependencies errors**: Create a new virtual environment and reinstall with `pip install -r requirements.txt`.
- **Database locked**: Delete `users.db` if it's corrupted (will remove user data) or inspect with `sqlite3 users.db`.
- **Secret key not found**: Set `SECRET_KEY` in `.env` or configure it directly in `app.py` for development.

---

## Future Improvements
- Add unit tests for model inference and routes.
- Improve UI responsiveness and mobile layout.
- Add logging and error monitoring.
- Add Dockerfile for containerized deployment.
- Replace pickle with joblib or model-server (e.g., FastAPI + Uvicorn) for production.
- Secure password storage (bcrypt) and email-based password reset.

---

## 📄 License

This project is licensed under the **MIT License** — feel free to use and modify it.

---

## 🌟 Acknowledgements

Developed by [**Diya Kansagara**](https://github.com/diya8405)  
Built with ❤️ using Flask and Machine Learning.

---

## Author & Contact
**Diya Kansagara**  
- Email: diyakansagara25@gmail.com  
- LinkedIn: https://www.linkedin.com/in/diya-kansagara-b70b7a245  

Feel free to open issues or submit pull requests.
