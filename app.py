from flask import Flask, render_template, request, redirect, url_for, flash,send_file, jsonify
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import pickle
import numpy as np
import pandas as pd
import sqlite3
import os
import uuid
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import plotly.express as px
import plotly.graph_objects as go
import json
from datetime import datetime, timedelta



load_dotenv()

EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

app = Flask(__name__)
app.secret_key = 'your_secret_key'


# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# User class
class User(UserMixin):
    def __init__(self, id_, username, password, role):
        self.id = id_
        self.username = username
        self.password = password
        self.role = role

# User loader
@login_manager.user_loader
def load_user(user_id):
    con = sqlite3.connect('users.db')
    cur = con.cursor()
    cur.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    row = cur.fetchone()
    con.close()
    if row:
        return User(row[0], row[2], row[3], row[11])
    return None

# Create DB and insert default admin
def init_db():
    con = sqlite3.connect('users.db')
    cur = con.cursor()

    # Create users table
    cur.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        email TEXT,
        phone TEXT,
        address TEXT,
        city TEXT,
        state TEXT,
        country TEXT,
        zip TEXT,
        role TEXT DEFAULT 'user'
    )''')

    # Create predictions table with feedback columns
    cur.execute('''CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        year INTEGER,
        odometer REAL,
        engine_size REAL,
        brand TEXT,
        model_name TEXT,
        fuel_type TEXT,
        owner INTEGER,
        predicted_price REAL,
        rating INTEGER DEFAULT 0,
        feedback_text TEXT DEFAULT '',
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )''')

    # Add rating and feedback_text columns if they don't exist
    try:
        cur.execute('ALTER TABLE predictions ADD COLUMN rating INTEGER DEFAULT 0')
    except sqlite3.OperationalError:
        pass  # Column already exists
    
    try:
        cur.execute('ALTER TABLE predictions ADD COLUMN feedback_text TEXT DEFAULT ""')
    except sqlite3.OperationalError:
        pass  # Column already exists

    cur.execute('''CREATE TABLE IF NOT EXISTS reset_tokens (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT,
        token TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')

    con.commit()

    # Check if admin exists
    cur.execute("SELECT * FROM users WHERE username = 'admin'")
    if not cur.fetchone():
        admin_pw = generate_password_hash("admin123")
        cur.execute('''INSERT INTO users 
            (name, username, password, email, phone, address, city, state, country, zip, role)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            ("Admin", "admin", admin_pw, "admin@example.com", "1234567890", "Admin Address",
             "Admin City", "Admin State", "Admin Country", "000000", "admin"))
        con.commit()

    con.close()

init_db()

# Load ML model and pre-processing tools
with open("car_price_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

with open("features.pkl", "rb") as f:
    feature_data = pickle.load(f)
    scaler_features = feature_data["scaler_features"]
    encoder_feature_names = feature_data["encoder_feature_names"]

# Routes
@app.route("/")
def home():
    return redirect(url_for('login'))


@app.route("/car")
@login_required
def car():
    con = sqlite3.connect('users.db')
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute("SELECT * FROM users WHERE id = ?", (current_user.id,))
    user_row = cur.fetchone()
    con.close()
    return render_template("car.html", username=current_user.username, user_details=user_row)

# Function to generate predictions per day chart
def generate_predictions_per_day_chart(predictions_df):
    try:
        # Ensure timestamp is datetime
        predictions_df['timestamp'] = pd.to_datetime(predictions_df['timestamp'])
        
        # Get last 30 days
        thirty_days_ago = datetime.now() - timedelta(days=30)
        recent_predictions = predictions_df[predictions_df['timestamp'] >= thirty_days_ago].copy()
        
        # Group by date and count
        daily_counts = recent_predictions.groupby(recent_predictions['timestamp'].dt.date).size().reset_index()
        daily_counts.columns = ['date', 'count']
        
        # Create the chart
        fig = px.line(daily_counts, x='date', y='count', 
                      title='Predictions Per Day (Last 30 Days)',
                      labels={'date': 'Date', 'count': 'Number of Predictions'})
        
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(gridcolor='lightgray'),
            yaxis=dict(gridcolor='lightgray', zeroline=True)
        )
        
        return fig.to_json()
    except Exception as e:
        print(f"Error in predictions chart: {str(e)}")
        # Return empty chart if there's an error
        fig = px.line(title='Predictions Per Day (Last 30 Days)')
        return fig.to_json()

# Function to generate average predicted prices chart
def generate_avg_prices_chart(predictions_df):
    try:
        # Ensure timestamp is datetime
        predictions_df['timestamp'] = pd.to_datetime(predictions_df['timestamp'])
        
        # Get last 30 days
        thirty_days_ago = datetime.now() - timedelta(days=30)
        recent_predictions = predictions_df[predictions_df['timestamp'] >= thirty_days_ago].copy()
        
        # Group by date and calculate average price
        daily_avg_prices = recent_predictions.groupby(recent_predictions['timestamp'].dt.date)['predicted_price'].mean().reset_index()
        daily_avg_prices.columns = ['date', 'avg_price']
        
        # Create the chart
        fig = px.line(daily_avg_prices, x='date', y='avg_price', 
                      title='Average Predicted Prices Over Time',
                      labels={'date': 'Date', 'avg_price': 'Average Price (₹)'})
        
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(gridcolor='lightgray'),
            yaxis=dict(gridcolor='lightgray', zeroline=True)
        )
        
        return fig.to_json()
    except Exception as e:
        print(f"Error in prices chart: {str(e)}")
        # Return empty chart if there's an error
        fig = px.line(title='Average Predicted Prices Over Time')
        return fig.to_json()

# Function to generate top brands chart
def generate_top_brands_chart(predictions_df):
    # Count occurrences of each brand
    brand_counts = predictions_df['brand'].value_counts().reset_index()
    brand_counts.columns = ['brand', 'count']
    
    # Get top 10 brands
    top_brands = brand_counts.head(10)
    
    # Create the chart
    fig = px.bar(top_brands, x='brand', y='count', 
                 title='Top 10 Most Searched Car Brands',
                 labels={'brand': 'Brand', 'count': 'Number of Searches'})
    
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(gridcolor='lightgray'),
        yaxis=dict(gridcolor='lightgray', zeroline=True)
    )
    
    return fig.to_json()

# Function to generate top models chart
def generate_top_models_chart(predictions_df):
    # Count occurrences of each model
    model_counts = predictions_df['model_name'].value_counts().reset_index()
    model_counts.columns = ['model_name', 'count']
    
    # Get top 10 models
    top_models = model_counts.head(10)
    
    # Create the chart
    fig = px.bar(top_models, x='model_name', y='count', 
                 title='Top 10 Most Searched Car Models',
                 labels={'model_name': 'Model', 'count': 'Number of Searches'})
    
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(gridcolor='lightgray'),
        yaxis=dict(gridcolor='lightgray', zeroline=True)
    )
    
    return fig.to_json()

# Function to generate fuel type distribution chart
def generate_fuel_type_chart(predictions_df):
    try:
        # Count fuel types
        fuel_counts = predictions_df['fuel_type'].value_counts().reset_index()
        fuel_counts.columns = ['fuel_type', 'count']
        
        # Create the chart
        fig = px.pie(fuel_counts, values='count', names='fuel_type',
                     title='Fuel Type Distribution',
                     color_discrete_sequence=px.colors.qualitative.Set3)
        
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            )
        )
        
        return fig.to_json()
    except Exception as e:
        print(f"Error in fuel type chart: {str(e)}")
        fig = px.pie(title='Fuel Type Distribution')
        return fig.to_json()

# Function to generate price range distribution chart
def generate_price_range_chart(predictions_df):
    try:
        # Create price ranges
        price_ranges = pd.cut(predictions_df['predicted_price'], 
                            bins=[0, 200000, 400000, 600000, 800000, 1000000, float('inf')],
                            labels=['0-2L', '2L-4L', '4L-6L', '6L-8L', '8L-10L', '10L+'])
        
        # Count cars in each range
        price_dist = price_ranges.value_counts().sort_index().reset_index()
        price_dist.columns = ['range', 'count']
        
        # Create the chart
        fig = px.bar(price_dist, x='range', y='count',
                     title='Price Range Distribution',
                     labels={'range': 'Price Range (₹)', 'count': 'Number of Cars'})
        
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(gridcolor='lightgray'),
            yaxis=dict(gridcolor='lightgray', zeroline=True)
        )
        
        return fig.to_json()
    except Exception as e:
        print(f"Error in price range chart: {str(e)}")
        fig = px.bar(title='Price Range Distribution')
        return fig.to_json()

# Function to generate year distribution chart
def generate_year_distribution_chart(predictions_df):
    try:
        # Count cars by year
        year_counts = predictions_df['year'].value_counts().sort_index().reset_index()
        year_counts.columns = ['year', 'count']
        
        # Create the chart
        fig = px.line(year_counts, x='year', y='count',
                      title='Year-wise Car Distribution',
                      labels={'year': 'Year', 'count': 'Number of Cars'})
        
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(gridcolor='lightgray'),
            yaxis=dict(gridcolor='lightgray', zeroline=True)
        )
        
        return fig.to_json()
    except Exception as e:
        print(f"Error in year distribution chart: {str(e)}")
        fig = px.line(title='Year-wise Car Distribution')
        return fig.to_json()

# Function to generate engine size distribution chart
def generate_engine_size_chart(predictions_df):
    try:
        # Create engine size ranges
        engine_ranges = pd.cut(predictions_df['engine_size'], 
                             bins=[0, 1.0, 1.5, 2.0, 2.5, 3.0, float('inf')],
                             labels=['<1.0L', '1.0-1.5L', '1.5-2.0L', '2.0-2.5L', '2.5-3.0L', '3.0L+'])
        
        # Count cars in each range
        engine_dist = engine_ranges.value_counts().sort_index().reset_index()
        engine_dist.columns = ['range', 'count']
        
        # Create the chart
        fig = px.bar(engine_dist, x='range', y='count',
                     title='Engine Size Distribution',
                     labels={'range': 'Engine Size', 'count': 'Number of Cars'})
        
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            xaxis=dict(gridcolor='lightgray'),
            yaxis=dict(gridcolor='lightgray', zeroline=True)
        )
        
        return fig.to_json()
    except Exception as e:
        print(f"Error in engine size chart: {str(e)}")
        fig = px.bar(title='Engine Size Distribution')
        return fig.to_json()

@app.route("/admin")
@login_required
def admin():
    if current_user.role != 'admin':
        return redirect(url_for('car'))  # Prevent non-admins

    con = sqlite3.connect('users.db')
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    # Fetch prediction records with user details
    cur.execute('''
        SELECT predictions.*, users.username 
        FROM predictions 
        JOIN users ON predictions.user_id = users.id
        ORDER BY predictions.timestamp DESC
    ''')
    prediction_records = cur.fetchall()

    # Fetch registered user data
    cur.execute('''
        SELECT id, name, username, email, phone, address, city, state, country, zip, role
        FROM users
        ORDER BY id DESC
    ''')
    user_records = cur.fetchall()

    # Fetch all predictions for chart generation
    cur.execute('SELECT * FROM predictions')
    predictions_data = cur.fetchall()
    
    # Convert to DataFrame for easier manipulation
    predictions_df = pd.DataFrame(predictions_data, columns=[
        'id', 'user_id', 'year', 'odometer', 'engine_size', 'brand', 
        'model_name', 'fuel_type', 'owner', 'predicted_price', 'rating', 'feedback_text', 'timestamp'
    ])
    
    # Generate charts using Python/Plotly
    top_brands_chart = generate_top_brands_chart(predictions_df)
    top_models_chart = generate_top_models_chart(predictions_df)
    fuel_type_chart = generate_fuel_type_chart(predictions_df)
    price_range_chart = generate_price_range_chart(predictions_df)
    year_distribution_chart = generate_year_distribution_chart(predictions_df)
    engine_size_chart = generate_engine_size_chart(predictions_df)

    con.close()

    return render_template("admin.html", username=current_user.username,
                           predictions=prediction_records,
                           users=user_records,
                           top_brands_chart=top_brands_chart,
                           top_models_chart=top_models_chart,
                           fuel_type_chart=fuel_type_chart,
                           price_range_chart=price_range_chart,
                           year_distribution_chart=year_distribution_chart,
                           engine_size_chart=engine_size_chart)


@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    predicted_price = None
    error_message = None
    prediction_id = None

    if request.method == "POST":
        try:
            # Get data from form
            year = int(request.form["Year"])
            odometer = float(request.form["Odometer Reading (km)"])
            engine_size = float(request.form["Engine Capacity (L)"])
            brand = request.form["Brand"]
            model_name = request.form["Model"]
            fuel_type = request.form["Fuel Type"]
            owner = int(request.form["Number of Owners"])

            num_features = np.array([[year, odometer, engine_size, owner]])
            num_features_scaled = scaler.transform(num_features)

            cat_features = pd.DataFrame([[brand, fuel_type, model_name]],
                                        columns=["Brand", "Fuel Type", "Model"])
            cat_features_encoded = encoder.transform(cat_features).toarray()

            features = np.hstack((num_features_scaled, cat_features_encoded))
            features = features.reshape(1, -1)

            predicted_price = round(model.predict(features)[0], 2)

            # Save the prediction in the database
            con = sqlite3.connect('users.db')
            cur = con.cursor()
            cur.execute('''INSERT INTO predictions 
                (user_id, year, odometer, engine_size, brand, model_name, fuel_type, owner, predicted_price, rating, feedback_text)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                        (current_user.id, year, odometer, engine_size, brand, model_name, fuel_type, owner, predicted_price, 0, ""))
            prediction_id = cur.lastrowid
            con.commit()
            con.close()

        except Exception as e:
            error_message = f"Error during prediction: {str(e)}"

    return render_template("index.html", price=predicted_price, error=error_message, prediction_id=prediction_id)

@app.route("/submit_feedback", methods=["POST"])
@login_required
def submit_feedback():
    try:
        prediction_id = request.form.get('prediction_id')
        rating = int(request.form.get('feedback_rating'))
        feedback_text = request.form.get('feedback_text')

        con = sqlite3.connect('users.db')
        cur = con.cursor()
        cur.execute('''UPDATE predictions SET rating = ?, feedback_text = ? WHERE id = ?''',
            (rating, feedback_text, prediction_id))
        con.commit()
        con.close()

        return jsonify({"status": "success", "message": "Thank you for your feedback!"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route("/download_report")
@login_required
def download_report():
    con = sqlite3.connect('users.db')
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    # Get latest prediction for current user
    cur.execute('''
        SELECT * FROM predictions 
        WHERE user_id = ? 
        ORDER BY timestamp DESC 
        LIMIT 1
    ''', (current_user.id,))
    prediction = cur.fetchone()
    con.close()

    if not prediction:
        flash("No predictions found to generate a report.")
        return redirect(url_for('car'))

    # Generate PDF
    pdf_path = f"static/user_prediction_report_{uuid.uuid4().hex}.pdf"
    c = canvas.Canvas(pdf_path, pagesize=letter)

    y = 770  # starting height

    # --- Header ---
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, "Car Price Prediction Report")
    y -= 20

    c.setFont("Helvetica", 11)
    c.drawString(50, y, "Generated by CarXpert - Smarter Vehicle Insights")
    y -= 30

    # --- Intro / Description ---
    c.setFont("Helvetica", 10)
    c.drawString(50, y, "This report contains the most recent car price prediction based on your input.")
    y -= 15
    c.drawString(50, y, "We use machine learning to estimate the expected market price of your vehicle.")
    y -= 30

    # --- Prediction Data ---
    c.setFont("Helvetica", 12)
    c.drawString(50, y, f"Username: {current_user.username}"); y -= 20
    c.drawString(50, y, f"Year: {prediction['year']}"); y -= 20
    c.drawString(50, y, f"Odometer: {prediction['odometer']} km"); y -= 20
    c.drawString(50, y, f"Engine Size: {prediction['engine_size']} L"); y -= 20
    c.drawString(50, y, f"Brand: {prediction['brand']}"); y -= 20
    c.drawString(50, y, f"Model: {prediction['model_name']}"); y -= 20
    c.drawString(50, y, f"Fuel Type: {prediction['fuel_type']}"); y -= 20
    c.drawString(50, y, f"Number of Owners: {prediction['owner']}"); y -= 30

    c.setFont("Helvetica-Bold", 13)
    c.drawString(50, y, f"Predicted Price: ₹ {prediction['predicted_price']:,}"); y -= 40

    # --- Footer / Note ---
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(50, y, "Note: This prediction is an estimate based on your input and may vary due to market conditions.")
    y -= 15
    c.drawString(50, y, "Thank you for using CarXpert!")

    c.save()

    return send_file(pdf_path, as_attachment=True)


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form['username']
        password = request.form['password']

        con = sqlite3.connect('users.db')
        cur = con.cursor()
        cur.execute("SELECT * FROM users WHERE username = ?", (username,))
        row = cur.fetchone()
        con.close()

        if row and check_password_hash(row[3], password):
            user = User(row[0], row[2], row[3], row[11])
            login_user(user)
            if user.role == 'admin':
                return redirect(url_for('admin'))
            else:
                return redirect(url_for('car'))
        else:
            flash("Invalid username or password!", "danger")

    return render_template("login.html")


def send_reset_email(to_email, token):
    reset_url = f"http://127.0.0.1:5000/reset/{token}"
    subject = "Password Reset Request"
    body = f"""
Hi,

You requested a password reset. Click the link below to reset your password:

{reset_url}

If you didn't request this, you can ignore this email.
"""
    msg = MIMEMultipart()
    msg["From"] = EMAIL_USER
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_USER, EMAIL_PASSWORD)
            server.send_message(msg)
        print("Reset email sent successfully.")
    except Exception as e:
        print(f"Error sending email: {e}")

@app.route("/forgot", methods=["GET", "POST"])
def forgot():
    if request.method == "POST":
        email = request.form['email']
        token = str(uuid.uuid4())
        con = sqlite3.connect('users.db')
        cur = con.cursor()
        cur.execute("SELECT * FROM users WHERE email = ?", (email,))
        user = cur.fetchone()
        if user:
            cur.execute("INSERT INTO reset_tokens (email, token) VALUES (?, ?)", (email, token))
            con.commit()
            send_reset_email(email, token)
            flash("Password reset link has been sent to your email.", "info")
        else:
            flash("Email not found!", "danger")
        con.close()
    return render_template("forgot.html")

@app.route("/reset/<token>", methods=["GET", "POST"])
def reset(token):
    con = sqlite3.connect('users.db')
    cur = con.cursor()
    cur.execute("SELECT email FROM reset_tokens WHERE token = ?", (token,))
    row = cur.fetchone()
    if not row:
        flash("Invalid or expired token.", "danger")
        return redirect(url_for('login'))
    if request.method == "POST":
        password = request.form['password']
        confirm = request.form['confirm_password']
        if password != confirm:
            flash("Passwords do not match!", "danger")
        else:
            hashed_pw = generate_password_hash(password)
            cur.execute("UPDATE users SET password = ? WHERE email = ?", (hashed_pw, row[0]))
            cur.execute("DELETE FROM reset_tokens WHERE token = ?", (token,))
            con.commit()
            flash("Password updated successfully.", "success")
            print(f"Updated password hash for {row[0]}: {hashed_pw}")

            con.close()
            return redirect(url_for('login'))
    con.close()
    return render_template("reset.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form['name']
        username = request.form['username']
        email = request.form['email']
        phone = request.form['phone']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        address = request.form['address']
        city = request.form['city']
        state = request.form['state']
        country = request.form['country']
        zip_code = request.form['zip']

        if password != confirm_password:
            flash("Passwords do not match.", "danger")
            return redirect(url_for('register'))

        hashed_pw = generate_password_hash(password)

        try:
            con = sqlite3.connect('users.db')
            cur = con.cursor()
            cur.execute('''INSERT INTO users 
                (name, username, password, email, phone, address, city, state, country, zip, role) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (name, username, hashed_pw, email, phone, address, city, state, country, zip_code, 'user'))
            con.commit()
            flash("Registered successfully! Please login.", "success")
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash("Username already exists!", "danger")
        finally:
            con.close()

    return render_template("register.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("Logged out successfully.", "info")
    return redirect(url_for('login'))

@app.route("/aboutus")
def aboutus():
    return render_template('aboutus.html')

if __name__ == "__main__":
    app.run(debug=True)
