from flask import Flask, render_template, request, redirect, url_for, flash
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from flask_mail import Mail, Message
import random
import string

app = Flask(__name__)
app.config['SECRET_KEY'] = 'dev-secret-key-change-this-for-production'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Mail configuration - Use Gmail App Password
# 1. Enable 2-step verification on your Gmail
# 2. Go to App Passwords and create one for 'Mail'
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'your-email@gmail.com'  # Replace with actual email
app.config['MAIL_PASSWORD'] = 'your-app-password'      # Replace with actual app password
app.config['MAIL_DEFAULT_SENDER'] = ('Auth System', 'your-email@gmail.com')

db = SQLAlchemy(app)
mail = Mail(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Ensure upload directory exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# User Model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    otp = db.Column(db.String(6), nullable=True)

with app.app_context():
    db.create_all()

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Load the trained model
MODEL_PATH = 'models/model.h5'
model = None
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)

classes = ['Active', 'Focused', 'Relaxed', 'Stressed']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('home'))
        flash('Invalid username or password')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists')
            return redirect(url_for('signup'))
        
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        new_user = User(username=username, email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful! Please login.')
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form.get('email')
        user = User.query.filter_by(email=email).first()
        if user:
            otp = ''.join(random.choices(string.digits, k=6))
            user.otp = otp
            db.session.commit()
            
            msg = Message('Password Reset OTP', recipients=[email])
            msg.body = f'Your OTP for password reset is: {otp}'
            try:
                mail.send(msg)
                flash('OTP sent to your email')
                return redirect(url_for('verify_otp', email=email))
            except Exception as e:
                flash(f'Error sending email: {str(e)}')
        else:
            flash('Email not found')
    return render_template('forgot_password.html')

@app.route('/verify_otp', methods=['GET', 'POST'])
def verify_otp():
    email = request.args.get('email')
    if request.method == 'POST':
        user_otp = request.form.get('otp')
        user = User.query.filter_by(email=email, otp=user_otp).first()
        if user:
            return redirect(url_for('reset_password', email=email))
        flash('Invalid OTP')
    return render_template('verify_otp.html', email=email)

@app.route('/reset_password', methods=['GET', 'POST'])
def reset_password():
    email = request.args.get('email')
    if request.method == 'POST':
        new_password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        if user:
            user.password = generate_password_hash(new_password, method='pbkdf2:sha256')
            user.otp = None
            db.session.commit()
            flash('Password reset successful! Please login.')
            return redirect(url_for('login'))
    return render_template('reset_password.html', email=email)

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Preprocess image
            img = image.load_img(filepath, target_size=(128, 128))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0

            # Predict
            if model:
                prediction = model.predict(img_array)
                result_idx = np.argmax(prediction[0])
                confidence = np.max(prediction[0]) * 100
                result = classes[result_idx]
            else:
                result = "Model not found. Please train the model first."
                confidence = 0

            return render_template('predict.html', result=result, confidence=f"{confidence:.2f}%", image_url=filepath)
    
    return render_template('predict.html')

@app.route('/dashboard')
@login_required
def dashboard():
    # Accuracy and Loss graphs should be in static/graphs/
    accuracy_graph = 'graphs/accuracy_plot.png'
    loss_graph = 'graphs/loss_plot.png'
    
    # Dataset details
    dataset_details = {
        "Total Samples": 20000,
        "Classes": classes,
        "Train/Val Split": "80/20",
        "Target Accuracy": "93%"
    }
    
    return render_template('dashboard.html', 
                           accuracy_graph=accuracy_graph, 
                           loss_graph=loss_graph,
                           dataset_details=dataset_details)

if __name__ == '__main__':
    app.run(debug=True)
