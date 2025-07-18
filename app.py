# --- START OF FILE app.py ---

import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import json
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI
from dotenv import load_dotenv
import hashlib # Added for password hashing

# Imports for Brain Tumor Detection Module
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore') # Suppress TensorFlow warnings


>>>>>>> b11d9b7f112ada933af127f161ee2a0dc30a9d51

# Set page configuration
st.set_page_config(
    page_title="🏥 Health Assistant",
    layout="wide",
    page_icon="🧑‍⚕️",
    initial_sidebar_state="expanded"
)

# --- USER AUTHENTICATION & STORAGE ---
# Define the path for the users file
USERS_FILE = 'users.json'

def load_users():
    """Loads users from the JSON file. Returns an empty dict if file not found."""
    try:
        with open(USERS_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_users(users):
    """Saves the users dictionary to the JSON file."""
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=4)

def hash_password(password):
    """Hashes a password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()

def check_password(username, password):
    """Checks if the provided password is correct for the user."""
    users = load_users()
    if username in users:
        hashed_password = hash_password(password)
        return users[username] == hashed_password
    return False

def add_user(username, password):
    """Adds a new user to the database. Returns False if user already exists."""
    users = load_users()
    if username in users:
        return False  # Username already exists
    users[username] = hash_password(password)
<<<<<<< HEAD
    save_users(users) # Save changes after adding user
    return True # User added successfully
=======
    save_users(users)
    return True
>>>>>>> b11d9b7f112ada933af127f161ee2a0dc30a9d51

# --- LOGIN/REGISTER SYSTEM ---
# Initialize session states
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "page" not in st.session_state:
    st.session_state.page = "Login"

# Login form function
def login():
    st.markdown("## 🔐 Login to Health Assistant")
<<<<<<< HEAD
    username = st.text_input("👤 Username", key="login_username")
    password = st.text_input("🔑 Password", type="password", key="login_password")

    col1, col2 = st.columns([1, 2.5])
    with col1:
        if st.button("🔓 Login", use_container_width=True, key="login_button"):
=======
    username = st.text_input("👤 Username")
    password = st.text_input("🔑 Password", type="password")

    col1, col2 = st.columns([1, 2.5])
    with col1:
        if st.button("🔓 Login", use_container_width=True):
>>>>>>> b11d9b7f112ada933af127f161ee2a0dc30a9d51
            if check_password(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success("✅ Logged in successfully!")
                st.rerun()
            else:
                st.error("❌ Invalid username or password.")
    with col2:
<<<<<<< HEAD
        if st.button("Don't have an account? Register", use_container_width=True, key="register_button_from_login"):
=======
        if st.button("Don't have an account? Register", use_container_width=True):
>>>>>>> b11d9b7f112ada933af127f161ee2a0dc30a9d51
            st.session_state.page = "Register"
            st.rerun()

# Registration form function
def register():
    st.markdown("## 📝 Register New Account")
<<<<<<< HEAD
    new_username = st.text_input("👤 Choose a Username", key="register_username")
    new_password = st.text_input("🔑 Choose a Password", type="password", key="register_password")
    confirm_password = st.text_input("🔑 Confirm Password", type="password", key="confirm_password")

    col1, col2 = st.columns([1, 2.5])
    with col1:
        if st.button("✍️ Register", use_container_width=True, key="register_submit_button"):
=======
    new_username = st.text_input("👤 Choose a Username")
    new_password = st.text_input("🔑 Choose a Password", type="password")
    confirm_password = st.text_input("🔑 Confirm Password", type="password")

    col1, col2 = st.columns([1, 2.5])
    with col1:
        if st.button("✍️ Register", use_container_width=True):
>>>>>>> b11d9b7f112ada933af127f161ee2a0dc30a9d51
            if not new_username or not new_password or not confirm_password:
                st.warning("Please fill out all fields.")
            elif new_password != confirm_password:
                st.error("Passwords do not match.")
            else:
                if add_user(new_username, new_password):
                    st.success("✅ Account created successfully! Please log in.")
                    st.session_state.page = "Login"
                    st.rerun()
                else:
                    st.error("❌ This username is already taken. Please choose another one.")
    with col2:
<<<<<<< HEAD
        if st.button("Already have an account? Login", use_container_width=True, key="login_button_from_register"):
=======
        if st.button("Already have an account? Login", use_container_width=True):
>>>>>>> b11d9b7f112ada933af127f161ee2a0dc30a9d51
            st.session_state.page = "Login"
            st.rerun()

# Show login/register page if not logged in
if not st.session_state.logged_in:
    # Place authentication forms in the center
    _, mid_col, _ = st.columns([1,2,1])
    with mid_col:
        if st.session_state.page == "Login":
            login()
        elif st.session_state.page == "Register":
            register()
    st.stop()
# --- END LOGIN/REGISTER SYSTEM ---


# Initialize session state for history
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# Initialize session state for chatbot history
if 'chatbot_history' not in st.session_state:
    st.session_state.chatbot_history = []

<<<<<<< HEAD
# Initialize session states for Brain Tumor Detection Module
if 'model' not in st.session_state:
    st.session_state.model = None
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'dataset_loaded' not in st.session_state:
    st.session_state.dataset_loaded = False
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'training_history' not in st.session_state:
    st.session_state.training_history = None

# Configure DeepSeek API using environment variables for better security
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

# Constants for Brain Tumor Detection
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
=======
# Configure DeepSeek API using environment variables for better security
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
>>>>>>> b11d9b7f112ada933af127f161ee2a0dc30a9d51

# Custom CSS for black theme styling
st.markdown("""
<style>
    /* Main theme colors - Black based */
    .stApp {
        background-color: #000000;
        color: #ffffff;
    }
    
    .main-header {
        font-size: 3rem;
        color: #ffffff;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(255,255,255,0.1);
        background: linear-gradient(45deg, #1a1a1a, #2d2d2d);
        padding: 2rem;
        border-radius: 15px;
        border: 2px solid #333333;
    }
    
    .prediction-container {
        background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 8px 32px rgba(255,255,255,0.1);
        border: 1px solid #333333;
    }
    
    .input-section {
        background: #1a1a1a;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #ffffff;
        border: 1px solid #333333;
    }
    
    .result-positive {
        background: linear-gradient(135deg, #cc0000, #990000);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        font-weight: bold;
        text-align: center;
        margin: 1rem 0;
        border: 2px solid #ff3333;
    }
    
    .result-negative {
        background: linear-gradient(135deg, #006600, #004d00);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        font-weight: bold;
        text-align: center;
        margin: 1rem 0;
        border: 2px solid #00cc00;
    }
    
    .info-card {
        background: #1a1a1a;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffffff;
        margin: 1rem 0;
        border: 1px solid #333333;
        color: #ffffff;
    }
    
    .history-card {
        background: #1a1a1a;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border: 1px solid #333333;
        color: #ffffff;
    }
    
    .metric-container {
        background: #1a1a1a;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #333333;
        color: #ffffff;
    }
    
    /* Override Streamlit's default colors */
    .stSelectbox > div > div {
        background-color: #1a1a1a;
        color: #ffffff;
        border: 1px solid #333333;
    }
    
<<<<<<< HEAD
    .stNumberInput > div > div > input, .stTextInput > div > div > input, .stFileUploader > div > div > button, .stFileUploader > div > div > div > p {
=======
    .stNumberInput > div > div > input, .stTextInput > div > div > input {
>>>>>>> b11d9b7f112ada933af127f161ee2a0dc30a9d51
        background-color: #1a1a1a;
        color: #ffffff;
        border: 1px solid #333333;
    }
    .stFileUploader > div > div > div > p {
        color: #cccccc; /* text color for file uploader description */
    }

    .stButton > button {
        background-color: #1a1a1a;
        color: #ffffff;
        border: 2px solid #ffffff;
        border-radius: 10px;
        font-weight: bold;
    }
    
    .stButton > button:hover {
        background-color: #2d2d2d;
        border-color: #cccccc;
    }
    
    .stExpander {
        background-color: #1a1a1a;
        border: 1px solid #333333;
        border-radius: 10px;
    }
    
    .stDataFrame {
        background-color: #1a1a1a;
    }
    
    /* Sidebar styling */
    .css-1d391kg { /* Target sidebar background */
        background-color: #0d0d0d;
    }
    
    /* This might target specific elements, keep an eye on it */
    .css-17eq0hr { 
        background-color: #1a1a1a;
    }

    /* Custom styles for Brain Tumor Detection module */
    .sub-header-brain {
        font-size: 1.5rem;
        color: #ffffff; /* White to fit dark theme */
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #444444; /* Subtle line */
        padding-bottom: 0.5rem;
    }
    .brain-info-box {
        background-color: #1a1a1a; /* Dark background */
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffffff; /* White border */
        margin: 1rem 0;
        color: #ffffff; /* White text */
        border: 1px solid #333333;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🏥 AI Health Assistant - Advanced Edition</h1>', unsafe_allow_html=True)
st.markdown("---")

# getting the working directory of the main.py
working_dir = os.path.dirname(os.path.abspath(__file__))

# Define path for images
IMAGE_DIR = os.path.join(working_dir, 'images')
# Ensure the images directory exists (optional, good for first run setup)
os.makedirs(IMAGE_DIR, exist_ok=True)


# loading the saved models
try:
    diabetes_model = pickle.load(open(f'{working_dir}/saved_models/diabetes_model.sav', 'rb'))
    heart_disease_model = pickle.load(open(f'{working_dir}/saved_models/heart_disease_model.sav', 'rb'))
    parkinsons_model = pickle.load(open(f'{working_dir}/saved_models/parkinsons_model.sav', 'rb'))
except FileNotFoundError:
    st.error("⚠️ Model files not found. Please ensure the saved_models directory exists with the required .sav files.")
    st.stop()

# Function to add prediction to history
def add_to_history(patient_name, prediction_type, inputs, result, risk_level):
    history_entry = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'patient_name': patient_name,
        'type': prediction_type,
        'inputs': inputs,
        'result': result,
        'risk_level': risk_level
    }
    st.session_state.prediction_history.append(history_entry)

# Function to export history as JSON
def export_history_json():
    return json.dumps(st.session_state.prediction_history, indent=2)

<<<<<<< HEAD
=======
# --- CORRECTED FUNCTION ---
>>>>>>> b11d9b7f112ada933af127f161ee2a0dc30a9d51
# This version creates a "wide" CSV with a separate column for each input parameter.
def export_history_csv():
    if not st.session_state.prediction_history:
        return None
    
    df_data = []
    for entry in st.session_state.prediction_history:
        # Start with the base information for the row
        row = {
            'Timestamp': entry['timestamp'],
            'Patient Name': entry['patient_name'],
            'Prediction Type': entry['type'],
            'Result': entry['result'],
            'Risk Level': entry['risk_level']
        }
        # Add each input parameter as a new column, prefixed with 'Input_'
        for key, value in entry['inputs'].items():
            row[f'Input_{key}'] = value
        df_data.append(row)
    
    # Pandas will automatically create all necessary columns and fill missing values with NaN
    df = pd.DataFrame(df_data)
    return df.to_csv(index=False)

# Function to create trend analysis charts
def create_trend_charts():
    if not st.session_state.prediction_history:
        st.info("No prediction history available for trend analysis.")
        return
    
    df_data = []
    for entry in st.session_state.prediction_history:
        df_data.append({
            'timestamp': pd.to_datetime(entry['timestamp']),
            'patient_name': entry['patient_name'],
            'type': entry['type'],
            'risk_level': entry['risk_level'],
            'result': 1 if entry['risk_level'] == 'High' else 0 # Simple numerical representation for plotting
        })
    df = pd.DataFrame(df_data)
    
    # Filter by patient
    patient_list = ['All'] + sorted(df['patient_name'].unique().tolist())
<<<<<<< HEAD
    selected_patient = st.selectbox("Analyze Trends for a Specific Patient", patient_list, key="trend_patient_select")
=======
    selected_patient = st.selectbox("Analyze Trends for a Specific Patient", patient_list)
>>>>>>> b11d9b7f112ada933af127f161ee2a0dc30a9d51
    if selected_patient != 'All':
        df = df[df['patient_name'] == selected_patient]
    
    if df.empty:
        st.warning(f"No data for patient: {selected_patient}")
        return

    # Risk level distribution pie chart
    risk_counts = df['risk_level'].value_counts()
    fig_pie = px.pie(values=risk_counts.values, names=risk_counts.index, 
                     title=f"Risk Level Distribution for {selected_patient}",
                     color_discrete_map={'High': '#cc0000', 'Low': '#006600'})
    fig_pie.update_layout(
        plot_bgcolor='#000000', paper_bgcolor='#1a1a1a', font_color='#ffffff'
    )
    st.plotly_chart(fig_pie, use_container_width=True, key=f"trend_risk_pie_{selected_patient}") # Added unique key
    
    # Prediction types distribution
    type_counts = df['type'].value_counts()
    fig_bar = px.bar(x=type_counts.index, y=type_counts.values,
                     title=f"Predictions by Type for {selected_patient}",
                     color=type_counts.values, color_continuous_scale='Viridis')
    fig_bar.update_layout(
        plot_bgcolor='#000000', paper_bgcolor='#1a1a1a', font_color='#ffffff',
        xaxis=dict(gridcolor='#333333'), yaxis=dict(gridcolor='#333333')
    )
    st.plotly_chart(fig_bar, use_container_width=True, key=f"trend_type_bar_{selected_patient}") # Added unique key
    
    # Timeline of predictions
    if len(df) > 1:
        df_sorted = df.sort_values('timestamp')
        fig_timeline = px.scatter(df_sorted, x='timestamp', y='type', 
                                 color='risk_level', size_max=15,
                                 title=f"Prediction Timeline for {selected_patient}",
                                 color_discrete_map={'High': '#cc0000', 'Low': '#006600'})
        fig_timeline.update_layout(
            plot_bgcolor='#000000', paper_bgcolor='#1a1a1a', font_color='#ffffff',
            xaxis=dict(gridcolor='#333333'), yaxis=dict(gridcolor='#333333')
        )
        st.plotly_chart(fig_timeline, use_container_width=True, key=f"trend_timeline_{selected_patient}") # Added unique key

# Helper functions for Brain Tumor Detection
def preprocess_image(image, target_size=(IMG_SIZE, IMG_SIZE)):
    """Preprocess image for model prediction"""
    # If image is a path string, load it. Otherwise, assume it's a PIL Image object
    if isinstance(image, str):
        image = load_img(image, target_size=target_size)
    else: # Assume it's a PIL.Image.Image object
        image = image.resize(target_size)
    
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0) # Add batch dimension
    image_array = image_array / 255.0 # Normalize to [0, 1]
    return image_array

def create_cnn_model():
    """Create CNN model architecture"""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Conv2D(256, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')  # Binary classification: Tumor vs No Tumor
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def generate_sample_data_for_demo():
    """Generate sample data for demonstration when actual dataset is not available"""
    np.random.seed(42)
    n_samples = 200
    
    # Generate sample images (random noise with some patterns)
    X_data = np.random.rand(n_samples, IMG_SIZE, IMG_SIZE, 3)
    
    # Add some pattern to simulate brain scans
    for i in range(n_samples):
        # Add circular pattern to simulate brain structure
        center_x, center_y = IMG_SIZE // 2, IMG_SIZE // 2
        y_coords, x_coords = np.ogrid[:IMG_SIZE, :IMG_SIZE]
        mask = (x_coords - center_x)**2 + (y_coords - center_y)**2 <= (IMG_SIZE // 3)**2
        X_data[i][mask] = X_data[i][mask] * 0.5 + 0.3 # Make central part slightly different
    
    # Generate labels (50% tumor, 50% no tumor)
    y_data = np.random.choice([0, 1], n_samples)
    y_data = to_categorical(y_data, 2)
    
    return X_data, y_data

def load_and_preprocess_dataset():
    """Load and preprocess the brain tumor dataset"""
    
    # Check if actual dataset exists
    dataset_path = os.path.join(working_dir, "brain_tumor_dataset")
    X_loaded, y_loaded = None, None

    if os.path.exists(dataset_path):
        st.info("✅ Attempting to load real brain tumor images from `brain_tumor_dataset` folder...")
        
        images = []
        labels = []
        
        # Load tumor images
        tumor_path = os.path.join(dataset_path, "yes")
        if os.path.exists(tumor_path):
            for img_name in os.listdir(tumor_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(tumor_path, img_name)
                    try:
                        img = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
                        img_array = img_to_array(img) / 255.0
                        images.append(img_array)
                        labels.append(1)  # Tumor
                    except Exception as e:
                        st.warning(f"Could not load image {img_name}: {e}")
        
        # Load no tumor images
        no_tumor_path = os.path.join(dataset_path, "no")
        if os.path.exists(no_tumor_path):
            for img_name in os.listdir(no_tumor_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(no_tumor_path, img_name)
                    try:
                        img = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
                        img_array = img_to_array(img) / 255.0
                        images.append(img_array)
                        labels.append(0)  # No tumor
                    except Exception as e:
                        st.warning(f"Could not load image {img_name}: {e}")
        
        if images:
            X_loaded = np.array(images)
            y_loaded = to_categorical(labels, 2)
            st.success(f"✅ Loaded {len(X_loaded)} real brain MRI images!")
        else:
            st.warning("⚠️ No valid images found in the `brain_tumor_dataset` folder. Using simulated data.")
            X_loaded, y_loaded = generate_sample_data_for_demo()
            st.info(f"Generated {len(X_loaded)} sample images for demonstration.")
        
    else:
        st.warning("⚠️ `brain_tumor_dataset` folder not found. Using simulated data for demonstration.")
        X_loaded, y_loaded = generate_sample_data_for_demo()
        st.info(f"Generated {len(X_loaded)} sample images for demonstration.")
    
    return X_loaded, y_loaded


# sidebar for navigation
with st.sidebar:
    st.markdown(f"### Welcome, {st.session_state.get('username', 'Guest')}! 👋")
<<<<<<< HEAD
    if st.button("🚪 Logout", key="sidebar_logout_btn"):
=======
    if st.button("🚪 Logout"):
>>>>>>> b11d9b7f112ada933af127f161ee2a0dc30a9d51
        st.session_state.logged_in = False
        if 'username' in st.session_state: del st.session_state.username
        st.session_state.page = "Login"
        st.rerun()

    st.markdown("### 🩺 Navigation")
    selected = option_menu(
        'Health Assistant',
        ['Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Prediction', 'Brain Tumor Prediction', 
         'Prediction History', 'Trend Analysis', 'Export Data', 'Health Chatbot'],
<<<<<<< HEAD
        icons=['🩸', '❤️', '🧠', '🖼️', '📋', '📊', '💾', '🤖'], # Changed Brain Tumor icon
=======
        icons=['🩸', '❤️', '🧠', '📋', '📊', '💾', '🤖'],
>>>>>>> b11d9b7f112ada933af127f161ee2a0dc30a9d51
        menu_icon='hospital-fill', default_index=0,
        styles={
            "container": {"padding": "10px!important", "background": "linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%)", "border-radius": "15px", "box-shadow": "0 8px 32px rgba(255,255,255,0.1)", "border": "2px solid #333333"},
            "icon": {"color": "#ffffff", "font-size": "28px", "text-shadow": "2px 2px 4px rgba(0,0,0,0.3)"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "5px 0px", "padding": "12px 16px", "border-radius": "10px", "color": "#ffffff", "font-weight": "500", "transition": "all 0.3s ease", "--hover-color": "rgba(255,255,255,0.2)", "background": "rgba(255,255,255,0.1)", "backdrop-filter": "blur(10px)", "border": "1px solid #333333"},
            "nav-link-selected": {"background": "linear-gradient(135deg, #333333, #4d4d4d)", "color": "#ffffff", "font-weight": "bold", "box-shadow": "0 4px 15px rgba(255,255,255,0.2)", "transform": "translateY(-2px)", "border": "2px solid #ffffff"},
            "menu-title": {"color": "#ffffff", "font-weight": "bold", "text-align": "center", "font-size": "18px", "text-shadow": "2px 2px 4px rgba(0,0,0,0.3)", "margin-bottom": "20px"}
        }
    )
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.info("""Enhanced AI health assistant with disease risk prediction, history tracking, trend analysis, and a health chatbot.""")
    
    if st.session_state.prediction_history:
        st.markdown("### 📊 Quick Stats")
        total_predictions = len(st.session_state.prediction_history)
        high_risk_count = sum(1 for entry in st.session_state.prediction_history if entry['risk_level'] == 'High')
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f'<div class="metric-container"><h3>{total_predictions}</h3><p>Total Tests</p></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-container"><h3>{high_risk_count}</h3><p>High Risk</p></div>', unsafe_allow_html=True)

# Diabetes Prediction Page
if selected == 'Diabetes Prediction':
    st.markdown("## 🩸 Diabetes Prediction using Machine Learning")
    with st.expander("📋 About Diabetes Prediction"):
        st.markdown("This prediction model analyzes various health parameters to assess diabetes risk.")

    st.markdown("### 📝 Enter Patient and Health Parameters")
    patient_name = st.text_input("👤 **Patient Name**", placeholder="e.g., Jane Doe", key="diabetes_patient_name")
    
    col1, col2, col3 = st.columns(3)
    with col1:
<<<<<<< HEAD
        Pregnancies = st.number_input('🤰 Number of Pregnancies', min_value=0, max_value=20, value=0, help="Total number of pregnancies", key="dp_pregnancies")
        SkinThickness = st.number_input('📏 Skin Thickness (mm)', min_value=0.0, max_value=100.0, value=20.0, help="Triceps skin fold thickness", key="dp_skinthickness")
        DiabetesPedigreeFunction = st.number_input('🧬 Diabetes Pedigree Function', min_value=0.0, max_value=3.0, value=0.5, help="Family history factor", key="dp_dpf")
    with col2:
        Glucose = st.number_input('🍭 Glucose Level (mg/dL)', min_value=0.0, max_value=300.0, value=120.0, help="Plasma glucose concentration", key="dp_glucose")
        Insulin = st.number_input('💉 Insulin Level (μU/mL)', min_value=0.0, max_value=900.0, value=80.0, help="2-Hour serum insulin", key="dp_insulin")
=======
        Pregnancies = st.number_input('🤰 Number of Pregnancies', min_value=0, max_value=20, value=0, help="Total number of pregnancies")
        SkinThickness = st.number_input('📏 Skin Thickness (mm)', min_value=0.0, max_value=100.0, value=20.0, help="Triceps skin fold thickness")
        DiabetesPedigreeFunction = st.number_input('🧬 Diabetes Pedigree Function', min_value=0.0, max_value=3.0, value=0.5, help="Family history factor")
    with col2:
        Glucose = st.number_input('🍭 Glucose Level (mg/dL)', min_value=0.0, max_value=300.0, value=120.0, help="Plasma glucose concentration")
        Insulin = st.number_input('💉 Insulin Level (μU/mL)', min_value=0.0, max_value=900.0, value=80.0, help="2-Hour serum insulin")
>>>>>>> b11d9b7f112ada933af127f161ee2a0dc30a9d51
    with col3:
        BloodPressure = st.number_input('💓 Blood Pressure (mmHg)', min_value=0.0, max_value=200.0, value=80.0, help="Diastolic blood pressure", key="dp_bloodpressure")
        BMI = st.number_input('⚖️ BMI (kg/m²)', min_value=0.0, max_value=70.0, value=25.0, help="Body mass index", key="dp_bmi")
        Age = st.number_input('🎂 Age (years)', min_value=1, max_value=120, value=30, help="Age in years", key="dp_age")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
<<<<<<< HEAD
        if st.button('🔬 Run Diabetes Test', type="primary", use_container_width=True, key="run_diabetes_test"):
=======
        if st.button('🔬 Run Diabetes Test', type="primary", use_container_width=True):
>>>>>>> b11d9b7f112ada933af127f161ee2a0dc30a9d51
            if not patient_name:
                st.warning("⚠️ Please enter a patient name before running the test.")
            else:
                try:
                    user_input = [float(Pregnancies), float(Glucose), float(BloodPressure), 
                                 float(SkinThickness), float(Insulin), float(BMI), 
                                 float(DiabetesPedigreeFunction), float(Age)]
                    diab_prediction = diabetes_model.predict([user_input])
                    inputs = {'Pregnancies': Pregnancies, 'Glucose': Glucose, 'BloodPressure': BloodPressure, 'SkinThickness': SkinThickness, 'Insulin': Insulin, 'BMI': BMI, 'DiabetesPedigreeFunction': DiabetesPedigreeFunction, 'Age': Age}

                    if diab_prediction[0] == 1:
                        st.markdown('<div class="result-positive">⚠️ High Risk: The model suggests increased diabetes risk</div>', unsafe_allow_html=True)
                        st.warning("Please consult with a healthcare professional for proper evaluation.")
                        add_to_history(patient_name, 'Diabetes Prediction', inputs, 'Positive', 'High')
                    else:
                        st.markdown('<div class="result-negative">✅ Low Risk: The model suggests lower diabetes risk</div>', unsafe_allow_html=True)
                        st.success("Continue maintaining a healthy lifestyle!")
                        add_to_history(patient_name, 'Diabetes Prediction', inputs, 'Negative', 'Low')
                except ValueError:
                    st.error("❌ Please ensure all fields are filled with valid numbers.")

# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':
    st.markdown("## ❤️ Heart Disease Prediction using Machine Learning")
    with st.expander("📋 About Heart Disease Prediction"):
        st.markdown("This model analyzes cardiovascular parameters to assess heart disease risk.")

    st.markdown("### 📝 Enter Patient and Cardiovascular Parameters")
    patient_name = st.text_input("👤 **Patient Name**", placeholder="e.g., John Smith", key="heart_patient_name")

    col1, col2, col3 = st.columns(3)
    with col1:
<<<<<<< HEAD
        age = st.number_input('🎂 Age (years)', min_value=1, max_value=120, value=50, key="hd_age")
        trestbps = st.number_input('💓 Resting Blood Pressure (mmHg)', min_value=50.0, max_value=250.0, value=120.0, key="hd_trestbps")
        restecg = st.selectbox('📊 Resting ECG Results', options=[0, 1, 2], format_func=lambda x: {0: "Normal", 1: "ST-T Abnormality", 2: "LV Hypertrophy"}[x], key="hd_restecg")
        oldpeak = st.number_input('📈 ST Depression', min_value=0.0, max_value=10.0, value=1.0, step=0.1, key="hd_oldpeak")
    with col2:
        sex = st.selectbox('👤 Sex', options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male", key="hd_sex")
        chol = st.number_input('🧪 Serum Cholesterol (mg/dl)', min_value=100.0, max_value=600.0, value=200.0, key="hd_chol")
        thalach = st.number_input('💨 Max Heart Rate Achieved', min_value=50.0, max_value=250.0, value=150.0, key="hd_thalach")
        slope = st.selectbox('📊 ST Segment Slope', options=[0, 1, 2], format_func=lambda x: {0: "Upsloping", 1: "Flat", 2: "Downsloping"}[x], key="hd_slope")
    with col3:
        cp = st.selectbox('🫀 Chest Pain Type', options=[0, 1, 2, 3], format_func=lambda x: {0: "Typical Angina", 1: "Atypical Angina", 2: "Non-anginal", 3: "Asymptomatic"}[x], key="hd_cp")
        fbs = st.selectbox('🍭 Fasting Blood Sugar > 120 mg/dl', options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key="hd_fbs")
        exang = st.selectbox('🏃 Exercise Induced Angina', options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key="hd_exang")
        ca = st.selectbox('🔬 Major Vessels (0-3)', options=[0, 1, 2, 3], key="hd_ca")
    thal = st.selectbox('🫀 Thalassemia', options=[0, 1, 2], format_func=lambda x: {0: "Normal", 1: "Fixed Defect", 2: "Reversible Defect"}[x], key="hd_thal")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button('🔬 Run Heart Disease Test', type="primary", use_container_width=True, key="run_heart_disease_test"):
=======
        age = st.number_input('🎂 Age (years)', min_value=1, max_value=120, value=50)
        trestbps = st.number_input('💓 Resting Blood Pressure (mmHg)', min_value=50.0, max_value=250.0, value=120.0)
        restecg = st.selectbox('📊 Resting ECG Results', options=[0, 1, 2], format_func=lambda x: {0: "Normal", 1: "ST-T Abnormality", 2: "LV Hypertrophy"}[x])
        oldpeak = st.number_input('📈 ST Depression', min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    with col2:
        sex = st.selectbox('👤 Sex', options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        chol = st.number_input('🧪 Serum Cholesterol (mg/dl)', min_value=100.0, max_value=600.0, value=200.0)
        thalach = st.number_input('💨 Max Heart Rate Achieved', min_value=50.0, max_value=250.0, value=150.0)
        slope = st.selectbox('📊 ST Segment Slope', options=[0, 1, 2], format_func=lambda x: {0: "Upsloping", 1: "Flat", 2: "Downsloping"}[x])
    with col3:
        cp = st.selectbox('🫀 Chest Pain Type', options=[0, 1, 2, 3], format_func=lambda x: {0: "Typical Angina", 1: "Atypical Angina", 2: "Non-anginal", 3: "Asymptomatic"}[x])
        fbs = st.selectbox('🍭 Fasting Blood Sugar > 120 mg/dl', options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        exang = st.selectbox('🏃 Exercise Induced Angina', options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        ca = st.selectbox('🔬 Major Vessels (0-3)', options=[0, 1, 2, 3])
    thal = st.selectbox('🫀 Thalassemia', options=[0, 1, 2], format_func=lambda x: {0: "Normal", 1: "Fixed Defect", 2: "Reversible Defect"}[x])

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button('🔬 Run Heart Disease Test', type="primary", use_container_width=True):
>>>>>>> b11d9b7f112ada933af127f161ee2a0dc30a9d51
            if not patient_name:
                st.warning("⚠️ Please enter a patient name before running the test.")
            else:
                try:
                    user_input = [float(age), float(sex), float(cp), float(trestbps), float(chol), float(fbs), float(restecg), float(thalach), float(exang), float(oldpeak), float(slope), float(ca), float(thal)]
                    heart_prediction = heart_disease_model.predict([user_input])
                    inputs = {'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol, 'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang, 'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal}

                    if heart_prediction[0] == 1:
                        st.markdown('<div class="result-positive">⚠️ High Risk: The model suggests increased heart disease risk</div>', unsafe_allow_html=True)
                        st.warning("Please consult with a cardiologist for proper evaluation.")
                        add_to_history(patient_name, 'Heart Disease Prediction', inputs, 'Positive', 'High')
<<<<<<< HEAD
                        
                        st.markdown("---")
                        st.subheader("Visualizing Potential Heart Condition")
                        image_path_high_risk = os.path.join(IMAGE_DIR, 'blocked_heart.png') 
                        if os.path.exists(image_path_high_risk):
                            st.image(image_path_high_risk, caption="Image: Potential indication of blockages", use_column_width=True)
                        else:
                            st.info("Visual cue for high risk: 💔 (Please place 'blocked_heart.png' in the 'images' folder for a visual representation.)")
                        
=======
>>>>>>> b11d9b7f112ada933af127f161ee2a0dc30a9d51
                    else:
                        st.markdown('<div class="result-negative">✅ Low Risk: The model suggests lower heart disease risk</div>', unsafe_allow_html=True)
                        st.success("Keep up the healthy lifestyle!")
                        add_to_history(patient_name, 'Heart Disease Prediction', inputs, 'Negative', 'Low')
<<<<<<< HEAD

                        st.markdown("---")
                        st.subheader("Visualizing Healthy Heart Status")
                        image_path_low_risk = os.path.join(IMAGE_DIR, 'healthy_heart.png') 
                        if os.path.exists(image_path_low_risk):
                            st.image(image_path_low_risk, caption="Image: Indicating a healthy cardiovascular system", use_column_width=True)
                        else:
                            st.info("Visual cue for low risk: ❤️‍🩹 (Please place 'healthy_heart.png' in the 'images' folder for a visual representation.)")

=======
>>>>>>> b11d9b7f112ada933af127f161ee2a0dc30a9d51
                except ValueError:
                    st.error("❌ Please ensure all fields are filled with valid values.")

# Parkinson's Prediction Page
if selected == "Parkinsons Prediction":
    st.markdown("## 🧠 Parkinson's Disease Prediction using Machine Learning")
    with st.expander("📋 About Parkinson's Disease Prediction"):
        st.markdown("This model analyzes voice measurements to assess Parkinson's disease risk.")

    st.markdown("### 📝 Enter Patient and Voice Analysis Parameters")
    patient_name = st.text_input("👤 **Patient Name**", placeholder="e.g., Robert Paulson", key="parkinsons_patient_name")
    st.info("🎤 These parameters are typically obtained from voice recording analysis.")

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
<<<<<<< HEAD
        fo = st.number_input('🎵 MDVP:Fo(Hz)', min_value=50.0, max_value=300.0, value=150.0, help="Average vocal fundamental frequency", key="pk_fo")
        RAP = st.number_input('📊 MDVP:RAP', min_value=0.0, max_value=1.0, value=0.01, step=0.001, format="%.4f", key="pk_rap")
        Shimmer = st.number_input('🌊 MDVP:Shimmer', min_value=0.0, max_value=1.0, value=0.03, step=0.001, format="%.4f", key="pk_shimmer")
        APQ = st.number_input('📈 MDVP:APQ', min_value=0.0, max_value=1.0, value=0.02, step=0.001, format="%.4f", key="pk_apq")
        RPDE = st.number_input('🔢 RPDE', min_value=0.0, max_value=1.0, value=0.5, step=0.01, key="pk_rpde")
    with col2:
        fhi = st.number_input('🎵 MDVP:Fhi(Hz)', min_value=50.0, max_value=500.0, value=200.0, key="pk_fhi")
        PPQ = st.number_input('📊 MDVP:PPQ', min_value=0.0, max_value=1.0, value=0.01, step=0.001, format="%.4f", key="pk_ppq")
        Shimmer_dB = st.number_input('🌊 MDVP:Shimmer(dB)', min_value=0.0, max_value=2.0, value=0.3, step=0.01, key="pk_shimmerdb")
        DDA = st.number_input('📈 Shimmer:DDA', min_value=0.0, max_value=1.0, value=0.05, step=0.001, format="%.4f", key="pk_dda")
        DFA = st.number_input('🔢 DFA', min_value=0.0, max_value=1.0, value=0.7, step=0.01, key="pk_dfa")
    with col3:
        flo = st.number_input('🎵 MDVP:Flo(Hz)', min_value=50.0, max_value=300.0, value=100.0, key="pk_flo")
        DDP = st.number_input('📊 Jitter:DDP', min_value=0.0, max_value=1.0, value=0.03, step=0.001, format="%.4f", key="pk_ddp")
        APQ3 = st.number_input('🌊 Shimmer:APQ3', min_value=0.0, max_value=1.0, value=0.02, step=0.001, format="%.4f", key="pk_apq3")
        NHR = st.number_input('📈 NHR', min_value=0.0, max_value=1.0, value=0.02, step=0.001, format="%.4f", key="pk_nhr")
        spread1 = st.number_input('📊 spread1', min_value=-10.0, max_value=0.0, value=-5.0, step=0.1, key="pk_spread1")
    with col4:
        Jitter_percent = st.number_input('📊 MDVP:Jitter(%)', min_value=0.0, max_value=10.0, value=0.5, step=0.01, key="pk_jitter_percent")
        APQ5 = st.number_input('🌊 Shimmer:APQ5', min_value=0.0, max_value=1.0, value=0.02, step=0.001, format="%.4f", key="pk_apq5")
        HNR = st.number_input('📈 HNR', min_value=0.0, max_value=50.0, value=20.0, step=0.1, key="pk_hnr")
        spread2 = st.number_input('📊 spread2', min_value=0.0, max_value=1.0, value=0.2, step=0.01, key="pk_spread2")
=======
        fo = st.number_input('🎵 MDVP:Fo(Hz)', min_value=50.0, max_value=300.0, value=150.0, help="Average vocal fundamental frequency")
        RAP = st.number_input('📊 MDVP:RAP', min_value=0.0, max_value=1.0, value=0.01, step=0.001, format="%.4f")
        Shimmer = st.number_input('🌊 MDVP:Shimmer', min_value=0.0, max_value=1.0, value=0.03, step=0.001, format="%.4f")
        APQ = st.number_input('📈 MDVP:APQ', min_value=0.0, max_value=1.0, value=0.02, step=0.001, format="%.4f")
        RPDE = st.number_input('🔢 RPDE', min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    with col2:
        fhi = st.number_input('🎵 MDVP:Fhi(Hz)', min_value=50.0, max_value=500.0, value=200.0)
        PPQ = st.number_input('📊 MDVP:PPQ', min_value=0.0, max_value=1.0, value=0.01, step=0.001, format="%.4f")
        Shimmer_dB = st.number_input('🌊 MDVP:Shimmer(dB)', min_value=0.0, max_value=2.0, value=0.3, step=0.01)
        DDA = st.number_input('📈 Shimmer:DDA', min_value=0.0, max_value=1.0, value=0.05, step=0.001, format="%.4f")
        DFA = st.number_input('🔢 DFA', min_value=0.0, max_value=1.0, value=0.7, step=0.01)
    with col3:
        flo = st.number_input('🎵 MDVP:Flo(Hz)', min_value=50.0, max_value=300.0, value=100.0)
        DDP = st.number_input('📊 Jitter:DDP', min_value=0.0, max_value=1.0, value=0.03, step=0.001, format="%.4f")
        APQ3 = st.number_input('🌊 Shimmer:APQ3', min_value=0.0, max_value=1.0, value=0.02, step=0.001, format="%.4f")
        NHR = st.number_input('📈 NHR', min_value=0.0, max_value=1.0, value=0.02, step=0.001, format="%.4f")
        spread1 = st.number_input('📊 spread1', min_value=-10.0, max_value=0.0, value=-5.0, step=0.1)
    with col4:
        Jitter_percent = st.number_input('📊 MDVP:Jitter(%)', min_value=0.0, max_value=10.0, value=0.5, step=0.01)
        APQ5 = st.number_input('🌊 Shimmer:APQ5', min_value=0.0, max_value=1.0, value=0.02, step=0.001, format="%.4f")
        HNR = st.number_input('📈 HNR', min_value=0.0, max_value=50.0, value=20.0, step=0.1)
        spread2 = st.number_input('📊 spread2', min_value=0.0, max_value=1.0, value=0.2, step=0.01)
>>>>>>> b11d9b7f112ada933af127f161ee2a0dc30a9d51
    with col5:
        Jitter_Abs = st.number_input('📊 MDVP:Jitter(Abs)', min_value=0.0, max_value=1.0, value=0.0001, step=0.0001, format="%.6f", key="pk_jitter_abs")
        D2 = st.number_input('🔢 D2', min_value=0.0, max_value=5.0, value=2.0, step=0.1, key="pk_d2")
        PPE = st.number_input('🔢 PPE', min_value=0.0, max_value=1.0, value=0.2, step=0.01, key="pk_ppe")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
<<<<<<< HEAD
        if st.button("🔬 Run Parkinson's Test", type="primary", use_container_width=True, key="run_parkinsons_test"):
=======
        if st.button("🔬 Run Parkinson's Test", type="primary", use_container_width=True):
>>>>>>> b11d9b7f112ada933af127f161ee2a0dc30a9d51
            if not patient_name:
                st.warning("⚠️ Please enter a patient name before running the test.")
            else:
                try:
                    user_input = [float(fo), float(fhi), float(flo), float(Jitter_percent), float(Jitter_Abs), float(RAP), float(PPQ), float(DDP), float(Shimmer), float(Shimmer_dB), float(APQ3), float(APQ5), float(APQ), float(DDA), float(NHR), float(HNR), float(RPDE), float(DFA), float(spread1), float(spread2), float(D2), float(PPE)]
                    parkinsons_prediction = parkinsons_model.predict([user_input])
                    inputs = {'fo': fo, 'fhi': fhi, 'flo': flo, 'Jitter_percent': Jitter_percent, 'Jitter_Abs': Jitter_Abs, 'RAP': RAP, 'PPQ': PPQ, 'DDP': DDP, 'Shimmer': Shimmer, 'Shimmer_dB': Shimmer_dB, 'APQ3': APQ3, 'APQ5': APQ5, 'APQ': APQ, 'DDA': DDA, 'NHR': NHR, 'HNR': HNR, 'RPDE': RPDE, 'DFA': DFA, 'spread1': spread1, 'spread2': spread2, 'D2': D2, 'PPE': PPE}

                    if parkinsons_prediction[0] == 1:
                        st.markdown('<div class="result-positive">⚠️ High Risk: The model suggests increased Parkinson\'s disease risk</div>', unsafe_allow_html=True)
                        st.warning("Please consult with a neurologist for proper evaluation.")
                        add_to_history(patient_name, 'Parkinsons Prediction', inputs, 'Positive', 'High')
                    else:
                        st.markdown('<div class="result-negative">✅ Low Risk: The model suggests lower Parkinson\'s disease risk</div>', unsafe_allow_html=True)
                        st.success("Regular health monitoring is still recommended.")
                        add_to_history(patient_name, 'Parkinsons Prediction', inputs, 'Negative', 'Low')
                except ValueError:
                    st.error("❌ Please ensure all fields are filled with valid numbers.")
<<<<<<< HEAD

# Brain Tumor Prediction Page
if selected == "Brain Tumor Prediction":
    st.markdown('<h2 class="sub-header-brain">🧠 Brain Tumor Detection System</h2>', unsafe_allow_html=True)
    st.markdown("""
        This module provides an AI-powered system for detecting brain tumors from MRI images.
        You can load and preview the dataset, train a Convolutional Neural Network (CNN) model,
        make predictions on new images, and analyze the model's performance.
        """)

    tab_dataset, tab_training, tab_prediction, tab_analytics = st.tabs(["Dataset Info", "Model Training", "Prediction", "Analytics"])

    with tab_dataset:
        st.markdown('<h3 class="sub-header-brain">Dataset Information</h3>', unsafe_allow_html=True)
        st.info("📝 **Dataset Instructions:**\n\n"
        "To use this application with the actual Kaggle dataset:\n\n"
        "1.  **Download the dataset** from: https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection\n"
        "2.  **Extract the zip file** to a folder named `brain_tumor_dataset`\n"
        "3.  **Organize the folder structure** as:\n"
        "    ```\n"
        "    brain_tumor_dataset/\n"
        "    ├── yes/  (tumor images)\n"
        "    └── no/   (no tumor images)\n"
        "    ```\n"
        "4.  **Place the folder** in the same directory as this script\n\n"
        "If the actual dataset is not found, simulated data will be used for demonstration purposes.")
        
        if st.button("Load Dataset for Preview", key="load_dataset_btn_bt"):
            with st.spinner("Loading dataset..."):
                X_data, y_data = load_and_preprocess_dataset()
                st.session_state.X_train = X_data
                st.session_state.y_train = y_data
                st.session_state.dataset_loaded = True
                st.success("✅ Dataset loaded successfully!")
        
        if st.session_state.dataset_loaded:
            st.success("✅ Dataset loaded for preview!")
            
            # Dataset details
            st.markdown("### 📋 Kaggle Brain MRI Images Dataset (or Simulated)")
            
            col_info1, col_info2 = st.columns([1, 1])
            
            with col_info1:
                st.markdown("""
                **Dataset Details:**
                - **Name**: Brain MRI Images for Brain Tumor Detection
                - **Source**: Kaggle (navoneel) or Simulated
                - **Total Images**: Depends on actual/simulated data
                - **Classes**: 2 (Tumor, No Tumor)
                - **Format**: JPG/PNG images
                - **Use Case**: Binary classification
                """)
            
            with col_info2:
                st.markdown("""
                **Image Characteristics:**
                - **Color**: RGB (internal conversion for CNN)
                - **Size**: Variable (resized to 224x224)
                - **Quality**: Dependent on source (simulated data is synthetic)
                - **Annotation**: Manual (for real data)
                """)

            # Display sample images
            st.markdown("### 🖼️ Sample Images")
            
            X_data = st.session_state.X_train
            y_data = st.session_state.y_train
            
            # Show sample images
            cols_sample = st.columns(4)
            labels = ['No Tumor', 'Tumor']
            
            for i in range(4):
                with cols_sample[i]:
                    idx = np.random.randint(0, len(X_data))
                    img = X_data[idx]
                    label = labels[np.argmax(y_data[idx])]
                    
                    st.image(img, caption=f"Sample {i+1}: {label}", use_column_width=True)
            
            # Dataset statistics
            st.markdown("### 📊 Dataset Statistics")
            
            tumor_count = np.sum(np.argmax(y_data, axis=1))
            no_tumor_count = len(y_data) - tumor_count
            
            col_stats1, col_stats2 = st.columns(2)
            
            with col_stats1:
                # Class distribution
                fig_pie_data = px.pie(
                    values=[no_tumor_count, tumor_count],
                    names=['No Tumor', 'Tumor'],
                    title="Class Distribution",
                    color_discrete_sequence=['#006600', '#cc0000'] # Adjusted colors for black theme
                )
                fig_pie_data.update_layout(plot_bgcolor='#000000', paper_bgcolor='#1a1a1a', font_color='#ffffff')
                st.plotly_chart(fig_pie_data, use_container_width=True, key="bt_dataset_class_dist_pie") # Added unique key
            
            with col_stats2:
                st.metric("Total Images", len(X_data))
                st.metric("Tumor Images", tumor_count)
                st.metric("No Tumor Images", no_tumor_count)
                st.metric("Image Shape", f"{X_data.shape[1]}x{X_data.shape[2]}x{X_data.shape[3]}")

    with tab_training:
        st.markdown('<h3 class="sub-header-brain">CNN Model Training</h3>', unsafe_allow_html=True)
        
        # Load dataset if not already loaded
        if not st.session_state.dataset_loaded:
            st.info("Please load the dataset first in the 'Dataset Info' tab or by clicking the button below.")
            if st.button("Load Dataset for Training", key="load_dataset_for_training_btn_bt"):
                with st.spinner("Loading brain tumor dataset..."):
                    X_data, y_data = load_and_preprocess_dataset()
                    st.session_state.X_train = X_data
                    st.session_state.y_train = y_data
                    st.session_state.dataset_loaded = True
                    st.success("✅ Dataset loaded successfully!")
                    st.rerun() # Rerun to update UI after loading
        
        if st.session_state.dataset_loaded:
            X_train_data = st.session_state.X_train
            y_train_data = st.session_state.y_train
            
            # Display dataset info
            col_train1, col_train2 = st.columns(2)
            
            with col_train1:
                st.markdown("**Dataset Information:**")
                st.write(f"• Total samples: {len(X_train_data)}")
                st.write(f"• Image shape: {X_train_data.shape[1:]}")
                st.write(f"• Classes: {y_train_data.shape[1]}")
                st.write(f"• Tumor samples: {np.sum(np.argmax(y_train_data, axis=1))}")
                st.write(f"• No tumor samples: {len(y_train_data) - np.sum(np.argmax(y_train_data, axis=1))}")
            
            with col_train2:
                st.markdown("**Model Architecture:**")
                st.write("• Input: 224x224x3 images")
                st.write("• Conv2D layers: 32, 64, 128, 256 filters")
                st.write("• MaxPooling after each Conv block")
                st.write("• Batch normalization for stability")
                st.write("• Dropout for regularization")
                st.write("• Dense layers: 512, 256, 2 neurons")
            
            # Training parameters
            st.markdown("### 🛠️ Training Configuration")
            col_config1, col_config2, col_config3 = st.columns(3)
            
            with col_config1:
                epochs_val = st.slider("Epochs", 5, 50, EPOCHS, key="epochs_slider_bt")
            with col_config2:
                batch_size_val = st.selectbox("Batch Size", [16, 32, 64], index=1, key="batch_size_select_bt")
            with col_config3:
                validation_split_val = st.slider("Validation Split", 0.1, 0.3, 0.2, key="val_split_slider_bt")
            
            # Train model
            if st.button("🚀 Train CNN Model", type="primary", key="train_model_btn_bt"):
                with st.spinner("Training CNN model... This may take a few minutes."):
                    # Create model
                    model = create_cnn_model()
                    
                    # Display model summary
                    st.markdown("### 📋 Model Summary")
                    model_summary = []
                    model.summary(print_fn=lambda x: model_summary.append(x))
                    st.text('\n'.join(model_summary))
                    
                    # Split data
                    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                        X_train_data, y_train_data, test_size=validation_split_val, random_state=42
                    )
                    
                    # Train model
                    history = model.fit(
                        X_train_split, y_train_split,
                        validation_data=(X_val_split, y_val_split),
                        epochs=epochs_val,
                        batch_size=batch_size_val,
                        verbose=0
                    )
                    
                    # Save model and history
                    st.session_state.model = model
                    st.session_state.training_history = history
                    st.session_state.model_trained = True
                    
                    st.success("✅ Model trained successfully!")
                    
                    # Display training results
                    st.markdown("### 📈 Training Results")
                    
                    # Training history plots
                    col_results1, col_results2 = st.columns(2)
                    
                    with col_results1:
                        # Accuracy plot
                        fig_acc = go.Figure()
                        fig_acc.add_trace(go.Scatter(
                            y=history.history['accuracy'],
                            name='Training Accuracy',
                            mode='lines+markers'
                        ))
                        fig_acc.add_trace(go.Scatter(
                            y=history.history['val_accuracy'],
                            name='Validation Accuracy',
                            mode='lines+markers'
                        ))
                        fig_acc.update_layout(title="Model Accuracy", xaxis_title="Epoch", yaxis_title="Accuracy",
                                             plot_bgcolor='#000000', paper_bgcolor='#1a1a1a', font_color='#ffffff')
                        st.plotly_chart(fig_acc, use_container_width=True, key="bt_training_accuracy_plot") # Added unique key
                    
                    with col_results2:
                        # Loss plot
                        fig_loss = go.Figure()
                        fig_loss.add_trace(go.Scatter(
                            y=history.history['loss'],
                            name='Training Loss',
                            mode='lines+markers'
                        ))
                        fig_loss.add_trace(go.Scatter(
                            y=history.history['val_loss'],
                            name='Validation Loss',
                            mode='lines+markers'
                        ))
                        fig_loss.update_layout(title="Model Loss", xaxis_title="Epoch", yaxis_title="Loss",
                                               plot_bgcolor='#000000', paper_bgcolor='#1a1a1a', font_color='#ffffff')
                        st.plotly_chart(fig_loss, use_container_width=True, key="bt_training_loss_plot") # Added unique key
                    
                    # Final metrics
                    final_acc_train = history.history['accuracy'][-1]
                    final_val_acc_train = history.history['val_accuracy'][-1]
                    
                    col_final_metrics1, col_final_metrics2, col_final_metrics3 = st.columns(3)
                    with col_final_metrics1:
                        st.metric("Final Training Accuracy", f"{final_acc_train:.2%}")
                    with col_final_metrics2:
                        st.metric("Final Validation Accuracy", f"{final_val_acc_train:.2%}")
                    with col_final_metrics3:
                        st.metric("Training Epochs", epochs_val)

    with tab_prediction:
        st.markdown('<h3 class="sub-header-brain">Brain Tumor Detection</h3>', unsafe_allow_html=True)
        patient_name_brain = st.text_input("👤 **Patient Name**", placeholder="e.g., Sarah Connor", key="brain_tumor_patient_name_tab")

        if not st.session_state.model_trained:
            st.warning("⚠️ Please train the model first in the 'Model Training' tab.")
            st.stop()
        
        col_pred1, col_pred2 = st.columns([1, 1])
        
        with col_pred1:
            st.markdown("### 📤 Upload MRI Image")
            uploaded_file_pred = st.file_uploader(
                "Choose a brain MRI image",
                type=['jpg', 'jpeg', 'png'],
                help="Upload a brain MRI scan image for tumor detection",
                key="brain_tumor_uploader"
            )
            
            if uploaded_file_pred is not None:
                # Display uploaded image
                image_pred = Image.open(uploaded_file_pred)
                st.image(image_pred, caption="Uploaded MRI Image", use_column_width=True)
                
                # Predict button
                if st.button("🔍 Analyze Image", type="primary", key="analyze_image_btn_bt"):
                    if not patient_name_brain:
                        st.warning("⚠️ Please enter a patient name before analyzing the image.")
                    else:
                        with st.spinner("Analyzing brain MRI image..."):
                            # Preprocess image
                            processed_image_pred = preprocess_image(image_pred)
                            
                            # Make prediction
                            prediction_prob = st.session_state.model.predict(processed_image_pred)
                            predicted_class_idx = np.argmax(prediction_prob[0])
                            confidence_score = np.max(prediction_prob[0])
                            
                            # Display results
                            st.markdown("### 🎯 Detection Results")
                            
                            result_text = ""
                            risk_level = ""

                            if predicted_class_idx == 1:  # Tumor detected
                                st.markdown('<div class="result-positive">⚠️ TUMOR DETECTED</div>', unsafe_allow_html=True)
                                st.error(f"Confidence: {confidence_score:.2%}")
                                result_text = "Tumor Detected"
                                risk_level = "High"
                            else:  # No tumor
                                st.markdown('<div class="result-negative">✅ NO TUMOR DETECTED</div>', unsafe_allow_html=True)
                                st.success(f"Confidence: {confidence_score:.2%}")
                                result_text = "No Tumor Detected"
                                risk_level = "Low"
                            
                            inputs_brain_pred = {
                                'Image File': uploaded_file_pred.name,
                                'Confidence': f"{confidence_score:.2%}",
                                'Predicted Class': ['No Tumor', 'Tumor'][predicted_class_idx]
                            }
                            add_to_history(patient_name_brain, 'Brain Tumor Prediction', inputs_brain_pred, result_text, risk_level)

                            # Prediction probabilities
                            st.markdown("### 📊 Prediction Probabilities")
                            
                            prob_data_chart = {
                                'Class': ['No Tumor', 'Tumor'],
                                'Probability': [prediction_prob[0][0], prediction_prob[0][1]]
                            }
                            
                            fig_prob = px.bar(
                                prob_data_chart,
                                x='Class',
                                y='Probability',
                                title="Prediction Confidence",
                                color='Probability',
                                color_continuous_scale='RdYlBu_r'
                            )
                            fig_prob.update_layout(plot_bgcolor='#000000', paper_bgcolor='#1a1a1a', font_color='#ffffff')
                            st.plotly_chart(fig_prob, use_container_width=True, key="bt_prediction_probability_bar") # Added unique key
                            
                            # Additional information
                            st.markdown("### ℹ️ Important Medical Disclaimer")
                            st.info("""
                            - This is an AI-assisted diagnostic tool for educational and research purposes.
                            - **Results should NOT replace professional medical diagnosis.**
                            - Always consult qualified healthcare professionals for medical decisions.
                            - The model's predictions are probabilistic and may contain errors.
                            """)
            else:
                st.info("Upload an image to get a prediction.")
        
        with col_pred2:
            st.markdown("### 📋 How to Use")
            st.write("""
            1. **Upload Image**: Select a brain MRI scan image from your device.
            2. **Supported Formats**: JPG, JPEG, PNG.
            3. **Click Analyze**: The system will preprocess the image and run the trained CNN model.
            4. **Review Results**: The predicted class (Tumor/No Tumor) and confidence score will be displayed.
            
            **Tips for Best Results:**
            - Use clear, high-quality MRI images.
            - Ensure the image clearly shows brain tissue.
            - Avoid rotated or distorted images.
            - Use standard medical imaging formats if possible.
            """)
            
            st.markdown("### 🔬 Model Performance Quick View")
            if st.session_state.model_trained and st.session_state.training_history:
                history_for_perf = st.session_state.training_history
                final_val_acc_perf = history_for_perf.history['val_accuracy'][-1]
                st.success(f"Model Validation Accuracy: {final_val_acc_perf:.2%}")
            else:
                st.info("Model not yet trained. Validation accuracy will appear here after training.")
            
            st.markdown("### 📈 Sample Predictions from Dataset")
            if st.session_state.dataset_loaded and st.session_state.model_trained:
                X_sample_pred = st.session_state.X_train
                y_sample_pred = st.session_state.y_train
                
                # Random sample for display
                idx_sample = np.random.randint(0, len(X_sample_pred))
                sample_img_display = X_sample_pred[idx_sample]
                sample_img_predict = np.expand_dims(sample_img_display, axis=0) # Add batch dimension
                
                true_label_display = ['No Tumor', 'Tumor'][np.argmax(y_sample_pred[idx_sample])]
                
                # Make prediction with the trained model
                pred_sample = st.session_state.model.predict(sample_img_predict)
                pred_label_display = ['No Tumor', 'Tumor'][np.argmax(pred_sample)]
                
                st.image(sample_img_display, caption=f"True: {true_label_display} | Predicted: {pred_label_display}", use_column_width=True)
            else:
                st.info("Load dataset and train model to see sample predictions from the dataset.")

    with tab_analytics:
        st.markdown('<h3 class="sub-header-brain">Model Analytics & Performance</h3>', unsafe_allow_html=True)
        
        if not st.session_state.model_trained:
            st.warning("⚠️ Please train the model first to view analytics.")
        else:
            history_analytics = st.session_state.training_history
            
            sub_tab1, sub_tab2, sub_tab3 = st.tabs(["Training Metrics", "Model Architecture", "Dataset Analysis"])
            
            with sub_tab1:
                st.markdown("### Training History")
                
                col_hist1, col_hist2 = st.columns(2)
                
                with col_hist1:
                    # Detailed accuracy plot
                    fig_acc_det = go.Figure()
                    fig_acc_det.add_trace(go.Scatter(
                        y=history_analytics.history['accuracy'],
                        name='Training Accuracy',
                        mode='lines+markers',
                        line=dict(color='blue', width=2)
                    ))
                    fig_acc_det.add_trace(go.Scatter(
                        y=history_analytics.history['val_accuracy'],
                        name='Validation Accuracy',
                        mode='lines+markers',
                        line=dict(color='red', width=2)
                    ))
                    fig_acc_det.update_layout(
                        title="Model Accuracy Over Time",
                        xaxis_title="Epoch",
                        yaxis_title="Accuracy",
                        hovermode='x unified',
                        plot_bgcolor='#000000', paper_bgcolor='#1a1a1a', font_color='#ffffff'
                    )
                    st.plotly_chart(fig_acc_det, use_container_width=True, key="bt_analytics_acc_det") # Added unique key
                
                with col_hist2:
                    # Detailed loss plot
                    fig_loss_det = go.Figure()
                    fig_loss_det.add_trace(go.Scatter(
                        y=history_analytics.history['loss'],
                        name='Training Loss',
                        mode='lines+markers',
                        line=dict(color='blue', width=2)
                    ))
                    fig_loss_det.add_trace(go.Scatter(
                        y=history_analytics.history['val_loss'],
                        name='Validation Loss',
                        mode='lines+markers',
                        line=dict(color='red', width=2)
                    ))
                    fig_loss_det.update_layout(
                        title="Model Loss Over Time",
                        xaxis_title="Epoch",
                        yaxis_title="Loss",
                        hovermode='x unified',
                        plot_bgcolor='#000000', paper_bgcolor='#1a1a1a', font_color='#ffffff'
                    )
                    st.plotly_chart(fig_loss_det, use_container_width=True, key="bt_analytics_loss_det") # Added unique key
                
                # Training summary
                st.markdown("### 📋 Training Summary")
                col_sum1, col_sum2, col_sum3, col_sum4 = st.columns(4)
                
                with col_sum1:
                    st.metric("Final Training Accuracy", f"{history_analytics.history['accuracy'][-1]:.2%}")
                with col_sum2:
                    st.metric("Final Validation Accuracy", f"{history_analytics.history['val_accuracy'][-1]:.2%}")
                with col_sum3:
                    st.metric("Best Validation Accuracy", f"{max(history_analytics.history['val_accuracy']):.2%}")
                with col_sum4:
                    st.metric("Total Epochs", len(history_analytics.history['accuracy']))
            
            with sub_tab2:
                st.markdown("### Model Architecture Analysis")

                # Model summary
                if st.session_state.model:
                    model_arch = st.session_state.model
                    
                    st.markdown("**Model Summary (Text):**")
                    # Capture the model summary text output using StringIO
                    import io
                    buffer = io.StringIO()
                    model_arch.summary(print_fn=lambda x: buffer.write(x + '\n'))
                    summary_string = buffer.getvalue()
                    st.text(summary_string) # Display the captured summary directly
                    
                    # Model parameters (these are safe to access directly)
                    total_params = model_arch.count_params()
                    trainable_params = sum([layer.count_params() for layer in model_arch.layers if layer.trainable])
                    
                    col_params1, col_params2, col_params3 = st.columns(3)
                    with col_params1:
                        st.metric("Total Parameters", f"{total_params:,}")
                    with col_params2:
                        st.metric("Trainable Parameters", f"{trainable_params:,}")
                    with col_params3:
                        st.metric("Model Size", f"{total_params * 4 / 1024 / 1024:.2f} MB")
                else:
                    st.info("Train the model first to view architecture analysis.")
            
            with sub_tab3:
                if st.session_state.dataset_loaded:
                    st.markdown("### Dataset Statistics")
                    
                    X_analytics = st.session_state.X_train
                    y_analytics = st.session_state.y_train
                    
                    # Class distribution
                    tumor_count_an = np.sum(np.argmax(y_analytics, axis=1))
                    no_tumor_count_an = len(y_analytics) - tumor_count_an
                    
                    col_ds_stats1, col_ds_stats2 = st.columns(2)
                    
                    with col_ds_stats1:
                        # Class distribution pie chart
                        fig_ds_pie = px.pie(
                            values=[no_tumor_count_an, tumor_count_an],
                            names=['No Tumor', 'Tumor'],
                            title="Class Distribution",
                            color_discrete_sequence=['#006600', '#cc0000'] # Adjusted colors
                        )
                        fig_ds_pie.update_layout(plot_bgcolor='#000000', paper_bgcolor='#1a1a1a', font_color='#ffffff')
                        st.plotly_chart(fig_ds_pie, use_container_width=True, key="bt_analytics_dataset_class_dist_pie") # Added unique key
                    
                    with col_ds_stats2:
                        # Dataset metrics
                        st.metric("Total Images", len(X_analytics))
                        st.metric("Image Dimensions", f"{X_analytics.shape[1]} x {X_analytics.shape[2]}")
                        st.metric("Color Channels", X_analytics.shape[3])
                        st.metric("Approx. Data Size", f"{X_analytics.nbytes / 1024 / 1024:.2f} MB")
                    
                    # Image statistics
                    st.markdown("### 🖼️ Image Analysis")
                    
                    # Sample images grid
                    st.markdown("**Sample Images from Dataset:**")
                    cols_sample_an = st.columns(6)
                    
                    for i in range(6):
                        with cols_sample_an[i]:
                            idx_an = np.random.randint(0, len(X_analytics))
                            label_an = ['No Tumor', 'Tumor'][np.argmax(y_analytics[idx_an])]
                            st.image(X_analytics[idx_an], caption=label_an, use_column_width=True)
                    
                    # Pixel intensity analysis
                    st.markdown("**Pixel Intensity Distribution:**")
                    
                    # Calculate mean pixel intensities for each class
                    tumor_indices_an = np.where(np.argmax(y_analytics, axis=1) == 1)[0]
                    no_tumor_indices_an = np.where(np.argmax(y_analytics, axis=1) == 0)[0]
                    
                    tumor_intensities_an = np.mean(X_analytics[tumor_indices_an], axis=(1, 2, 3))
                    no_tumor_intensities_an = np.mean(X_analytics[no_tumor_indices_an], axis=(1, 2, 3))
                    
                    # Create histogram
                    fig_hist_intensity = go.Figure()
                    fig_hist_intensity.add_trace(go.Histogram(
                        x=no_tumor_intensities_an,
                        name='No Tumor',
                        opacity=0.7,
                        nbinsx=30
                    ))
                    fig_hist_intensity.add_trace(go.Histogram(
                        x=tumor_intensities_an,
                        name='Tumor',
                        opacity=0.7,
                        nbinsx=30
                    ))
                    fig_hist_intensity.update_layout(
                        title="Pixel Intensity Distribution by Class",
                        xaxis_title="Mean Pixel Intensity",
                        yaxis_title="Frequency",
                        barmode='overlay',
                        plot_bgcolor='#000000', paper_bgcolor='#1a1a1a', font_color='#ffffff'
                    )
                    st.plotly_chart(fig_hist_intensity, use_container_width=True, key="bt_analytics_pixel_intensity_hist") # Added unique key
                
                else:
                    st.warning("Dataset not loaded. Please load the dataset in the 'Dataset Info' tab.")

=======
>>>>>>> b11d9b7f112ada933af127f161ee2a0dc30a9d51

# Prediction History Page
if selected == 'Prediction History':
    st.markdown("## 📋 Prediction History")
    
    if not st.session_state.prediction_history:
        st.info("No prediction history available. Make some predictions first!")
    else:
        st.markdown(f"### Total Predictions: {len(st.session_state.prediction_history)}")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            patient_names = ['All'] + sorted(list(set(entry['patient_name'] for entry in st.session_state.prediction_history)))
<<<<<<< HEAD
            selected_patient = st.selectbox("Filter by Patient", patient_names, key="filter_patient")
        with col2:
            prediction_types = ['All'] + list(set(entry['type'] for entry in st.session_state.prediction_history))
            selected_type = st.selectbox("Filter by Prediction Type", prediction_types, key="filter_type")
        with col3:
            risk_levels = ['All', 'High', 'Low']
            selected_risk = st.selectbox("Filter by Risk Level", risk_levels, key="filter_risk")
        with col4:
            if st.button("🗑️ Clear History", type="secondary", key="clear_history_btn"):
=======
            selected_patient = st.selectbox("Filter by Patient", patient_names)
        with col2:
            prediction_types = ['All'] + list(set(entry['type'] for entry in st.session_state.prediction_history))
            selected_type = st.selectbox("Filter by Prediction Type", prediction_types)
        with col3:
            risk_levels = ['All', 'High', 'Low']
            selected_risk = st.selectbox("Filter by Risk Level", risk_levels)
        with col4:
            if st.button("🗑️ Clear History", type="secondary"):
>>>>>>> b11d9b7f112ada933af127f161ee2a0dc30a9d51
                st.session_state.prediction_history = []
                st.rerun()
        
        # Filter history
        filtered_history = st.session_state.prediction_history
        if selected_patient != 'All':
            filtered_history = [entry for entry in filtered_history if entry['patient_name'] == selected_patient]
        if selected_type != 'All':
            filtered_history = [entry for entry in filtered_history if entry['type'] == selected_type]
        if selected_risk != 'All':
            filtered_history = [entry for entry in filtered_history if entry['risk_level'] == selected_risk]
        
        for i, entry in enumerate(reversed(filtered_history)):
            with st.container():
                st.markdown(f'<div class="history-card">', unsafe_allow_html=True)
                
                col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
                with col1:
                    st.markdown(f"**📅 {entry['timestamp']}**")
                with col2:
                    st.markdown(f"**👤 {entry['patient_name']}**")
                with col3:
                    st.markdown(f"**🔬 {entry['type']}**")
                with col4:
                    risk_color = "#cc0000" if entry['risk_level'] == 'High' else "#006600"
                    st.markdown(f"<span style='color: {risk_color}; font-weight: bold;'>{entry['risk_level']} Risk</span>", unsafe_allow_html=True)
                
                with st.expander(f"View Input Parameters - Test #{len(filtered_history) - i}"):
                    input_cols = st.columns(3)
                    
                    # Special handling for Brain Tumor Prediction inputs
                    if entry['type'] == 'Brain Tumor Prediction':
                        st.write(f"**Image File:** {entry['inputs'].get('Image File', 'N/A')}")
                        st.write(f"**Predicted Class:** {entry['inputs'].get('Predicted Class', 'N/A')}")
                        st.write(f"**Confidence:** {entry['inputs'].get('Confidence', 'N/A')}")
                    else:
                        for idx, (key, value) in enumerate(entry['inputs'].items()):
                            with input_cols[idx % 3]:
                                st.write(f"**{key}:** {value}")
                
                st.markdown('</div>', unsafe_allow_html=True)
                st.markdown("---")

# Trend Analysis Page
if selected == 'Trend Analysis':
    st.markdown("## 📊 Trend Analysis & Progress Monitoring")
    if not st.session_state.prediction_history:
        st.info("No prediction history available for trend analysis.")
    else:
        st.markdown("### 📈 Visual Trend Analysis")
        create_trend_charts()

# Export Data Page
if selected == 'Export Data':
    st.markdown("## 💾 Export Data & Reports")
    
    if not st.session_state.prediction_history:
        st.info("No prediction history available for export.")
    else:
        st.markdown("### 📊 Export Options")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="info-card">', unsafe_allow_html=True)
            st.markdown("#### 📄 JSON Export")
            json_data = export_history_json()
<<<<<<< HEAD
            st.download_button(label="📥 Download JSON", data=json_data, file_name=f"health_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", mime="application/json", use_container_width=True, key="download_json")
=======
            st.download_button(label="📥 Download JSON", data=json_data, file_name=f"health_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", mime="application/json", use_container_width=True)
>>>>>>> b11d9b7f112ada933af127f161ee2a0dc30a9d51
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="info-card">', unsafe_allow_html=True)
            st.markdown("#### 📊 CSV Export")
            st.markdown("Export history with each input parameter in its own column.")
            csv_data = export_history_csv()
            if csv_data:
<<<<<<< HEAD
                st.download_button(label="📥 Download CSV", data=csv_data, file_name=f"health_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv", use_container_width=True, key="download_csv")
=======
                st.download_button(label="📥 Download CSV", data=csv_data, file_name=f"health_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv", use_container_width=True)
>>>>>>> b11d9b7f112ada933af127f161ee2a0dc30a9d51
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 📋 Comprehensive Health Report")
        
<<<<<<< HEAD
        if st.button("📊 Generate Report", type="primary", use_container_width=True, key="generate_report_btn"):
=======
        if st.button("📊 Generate Report", type="primary", use_container_width=True):
>>>>>>> b11d9b7f112ada933af127f161ee2a0dc30a9d51
            report = f"# 🏥 Comprehensive Health Report\n**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            report += f"## 📊 Summary Statistics\n- **Total Predictions:** {len(st.session_state.prediction_history)}\n- **High Risk Results:** {sum(1 for entry in st.session_state.prediction_history if entry['risk_level'] == 'High')}\n"
            report += "## 🔬 Test Breakdown\n"
            prediction_counts = {}
            for entry in st.session_state.prediction_history:
                pred_type = entry['type']
                if pred_type not in prediction_counts: prediction_counts[pred_type] = {'total': 0, 'high_risk': 0}
                prediction_counts[pred_type]['total'] += 1
                if entry['risk_level'] == 'High': prediction_counts[pred_type]['high_risk'] += 1
            for pred_type, counts in prediction_counts.items():
                risk_rate = (counts['high_risk'] / counts['total'] * 100) if counts['total'] > 0 else 0
                report += f"- **{pred_type}:** {counts['total']} tests, {counts['high_risk']} high risk ({risk_rate:.1f}%)\n"

            report += "\n## 📈 Recent Activity (Last 5 Predictions)\n"
            recent_predictions = sorted(st.session_state.prediction_history, key=lambda x: x['timestamp'], reverse=True)[:5]
            for entry in recent_predictions:
                risk_indicator = "🔴" if entry['risk_level'] == 'High' else "🟢"
                report += f"- {risk_indicator} {entry['timestamp']} - **{entry['patient_name']}** - {entry['type']} - {entry['risk_level']} Risk\n"
            
            report += "\n## ⚠️ Important Disclaimer\nThis AI-powered health assistant is for informational purposes only..."
            st.markdown(report)
<<<<<<< HEAD
            st.download_button(label="📥 Download Report", data=report, file_name=f"health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md", mime="text/markdown", use_container_width=True, key="download_report")
=======
            st.download_button(label="📥 Download Report", data=report, file_name=f"health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md", mime="text/markdown", use_container_width=True)
>>>>>>> b11d9b7f112ada933af127f161ee2a0dc30a9d51

# Health Chatbot page
if selected == 'Health Chatbot':
    st.markdown("## 🤖 Health Chatbot - Powered by Poko")
    st.info("💡 Ask me anything about symptoms, diseases, or healthy living. I am an AI and my advice is not a substitute for a real doctor.")

    if not DEEPSEEK_API_KEY:
        st.error("⚠️ DEEPSEEK_API_KEY environment variable not set. Please create a `.env` file and add your key.")
        st.stop()

    if "deepseek_client" not in st.session_state:
        st.session_state.deepseek_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
        
    if "chatbot_history" not in st.session_state:
        st.session_state.chatbot_history = []

    for message in st.session_state.chatbot_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

<<<<<<< HEAD
    if user_prompt := st.chat_input("Ask a health-related question...", key="chatbot_input"):
=======
    if user_prompt := st.chat_input("Ask a health-related question..."):
>>>>>>> b11d9b7f112ada933af127f161ee2a0dc30a9d51
        st.session_state.chatbot_history.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            with st.spinner("🤖 Thinking..."):
                try:
                    messages_for_api = [{"role": m["role"], "content": m["content"]} for m in st.session_state.chatbot_history]
                    response = st.session_state.deepseek_client.chat.completions.create(model="deepseek-chat", messages=messages_for_api)
                    bot_reply = response.choices[0].message.content
                    st.session_state.chatbot_history.append({"role": "assistant", "content": bot_reply})
                    st.markdown(bot_reply)
                except Exception as e:
                    error_message = f"❌ An error occurred: {e}. Please check your API key and account balance."
                    st.error(error_message)
<<<<<<< HEAD
                    st.session_state.chatbot_history.append({"role": "assistant", "content": error_message})
=======
                    st.session_state.chatbot_history.append({"role": "assistant", "content": error_message})
>>>>>>> b11d9b7f112ada933af127f161ee2a0dc30a9d51
