import streamlit as st
import numpy as np
import os  
import pandas as pd
import pickle
import cv2
import gdown
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.utils import load_img, img_to_array
import random

# --- Set Background Using Custom Image ---
def set_bg_image(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("{image_url}") no-repeat center center fixed;
            background-size: cover;
            font-family: 'Roboto', sans-serif;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Load custom Google Fonts
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap');
    </style>
    """, unsafe_allow_html=True)

# Set background image
set_bg_image("https://c0.wallpaperflare.com/preview/674/919/828/rice-rice-seeds-agriculture-harvesting.jpg")

# --- Leaf Disease Prediction ---
# Defining the number of classes
num_classes = 5

# Defining the class labels and remedies
class_labels = {
    0: ('Brown Spot', "🌱 Spray neem oil and ensure good air circulation."),
    1: ('Healthy', "✅ No action needed. Keep monitoring."),
    2: ('Hispa', "🐞 Use neem spray and remove pests by hand."),
    3: ('Leaf Blast', "🔥 Apply compost tea and avoid overwatering."),
    4: ('Tungro', "🚜 Remove infected plants and use light traps.")
}

# Function to download the model from Google Drive
@st.cache_resource
def load_cached_model(file_id, output_path):
    if not os.path.exists(output_path):
        gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False)
    model = load_model(output_path)
    return model

file_id = "1lHCCMfbvEPjjxFC4JqlwYowrSX-6PfUt"
model_path = "weight.h5"

# Loading the trained model
model = load_cached_model(file_id, model_path)

# Function to preprocess the input image
def preprocess_image(img):
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)  # Apply VGG16 preprocessing
    return img

# Leaf disease prediction logic
def leaf_disease_prediction():
    st.subheader("🌿 Leaf Disease Prediction App")
    st.markdown("---")

    # Upload image through Streamlit
    uploaded_file = st.file_uploader("📸 Upload a leaf image for disease prediction...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image with improved styling
        file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        image_display = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        st.image(image_display, caption="🖼️ Uploaded Image", use_container_width=True)

        st.markdown("---")

        # Preprocess the uploaded image
        img = load_img(uploaded_file, target_size=(224, 224))
        img = preprocess_image(img)

        # Extract features using the VGG16 base model
        base_model = VGG16(weights='imagenet', include_top=False)
        img_features = base_model.predict(img)
        img_features = img_features.reshape(1, 7, 7, 512)

        # Make predictions using the student model
        predictions = model.predict(img_features)
        predicted_class_index = np.argmax(predictions[0])
        predicted_class_label, remedy = class_labels[predicted_class_index]

        # Display the prediction result with a styled box
        st.markdown(f"""
        <style>
        .prediction-box {{
            font-size: 22px;
            font-weight: bold;
            color: white;
            background-color: {'#FF6347' if predicted_class_index != 1 else '#4CAF50'};
            border-radius: 10px;
            padding: 15px;
            text-align: center;
        }}
        </style>
        <div class="prediction-box">
            🔍 Disease Prediction: {predicted_class_label}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Display remedy inside an info box
        st.markdown(f"""
        <div style="border: 2px solid #2E86C1; padding: 15px; border-radius: 10px; background-color: #D6EAF8;">
            <h4>💡 Recommended Remedy</h4>
            <p style="font-size: 16px;">{remedy}</p>
        </div>
        """, unsafe_allow_html=True)

# --- Agricultural Yield Prediction ---
# Load the pre-trained models for yield prediction
try:
    # Yield prediction models
    dtr = pickle.load(open('models/dtr_model.pkl', 'rb'))
    preprocessor = pickle.load(open('models/preprocesser.pkl', 'rb'))
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# Yield Prediction
def yield_prediction():
    st.title("🌱 Agricultural Yield Prediction", anchor="yield-prediction")

    st.sidebar.header("Enter Input Features", help="Fill in the required details")
    year = st.sidebar.number_input("Year", min_value=2000, max_value=2100, step=1, value=2024)
    rainfall = st.sidebar.number_input("Rainfall (mm)", min_value=0.0, max_value=5000.0, step=0.1, value=1000.0)
    pesticides = st.sidebar.number_input("Pesticides Used (tonnes)", min_value=0.0, max_value=1000.0, step=0.1, value=50.0)
    avg_temp = st.sidebar.number_input("Temperature (°C)", min_value=-50.0, max_value=60.0, step=0.1, value=25.0)

    area = st.sidebar.selectbox("Area", [
    "Albania", "Algeria", "Angola", "Argentina", "Armenia", "Australia", "Austria",
    "Azerbaijan", "Bahamas", "Bahrain", "Bangladesh", "Belarus", "Belgium",
    "Botswana", "Brazil", "Bulgaria", "Burkina Faso", "Burundi", "Cameroon",
    "Canada", "Central African Republic", "Chile", "Colombia", "Croatia",
    "Denmark", "Dominican Republic", "Ecuador", "Egypt", "El Salvador", "Eritrea",
    "Estonia", "Finland", "France", "Germany", "Ghana", "Greece", "Guatemala",
    "Guinea", "Guyana", "Haiti", "Honduras", "Hungary", "India", "Indonesia", "Iraq",
    "Ireland", "Italy", "Jamaica", "Japan", "Kazakhstan", "Kenya", "Latvia",
    "Lebanon", "Lesotho", "Libya", "Lithuania", "Madagascar", "Malawi", "Malaysia",
    "Mali", "Mauritania", "Mauritius", "Mexico", "Montenegro", "Morocco",
    "Mozambique", "Namibia", "Nepal", "Netherlands", "New Zealand", "Nicaragua",
    "Niger", "Norway", "Pakistan", "Papua New Guinea", "Peru", "Poland", "Portugal",
    "Qatar", "Romania", "Rwanda", "Saudi Arabia", "Senegal", "Slovenia",
    "South Africa", "Spain", "Sri Lanka", "Sudan", "Suriname", "Sweden",
    "Switzerland", "Tajikistan", "Thailand", "Tunisia", "Turkey", "Uganda",
    "Ukraine", "United Kingdom", "Uruguay", "Zambia", "Zimbabwe"
])

    item = st.sidebar.selectbox("Item", [
    "Maize", "Potatoes", "Rice, paddy", "Sorghum", "Soybeans", "Wheat", "Cassava",
    "Sweet potatoes", "Plantains and others", "Yams"
])

    if st.sidebar.button("Predict Yield", key="predict-button"):
        try:
            features = pd.DataFrame([[year, rainfall, pesticides, avg_temp, area, item]], columns=['Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp', 'Area', 'Item'])
            transformed_features = preprocessor.transform(features)
            prediction = dtr.predict(transformed_features)
            
            # Styled Output for Yield Prediction
            st.markdown(f"""
            <style>
            .prediction-box {{
                font-size: 24px;
                font-weight: bold;
                color: white;
                background-color: #4CAF50;
                border-radius: 10px;
                padding: 10px;
                text-align: center;
            }}
            </style>
            <div class="prediction-box">
                🌱 The predicted agricultural yield is: {prediction[0]:.2f} tonnes
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error in prediction: {e}")

# Main App Navigation
def main_app():
    st.title("🚜 Farmer Dashboard")
    st.write("Choose an option:")
    option = st.radio("Select the prediction type", ("Leaf Disease Prediction", "Yield Prediction"))

    if option == "Leaf Disease Prediction":
        leaf_disease_prediction()
    elif option == "Yield Prediction":
        yield_prediction()

# Start the app directly
main_app()
