import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import random
import json
import time
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
import os
import torch.nn as nn
from tqdm.auto import tqdm
import streamlit as st
from streamlit_lottie import st_lottie

from Model import ResNet34Custom
from helpful_Functions import test_step, train, train_step, plot_transformed_images
from helpful_Functions import plot_loss_curves, pred_and_plot_image

# Load animations
def load_lottie_json(path):
    with open(path, "r") as f:
        return json.load(f)

lottie_medical = load_lottie_json("1.json")
success_animation = load_lottie_json("2.json")  # Benign
sad_animation = load_lottie_json("3.json")      # Malignant

# Streamlit settings
st.set_page_config(page_title="Skin Cancer Detector", layout="centered")

# Custom CSS
st.markdown("""
    <style>
    .block-container {
        padding-top: 2rem;
        font-family: 'Segoe UI', sans-serif;
    }
    </style>
""", unsafe_allow_html=True)

# Session state to toggle pages
if "started" not in st.session_state:
    st.session_state.started = False

# Data & Model Loading
device = "cuda" if torch.cuda.is_available() else "cpu"
data_path = Path("data/")
train_path = data_path / "train"
test_path = data_path / "test"

simple_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

data_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])

train_data = datasets.ImageFolder(root=train_path, transform=data_transform)
test_data = datasets.ImageFolder(root=test_path, transform=data_transform)

train_dataloader = DataLoader(train_data, batch_size=1, shuffle=True)
class_names = train_data.classes

model_0 = ResNet34Custom(input_shape=3, output_shape=len(class_names)).to(device)
model_0.load_state_dict(torch.load('saved_model.pth', map_location=device))

# Prediction function
def predict_image(uploaded_image):
    img = Image.open(uploaded_image).convert("RGB")
    img_transformed = simple_transform(img).unsqueeze(0).to(device)

    model_0.eval()
    with torch.inference_mode():
        pred = model_0(img_transformed)
    
    class_idx = pred.argmax(dim=1).item()
    class_name = class_names[class_idx]
    confidence = torch.nn.functional.softmax(pred, dim=1)[0][class_idx].item() * 100

    return class_name, confidence, pred, img_transformed

# ---------------------- INTRO PAGE ---------------------- #
if not st.session_state.started:
    st.title("🩺 Skin Cancer Detection Using Machine Learning")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        Welcome to the **Skin Cancer Detection System**.

        This app uses AI to detect whether a skin lesion is **Benign** or **Malignant** from uploaded images.

        ### 🔍 Features:
        - Upload skin lesion images
        - Get predictions with confidence score
        - Dynamic animations for feedback


        """)
    with col2:
        st_lottie(lottie_medical, height=300)

    if st.button("🚀 Get Started", use_container_width=True):
        st.session_state.started = True
        st.rerun()

# ---------------------- PREDICTION PAGE ---------------------- #
else:
    st.title("🔬 Prediction Dashboard")

    uploaded_files = st.file_uploader("📤 Upload your skin image", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files:
        for img in uploaded_files:
            image = Image.open(img)
            st.image(image, caption=img.name,  use_container_width=True)

            with st.spinner("🔎 Analyzing..."):
                predicted_class, confidence, prediction_tensor, transformed = predict_image(img)
                time.sleep(1.5)

            card_color = "#d4edda" if predicted_class.lower() == "benign" else "#f8d7da"
            text_color = "#155724" if predicted_class.lower() == "benign" else "#721c24"

            st.markdown(f"""
                <div style='
                    background-color: {card_color};
                    color: {text_color};
                    padding: 25px;
                    border-radius: 12px;
                    border: 2px solid {"#c3e6cb" if predicted_class.lower() == "benign" else "#f5c6cb"};
                    margin-bottom: 20px;
                    text-align: center;
                '>
                    <h3>🩺 Prediction: {predicted_class}</h3>
                    <p style="font-size: 18px;"><b>Confidence Score:</b> {confidence:.2f}%</p>
                    <p style="font-size: 16px;">{ "👍 This appears to be a benign lesion." if predicted_class.lower() == "benign" else "⚠️ This appears to be malignant. Please consult a doctor." }</p>
                </div>
            """, unsafe_allow_html=True)

            if predicted_class.lower() == "benign":
                st_lottie(success_animation, height=200)
            else:
                st_lottie(sad_animation, height=200)

        # Model metrics (real metrics could be calculated during training and saved in JSON)
        st.subheader("📊 Model Performance Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", confidence)


        # Accuracy & Loss graph
        st.subheader("🖼️ Accuracy and Loss Graph")

        image1_path = "accuracy_graph.png"
        image2_path = "loss_graph.png"

        col1, col2 = st.columns(2)

        with col1:
            st.image(image1_path, caption="Sample Image 1",  use_container_width=True)

        with col2:
            st.image(image2_path, caption="Sample Image 2",  use_container_width=True)

    if st.button("🔙 Back to Home"):
        st.session_state.started = False
        st.rerun()