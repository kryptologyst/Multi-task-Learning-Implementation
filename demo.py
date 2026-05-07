#!/usr/bin/env python3
"""
Interactive Demo for Multi-task Learning

This script creates an interactive Streamlit demo for the multi-task learning model.

Author: kryptologyst
GitHub: https://github.com/kryptologyst
"""

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import io
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from mtl.models import ResNetMTL, BaselineModel
from mtl.utils import get_device, set_seed


# Page configuration
st.set_page_config(
    page_title="Multi-task Learning Demo",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Safety disclaimer
st.markdown("""
<div class="warning-box">
    <h4>⚠️ Safety and Ethics Disclaimer</h4>
    <p><strong>This is a research demonstration only.</strong></p>
    <ul>
        <li>This demo uses synthetic attributes for educational purposes</li>
        <li>Not suitable for production decisions or real-world applications</li>
        <li>Age and gender predictions are synthetic and not based on real biometric data</li>
        <li>Use responsibly and in accordance with applicable laws and regulations</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🧠 Multi-task Learning Demo</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Configuration")
st.sidebar.markdown("---")

# Model selection
model_type = st.sidebar.selectbox(
    "Select Model Type",
    ["ResNet MTL", "Baseline CNN", "Uncertainty Weighted MTL"],
    help="Choose the multi-task learning model architecture"
)

# Task weights
st.sidebar.markdown("### Task Weights")
classification_weight = st.sidebar.slider("Classification Weight", 0.0, 2.0, 1.0, 0.1)
age_weight = st.sidebar.slider("Age Regression Weight", 0.0, 2.0, 1.0, 0.1)
gender_weight = st.sidebar.slider("Gender Classification Weight", 0.0, 2.0, 1.0, 0.1)

# Advanced options
st.sidebar.markdown("### Advanced Options")
use_uncertainty_weighting = st.sidebar.checkbox("Use Uncertainty Weighting", False)
use_gradient_surgery = st.sidebar.checkbox("Use Gradient Surgery", False)
freeze_backbone = st.sidebar.checkbox("Freeze Backbone", False)

# Load model function
@st.cache_resource
def load_model(model_type: str, device: torch.device):
    """Load the selected model."""
    set_seed(42)
    
    if model_type == "ResNet MTL":
        model = ResNetMTL(
            backbone="resnet50",
            pretrained=True,
            freeze_backbone=freeze_backbone,
            dropout_rate=0.5,
            classification={"num_classes": 10, "hidden_dim": 512},
            age_regression={"hidden_dim": 256},
            gender_classification={"num_classes": 2, "hidden_dim": 256},
            feature_dim=2048,
            shared_layers=[512, 256]
        )
    elif model_type == "Baseline CNN":
        model = BaselineModel()
    else:  # Uncertainty Weighted MTL
        base_model = ResNetMTL(
            backbone="resnet50",
            pretrained=True,
            freeze_backbone=freeze_backbone,
            dropout_rate=0.5,
            classification={"num_classes": 10, "hidden_dim": 512},
            age_regression={"hidden_dim": 256},
            gender_classification={"num_classes": 2, "hidden_dim": 256},
            feature_dim=2048,
            shared_layers=[512, 256]
        )
        from mtl.models import UncertaintyWeightedMTL
        model = UncertaintyWeightedMTL(base_model)
    
    model.to(device)
    model.eval()
    return model

# Get device
device = get_device("auto")

# Load model
with st.spinner("Loading model..."):
    model = load_model(model_type, device)

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📸 Upload Image")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg'],
        help="Upload an image to test the multi-task learning model"
    )
    
    # Sample images
    st.markdown("### 🎯 Sample Images")
    sample_images = {
        "Airplane": "https://images.unsplash.com/photo-1436491865332-7a61a109cc05?w=200",
        "Car": "https://images.unsplash.com/photo-1552519507-da3b142c6e3d?w=200",
        "Bird": "https://images.unsplash.com/photo-1444464666168-49d633b86797?w=200",
        "Cat": "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=200",
        "Dog": "https://images.unsplash.com/photo-1552053831-71594a27632d?w=200"
    }
    
    selected_sample = st.selectbox("Or select a sample image:", list(sample_images.keys()))
    
    if selected_sample:
        st.image(sample_images[selected_sample], width=200)

with col2:
    st.markdown("### 🔮 Predictions")
    
    # Process image and make predictions
    if uploaded_file is not None or selected_sample:
        # Load image
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
        else:
            # For demo purposes, create a random image
            image = Image.new('RGB', (224, 224), color='lightblue')
        
        # Preprocess image
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Convert PIL to tensor
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Make predictions
        with torch.no_grad():
            predictions = model(image_tensor)
        
        # Display image
        st.image(image, caption="Input Image", width=300)
        
        # Process predictions
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                      'dog', 'frog', 'horse', 'ship', 'truck']
        
        # Classification prediction
        if "classification" in predictions:
            class_probs = torch.softmax(predictions["classification"], dim=1)
            predicted_class = torch.argmax(class_probs, dim=1).item()
            confidence = class_probs[0][predicted_class].item()
            
            st.markdown(f"""
            <div class="metric-card">
                <h4>🏷️ Classification</h4>
                <p><strong>Predicted Class:</strong> {class_names[predicted_class].title()}</p>
                <p><strong>Confidence:</strong> {confidence:.3f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Classification probabilities
            fig, ax = plt.subplots(figsize=(10, 6))
            probs = class_probs[0].cpu().numpy()
            bars = ax.bar(class_names, probs, color='skyblue')
            ax.set_xlabel('Classes')
            ax.set_ylabel('Probability')
            ax.set_title('Classification Probabilities')
            ax.tick_params(axis='x', rotation=45)
            
            # Highlight predicted class
            bars[predicted_class].set_color('red')
            
            st.pyplot(fig)
        
        # Age regression prediction
        if "age_regression" in predictions:
            predicted_age = predictions["age_regression"].item()
            
            st.markdown(f"""
            <div class="metric-card">
                <h4>👤 Age Prediction</h4>
                <p><strong>Predicted Age:</strong> {predicted_age:.1f} years</p>
                <p><em>Note: This is a synthetic prediction for demonstration purposes</em></p>
            </div>
            """, unsafe_allow_html=True)
        
        # Gender classification prediction
        if "gender_classification" in predictions:
            gender_probs = torch.softmax(predictions["gender_classification"], dim=1)
            predicted_gender = torch.argmax(gender_probs, dim=1).item()
            gender_confidence = gender_probs[0][predicted_gender].item()
            gender_labels = ['Male', 'Female']
            
            st.markdown(f"""
            <div class="metric-card">
                <h4>⚧ Gender Classification</h4>
                <p><strong>Predicted Gender:</strong> {gender_labels[predicted_gender]}</p>
                <p><strong>Confidence:</strong> {gender_confidence:.3f}</p>
                <p><em>Note: This is a synthetic prediction for demonstration purposes</em></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Gender probabilities
            fig, ax = plt.subplots(figsize=(6, 4))
            gender_probs_np = gender_probs[0].cpu().numpy()
            bars = ax.bar(gender_labels, gender_probs_np, color=['lightblue', 'lightpink'])
            ax.set_ylabel('Probability')
            ax.set_title('Gender Classification Probabilities')
            
            # Highlight predicted gender
            bars[predicted_gender].set_color('red')
            
            st.pyplot(fig)

# Model information
st.markdown("---")
st.markdown("### 📊 Model Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="metric-card">
        <h4>Model Type</h4>
        <p>{}</p>
    </div>
    """.format(model_type), unsafe_allow_html=True)

with col2:
    from mtl.utils import count_parameters
    param_info = count_parameters(model)
    st.markdown(f"""
    <div class="metric-card">
        <h4>Parameters</h4>
        <p>Total: {param_info['total']:,}</p>
        <p>Trainable: {param_info['trainable']:,}</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <h4>Device</h4>
        <p>{device}</p>
    </div>
    """, unsafe_allow_html=True)

# Task weights visualization
st.markdown("### ⚖️ Task Weights")
fig, ax = plt.subplots(figsize=(8, 4))
tasks = ['Classification', 'Age Regression', 'Gender Classification']
weights = [classification_weight, age_weight, gender_weight]
bars = ax.bar(tasks, weights, color=['skyblue', 'lightgreen', 'lightcoral'])
ax.set_ylabel('Weight')
ax.set_title('Current Task Weights')
ax.tick_params(axis='x', rotation=45)

# Add value labels on bars
for bar, weight in zip(bars, weights):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{weight:.1f}', ha='center', va='bottom')

st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
    <p><strong>Multi-task Learning Implementation</strong></p>
    <p>Author: <a href="https://github.com/kryptologyst">kryptologyst</a></p>
    <p>GitHub: <a href="https://github.com/kryptologyst">https://github.com/kryptologyst</a></p>
</div>
""", unsafe_allow_html=True)
