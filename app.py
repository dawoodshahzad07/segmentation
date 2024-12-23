import os
import tensorflow as tf
import cv2
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import requests
from io import BytesIO

# Constants
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

# CSS for cyberpunk theme
st.markdown("""
    <style>
        body {
            background-color: #121212;
            color: #00FF99;
            font-family: 'Courier New', monospace;
        }
        .stButton>button {
            background-color: #00FF99;
            color: #121212;
            font-weight: bold;
            border-radius: 5px;
            padding: 10px;
            box-shadow: 0 4px 8px rgba(0, 255, 153, 0.6);
        }
        .stButton>button:hover {
            background-color: #00FF66;
            box-shadow: 0 6px 12px rgba(0, 255, 102, 0.8);
        }
        .stTextInput input {
            background-color: #333333;
            color: #00FF99;
            border: 1px solid #00FF99;
        }
        .stTextInput input:focus {
            border-color: #00FF66;
        }
        .stFileUploader {
            color: #00FF99;
        }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_segmentation_model():
    """Load and cache the pre-trained segmentation model"""
    segmentation_model = tf.keras.models.load_model('seg_model.h5')
    return segmentation_model

@st.cache_resource
def load_classification_model():
    """Load and cache the pre-trained classification model"""
    classification_model = tf.keras.models.load_model('best_classifier_final.keras')
    return classification_model

def process_image(image_source):
    """Process uploaded image or PIL Image"""
    # If image_source is already a PIL Image (from URL)
    if isinstance(image_source, Image.Image):
        image = image_source
    else:
        # If image_source is a file upload
        image = Image.open(image_source)
    
    # Convert to numpy array
    image_array = np.array(image)
    
    # Convert to RGB if needed
    if len(image_array.shape) == 2:  # Grayscale
        image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
    elif image_array.shape[2] == 4:  # RGBA
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
    
    # Resize
    image_array = cv2.resize(image_array, (IMG_WIDTH, IMG_HEIGHT))
    
    # Normalize
    image_array = image_array.astype('float32') / 255.0
    
    return image_array

def predict_with_segmentation_and_classification(image):
    """Generate segmentation mask and use classification model on the image"""
    segmentation_model = load_segmentation_model()
    classification_model = load_classification_model()

    # 1. Segmentation prediction
    mask_pred = segmentation_model.predict(np.expand_dims(image, 0), verbose=0)[0]
    mask_img = (np.argmax(mask_pred, axis=-1) * 255).astype(np.uint8)
    mask_img = cv2.resize(mask_img, (IMG_WIDTH, IMG_HEIGHT))  # resize mask to match input size of classification model
    mask_img = np.expand_dims(mask_img, axis=-1)
    mask_img = np.repeat(mask_img, 3, axis=-1)  # Create RGB mask (replicate the mask channel 3 times for 3-channel RGB input)
    mask_img = mask_img.astype('float32') / 255.0

    # 2. Combine original image and mask
    combined_image = np.concatenate((image, mask_img), axis=-1)
    combined_image = np.expand_dims(combined_image, axis=0)

    # 3. Classification prediction
    class_probabilities = classification_model.predict(combined_image, verbose=0)[0]
    predicted_class = np.argmax(class_probabilities)

    return predicted_class, class_probabilities, mask_img

def plot_class_probabilities(probabilities, class_names):
    """Plot a bar chart for class probabilities"""
    fig, ax = plt.subplots()
    ax.bar(class_names, probabilities, color=['#FF007F', '#00FF99', '#00FFFF'])
    ax.set_xlabel('Class')
    ax.set_ylabel('Probability')
    ax.set_title('Classification Probabilities')
    st.pyplot(fig)

def load_image_from_url(url):
    """Load an image from a URL"""
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        return Image.open(BytesIO(response.content))
    except Exception as e:
        st.error(f"Error loading image from URL: {str(e)}")
        return None

def main():
    st.title("AutoMobile - Airplane - Truck, Classification and Segmentation")
    st.write("Upload an image or paste an image URL to generate a segmentation mask and classify the vehicle!")

    # Add tabs for upload methods
    tab1, tab2 = st.tabs(["Upload Image", "Image URL"])

    with tab1:
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        image_source = uploaded_file
        if uploaded_file is not None:
            input_image = Image.open(uploaded_file)
    
    with tab2:
        url = st.text_input("Enter image URL:")
        if url:
            input_image = load_image_from_url(url)
            image_source = input_image

    # Process image if available from either source
    if 'input_image' in locals() and input_image is not None:
        # Display uploaded/loaded image
        st.image(input_image, caption="Input Image", use_container_width=True)

        # Process image when button is clicked
        if st.button("Generate Segmentation Mask and Classify Vehicle"):
            with st.spinner("Generating mask and classification..."):
                try:
                    # Process image
                    image = process_image(image_source)
                    
                    # Predict segmentation and classification
                    predicted_class, probabilities, mask = predict_with_segmentation_and_classification(image)
                    
                    # Display segmentation mask
                    st.subheader("Segmentation Mask")
                    st.image(mask, caption="Generated Segmentation Mask", use_container_width=True)

                    # Display classification result
                    class_names = ["airplane", "automobile", "truck"]
                    st.subheader("Classification Result")
                    st.write(f"Predicted class: {class_names[predicted_class]}")
                    st.write(f"Class probabilities: {probabilities}")

                    # Plot class probabilities
                    plot_class_probabilities(probabilities, class_names)

                except Exception as e:
                    st.error(f"Error generating mask and classification: {str(e)}")

if __name__ == "__main__":
    main()
