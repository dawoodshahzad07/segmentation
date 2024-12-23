import os
import tensorflow as tf
import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Constants
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

@st.cache_resource
def load_segmentation_model():
    """Load and cache the pre-trained segmentation model"""
    # Directly load the pre-trained segmentation model
    segmentation_model = tf.keras.models.load_model('seg_model.h5')
    return segmentation_model

@st.cache_resource
def load_classification_model():
    """Load and cache the pre-trained classification model"""
    # Load the classification model
    classification_model = tf.keras.models.load_model('best_classifier_final.keras')
    return classification_model

def process_image(upload):
    """Process uploaded image"""
    # Read image
    image = Image.open(upload)
    
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

def main():
    st.title("Vehicle Segmentation Mask and Classification")
    st.write("Upload an image to generate a segmentation mask and classify the vehicle!")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

        # Process image when button is clicked
        if st.button("Generate Segmentation Mask and Classify Vehicle"):
            with st.spinner("Generating mask and classification..."):
                try:
                    # Process image
                    image = process_image(uploaded_file)
                    
                    # Predict segmentation and classification
                    predicted_class, probabilities, mask = predict_with_segmentation_and_classification(image)
                    
                    # Display segmentation mask
                    st.subheader("Segmentation Mask")
                    st.image(mask, caption="Generated Segmentation Mask", use_container_width=True)

                    # Display classification result
                    class_names = ["airplane", "automobile", "truck"]  # Replace with actual class names
                    st.subheader("Classification Result")
                    st.write(f"Predicted class: {class_names[predicted_class]}")
                    st.write(f"Class probabilities: {probabilities}")

                except Exception as e:
                    st.error(f"Error generating mask and classification: {str(e)}")

if __name__ == "__main__":
    main()
