import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
import io

# Constants
IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

def unet_model(input_shape=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS)):
    inputs = tf.keras.layers.Input(input_shape)

    # Add BatchNormalization after inputs
    x = layers.BatchNormalization()(inputs)

    # Encoder
    conv1 = layers.Conv2D(64, 3, padding='same')(x)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.Activation('relu')(conv1)
    conv1 = layers.Conv2D(64, 3, padding='same')(conv1)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.Activation('relu')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = layers.Dropout(0.25)(pool1)  # Add dropout

    conv2 = layers.Conv2D(128, 3, padding='same')(pool1)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.Activation('relu')(conv2)
    conv2 = layers.Conv2D(128, 3, padding='same')(conv2)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.Activation('relu')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = layers.Dropout(0.3)(pool2)  # Add dropout

    # Bottleneck
    conv3 = layers.Conv2D(256, 3, padding='same')(pool2)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.Activation('relu')(conv3)
    conv3 = layers.Conv2D(256, 3, padding='same')(conv3)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.Activation('relu')(conv3)
    conv3 = layers.Dropout(0.4)(conv3)  # Add dropout

    # Decoder
    up1 = layers.UpSampling2D(size=(2, 2))(conv3)
    up1 = layers.Conv2D(128, 2, padding='same')(up1)
    up1 = layers.BatchNormalization()(up1)
    up1 = layers.Activation('relu')(up1)
    merge1 = layers.concatenate([conv2, up1], axis=3)
    merge1 = layers.Dropout(0.3)(merge1)  # Add dropout

    conv4 = layers.Conv2D(128, 3, padding='same')(merge1)
    conv4 = layers.BatchNormalization()(conv4)
    conv4 = layers.Activation('relu')(conv4)
    conv4 = layers.Conv2D(128, 3, padding='same')(conv4)
    conv4 = layers.BatchNormalization()(conv4)
    conv4 = layers.Activation('relu')(conv4)

    up2 = layers.UpSampling2D(size=(2, 2))(conv4)
    up2 = layers.Conv2D(64, 2, padding='same')(up2)
    up2 = layers.BatchNormalization()(up2)
    up2 = layers.Activation('relu')(up2)
    merge2 = layers.concatenate([conv1, up2], axis=3)
    merge2 = layers.Dropout(0.25)(merge2)  # Add dropout

    conv5 = layers.Conv2D(64, 3, padding='same')(merge2)
    conv5 = layers.BatchNormalization()(conv5)
    conv5 = layers.Activation('relu')(conv5)
    conv5 = layers.Conv2D(64, 3, padding='same')(conv5)
    conv5 = layers.BatchNormalization()(conv5)
    conv5 = layers.Activation('relu')(conv5)

    # Changed output to 2 channels (binary segmentation) with sigmoid activation
    outputs = layers.Conv2D(2, 1, activation='sigmoid')(conv5)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    return model


# Class mapping
CLASS_NAMES = {
    0: 'airplane',
    1: 'automobile',
    2: 'truck'
}

@st.cache_resource
def load_models():
    """Load and cache the models"""
    # Load classification model
    classification_model = tf.keras.models.load_model('best_classifier_new.keras')
    
    # Create and load segmentation model
    segmentation_model = unet_model((IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS))
    segmentation_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=2)]
    )
    segmentation_model.load_weights('seg_best_model_50_epochs.keras')
    
    return classification_model, segmentation_model

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

def main():
    st.title("Vehicle Image Classifier")
    st.write("Upload an image of an airplane, automobile, or truck to classify it!")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=["jpg", "jpeg", "png"]
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        
        # Process image when button is clicked
        if st.button("Classify Image"):
            with st.spinner("Processing..."):
                try:
                    # Load models
                    classification_model, segmentation_model = load_models()
                    
                    # Process image
                    image = process_image(uploaded_file)
                    
                    # Generate mask
                    mask_prediction = segmentation_model.predict(
                        np.expand_dims(image, 0),
                        verbose=0
                    )
                    mask = (np.argmax(mask_prediction[0], axis=-1) * 255).astype(np.uint8)
                    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
                    mask = cv2.resize(mask, (IMG_WIDTH, IMG_HEIGHT))
                    mask = mask.astype('float32') / 255.0
                    
                    # Combine image and mask
                    combined = np.concatenate([image, mask], axis=-1)
                    combined = np.expand_dims(combined, 0)
                    
                    # Make prediction
                    prediction = classification_model.predict(combined, verbose=0)
                    predicted_class = np.argmax(prediction)
                    confidence = prediction[0][predicted_class]
                    
                    # Display results
                    st.success(f"Predicted class: {CLASS_NAMES[predicted_class]}")
                    st.progress(float(confidence))
                    st.write(f"Confidence: {confidence:.2%}")
                    
                    # Display segmentation mask
                    st.subheader("Segmentation Mask")
                    st.image(mask, caption="Generated Mask", use_column_width=True)
                    
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    main()