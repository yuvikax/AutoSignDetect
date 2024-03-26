import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import os
import random
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense, Lambda
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.layers import ZeroPadding2D


# Define the Euclidean distance function
def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

# Define the base network for the Siamese Network
def create_base_network_signet(input_shape):
    '''Base Siamese Network'''

    seq = Sequential()
    seq.add(Conv2D(96, (11, 11), activation='relu', strides=4, input_shape=input_shape,
                   kernel_initializer='glorot_uniform', data_format='channels_last'))
    seq.add(BatchNormalization(axis=-1, momentum=0.9))
    seq.add(MaxPooling2D((3, 3), strides=(2, 2)))
    seq.add(ZeroPadding2D((2, 2)))

    seq.add(Conv2D(256, (5, 5), activation='relu', kernel_initializer='glorot_uniform', data_format='channels_last'))
    seq.add(BatchNormalization(axis=-1, momentum=0.9))
    seq.add(MaxPooling2D((3, 3), strides=(2, 2)))
    seq.add(Dropout(0.3))
    seq.add(ZeroPadding2D((1, 1)))

    seq.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='glorot_uniform', data_format='channels_last'))
    seq.add(MaxPooling2D((3, 3), strides=(2, 2)))
    seq.add(Dropout(0.3))

    seq.add(Flatten())
    seq.add(Dense(1024, activation='relu', kernel_regularizer=l2(0.0005), kernel_initializer='glorot_uniform'))
    seq.add(Dropout(0.5))

    seq.add(Dense(128, activation='relu', kernel_regularizer=l2(0.0005), kernel_initializer='glorot_uniform'))

    return seq

# Streamlit App Interface
st.title("Signature Verification")

# Input for User ID
user_id = st.text_input("Enter User ID (001 to 050):")

# Initialize the Siamese Network
input_shape = (155, 220, 1)
input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)
base_network = create_base_network_signet(input_shape)
processed_a = base_network(input_a)
processed_b = base_network(input_b)
distance = Lambda(euclidean_distance)([processed_a, processed_b])
model = Model(inputs=[input_a, input_b], outputs=distance)

# Load the saved weights
MODEL_PATH = "C:/Users/hittu/Downloads/ml_project/extracted_files/BHSig260/Weights_1/signet-bhsig260-full-xavier-007.h5"
model.load_weights(MODEL_PATH)

# Image processing function
def process_image(img):
    img_h, img_w = 155, 220
    img = cv2.resize(img, (img_w, img_h))
    img = np.array(img, dtype=np.float64)
    img /= 255.0
    img = img[..., np.newaxis]
    return img

# Function to predict the signature match
def predict_signature(model, img1, img2):
    threshold = 0.21031622775713912 # Threshold for Euclidean distance
    pred = model.predict([np.array([img1]), np.array([img2])])
    return "Genuine" if pred[0][0] < threshold else "Forged", pred[0][0]

# Image upload feature for signature verification
uploaded_file = st.file_uploader("Choose a Signature File to Verify", type=["tif", "jpg", "png"])

if uploaded_file and user_id:
    # Directory containing the user's genuine signatures
    user_dir_path = f"C:/Users/hittu/Downloads/ml_project/extracted_files/BHSig260/Hindi/{user_id}"
    if os.path.exists(user_dir_path):
        # List all genuine signature files for the user
        genuine_signatures = [file for file in os.listdir(user_dir_path) if 'G' in file and file.endswith('.tif')]
        if genuine_signatures:
            # Randomly select a genuine signature
            selected_signature = random.choice(genuine_signatures)
            genuine_signature_path = os.path.join(user_dir_path, selected_signature)

            # Load the genuine signature
            genuine_signature = cv2.imread(genuine_signature_path, cv2.IMREAD_GRAYSCALE)
            processed_genuine_img = process_image(genuine_signature)

            # Load and process the uploaded signature
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            uploaded_image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
            processed_uploaded_img = process_image(uploaded_image)

            # Predict if the uploaded signature matches the genuine signature
            prediction, score = predict_signature(model, processed_genuine_img, processed_uploaded_img)
            st.write(f"Prediction: {prediction}")
            st.write(f"Score: {score:.3f}")
            
            # Display the images
            st.image(genuine_signature, caption="Genuine Signature", use_column_width=True)
            st.image(uploaded_image, caption="Uploaded Signature", use_column_width=True)
        else:
            st.error(f"No genuine signatures found for User ID {user_id}.")
    else:
        st.error(f"Directory for User ID {user_id} does not exist. Please check the User ID and try again.")
else:
    if not user_id:
        st.warning("Please enter a User ID.")
    if not uploaded_file:
        st.warning("Please upload a signature image for verification.")
