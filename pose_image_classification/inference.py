import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
import mediapipe as mp
import cv2
import os

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()


def test_image(csv_path, image):
    data = pd.read_csv(csv_path)
    classes_names = np.unique(data['output'])
    loaded_model = tf.keras.models.load_model(os.path.join('static', 'final_model.h5'))
    # Check for NaN values
    if data.isnull().sum().sum() > 0:
        raise ValueError("Dataset contains NaN values")

    # Split data into features and labels
    X = data.iloc[:, :-1].values  # Features
    y = data.iloc[:, -1].values  # Labels

    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    num_classes = len(label_encoder.classes_)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize features
    X_train_mean, X_train_std = X_train.mean(axis=0), X_train.std(axis=0)

    frame = image.copy()
    results = pose.process(frame)
    your_data = []

    # Draw pose landmarks on the image
    if results.pose_landmarks:
        # Iterate over all landmarks
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            h, w, c = frame.shape
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            if 0 < cx < w and 0 < cy < h:
                your_data.append(cx)
                your_data.append(cy)

            else:
                your_data.append(1)
                your_data.append(1)

    new_row = np.array([your_data])  # Replace 'your_data' with the actual data for the new row
    if len(your_data) <= 0:
        return 'Pose Not Found', [], list(classes_names)
    # Normalize the new row of features using the same mean and standard deviation as used for training data
    normalized_new_row = (new_row - X_train_mean) / (X_train_std + 1e-8)  # Using the same mean and std as training data

    # Perform inference
    prediction = loaded_model.predict(normalized_new_row)

    # Decode prediction (if needed)
    decoded_prediction = label_encoder.inverse_transform(np.argmax(prediction, axis=1))

    return decoded_prediction[0], list(prediction[0]), list(classes_names)
