import os

import cv2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
import csv
import mediapipe as mp

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()


def train(dir_path):
    fieldnames = ['0x', '0y', '1x', '1y', '2x', '2y', '3x', '3y', '4x', '4y', '5x', '5y', '6x', '6y',
                  '7x', '7y', '8x', '8y', '9x', '9y', '10x', '10y', '11x', '11y', '12x', '12y', '13x',
                  '13y', '14x', '14y', '15x', '15y', '16x', '16y', '17x', '17y', '18x', '18y', '19x', '19y',
                  '20x', '20y', '21x', '21y', '22x', '22y', '23x', '23y', '24x', '24y', '25x', '25y', '26x', '26y',
                  '27x', '27y', '28x', '28y', '29x', '29y', '30x', '30y', '31x', '31y', '32x', '32y', 'output']

    with open(os.path.join('static', 'data.csv'), 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for r, d, f in os.walk(dir_path):
            for dr in d:
                dr_path = os.path.join(r, dr)
                files = os.listdir(dr_path)
                for file in files:
                    file_path = os.path.join(dr_path, file)
                    img = cv2.imread(file_path)
                    results = pose.process(img)
                    pose_points = {}
                    for i in range(33):
                        pose_points[f'{i}x'] = 1
                        pose_points[f'{i}y'] = 1

                    # Draw pose landmarks on the image
                    if results.pose_landmarks:
                        # Iterate over all landmarks
                        for idx, landmark in enumerate(results.pose_landmarks.landmark):
                            h, w, c = img.shape
                            cx, cy = int(landmark.x * w), int(landmark.y * h)
                            if 0 < cx < w and 0 < cy < h:
                                pose_points[f'{idx}x'] = cx
                                pose_points[f'{idx}y'] = cy
                    pose_points['output'] = dr
                    writer.writerow(pose_points)
    # Load your CSV data
    data = pd.read_csv(os.path.join('static', 'data.csv'))

    # Check for NaN values
    if data.isnull().sum().sum() > 0:
        raise ValueError("Dataset contains NaN values")

    # Split data into features and labels
    X = data.iloc[:, :-1].values  # Features
    y = data.iloc[:, -1].values   # Labels

    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    num_classes = len(label_encoder.classes_)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize features
    X_train_mean, X_train_std = X_train.mean(axis=0), X_train.std(axis=0)
    X_train = (X_train - X_train_mean) / (X_train_std + 1e-8)  # Add small epsilon to avoid division by zero
    X_test = (X_test - X_train_mean) / (X_train_std + 1e-8)

    # Define the model
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    # Compile the model with a lower learning rate
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)
    # Evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    model.save(os.path.join('static', 'final_model.h5'))
    return str(test_accuracy*100)
