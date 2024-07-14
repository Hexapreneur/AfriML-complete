import librosa
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import csv
import pandas as pd


def get_data2audio(sound_name):
    y, sr = librosa.load(sound_name, mono=True)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    rmse = librosa.feature.rms(y=y)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    to_append = f'{np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'

    for e in mfcc:
        to_append += f' {np.mean(e)}'
    to_append = to_append.split()
    to_append = [float(d) for d in to_append]
    return to_append


def inference_audio_file(dataset_path, model_path, input_file):
    scaler = StandardScaler()
    model = tf.keras.models.load_model(model_path)

    classes = eval(open(os.path.join(dataset_path, 'classes.txt'), 'r').readline())

    csv_path = os.path.join(dataset_path, 'data.csv')

    csv_data = pd.read_csv(csv_path, header=None)
    X_test_main = []
    for index, row in csv_data.iloc[:, 1:-1].iterrows():
        X_test_main.append(list(row))

    data_ls = get_data2audio(input_file)

    X_test_main.append(data_ls)
    X = scaler.fit_transform(np.array(X_test_main))
    predictions = model.predict(X)
    pred_classes = np.argmax(predictions, axis=1)

    main_class = classes[pred_classes[-1]]
    return True, main_class, list(predictions[-1]), classes
    # for row in csv_data.iloc[:, 1:-1]:
    #     print(row)


# inference_audio_file('static/dataset', 'static/dataset/class2/49001ch1_44kHz_12.wav')
