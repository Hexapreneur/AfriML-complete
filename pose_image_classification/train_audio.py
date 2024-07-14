import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import librosa
import csv


def create_dataset_audio(dataset_path):
    try:
        csv_file = os.path.join(dataset_path, 'data.csv')
        csv_writer = open(csv_file, 'w', newline='')
        writer = csv.writer(csv_writer)

        for root, _, files in os.walk(dataset_path):
            for file in files:
                if file.endswith('.csv'):
                    continue
                if file.endswith('.csv#'):
                    continue
                if file.endswith('.txt'):
                    continue
                if file.endswith('.txt#'):
                    continue
                animal = os.path.basename(root)
                # animal = '_'.join(animal.split())
                print(root, file)
                sound_name = os.path.join(root, file)
                y, sr = librosa.load(sound_name, mono=True, duration=30)
                chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
                rmse = librosa.feature.rms(y=y)
                spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
                spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
                rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
                zcr = librosa.feature.zero_crossing_rate(y)
                mfcc = librosa.feature.mfcc(y=y, sr=sr)
                to_append = f'{file} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'

                for e in mfcc:
                    to_append += f' {np.mean(e)}'
                # to_append += f' {animal}'
                to_append_array = to_append.split()
                to_append_array.append(animal)
                writer.writerow(to_append_array)

        csv_writer.close()
    except Exception as e:
        print(e)


def main_audio_train(dataset_path):
    try:
        csv_file = os.path.join(dataset_path, 'data.csv')
        df = pd.read_csv(csv_file)
        df = df.sample(frac=1)
        class_list = df.iloc[:, -1]
        c = list(class_list)
        uniques = (np.unique(c))
        
        indexes = [c.index(i) for i in uniques]
        encoder = LabelEncoder()
        y = encoder.fit_transform(class_list)
        labels = [y[i] for i in indexes]
        class_index = {}
        for (l, u) in zip(labels, uniques):
            class_index[l] = u
        f = open(os.path.join(dataset_path, "classes.txt"), "w")
        f.write(str(class_index))
        f.close()
        # print("y: ", y)
        input_parameters = df.iloc[:, 1:27]
        # print(input_parameters.shape)
        scaler = StandardScaler()
        X = scaler.fit_transform(np.array(input_parameters))
        # print("X:", X)

        # training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=True)
        # print(np.unique(y_train))
        # exit()

        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
            tf.keras.layers.Dropout(0.2),

            tf.keras.layers.Dense(256, activation='relu'),
            # keras.layers.Dropout(0.2),

            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),

            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),

            tf.keras.layers.Dense(len(uniques), activation='softmax'),
        ])

        print(model.summary())

        def trainModel(model, epochs, optimizer):
            batch_size = 4
            model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            return model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size)

        model_history = trainModel(model=model, epochs=100, optimizer='adam')
        # loss_train_curve = model_history.history["loss"]
        # loss_val_curve = model_history.history["val_loss"]
        # plt.plot(loss_train_curve, label="Train")
        # plt.plot(loss_val_curve, label="Validation")
        # plt.legend(loc='upper right')
        # plt.title("Loss")
        # plt.show()
        # #
        # acc_train_curve = model_history.history["accuracy"]
        # acc_val_curve = model_history.history["val_accuracy"]
        # plt.plot(acc_train_curve, label="Train")
        # plt.plot(acc_val_curve, label="Validation")
        # plt.legend(loc='lower right')
        # plt.title("Accuracy")
        # plt.show()
        # #
        #
        # df_test = pd.read_csv(csv_file)
        # df_test.head()
        # df_test = df_test.sample(frac=1)
        #
        # X_test = scaler.fit_transform(np.array(df_test.iloc[:, 1:27]))
        # X_label = np.array(df_test.iloc[:, -1])
        # # print(X_label)
        # # exit()
        # # print("X_test:", X_test)
        #
        # # generate predictions for samples
        # predictions = model.predict(X_test)
        # # print(predictions)
        #
        # # generate argmax for predictions
        # classes = np.argmax(predictions, axis=1)
        # # print(classes)
        #
        # # transform classes number into classes name
        # result = encoder.inverse_transform(classes)
        # # print(result)

        model.save(os.path.join('static', 'weights', 'audio_model.h5'))
        model_path = os.path.join('static', 'weights', 'audio_model.h5')
        return True, model_path
    except Exception as e:
        return False, str(e)



# if __name__ == '__main__':
#     main_audio_train('static/dataset')
