import tensorflow as tf
import time
import win32api
import win32con
import mss
from sklearn.model_selection import train_test_split
import mss.tools
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import load_model


# Ścieżka do folderu z obrazami
folder_path = "C:/RL_DSJ2/wiatr"

# Rozmiar obrazów
image_width = 50
image_height = 30

# Listy do przechowywania danych i etykiet
images = []
labels = []

# Iterowanie po plikach w folderze
for filename in os.listdir(folder_path):
    if filename.endswith(".png"):  # Możesz dostosować rozszerzenie plików
        image_path = os.path.join(folder_path, filename)

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # Sprawdzenie, czy obraz został poprawnie wczytany
        if img is not None:
            # Skalowanie obrazu
            img = cv2.resize(img, (image_width, image_height))

            # Normalizacja obrazu
            img = img / 255.0

            # Dodanie obrazu do listy
            images.append(img)

            # Wyciągnięcie etykiety z nazwy pliku (zakładając, że nazwa pliku zaczyna się od cyfry reprezentującej etykietę)
            # label = os.path.splitext(filename)[0]
            base_filename = os.path.splitext(filename)[0]  # Usunięcie rozszerzenia pliku
            label = base_filename.split('_')[0]  # Zakładając, że etykieta jest przed pierwszym znakiem podkreślenia
            if label is not None:
                labels.append(int(label))
            else:
                print(f"Etykieta dla pliku {filename} nie jest zdefiniowana.")
        else:
            print(f"Nie udało się wczytać obrazu: {filename}")

print(len(images))
print(len(images[0]))
print(len(images[0][0]))

images = np.array(images)  # Dodanie wymiaru kanału
labels = np.array(labels)

print(images.shape)
print(labels)


model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(256, activation='relu'),
                                    tf.keras.layers.Dense(128, activation='relu'),
                                    tf.keras.layers.Dense(11, activation='softmax')])


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


model.fit(images, labels, epochs=50)



classifications = model.predict(images)

predicted_classes = np.argmax(classifications, axis=1)

print(classifications)
print(labels)
print(predicted_classes)


model.save('model_wiatr.keras')

