import tensorflow as tf
import cv2
import os
import numpy as np


# Ścieżka do folderu z obrazami
folder_path = "C:/RL_DSJ2/cyfry"

# Rozmiar obrazów
image_width = 16
image_height = 20

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



# Konwersja list do tablic NumPy
print(len(images))
print(len(images[0]))
print(len(images[0][0]))

images = np.array(images)  # Dodanie wymiaru kanału
labels = np.array(labels)

print(images.shape)
print(labels)

# for i in range(11):
#     plt.imshow(images[i]*255, cmap='gray')
#     plt.title("Wyświetlony obraz")
#     plt.axis('off')  # Wyłączenie osi
#     plt.show()
#


# model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(20, 16, 1)),
#                                     tf.keras.layers.MaxPooling2D(2, 2),
#                                     tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
#                                     tf.keras.layers.MaxPooling2D(2, 2),
#                                     tf.keras.layers.Flatten(),
#                                     #The same 128 dense layers, and 10 output layers as in the pre-convolution example:
#                                     tf.keras.layers.Dense(128, activation='relu'),
#                                     tf.keras.layers.Dense(11, activation='softmax')
#                                     ])


model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(256, activation='relu'),
                                    tf.keras.layers.Dense(128, activation='relu'),
                                    tf.keras.layers.Dense(11, activation='softmax')])


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


model.fit(images, labels, epochs=100)



classifications = model.predict(images)

predicted_classes = np.argmax(classifications, axis=1)

print(classifications)
print(labels)
print(predicted_classes)


model.save('model_cyfry.keras')




