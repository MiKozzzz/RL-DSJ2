import numpy as np
import cv2
from Model_Rozpoznawanie_cyfr import SimpleCNN_Number
import torch
import time

class RozpoznawanieLiczb:
    def __init__(self):
        self.model = SimpleCNN_Number()
        # Wagi są w pliku o nazwie: model_cyfr_weights.pth
        state = torch.load("model_cyfr_weights.pth", weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()

    def rozpoznawanie_cyfr(self, img):
        # Filtr
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray_img, 135, 255, cv2.THRESH_BINARY)

        # Segmentacja
        segments = self.segmentacjaliczb(binary)
        # Resize
        processed_segments = []
        for i in range(len(segments)):
            resized = cv2.resize(segments[i], (28, 28))
            resized = resized.astype("float32") / 255.0
            resized = resized.reshape((28, 28, 1))
            processed_segments.append(resized)

        # Jeżeli brak elementów to wynik -168.0, ponieważ widzocznie była to dyskwalifikacja
        if len(processed_segments) == 0:
            return -168.0

        # Rozpoznawanie liczb
        else:
            X = np.array(processed_segments)
            classifications = self.predict(X)
            predicted_classes = np.argmax(classifications, axis=1)
            return self.wynik(predicted_classes)

    def predict(self, segments):
        """
        segments: numpy array (n_segments, 28, 28, 1)
        zwraca: numpy array z predykcjami dla wszystkich segmentów
        """
        # Konwersja do tensorów i zmiana kształtu na [batch, channels, H, W]
        segment_tensor = torch.from_numpy(segments).permute(0, 3, 1, 2).float()  # (n,1,28,28)
        with torch.no_grad():
            output = self.model(segment_tensor)
            return output.numpy()  # zwraca tablicę z logitami

    def segmentacjaliczb(self, image):
        # Znajdowanie konturów na obrazie
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        digits = []
        # Iteracja po konturach
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            digit = image[y:y + h, x:x + w]
            # Dodawanie wyciętej cyfry i jej pozycji x do listy
            digits.append((digit, x))
        # Sortowanie cyfr według pozycji x
        sorted_digits = sorted(digits, key=lambda item: item[1])
        # Zwracanie tylko wyciętych cyfr, bez pozycji x
        return [digit for digit, x in sorted_digits]

    def wynik(self, predicted_classes):
        """
        predicted_classes: tablica cyfr (0-9), 10 oznacza kropkę dziesiętną,
        zakładamy, że jeśli pierwsza cyfra to 10, to liczba jest ujemna
        """
        predicted_classes = np.array(predicted_classes)
        wynik = 0
        n = 0
        if predicted_classes[0] == 10:
            predicted_classes = predicted_classes[1:] * -1
        length = len(predicted_classes)
        for i in range(length):
            if predicted_classes[i] != 10 and predicted_classes[i] != -10:
                wynik = wynik * 10
                wynik += predicted_classes[i]
            else:
                n = i + 1
        wynik = wynik / 10 ** (length - n)
        return wynik


if __name__ == "__main__":
    name = "probka.png"
    img = cv2.imread(name, cv2.IMREAD_COLOR)

    start = time.time()
    rozpoznana_liczba = RozpoznawanieLiczb().rozpoznawanie_cyfr(img)
    end = time.time()
    print(f"Czas rozpoznawania liczby: {end-start}")
    print(f"Rozpoznana liczba: {rozpoznana_liczba}")

    # print(RozpoznawanieLiczb().wynik([3, 3, 5, 3, 10, 2, 3, 4, 5, 7]))

