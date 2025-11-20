import numpy as np
import cv2
from .Model_Rozpoznawania_kierunku_wiatru import SimpleCNN_Wind
import torch
import time

class RozpoznawanieWiatru:
    def __init__(self):
        self.model = SimpleCNN_Wind()
        # Wagi są w pliku o nazwie: model_cyfr_weights.pth
        state = torch.load("../models/vision/model_wiatru_weights.pth", weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()

    def rozpoznawanie_wiatru(self, img):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray_img, 80, 255, cv2.THRESH_BINARY)

        cropped = self.crop_digit(binary)
        resized = cv2.resize(cropped, (28, 28))
        resized = resized.astype("float32") / 255.0
        resized = resized.reshape((28, 28, 1))

        resized = np.transpose(resized, (2, 0, 1))  # HWC -> CHW
        resized = np.expand_dims(resized, axis=0)  # batch=1
        resized = torch.tensor(resized, dtype=torch.float32)

        with torch.no_grad():
            output = self.model(resized)
            pred = torch.argmax(output, dim=1)
        return pred.item()

    def crop_digit(self, image):
        """
        Usuwa czarne tło wokół cyfry i zwraca wycięty obraz.
        Zakłada, że cyfra jest jaśniejsza niż tło.
        """

        # Jeśli obraz jest kolorowy → konwertujemy
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Znajdź nie-czarne piksele (cyfra)
        coords = cv2.findNonZero(image)
        if coords is None:
            return image  # puste zdjęcie, nic nie wycinamy
        # Pobranie bounding-boxa cyfry
        x, y, w, h = cv2.boundingRect(coords)
        # Zwróć wycięty fragment
        cropped = image[y:y + h, x:x + w]
        return cropped


if __name__ == "__main__":
    name = "wiatr.png"
    img = cv2.imread(name, cv2.IMREAD_COLOR)
    start = time.time()
    rozpoznana_kierunek = RozpoznawanieWiatru().rozpoznawanie_wiatru(img)
    end = time.time()
    print(f"Czas rozpoznawania liczby: {end-start}")
    print(f"Rozpoznana kierunek: {rozpoznana_kierunek}")