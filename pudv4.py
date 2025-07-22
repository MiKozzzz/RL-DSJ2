import time
import win32api
import win32con
from mss import mss
from tensorflow.keras.models import load_model
import pyautogui
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import os
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common import env_checker
from matplotlib import pyplot as plt
from stable_baselines3 import DQN, PPO


class WindEnv(gym.Env):
    def __init__(self):
        super(WindEnv, self).__init__()

        # Definiujemy przestrzeń akcji: 4 akcje (w górę, w dół, kliknięcie, nic)
        self.action_space = spaces.Discrete(4)

        # Przestrzeń obserwacji: obraz w skali szarości 200x200
        self.observation_space = spaces.Box(low=0, high=255, shape=(1, 200, 200), dtype=np.uint8)

        # Lokacje obrazu
        self.cap = mss()
        # Obraz zawodnika
        self.jumper_observation = {"top": 110, "left": 215, "width": 200, "height": 200}
        # Kierunek wiatru
        self.wind_direction_observation = {"top": 36, "left": 580, "width": 50, "height": 30}
        # Sila wiatru
        self.wind_speed_observation = {"top": 64, "left": 580, "width": 50, "height": 17}
        # Długość skoku
        self.jump_length_observation = {"top": 402, "left": 300, "width": 62, "height": 20}
        # Warunek końca skoku
        self.done_observation = {"top": 402, "left": 300, "width": 80, "height": 20}
        # Wynik
        self.score_observation = {"top": 402, "left": 500, "width": 110, "height": 20}
        # Warunek pojawienia się wyniku
        self.done_score_observation = {"top": 402, "left": 450, "width": 150, "height": 20}

        # Stany
        # 0- dojazd do progu, 1- lot, 2-ladowanie
        self.state = 0
        self.liczba_klik = 0
        self.slownik = {0: "dojazd do progu",
                        1: "lot",
                        2: "ladowanie"}
        self.total_reward = 0
        self.max_score = 0
        self.max_jump = 0

    def step(self, action):
        # Definiowanie warunków zakończenia
        truncated = False
        terminated = self._check_done_condition()
        # Wykonujemy akcję, jeżeli warunek zakończenia nie jest spełniony
        if not terminated:
            reward = 0
            if action == 0:  # poruszanie myszką do góry
                self._move_mouse_up()
                if self.state == 1:
                    reward = 2
            elif action == 1:  # poruszanie myszką w dół
                self._move_mouse_down()
                if self.state == 1:
                    reward = 2
            elif action == 2:  # kliknięcie myszką
                self._click_mouse()
                self.liczba_klik += 1
                if self.state == 0:
                    reward = 1
                    self.state += 1
                elif self.state == 1:
                    reward = 2
                    self.state += 1

            elif action == 3:  # nic nie robienie
                if self.state == 1:
                    reward = 2
            # Sumowanie nagród
            self.total_reward += reward

        # Jeżeli warunek zakończenia spełniony
        else:
            print("koniec")
            # Czekanie aż pojawi się wynik za skok
            stop = True
            while stop:
                score = np.array(self.cap.grab(self.done_score_observation))[:, :, :3]
                gray_img = cv2.cvtColor(score, cv2.COLOR_BGR2GRAY)
                _, binary_image = cv2.threshold(gray_img, 130, 255, cv2.THRESH_BINARY)
                if np.sum(binary_image) > 0:
                    stop = False
            # Czytanie obrazu wyniku
            img = np.array(self.cap.grab(self.score_observation))[:, :, :3]
            reward = self.rozpoznawanie_cyfr(img) + 168
            # Jeżeli nie było dyskwalifikacji, czytanie długości skoku
            if reward != 0:
                img_len = np.array(self.cap.grab(self.jump_length_observation))[:, :, :3]
                jump_len = self.rozpoznawanie_cyfr(img_len)
            else:
                jump_len = 0

            # Kara za brak wybicia
            if self.state == 0:
                reward -= 400
            # Kara za brak lądowania
            if self.state == 1:
                reward -= 200
            # Sumowanie nagród
            self.total_reward += reward
            # Zapisywanie największego wyniku
            if self.total_reward > self.max_score:
                self.max_score = self.total_reward
            # Zapisywanie najdłuszego skoku
            if self.max_jump < jump_len:
                self.max_jump = jump_len
            # Wyświetlanie informacji
            print(f"Skonczył lot przy fazie: {self.slownik[self.state]}")
            print(f"Wynik za skok: {reward}")
            print(f"Zebrana nagroda: {self.total_reward}")
            print(f"Liczba click: {self.liczba_klik}")
            print(f"Największy wynik to: {self.max_score}")
            print(f"Najdłuższy skok: {self.max_jump}")


        # Zaktualizowanie obserwacji
        new_observation = self._get_observation()
        # Info
        info = {}
        return new_observation, reward, terminated, truncated, info

    def reset(self, seed=None):
        # Resetujemy stan środowiska
        time.sleep(1)
        self.state = 0
        self.total_reward = 0
        self.liczba_klik = 0
        self.click()
        print("Menu")
        time.sleep(1)
        self.click()
        time.sleep(1)
        # Czekanie aż załaduje się gra
        stop = True
        while stop:
            img = np.array(self.cap.grab(self.wind_direction_observation))[:, :, :3]
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, binary_image = cv2.threshold(gray_img, 80, 255, cv2.THRESH_BINARY)
            if np.sum(binary_image) > 0:
                stop = False
                print('juz')
        time.sleep(1)
        info = {}
        self.click()
        return self._get_observation(), info

    def render(self):
        # Renderowanie środowiska (opcjonalnie)
        pass

    def _get_observation(self):
        # Screeny zawodnika
        jumper = np.array(self.cap.grab(self.jumper_observation))[:, :, :3]
        jumper_done = self.odczyt_zawodnika(jumper)
        # wind_speed = np.array(self.cap.grab(self.wind_speed_observation))[:, :, :3]
        # wind_direction = np.array(self.cap.grab(self.wind_direction_observation))[:, :, :3]
        return jumper_done

    def _move_mouse_up(self):
        pyautogui.move(0, -3)
        return 1

    def _move_mouse_down(self):
        pyautogui.move(0, 3)
        return 1

    def _click_mouse(self):
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0)
        win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTDOWN, 0, 0)
        pyautogui.move(0, 30)
        time.sleep(0.10)
        pyautogui.move(0, -30)
        time.sleep(0.15)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0)
        win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP, 0, 0)
        return 1

    def click(self):
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0)
        time.sleep(0.1)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0)

    def _check_done_condition(self):
        # Czytanie obrazu, czy pojawiła się odległość skoku lub napis dyskfalifikacja
        score = np.array(self.cap.grab(self.done_observation))[:, :, :3]
        gray_img = cv2.cvtColor(score, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(gray_img, 130, 255, cv2.THRESH_BINARY)
        # Suma pixeli większa od 0 oznacza, że pojawił się napis (długość skoku lub napis dyskwalifikacji
        done = False
        if np.sum(binary_image) > 0:
            done = True
        return done

    def Powiekszanie_macierzy(self, original_matrix, target_rows, target_cols):
        # Oblicza różnicę w liczbie wierszy i kolumn
        diff_rows = target_rows - original_matrix.shape[0]
        diff_cols = target_cols - original_matrix.shape[1]

        # Oblicza liczbę wierszy i kolumn, które zostaną dodane na górze, na dole, z lewej i z prawej strony
        top_rows = diff_rows // 2
        left_cols = diff_cols // 2
        # Stwarza nową macierz o docelowym rozmiarze, wypełnioną zerami
        new_matrix = np.zeros((target_rows, target_cols))
        # Umieszcza pierwotną macierz na odpowiednim miejscu w nowej macierzy
        new_matrix[top_rows:top_rows + original_matrix.shape[0],
        left_cols:left_cols + original_matrix.shape[1]] = original_matrix

        return new_matrix

    def Segmentacjaliczb(self, image):
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

    def wynik(self, asd):
        wynik = 0
        lenght = len(asd)
        if asd[0] == 10:
            for i in range(1, lenght - 2):
                wynik += asd[i] * 10 ** (lenght - i - 3)
            wynik += asd[-1] / 10
            wynik = wynik * -1
        else:
            for i in range(lenght - 2):
                wynik += asd[i] * 10 ** (lenght - i - 3)
            wynik += asd[-1] / 10
        return wynik

    def rozpoznawanie_wiatru(self, img):
        # Konwersja do skali szarości
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(gray_img, 80, 255, cv2.THRESH_BINARY)
        gotowe = binary_image.reshape(1, 30, 50)
        classifications = model_wiatr.predict(gotowe)
        predicted_classes = np.argmax(classifications, axis=1)
        return predicted_classes[0]

    def rozpoznawanie_cyfr(self, img):
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(gray_img, 130, 255, cv2.THRESH_BINARY)
        segments = self.Segmentacjaliczb(binary_image)
        # Nowy rozmiar macierzy
        image_width = 16
        image_height = 20
        lista_liczb = np.zeros((len(segments), 20, 16))
        # Zwiększenie macierzy
        for i in range(len(segments)):
            increased_matrix = self.Powiekszanie_macierzy(segments[i] / 255, image_height, image_width)
            lista_liczb[i] = lista_liczb[i] + increased_matrix

        # Jeżeli brak elementów to wynik -168.0, ponieważ widzocznie była to dyskwalifikacja
        if len(lista_liczb) == 0:
            return -168.0
        # Rozpoznawanie liczb
        else:
            classifications = model_cyfr.predict(lista_liczb)
            predicted_classes = np.argmax(classifications, axis=1)
            return self.wynik(predicted_classes)

    def odczyt_zawodnika(self, img):
        # Konwersja do skali szarości
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        value_list = [76, 85, 73, 109, 42, 55, 65, 52, 108, 231, 92, 24]
        # 76 kask
        # 85 Gorna narta (Uwaga tlo)
        # 73 rekawiczka przod
        # 109 dolna narta
        # 42 nogawka tyl ciemny
        # 55 nogawka tyl jasniejszy
        # 65 nogawka przod jasniejszy
        # 52 tylna reka
        # 108 przod reka
        # 231 plastron klata
        # 92 Cien
        # 24 rekawiczka tył

        # Wartość zastępcza
        replacement_value = 255
        # Konwersja listy wartości i wartości zastępczej na numpy array
        value_list_np = np.array(value_list)
        replacement_value_np = np.uint8(replacement_value)
        # Tworzenie maski dla pikseli, które mają pozostać niezmienione
        mask = np.isin(gray_img, value_list_np)
        # Tworzenie nowego obrazu z wartością zastępczą
        filtered_image = np.full(gray_img.shape, replacement_value_np, dtype=np.uint8)
        # Nakładanie oryginalnych pikseli na maskę
        filtered_image[mask] = gray_img[mask]
        channel = np.reshape(filtered_image, (1, 200, 200))
        return channel


# BOT
model_cyfr = load_model("best_cyfry.keras")
model_wiatr = load_model("best_wiatr.keras")
model_kat = load_model("model_kat.keras")

# DANE WEJSCIOWE
# OBRAZ
# STANY
# WYBICIE, LOT, LADOWANIE
# AKCJE
# klik, ruch myszki w gore, ruch myszki w dol, czekanie


# Inicjalizacja środowiska
env = WindEnv()

# model = DQN("CnnPolicy", env, verbose=1, buffer_size=500000, learning_starts=1000)
#
# model = PPO("CnnPolicy", env, verbose=1)
# model.learn(total_timesteps=40000)
# model.save("ppo_DSJ")

# model = PPO.load("ppo_DSJ", env=env)
# model.learn(total_timesteps=10000)
# model.save("ppo_DSJ")
#
model = PPO.load("ppo_DSJ6_10", env=env)

obs, info = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        print("Uczenie zakończone")
        obs, info = env.reset()



# env_checker.check_env(env)
#
# plt.imshow(cv2.cvtColor(env._get_observation()[0], cv2.COLOR_BGR2RGB))
# plt.show()

# for episode in range(10):
#     obs, info = env.reset(seed=0)
#     done = False
#     total_reward = 0
#
#     while not done:
#         obs, reward, done, tru, info = env.step(env.action_space.sample())
#         total_reward += reward
#     print(f"Total reward for episode {episode} is {total_reward}")