import time
import win32api
import win32con
from mss import mss
import pyautogui
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
from stable_baselines3 import DQN, PPO
from Rozpoznawanie_Wiatru import RozpoznawanieWiatru
from Rozpoznawanie_Liczb import RozpoznawanieLiczb

class WindEnv(gym.Env):
    def __init__(self, center_x, center_y):
        super(WindEnv, self).__init__()
        # Definiujemy przestrzeń akcji: 4 akcje (w górę, w dół, kliknięcie, nic)
        self.action_space = spaces.Discrete(4)
        # Przestrzeń obserwacji: obraz w skali szarości 200x200
        self.observation_space = spaces.Box(low=0, high=255, shape=(1, 200, 200), dtype=np.uint8)
        # Modele
        self.Rozpoznawanie_wiatru = RozpoznawanieWiatru()
        self.Rozpoznawanie_liczb = RozpoznawanieLiczb()
        # Lokacje obrazu
        self.cap = mss()
        self.dict_windows = {"jumper_observation": {"top": center_y - 100, "left": center_x - 100, "width": 200,
                                                    "height": 200},
                             "wind_direction_observation": {"top": center_y - 178, "left": center_x + 260, "width": 45,
                                                            "height": 29},
                             "wind_speed_observation": {"top": center_y - 150, "left": center_x + 264, "width": 40,
                                                        "height": 17},
                             "jump_length_observation": {"top": center_y + 190, "left": center_x - 25, "width": 65,
                                                         "height": 20},
                             "score_observation": {"top": center_y + 190, "left": center_x + 160, "width": 110,
                                                   "height": 20}}
        # Stany
        # 0- dojazd do progu, 1- lot, 2-ladowanie
        self.state = 0
        self.slownik = {0: "dojazd do progu",
                        1: "lot",
                        2: "ladowanie"}
        self.total_reward = 0
        self.max_score = 0
        self.max_jump = 0

    def step(self, action):
        # Definiowanie warunków zakończenia
        truncated = False
        terminated = self._check_done_condition("jump_length_observation")
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
            while not self._check_done_condition("score_observation"):
                pass

            frame_score = self.grab_frame("score_observation")
            reward = self.Rozpoznawanie_liczb.rozpoznawanie_cyfr(frame_score) + 168
            # TODO: Przy różnych skoczniach różny minus dla AUS to -168 ale dla innych nie
            # Jeżeli nie było dyskwalifikacji, czytanie długości skoku
            if reward != 0:
                frame_len = self.grab_frame("jump_length_observation")
                jump_len = self.Rozpoznawanie_liczb.rozpoznawanie_cyfr(frame_len)
            else:
                jump_len = 0

            # TODO: Można pomyśleć czy da się znaleźć jakieś konkretne wartości kar niż takie z czapy
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
            print(f"Największy wynik to: {self.max_score}")
            print(f"Najdłuższy skok: {self.max_jump}")
        # Zaktualizowanie obserwacji
        new_observation = self._get_observation()
        # Info
        info = {}
        return new_observation, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        # Resetujemy stan środowiska
        self.state = 0
        self.total_reward = 0
        self.click()
        print("Menu")
        time.sleep(1)
        # TODO: tu można wrzucić jakąś funkcję zmiany skoczni w przyszłości
        self.click()
        time.sleep(0.5)
        # Czekanie aż załaduje się gra
        while not self._check_done_condition("wind_direction_observation"):
            pass
        time.sleep(0.5)
        info = {}
        # TODO: Dodać tutaj zapisywanie danych o wietrze
        # wind_speed = np.array(self.cap.grab(self.wind_speed_observation))[:, :, :3]
        # wind_direction = np.array(self.cap.grab(self.wind_direction_observation))[:, :, :3]
        self.click()
        return self._get_observation(), info

    def render(self):
        # Renderowanie środowiska (opcjonalnie)
        pass

    def _get_observation(self):
        # Screeny zawodnika
        jumper = self.grab_frame("jumper_observation")
        jumper_done = self.odczyt_zawodnika(jumper)
        # TODO: Można dodać tutaj aktualizację danych wiatru ale najpewniej co któryś krok, żeby nie z każdym
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
        # TODO: Znany trick to zrobienie szybkiego ruchu myszką w dół i w góre przy wybiciu można to dodać
        # pyautogui.move(0, 30)
        time.sleep(0.10)
        # pyautogui.move(0, -30)
        # time.sleep(0.15)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0)
        win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP, 0, 0)
        return 1

    def click(self):
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0)
        time.sleep(0.1)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0)

    def grab_frame(self, name):
        return np.array(self.cap.grab(self.dict_windows[name]))

    def _check_done_condition(self, name):
        frame = self.grab_frame(name)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # wykrycie jasnych pixeli > prog
        return bool(np.any(gray > 80))

    # TODO: Zobaczymy czy będę z tego korzystał
    def odczyt_zawodnika(self, frame):
        # Konwersja do skali szarości
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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

