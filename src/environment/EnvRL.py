import time
import win32api
import win32con
import win32gui
import subprocess
from mss import mss
import pyautogui
import gymnasium as gym
import numpy as np
import cv2
from src.vision.Rozpoznawanie_Wiatru import RozpoznawanieWiatru
from src.vision.Rozpoznawanie_Liczb import RozpoznawanieLiczb
from src.utils.Cursor import Cursor

class DSJEnv(gym.Env):
    def __init__(self):
        super(DSJEnv, self).__init__()
        # Definiujemy przestrzeń akcji: 4 akcje (w górę, w dół, kliknięcie, nic)
        self.action_space = gym.spaces.Discrete(4)
        # TODO: Można pomyśleć czy nie dodać do obserwacji w jakiej fazie się znajduje (prog, lot, ladowanie)
        # Przestrzeń obserwacji: obraz w skali szarości 200x200, siła, kierunek wiatru
        self.observation_space = gym.spaces.Dict({
            "frame": gym.spaces.Box(
                low=0, high=255, shape=(200, 200, 1), dtype=np.uint8
            ),
            "wind_direction": gym.spaces.Box(low=0, high=7, shape=(1,), dtype=np.int64),
            "wind_strength": gym.spaces.Box(
                low=0, high=5.0, shape=(1,), dtype=np.float32
            ),  # ciągła wartość
        })
        # Stany
        # 0- dojazd do progu, 1- lot, 2-ladowanie
        self.wind_speed = 0
        self.wind_direction = 0
        self.state = 0
        self.slownik = {0: "dojazd do progu",
                        1: "lot",
                        2: "ladowanie"}
        self.epiosde = 0
        self.total_reward = 0
        self.max_score = 0
        self.max_jump = 0

        self.Rozpoznawanie_wiatru = RozpoznawanieWiatru()
        self.Rozpoznawanie_liczb = RozpoznawanieLiczb()
        self.Cursor_game = Cursor(640, 400)
        self.center_x = None
        self.center_y = None
        self.hwnd = None
        self.dict_windows = None
        # Lokacje obrazu
        self.cap = mss()

        self._player_vals = np.array([76, 85, 73, 109, 42, 55, 65, 52, 108, 231, 92, 24], dtype=np.uint8)
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
        self._player_lut = np.zeros(256, dtype=np.uint8)
        self._player_lut[self._player_vals] = 1

        self.TIMEOUT_SCORE = 10  # sekund
        self.TIMEOUT_LOAD = 10  # sekund
        self.SLEEP_INTERVAL = 0.02  # sekund


    def initialize_game(self, path_game, window_game_name):
        subprocess.Popen([path_game])
        time.sleep(4)
        # Włączenie trybu oknowego
        pyautogui.hotkey('alt', 'enter')
        time.sleep(1)
        self.hwnd = win32gui.FindWindow(None, window_game_name)
        # Liczenie współrzędnych środka okna DSJ
        rect = win32gui.GetWindowRect(self.hwnd)
        # Środek okna DSJ 640x400. Teoretycznie okno większę ale to przez pasek menu, sama gra ma własciwie tyle
        # print(rect)
        left, top, right, bottom = rect
        print("Rozdziałka gry: ", right - left, bottom - top)
        # Środek okna
        self.center_x = int((right + left) / 2)
        self.center_y = int((bottom + top) / 2)
        # print(self.center_x, self.center_y)
        self.dict_windows = {
            "jumper_observation": {"top": self.center_y - 100, "left": self.center_x - 100, "width": 200,
                                   "height": 200},
            "wind_direction_observation": {"top": self.center_y - 178, "left": self.center_x + 260, "width": 45,
                                           "height": 29},
            "wind_speed_observation": {"top": self.center_y - 150, "left": self.center_x + 264, "width": 40,
                                       "height": 17},
            "jump_length_observation": {"top": self.center_y + 190, "left": self.center_x - 25, "width": 65,
                                        "height": 20},
            "score_observation": {"top": self.center_y + 190, "left": self.center_x + 160, "width": 110,
                                  "height": 20}}
        # Kliknięcie w okno gry DSJ
        pyautogui.moveTo(self.center_x, self.center_y)
        # Kursor dla okna DSJ 640x400
        self.Cursor_game.click()
        time.sleep(6)
        self.Cursor_game.move_to(160, 210)
        self.Cursor_game.click()
        self.Cursor_game.move_to(440, 310)


    def quit_game(self):
        win32gui.PostMessage(self.hwnd, win32con.WM_CLOSE, 0, 0)

    def step(self, action):
        truncated = False
        terminated = self._check_done_condition("jump_length_observation")
        # 1. Wykonaj akcję (jeśli skok jeszcze trwa)
        if not terminated:
            self._take_action(action)
            reward = self._calculate_step_reward(action)
            self.total_reward += reward
        else:
            # Czekanie na wynik z timeout
            start_time = time.time()
            while not self._check_done_condition("score_observation"):
                if time.time() - start_time > self.TIMEOUT_SCORE:  # 10 sekund timeout
                    raise Exception(f"Timeout after {self.TIMEOUT_SCORE}s waiting for score!")
                time.sleep(self.SLEEP_INTERVAL)
            # 2. Skok się zakończył - oblicz finalną nagrodę
            reward = self._calculate_final_reward()

        info = {}
        new_observation = self._get_observation()
        if terminated:
            # 3. Logowanie wyników
            self._log_episode_result()
            time.sleep(1)
            self.Cursor_game.click()  # Przejście do menu
        return new_observation, reward, terminated, truncated, info


    def reset(self, *, seed=None, options=None):
        # Resetujemy stan środowiska
        time.sleep(1)
        self.state = 0
        self.total_reward = 0
        self.Cursor_game.click()
        time.sleep(0.5)
        # Czekanie aż załaduje się gra
        start_time = time.time()
        while not self._check_done_condition("wind_direction_observation"):
            if time.time() - start_time > self.TIMEOUT_LOAD:  # 10 sekund timeout
                raise Exception(f"Timeout after {self.TIMEOUT_LOAD}s waiting for load game!")
            time.sleep(self.SLEEP_INTERVAL)
        time.sleep(1.5)
        # Pobieranie danych o wietrze
        frame_wind_speed = self.grab_frame("wind_speed_observation")
        self.wind_speed = self.Rozpoznawanie_liczb.rozpoznawanie_cyfr(frame_wind_speed)
        frame_wind_direction = self.grab_frame("wind_direction_observation")
        self.wind_direction = self.Rozpoznawanie_wiatru.rozpoznawanie_wiatru(frame_wind_direction)
        info = {}
        self.Cursor_game.click()
        return self._get_observation(), info

    def render(self):
        pass

    def _take_action(self, action):
        if action == 0:  # poruszanie myszką do góry
            self._move_mouse_up()
        elif action == 1:  # poruszanie myszką w dół
            self._move_mouse_down()
        elif action == 2:  # kliknięcie myszką
            self._click_mouse()
        elif action == 3:  # nic nie robienie
            time.sleep(0.01)

    def _calculate_step_reward(self, action):
        if self.state == 0:  # Stan: najazd na progu
            if action == 2:  # kliknięcie myszką
                reward = 1
                self.state += 1
            else:
                reward = 1
        elif self.state == 1:  # Stan: lot
            if action == 2:  # kliknięcie myszką
                reward = 2
                self.state += 1
            else:
                reward = 2
        else:  # Stan: lądowanie
            reward = 1
        return reward

    def _calculate_final_reward(self):
        frame_score = self.grab_frame("score_observation")
        reward = self.Rozpoznawanie_liczb.rozpoznawanie_cyfr(frame_score) + 168
        # TODO: Przy różnych skoczniach różny minus dla AUS to -168 ale dla innych nie
        # Jeżeli nie było dyskwalifikacji, czytanie długości skoku
        if reward != 0:
            frame_len = self.grab_frame("jump_length_observation")
            jump_len = self.Rozpoznawanie_liczb.rozpoznawanie_cyfr(frame_len)
        else:
            jump_len = 0
        # Kara za brak wybicia i ladowania
        if self.state == 0:
            reward -= 100
        elif self.state == 1:
            reward -= 50
        # Sumowanie nagród
        self.total_reward += reward
        # Zapisz statystyki (BEZ dodawania do total_reward - to zrobi step)
        if self.total_reward > self.max_score:
            self.max_score = self.total_reward
        if jump_len > self.max_jump:
            self.max_jump = jump_len
        return reward

    def _log_episode_result(self):
        self.epiosde += 1
        # Wyświetlanie informacji
        print(f"\nEpisde: {self.epiosde}")
        print(f"Wiatr kierunek: {self.wind_direction} siła: {self.wind_speed}")
        print(f"Skonczył lot przy fazie: {self.slownik[self.state]}")
        print(f"Zebrana nagroda: {self.total_reward}")
        print(f"Największy wynik to: {self.max_score}")
        print(f"Najdłuższy skok: {self.max_jump}")

    def _get_observation(self):
        jumper = self.grab_frame("jumper_observation")
        jumper_done = self.odczyt_zawodnika(jumper)
        # Dodaj kanał
        jumper_done = jumper_done[:, :, np.newaxis]  # (200,200,1)
        # TODO: Można dodać tutaj aktualizację danych wiatru ale najpewniej co któryś krok, żeby nie z każdym
        obs = {
            "frame": jumper_done,  # (200, 200, 1), dtype=np.uint8
            "wind_direction": np.array([self.wind_direction], dtype=np.int64),
            "wind_strength": np.array([self.wind_speed], dtype=np.float32)  # Normalizacja siły wiatru
        }

        return obs

    def _check_done_condition(self, name):
        frame = self.grab_frame(name)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # wykrycie jasnych pixeli > prog
        return bool(np.any(gray > 80))

    def _move_mouse_up(self):
        pyautogui.move(0, -2)
        time.sleep(0.01)

    def _move_mouse_down(self):
        pyautogui.move(0, 2)
        time.sleep(0.01)

    def _click_mouse(self):
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0)
        win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTDOWN, 0, 0)
        # TODO: Znany trick to zrobienie szybkiego ruchu myszką w dół i w góre przy wybiciu można to dodać
        # pyautogui.move(0, 30)
        time.sleep(0.15)
        # pyautogui.move(0, -30)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0)
        win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP, 0, 0)

    def grab_frame(self, name):
        return np.array(self.cap.grab(self.dict_windows[name]))

    # TODO: Zobaczymy czy będę z tego korzystał
    # Szybkie przetwarzanie obrazu z Look-Up Table
    # Zysk wydajności: 10-100x w porównaniu do pętli for
    def odczyt_zawodnika(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = self._player_lut[gray] == 1  # szybkie indeksowanie tablicą
        filtered_image = np.full(gray.shape, 255, dtype=np.uint8)
        filtered_image[mask] = gray[mask]
        return filtered_image


if __name__ == "__main__":
    env = DSJEnv()
    img = cv2.imread("../../data/images/skoczek.png", cv2.IMREAD_COLOR)  # kolor
    start = time.time()
    frame = env.odczyt_zawodnika(img)
    end = time.time()
    print(f"{end - start} seconds")
    cv2.imshow("name", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
