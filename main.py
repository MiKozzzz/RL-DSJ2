from stable_baselines3 import PPO
from EnvRL import DSJEnv
import time
import subprocess
import pyautogui
import win32api
import win32con
import win32gui
from stable_baselines3.common import env_checker
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn
from gymnasium import spaces
from datetime import datetime

# TODO: To trzeba jeszcze przejrzeć i zrozumieć, bo DeepSeek mi to zrobił kij wie czy dobrze, niby działa
class DSJFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict):
        super().__init__(observation_space, features_dim=128)

        # Optymalizowana CNN dla DSJ2
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=2, padding=2),  # 200x200 -> 100x100
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),  # 100x100 -> 50x50
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),  # 50x50 -> 25x25
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # 25x25 -> 4x4
            nn.Flatten()
        )

        n_flat = 64 * 4 * 4  # 1024 features

        # wind_dir embedding
        self.embed_dir = nn.Embedding(8, 4)

        # MLP dla połączonych cech
        self.fc = nn.Sequential(
            nn.Linear(n_flat + 1 + 4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),  # dodatkowa warstwa dla lepszej reprezentacji
            nn.ReLU()
        )

    def forward(self, obs):
        # --- Obraz ---
        img = obs["frame"].float()
        # Poprawka: jeśli obraz ma kształt (B, H, W, C), zmieniamy na (B, C, H, W)
        if len(img.shape) == 4 and img.shape[-1] == 1:  # jeśli kanał jest ostatni
            img = img.permute(0, 3, 1, 2)
        # --- CNN ---
        cnn_out = self.cnn(img)
        # --- Wind direction (embedding) ---
        wind_dir = obs["wind_direction"].long()
        # Sprawdź kształt i odpowiednio przetwórz
        if len(wind_dir.shape) > 1:
            wind_dir = wind_dir.squeeze(-1)  # usuń ostatni wymiar jeśli istnieje
        # Jeśli nadal ma więcej niż 1 wymiar, weź tylko pierwszy element
        if len(wind_dir.shape) > 1 and wind_dir.shape[1] > 1:
            wind_dir = wind_dir[:, 0]  # weź tylko pierwszy kierunek wiatru
        wind_dir_emb = self.embed_dir(wind_dir)  # (B, 4)
        # Jeśli wind_dir_emb ma 3 wymiary, zredukuj do 2
        if len(wind_dir_emb.shape) == 3:
            wind_dir_emb = wind_dir_emb.squeeze(1)  # (B, 4)
        # --- Wind strength ---
        wind_strength = obs["wind_strength"].float()
        if len(wind_strength.shape) > 1:
            wind_strength = wind_strength.squeeze(-1)  # (B,)
        wind_strength = wind_strength.unsqueeze(1)  # (B, 1)
        # --- Łączenie cech ---
        # Wszystkie tensory powinny mieć kształt (B, features)
        x = torch.cat([cnn_out, wind_strength, wind_dir_emb], dim=1)
        # --- MLP ---
        return self.fc(x)


class Cursor:
    def __init__(self, width, height):
        self.x_postion = width / 2
        self.y_postion = height / 2
        print(f"X: {self.x_postion}, Y: {self.y_postion}")

    def click(self):
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0)
        time.sleep(0.1)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0)

    def move_to(self, x, y):
        pyautogui.move(x - self.x_postion, y - self.y_postion)
        self.x_postion = x
        self.y_postion = y
        print(f"X: {self.x_postion}, Y: {self.y_postion}")

def checker_env(env, episodes=10):
    env_checker.check_env(env)

    for episode in range(episodes-1):
        obs, info = env.reset(seed=0)
        done = False
        total_reward = 0

        while not done:
            obs, reward, done, tru, info = env.step(env.action_space.sample())
            total_reward += reward
        print(f"Total reward for episode {episode} is {total_reward}")


def learn_PPO(env, timesteps, cuda):
    policy_kwargs = dict(
        features_extractor_class=DSJFeatureExtractor,
    )
    if cuda:
        model = PPO(
            "MultiInputPolicy",
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            device="cuda"
        )
    else:
        model = PPO(
            "MultiInputPolicy",
            env,
            policy_kwargs=policy_kwargs,
            verbose=1
        )

    model.learn(total_timesteps=timesteps)
    now = datetime.now()
    model.save("PPO_"+now.strftime("%Y-%m-%d %H:%M"))

def load_model():
    # model = PPO.load("ppo_DSJ", env=env)
    # model.learn(total_timesteps=10000)
    # model.save("ppo_DSJ")

    # obs, info = env.reset()
    # while True:
    #     action, _states = model.predict(obs, deterministic=True)
    #     obs, reward, terminated, truncated, info = env.step(action)
    #     if terminated or truncated:
    #         print("Uczenie zakończone")
    #         obs, info = env.reset()
    pass

def main(path_game):
    CUDA = torch.cuda.is_available()
    if CUDA:
        print("GPU z CUDA jest dostępne")
        print("Liczba dostępnych GPU:", torch.cuda.device_count())
        print("Aktualny GPU:", torch.cuda.current_device())
        print("Nazwa GPU:", torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        print("Brak GPU z CUDA, będzie używany CPU")

    subprocess.Popen([path_game], shell=True)
    time.sleep(3)
    # Włączenie trybu oknowego
    pyautogui.hotkey('alt', 'enter')
    time.sleep(5)
    hwnd = win32gui.FindWindow(None, "DOSBox 0.74-3, Cpu speed:   100000 cycles, Frameskip  0, Program:      DSJ")
    # Liczenie współrzędnych środka okna DSJ
    rect = win32gui.GetWindowRect(hwnd)
    # Środek okna DSJ 640x400. Teoretycznie okno większę ale to przez pasek menu, sama gra ma własciwie tyle
    print(rect)
    print("Rozdziałka gry: ", rect[2] - rect[0], rect[3] - rect[1])
    # Środek okna
    center_x = int((rect[2] + rect[0]) / 2)
    center_y = int((rect[3] + rect[1]) / 2)
    print(center_x, center_y)
    # Inicjalizacja środowiska
    env = DSJEnv(center_x, center_y)
    # Kliknięcie w okno gry DSJ
    pyautogui.moveTo(center_x, center_y)
    # Kursor dla okna DSJ 640x400
    cursor = Cursor(640, 400)
    cursor.click()
    time.sleep(3)
    cursor.move_to(160, 210)
    time.sleep(1)
    cursor.click()
    time.sleep(1)
    cursor.move_to(440, 310)
    time.sleep(1)
    cursor.click()
    time.sleep(2)
    cursor.click()
    time.sleep(0.5)
    cursor.click()
    time.sleep(3)

    # env = DSJEnv(349, 240)
    learn_PPO(env, 20000, CUDA)
    # checker_env(env)
    time.sleep(3)
    # Wyłączanie gry
    win32gui.PostMessage(hwnd, win32con.WM_CLOSE, 0, 0)


if __name__ == "__main__":
    path_game = r"C:\RL-DSJ2\dosbox\DSJ.bat"
    main(path_game)
