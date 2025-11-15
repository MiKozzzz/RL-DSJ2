from stable_baselines3 import DQN, PPO
from EnvRL import DSJEnv
import time
import subprocess
import pyautogui
import win32api
import win32con
import win32gui
from stable_baselines3.common import env_checker

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

    for episode in range(episodes):
        obs, info = env.reset(seed=0)
        done = False
        total_reward = 0

        while not done:
            obs, reward, done, tru, info = env.step(env.action_space.sample())
            total_reward += reward
        print(f"Total reward for episode {episode} is {total_reward}")

def main(path_game):
    subprocess.Popen([path_game], shell=True)
    time.sleep(2)
    hwnd = win32gui.FindWindow(None, "DOSBox 0.74-3, Cpu speed:   100000 cycles, Frameskip  0, Program:      DSJ")
    # Włączenie trybu oknowego
    pyautogui.hotkey('alt', 'enter')
    time.sleep(5)
    # Liczenie współrzędnych środka okna DSJ
    rect = win32gui.GetWindowRect(hwnd)
    # Środek okna DSJ 640x400. Teoretycznie okno większę ale to przez pasek menu, sama gra ma własciwie tyle
    print(rect)
    print("Rozdziałka gry: ", rect[2] - rect[0], rect[3] - rect[1])
    # Środek okna
    center_x = int((rect[2] + rect[0]) / 2)
    center_y = int((rect[3] + rect[1]) / 2)

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

    checker_env(env)


    # model = DQN("CnnPolicy", env, verbose=1, buffer_size=500000, learning_starts=1000)
    #
    # model = PPO("CnnPolicy", env, verbose=1)
    # model.learn(total_timesteps=40000)
    # model.save("ppo_DSJ")

    # model = PPO.load("ppo_DSJ", env=env)
    # model.learn(total_timesteps=10000)
    # model.save("ppo_DSJ")
    #

    # model = PPO("MultiInputPolicy", env, verbose=1, device="cuda")
    #
    # model = PPO.load("ppo_DSJ6_10", env=env)
    #
    # obs, info = env.reset()
    # while True:
    #     action, _states = model.predict(obs, deterministic=True)
    #     obs, reward, terminated, truncated, info = env.step(action)
    #     if terminated or truncated:
    #         print("Uczenie zakończone")
    #         obs, info = env.reset()


    time.sleep(3)
    # Wyłączanie gry
    win32gui.PostMessage(hwnd, win32con.WM_CLOSE, 0, 0)

if __name__ == "__main__":
    path_game = r"C:\RL-DSJ2\dosbox\DSJ.bat"
    main(path_game)