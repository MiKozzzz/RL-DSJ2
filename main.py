from stable_baselines3 import PPO
from EnvRL import DSJEnv
import time
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
            nn.Dropout(0.1),  # DODANE - 10% dropout
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.1),  # DODANE
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

def checker_env(env, episodes=10):
    # env_checker.check_env(env)

    for episode in range(episodes - 1):
        obs, info = env.reset()
        done = False
        total_reward = 0
        while not done:
            obs, reward, done, tru, info = env.step(env.action_space.sample())
            total_reward += reward


# TODO: Cuda czy jest czy nie, dodam reset_num_timesteps=False
def learn_PPO(env, timesteps, model=None, reset=True):
    policy_kwargs = dict(
        features_extractor_class=DSJFeatureExtractor,
    )
    # TODO: reset bufora trzeba będzie to zeminić
    if model is not None:
        model.ent_coef = 0.01  # WIĘCEJ eksploracji
        model.clip_range = 0.3  # WIĘKSZE zmiany strategii
        model.learning_rate = 1e-5  # MNIEJSZY krok
        model.n_epochs = 5  # MNIEJ przejść przez dane

        # # DEFAULT PARAMETERS:
        # n_steps = 2048
        # batch_size = 64
        # n_epochs = 10
        # learning_rate = 3e-4
        # clip_range = 0.2
        # gamma = 0.99
        # gae_lambda = 0.95
        # ent_coef = 0.0  # ← WYŁĄCZONE domyślnie!
        # vf_coef = 0.5
        # max_grad_norm = 0.5

    else:
        model = PPO(
            "MultiInputPolicy",
            env,
            policy_kwargs=policy_kwargs,
            learning_rate=3e-4,  # Możesz dostosować
            n_steps=8192,  # Dłuższe rollout
            batch_size=256,
            n_epochs=10,
            clip_range=0.3,
            ent_coef=0.01,  # Zachęca do eksploracji
            verbose=1,
            device="cuda",
        )

    model.learn(total_timesteps=timesteps, reset_num_timesteps=reset)
    # now = datetime.now()
    # name = "PPO_" + now.strftime("%Y-%m-%d_%H:%M")
    name = 'ppo_19_11_2025'
    model.save(name)


def load_model_test(env, model):
    # Uruchom episod
    obs, _ = env.reset()
    for _ in range(2000):  # maksymalna liczba kroków
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated:
            obs, info = env.reset()
            break


def main(path_game):
    CUDA = torch.cuda.is_available()
    if CUDA:
        print("GPU z CUDA jest dostępne")
        print("Liczba dostępnych GPU:", torch.cuda.device_count())
        print("Aktualny GPU:", torch.cuda.current_device())
        print("Nazwa GPU:", torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        print("Brak GPU z CUDA, będzie używany CPU")

    env = DSJEnv()
    env.initialize_game(path_game, "DOSBox 0.74-3, Cpu speed:   100000 cycles, Frameskip  0, Program:      DSJ")

    checker_env(env)

    # learn_PPO(env, 140000, reset=False)

    # model = PPO.load("ppo_4.zip", env)
    # learn_PPO(env, 20000, model, reset=False)

    # model = PPO.load("ppo_4.zip", env)
    # load_model_test(env, model)

    time.sleep(3)
    # Wyłączanie gry
    env.quit_game()


if __name__ == "__main__":
    path_game = r"C:\RL-DSJ2\dosbox\DSJ.bat"
    main(path_game)
