from stable_baselines3 import PPO
from src.environment.EnvRL import DSJEnv
import time
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn
from gymnasium import spaces
from datetime import datetime
import cv2

class DSJFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict):
        features_dim = 512
        super().__init__(observation_space, features_dim=features_dim)

        # Cnn oparte na NatureCnn + nn.AdaptiveMaxPool2d((7, 7)) na obrazie 200x200
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((7, 7)),
            nn.Flatten(),
        )

        n_flat = 64 * 7 * 7  # 3136 features

        # wind_dir embedding
        self.embed_dir = nn.Embedding(8, 4)

        # MLP dla połączonych cech
        self.fc = nn.Sequential(
            nn.Linear(n_flat + 1 + 4, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU(),
        )

    def forward(self, obs):
        # Obraz
        img = obs["frame"].float()
        if img.dim() == 4 and img.size(-1) == 1:
            img = img.permute(0, 3, 1, 2)
        # Dostosuj normalizację do zakresu
        if img.max() > 1.1:  # Pewnie [0, 255]
            img = (img / 255.0 - 0.5) / 0.5
        else:  # Pewnie już [0, 1]
            img = (img - 0.5) / 0.5
        cnn_out = self.cnn(img)
        wind_dir = obs["wind_direction"].view(-1).long()
        wind_dir_emb = self.embed_dir(wind_dir)
        # Wind strength - zawsze (B, 1)
        wind_strength = obs["wind_strength"].float()
        wind_strength = (wind_strength / 5.0 - 0.5) / 0.5  # Normalizacja gdzie max wiatr to 5.0
        # Łączenie
        x = torch.cat([cnn_out, wind_strength, wind_dir_emb], dim=1)
        return self.fc(x)

def checker_env(env, episodes=5):
    # env_checker.check_env(env)
    for episode in range(episodes):
        obs, info = env.reset()
        done = False
        while not done:
            obs, reward, done, tru, info = env.step(env.action_space.sample())


# TODO: Cuda czy jest czy nie
def learn_PPO(env, timesteps, name_model=None, reset=True):
    policy_kwargs = dict(
        features_extractor_class=DSJFeatureExtractor,
    )
    # TODO: reset bufora trzeba będzie to zeminić
    if name_model is not None:
        path_model = r"../models/rl/" + name_model
        model = PPO.load(path_model, env, device="cuda")

        # model.ent_coef = 0.01  # WIĘCEJ eksploracji
        # model.clip_range = 0.3  # WIĘKSZE zmiany strategii
        # model.learning_rate = 1e-5  # MNIEJSZY krok
        # model.n_epochs = 5  # MNIEJ przejść przez dane

        """
        DEFAULT PARAMETERS:
        n_steps = 2048
        batch_size = 64
        n_epochs = 10
        learning_rate = 3e-4
        clip_range = 0.2
        gamma = 0.99
        gae_lambda = 0.95
        ent_coef = 0.0  # ← WYŁĄCZONE domyślnie!
        vf_coef = 0.5
        max_grad_norm = 0.5
        
        od chat:
            n_steps=2048,  # 2048 kroków = ~8-12 epizodów (160-250 kroków/epizod)
            batch_size=128,  # Lepsze: 2048/128 = 16 batchów (więcej aktualizacji)
            n_epochs=10,  # 10 przejść przez dane
            learning_rate=3e-4,  # Dobry start
            clip_range=0.2,  # Standard
            ent_coef=0.01,  # WIĘCEJ - DSJ2 wymaga eksploracji różnych strategii!
            gamma=0.995,  # WYŻSZE - w DSJ2 przyszłe punkty są ważne
            gae_lambda=0.95,  # OK
        """
    else:
        model = PPO(
            "MultiInputPolicy",
            env,
            policy_kwargs=policy_kwargs,
            tensorboard_log="./tb",
            ent_coef=0.01,
            verbose=1,
            n_steps=4096,
            batch_size=256,
            n_epochs=10,
            learning_rate=3e-4,
            clip_range=0.2,
            gamma=0.995,
            gae_lambda=0.95,
            vf_coef=0.5,
            max_grad_norm=0.5,
            device="cuda",
        )

    model.learn(total_timesteps=timesteps, reset_num_timesteps=reset)
    now = datetime.now()
    name = "PPO_" + now.strftime("%Y-%m-%d_%H-%M")
    model.save(r"../models/rl/" + name)
    return name


def load_model_test(env, name_model, episodes=5):
    path_model = r"../models/rl/" + name_model
    model = PPO.load(path_model, env, device="cuda")
    model.policy.eval()
    for episode in range(episodes):
        obs, info = env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=False)
            obs, reward, done, tru, info = env.step(action)


def main(path_game, window_name):
    CUDA = torch.cuda.is_available()
    if CUDA:
        print("GPU z CUDA jest dostępne")
        print("Liczba dostępnych GPU:", torch.cuda.device_count())
        print("Aktualny GPU:", torch.cuda.current_device())
        print("Nazwa GPU:", torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        print("Brak GPU z CUDA, będzie używany CPU")

    env = DSJEnv()
    env.initialize_game(path_game, window_name)
    # name_model = "PPO_2025-11-25_09-45.zip"

    # checker_env(env)
    # load_model_test(env, name_model)

    # name_model = learn_PPO(env, 40_000, reset=False)

    # name_model = learn_PPO(env, 40_000, name_model, reset=False)
    #
    # env._click_mouse()
    #
    name_model = learn_PPO(env, 40_000, name_model, reset=False)

    env._click_mouse()

    name_model = learn_PPO(env, 40_000, name_model, reset=False)

    env._click_mouse()

    learn_PPO(env, 80_000, name_model, reset=False)

    time.sleep(3)
    # Wyłączanie gry
    env.quit_game()


if __name__ == "__main__":
    with open("path_game", "r", encoding="utf-8") as f:
        linie = f.readlines()
    path_game = linie[0].strip()
    window_name = linie[1]
    main(path_game, window_name)
