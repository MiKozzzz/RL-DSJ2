from stable_baselines3 import PPO
from src.environment.EnvRL import DSJEnv
import time
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch
import torch.nn as nn
from gymnasium import spaces

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
        # Obraz
        img = obs["frame"].float()
        if img.dim() == 4 and img.size(-1) == 1:
            img = img.permute(0, 3, 1, 2)

        cnn_out = self.cnn(img)

        wind_dir = obs["wind_direction"].long().view(-1)
        wind_dir_emb = self.embed_dir(wind_dir)

        # Wind strength - zawsze (B, 1)
        wind_strength = obs["wind_strength"].float()

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
            n_steps=2048,  # 2048 kroków = ~8-12 epizodów (160-250 kroków/epizod)
            batch_size=128,  # Lepsze: 2048/128 = 16 batchów (więcej aktualizacji)
            n_epochs=10,  # 10 przejść przez dane
            learning_rate=3e-4,  # Dobry start
            clip_range=0.2,  # Standard
            ent_coef=0.01,  # WIĘCEJ - DSJ2 wymaga eksploracji różnych strategii!
            gamma=0.995,  # WYŻSZE - w DSJ2 przyszłe punkty są ważne
            gae_lambda=0.95,  # OK
            verbose=1,
            device="cuda",
        )

    model.learn(total_timesteps=timesteps, reset_num_timesteps=reset)
    # now = datetime.now()
    # name = "PPO_" + now.strftime("%Y-%m-%d_%H:%M")
    name = 'ppo_20_11_2025'
    model.save(name)


def load_model_test(env, model, episodes=5):
    for episode in range(episodes):
        obs, info = env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs, deterministic=True)
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

    # checker_env(env)

    learn_PPO(env, 1000, reset=False)

    # model = PPO.load("ppo_4.zip", env)
    # learn_PPO(env, 20000, model, reset=False)

    # model = PPO.load("ppo_4.zip", env)
    # load_model_test(env, model)

    time.sleep(3)
    # Wyłączanie gry
    env.quit_game()


if __name__ == "__main__":
    with open("path_game", "r", encoding="utf-8") as f:
        linie = f.readlines()
    path_game = linie[0].strip()
    window_name = linie[1]
    main(path_game, window_name)
