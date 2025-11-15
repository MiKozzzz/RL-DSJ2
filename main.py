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

# model = PPO("MultiInputPolicy", env, verbose=1, device="cuda")

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