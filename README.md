# 🏔️ Ski Jumping Bot with Reinforcement Learning (DSJ2 + PPO)

This project is a reinforcement learning agent designed to play **Deluxe Ski Jump 2** using the **PPO algorithm** and visual input from the game screen.

The goal is to train an autonomous jumper capable of performing full ski jumps (take-off, flight, landing) in varying wind conditions — by interacting with the game through screenshots and mouse control.

## 🎮 Project Overview

- The environment is built using [Gymnasium](https://gymnasium.farama.org/), custom screen captures via `mss`, and action execution using `pyautogui` and `win32api`.
- The observations are grayscale images (200x200) filtered to highlight the jumper silhouette.
- The agent takes one of four discrete actions:
  1. Jump (press both mouse buttons)
  2. Move mouse down (lower body)
  3. Move mouse up (lift body)
  4. Do nothing (hold position)

## 🧠 Learning

- **Algorithm:** PPO (Proximal Policy Optimization) from `stable-baselines3`, with `CnnPolicy`.
- **Observation:** Only the filtered image of the jumper (for simplicity).
- **Reward strategy:**
  - Points awarded for proper take-off, flight stability, and landing.
  - Additional reward extracted from the game score using OCR with a custom CNN digit recognizer.
  - Penalties for failing to jump or land properly.
- Training was done over multiple sessions (40k steps and fine-tuning with 10k steps).

## 🧪 Results

- The best jump achieved: **251 meters**.
- Average jump length: ~180 meters.
- The agent mostly lands successfully.
- Results vary significantly due to wind conditions — which are not included in the observation space (yet).

## 🧰 Technologies Used

- Python 3
- Gymnasium
- Stable Baselines3
- PyAutoGUI
- OpenCV
- mss
- win32api / win32con

## 📦 Files & Structure

- `wind_env.py` – custom Gym environment
- `train.py` – training loop with PPO
- `digit_recognition.py` – CNN model for reading scores from screenshots
- `utils.py` – image processing and filtering functions
- `model/` – trained models and digit recognition networks

## 📈 Future Improvements

- Add wind data to observation space
- Improve digit recognition and reward normalization
- Consider using frame stacking or recurrent policies
- Use experience replay or curriculum learning

## 📸 Screenshots

*(You can include filtered screenshot examples of jump phases here)*

## 📄 License

This project was developed as part of a university course on reinforcement learning and is intended for educational purposes only. Deluxe Ski Jump 2 is a commercial game and its assets are not included in this repository.
