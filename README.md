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

- `pudv4.py` – custom Gym environment
- `dosbox` – DSJ game file
- `cyfry` – file with photos of digits
- `wiatr` – file with photos of wind
- `Model_Ropoznawania_kierunku_wiatru.py` – trains a model to recognize wind from game screenshots
- `Model_Rozpoznawanie_cyfr.py` – trains a model to recognize digits from game screenshots (used to extract jump length and scores)
- `best_cyfry.keras` – trained digit recognition model
- `best_wiatr.keras` – trained wind recognition model

## 📈 Future Improvements

- Add wind data to observation space
- Improve digit recognition and reward normalization
- Consider using frame stacking or recurrent policies
- Use experience replay or curriculum learning

## 📸 Screenshots

<img width="472" height="295" alt="obraz" src="https://github.com/user-attachments/assets/2d19fcbb-7c7a-4510-bac3-ef139edf4ac1" />

1. Image of the jumper – cropped screen region containing the ski jumper, used as the main observation input.
2. Wind direction image – screenshot region showing the current wind direction indicator.
3. Wind speed image – screenshot region showing the current wind speed value.
4. Jump length image – cropped area of the screen displaying the distance jumped after landing.
5. Jump score image – screen region showing the final score received for the jump.


<img width="200" height="200" alt="obraz" src="https://github.com/user-attachments/assets/847f60d0-9f5f-4993-b524-793ceaa05b21" />

Filtered screenshot – processed image highlighting only the ski jumper and key elements, removing unnecessary background to improve learning efficiency.

## 📄 License

This project was developed as part of a university course on reinforcement learning and is intended for educational purposes only. Deluxe Ski Jump 2 is a commercial game and its assets are not included in this repository.
