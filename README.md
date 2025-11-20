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
  - Points are awarded for each decision made during the jump (e.g., taking off, maintaining position).
    The longer the jump lasts, the more decisions are made, resulting in a higher total reward.
  - Additional reward extracted from the game score using OCR with a custom CNN digit recognizer.
  - Penalties for failing to jump or land properly.
- Training was done over multiple sessions (40k steps and fine-tuning with 10k steps).

## 🧪 Results

- The agent was trained and evaluated on the **Australia K240** ski jump hill.
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

RL-DSJ2/
├── 📁 models/                         # All trained models
│   ├── rl/                           # Reinforcement Learning models
│   │   ├── ppo_1.zip
│   │   └── ... (other model versions)
│   ├── vision/                       # Weights for computer Vision models
│   │   ├── model_cyfr_weights.pth
│   │   ├── model_wiatru_weights.pth
│   │   └── ...
│   └── training/                     # Training checkpoints and logs
│       ├── checkpoints/
│       └── tensorboard_logs/
├── 📁 src/                           # Source code
│   ├── environment/
│   │   ├── EnvRL.py                  # DSJ2 Gym environment
│   │   └── __init__.py
│   ├── vision/
│   │   ├── Model_Rozpoznawania_kierunku_wiatru.py
│   │   ├── Model_Rozpoznawanie_cyfr.py
│   │   ├── Rozpoznawanie_Liczb.py
│   │   ├── Rozpoznawanie_Wiatru.py
│   │   └── __init__.py
│   ├── utils/
│   │   ├── Cursor.py                 # Mouse control utilities
│   │   └── __init__.py
│   ├── main.py                       # Main training script
│   └── __init__.py
├── 📁 data/                          # Data and assets
│   ├── images/                       # Sample images and screenshots
│   │   ├── liczba.png
│   │   ├── skoczek.png
│   │   └── wiatr.png
│   ├── datasets/                     # Training datasets
│   │   ├── cyfry/                    # Digit dataset
│   │   └── wiatr/                    # Wind direction dataset
│   └── config/
│       └── pozycje w oknie DSJ.txt   # Window position config
├── 📁 docs/                          # Documentation
│   ├── README.md
│   └── .gitignore
├── 📁 venv/                          # Python virtual environment
└── 📁 dosbox/                        # DOSBox emulator files


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


<img width="200" height="200" alt="obraz" src="https://github.com/user-attachments/assets/847f60d0-9f5f-4993-b524-793ceaa05b21" /> <img width="200" height="200" alt="obraz" src="https://github.com/user-attachments/assets/688bd85a-30f0-4c44-a612-c7ab3fe02a34" />


Filtered screenshot – processed image highlighting only the ski jumper and key elements, removing unnecessary background to improve learning efficiency.

## 📄 License

This project was developed as part of a university course on reinforcement learning and is intended for educational purposes only. Deluxe Ski Jump 2 is a commercial game and its assets are not included in this repository.
