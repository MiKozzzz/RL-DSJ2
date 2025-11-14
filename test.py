from mss import mss
import cv2
import numpy as np
import time
import subprocess
import pyautogui
import win32api
import win32con
import win32gui

def sprawdzanie_obrazu(name, dictonary):
    time.sleep(1)
    img = mss().grab(dictonary[name])
    frame = np.array(img)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    cv2.imshow(name, frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # zapis do pliku JPG
    # cv2.imwrite(f"{name}.jpg", frame)


def _check_done_condition(name, dictonary):
    frame = np.array(mss().grab(dictonary[name]))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    print(gray)
    # wykrycie jasno-litery > prog
    return bool(np.any(gray > 130))

# def show_all_windows():
#     def callback(hwnd, extra):
#         title = win32gui.GetWindowText(hwnd)
#         if title:  # ignoruj okna bez tytułu
#             print(f"HWND: {hwnd}, Tytuł: {title}")
#     win32gui.EnumWindows(callback, None)
#
# show_all_windows()

# sprawdzanie_obrazu({"top": 110, "left": 215, "width": 200, "height": 200})
#
# while True:
#     x, y = pyautogui.position()
#     print(f"Pozycja kursora: X={x}, Y={y}")
#

subprocess.Popen([r"C:\RL-DSJ2\dosbox\DSJ.bat"], shell=True)

time.sleep(2)

hwnd = win32gui.FindWindow(None, "DOSBox 0.74-3, Cpu speed:   100000 cycles, Frameskip  0, Program:      DSJ")
# Włączenie trybu oknowego
pyautogui.hotkey('alt', 'enter')

time.sleep(5)

# Kliknięcie w okno gry DSJ
rect = win32gui.GetWindowRect(hwnd)
print(rect)
print("Rozdziałka gry: ", rect[2] - rect[0], rect[3] - rect[1])

# Środek okna
center_x = int((rect[2] + rect[0]) / 2)
center_y = int((rect[3] + rect[1]) / 2)
pyautogui.moveTo(center_x, center_y)
pyautogui.click()
# Środek okna DSJ 640x400. Teoretycznie okno większę ale to przez pasek menu, sama gra ma własciwie tyle
x_postion = 320
y_postion = 200

print(f"X: {x_postion}, Y: {y_postion}")

time.sleep(3)

move_to_x = 160
move_to_y = 210
pyautogui.move(move_to_x - x_postion, move_to_y - y_postion)
x_postion = move_to_x
y_postion = move_to_y
print(f"X: {x_postion}, Y: {y_postion}")

time.sleep(1)

win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0)
time.sleep(0.1)
win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0)

time.sleep(2)

move_to_x = 440
move_to_y = 310
pyautogui.move(move_to_x - x_postion, move_to_y - y_postion)
x_postion = move_to_x
y_postion = move_to_y
print(f"X: {x_postion}, Y: {y_postion}")

time.sleep(1)

win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0)
time.sleep(0.1)
win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0)

time.sleep(2)

win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0)
time.sleep(0.1)
win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0)

time.sleep(5)

dict_windows = {}

dict_windows["jumper_observation"] = {"top": center_y - 100, "left": center_x - 100, "width": 200, "height": 200}
dict_windows["wind_direction_observation"] = {"top": center_y - 178, "left": center_x + 260, "width": 45, "height": 29}
dict_windows["wind_speed_observation"] = {"top": center_y - 150, "left": center_x + 265, "width": 40, "height": 17}
dict_windows["jump_length_observation"] = {"top": center_y + 190, "left": center_x - 25, "width": 65, "height": 20}
dict_windows["score_observation"] = {"top": center_y + 190, "left": center_x + 160, "width": 110, "height": 20}

print(_check_done_condition("jump_length_observation", dict_windows))

# sprawdzanie_obrazu("jumper_observation", dict_windows)
# sprawdzanie_obrazu("wind_direction_observation", dict_windows)
# sprawdzanie_obrazu("wind_speed_observation", dict_windows)
# sprawdzanie_obrazu("jump_length_observation", dict_windows)
# sprawdzanie_obrazu("score_observation", dict_windows)


win32gui.PostMessage(hwnd, win32con.WM_CLOSE, 0, 0)

