import time
import pyautogui
import win32api
import win32con

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