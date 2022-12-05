from gpiozero import Servo
from picamera.array import PiRGBArray
from picamera import PiCamera

import numpy as np
import time

WIDTH = 640
HEIGHT = 480
Y_OFFSET = 0
RANGE_MAX = 10
VALUE_MIN = 20
DEFAULT_GRAY = (80, 80, 80)
MAX_DISTANCE_SQ = 2500

SERVO_PIN = 13

FPS = 5


def info(*args, **kwargs):
    print(end="\033[36m[INFO]\033[0m\t")
    print(*args, **kwargs)


class Driver(object):
    def __init__(self):
        self.camera = PiCamera()
        self.camera.resolution = WIDTH, HEIGHT
        self.raw_capture = PiRGBArray(self.camera, size=(WIDTH, HEIGHT))
        self.last_gray = np.array(DEFAULT_GRAY)

        self.servo = Servo(SERVO_PIN)

    def turn(self):
        self.camera.capture(self.raw_capture, format="bgr")

        image = self.raw_capture.array

        gray = np.zeros(3, dtype=int)
        gray_count = 0

        for x in range(WIDTH):
            color = image[HEIGHT - 1, x]
            channel_range = np.max(color) - np.min(color)
            value = np.sum(color) // 3

            if channel_range < RANGE_MAX and value > VALUE_MIN:
                gray += color
                gray_count += 1

        if gray_count:
            gray //= gray_count
        else:
            gray = self.last_gray

        self.last_gray = gray

        x_values = []

        for x in range(WIDTH):
            color = image[HEIGHT - 1, x]
            distance_sq = sum((c - g) ** 2 for c, g in zip(color, gray))

            if distance_sq < MAX_DISTANCE_SQ:
                x_values.append(x)

        cutoff = int(len(x_values) * 0.25)
        x_values = x_values[cutoff:-cutoff]
        average_x = sum(x_values) // len(x_values)
        turn_value = (average_x / WIDTH) * 0.8 - 0.4

        info("Turn value:", turn_value)

        self.servo.value = turn_value

        self.raw_capture.truncate(0)

    def run(self, fps):
        seconds_per_frame = 1 / fps

        while True:
            start = time.time()

            self.turn()

            while time.time() - start < seconds_per_frame:
                pass


if __name__ == "__main__":
    Driver().run(FPS)
