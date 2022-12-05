from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2

WIDTH = 640
HEIGHT = 480


class Driver(object):
    def __init__(self):
        self.camera = PiCamera()
        self.camera.resolution = WIDTH, HEIGHT
        self.rawCapture = PiRGBArray(self.camera, size=(WIDTH, HEIGHT))

    def turn(self):
        self.camera.capture(self.rawCapture, format="bgr")

        image = self.rawCapture.array

        for x in range(WIDTH):
            pass

