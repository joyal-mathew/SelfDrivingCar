import time
import 
import picamera
import numpy as np
import cv2
from gpiozero import Servo

servo = Servo(13)

with picamera.PiCamera() as camera:
    camera.resolution = (224, 224)
    camera.framerate = 24

    time.sleep(2)
    while True:
        image = np.empty((224 * 224 * 3,), dtype=np.uint8)
        camera.capture(image, 'bgr') #decide if should be bgr or rgb
        image = image.reshape((240, 320, 3))


        servo.value = 0
