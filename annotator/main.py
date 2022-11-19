import itertools
import cv2
import time
import numpy as np

video = cv2.VideoCapture("data/sample-5s.mp4")

if not video.isOpened():
    print("[ERR]\tCould not open video")
    exit(1)

fps = video.get(cv2.CAP_PROP_FPS)
mspf = int(1000 / fps)
windowName = "Annotator"
cv2.namedWindow(windowName)

delta = 0

for i in itertools.count():
    result, frame = video.read()

    if not result:
        break

    cv2.imwrite(f"dataset/input/img{i:03}.png", frame)

cv2.destroyWindow(windowName)
zeros = np.zeros(2)
print(zeros.shape)

np.save("dataset/output.npy", zeros)
