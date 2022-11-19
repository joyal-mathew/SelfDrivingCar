import itertools
import cv2

video = cv2.VideoCapture("sample-5s.mp4")

if not video.isOpened():
    print("[ERR]\tCould not open video")
    exit(1)

windowName = "Annotator"
cv2.namedWindow(windowName)

for i in itertools.count():
    print(f"Frame: {i}")

    result, frame = video.read()

    if not result:
        print("Could not read frame")
        cv2.destroyWindow(windowName)
        break

    cv2.imshow(windowName, frame)
