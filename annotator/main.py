import itertools
import cv2
import time
import numpy as np


def info(s, *args, **kwargs):
    print("\033[33m[INFO]\033[0m\t" + s, *args, **kwargs)


class Annotator(object):
    def __init__(self):
        self.mouseX = 0

    def mouseMove(event, x, y, flags, self):
        self.mouseX = x

    def save_video(self, path, outputdir):
        info(f"Saving video, {path} to {outputdir}")

        video = cv2.VideoCapture(path)

        if not video.isOpened():
            raise IOError(f"Could not open video, {path}")

        for i in itertools.count():
            result, frame = video.read()

            if not result:
                break

            frame = cv2.pyrDown(frame)
            cv2.imwrite(f"{outputdir}/img{i:03}.png", frame)

    def annotate(self, path):
        info(f"Starting annotation of video, {path}")

        video = cv2.VideoCapture(path)

        if not video.isOpened():
            raise IOError(f"Could not open video, {path}")

        self.mouseX = video.get(cv2.CAP_PROP_FRAME_WIDTH) / 2

        fps = video.get(cv2.CAP_PROP_FPS)
        mspf = int(1000 / fps)
        windowName = "Annotator"
        cv2.namedWindow(windowName)
        cv2.setMouseCallback(windowName, Annotator.mouseMove, self)

        delta = 0
        values = []

        while True:
            key = cv2.waitKey(mspf) & 0xFF
            result, frame = video.read()

            if not result or key == ord("q"):
                break

            frame = cv2.pyrDown(frame)
            cv2.imshow(windowName, frame)
            values.append(self.mouseX / frame.shape[1])

        cv2.destroyWindow(windowName)
        np.save("dataset/output.npy", np.array(values))


if __name__ == "__main__":
    annotator = Annotator()
    annotator.annotate("data/sample-5s.mp4")
    annotator.save_video("data/sample-5s.mp4", "data/input")
