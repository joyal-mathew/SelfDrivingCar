import itertools
import cv2
import time
import numpy as np


def info(s, *args, **kwargs):
    print("\033[33m[INFO]\033[0m\t" + s, *args, **kwargs)


class Annotator(object):
    def __init__(self):
        self.mouseX = 0
        self.i = 0
        self.values = []

    def mouseMove(event, x, _y, _flags, self):
        self.mouseX = x

    def save_video(self, path, outputdir):
        info(f"Saving video, {path} to {outputdir}")

        video = cv2.VideoCapture(path)

        if not video.isOpened():
            raise IOError(f"Could not open video, {path}")

        while True:
            result, frame = video.read()

            if not result:
                break

            frame = cv2.pyrDown(frame)
            cv2.imwrite(f"{outputdir}/img{self.i:03}.png", frame)

            self.i += 1

        return self

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

        while True:
            key = cv2.waitKey(mspf) & 0xFF
            result, frame = video.read()

            if not result or key == ord("q"):
                break

            frame = cv2.pyrDown(frame)
            cv2.imshow(windowName, frame)
            self.values.append(self.mouseX / frame.shape[1])

        cv2.destroyWindow(windowName)

        return self

    def finish(self):
        info("Saving control values")
        np.save("dataset/output.npy", np.array(self.values))


if __name__ == "__main__":
    Annotator() \
        .annotate("data/sample-5s.mp4") \
        .save_video("data/sample-5s.mp4", "dataset/input") \
        .finish()
