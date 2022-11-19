import itertools
import cv2
import time
import os
import numpy as np

FRAME_SKIP = 5 # Count every n frames


def info(s, *args, **kwargs):
    print("\033[33m[INFO]\033[0m\t" + s, *args, **kwargs)


class Annotator(object):
    def __init__(self):
        self.mouseX = 0
        self.capture = False
        self.i = 0
        self.written = 0
        self.values = []

    def mouseMove(event, x, _y, _flags, self):
        self.mouseX = x

        if event == cv2.EVENT_LBUTTONDOWN:
            info("Capturing")
            self.capture = True
        elif event == cv2.EVENT_LBUTTONUP:
            info("Not Capturing")
            self.capture = False

    def save_video(self, path, outputdir):
        width = len(str(len(self.values)))
        print(width)
        info(f"Saving video, {path} to {outputdir}")

        video = cv2.VideoCapture(path)

        if not video.isOpened():
            raise IOError(f"Could not open video, {path}")

        while True:
            result, frame = video.read()

            if not result:
                break

            frame = cv2.pyrDown(frame)
            if self.values[self.i] >= 0:
                cv2.imwrite(f"{outputdir}/img{str(self.i).rjust(width, '0')}.png", frame)
                self.written += 1
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

        delay = 0

        for i in itertools.count():
            result, frame = video.read()

            if not result:
                break

            frame = cv2.pyrDown(frame)
            cv2.imshow(windowName, frame)
            if self.capture and i % FRAME_SKIP == 0:
                self.values.append(self.mouseX / frame.shape[1])
            else:
                self.values.append(-1)

            key = cv2.waitKey(delay) & 0xFF
            delay = mspf

            if key == ord("q"):
                break

        cv2.destroyWindow(windowName)

        return self

    def finish(self):
        info("Saving control values")
        self.values = [v for v in self.values if v >= 0]
        assert(len(self.values) == self.written)
        np.save("annotator/dataset/output.npy", np.array(self.values))


if __name__ == "__main__":
    if not os.path.exists("annotator/dataset/input"):
        os.makedirs("annotator/dataset/input")

    for file in os.listdir("annotator/dataset/input"):
        os.remove("annotator/dataset/input/" + file)

    annotator = Annotator()

    for file in os.listdir("annotator/data"):
        annotator.annotate("annotator/data/" + file)

    for file in os.listdir("annotator/data"):
        annotator.save_video("annotator/data/" + file, "annotator/dataset/input")

    annotator.finish()
