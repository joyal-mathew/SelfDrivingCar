import itertools
import cv2
import os
import numpy as np

FRAME_SKIP = 5 # Count every n frames
SLOW_DOWN_FACTOR = 5


def warn(*args, **kwargs):
    print(end="\033[33m[WARN]\033[0m\t")
    print(*args, **kwargs)


def info(*args, **kwargs):
    print(end="\033[36m[INFO]\033[0m\t")
    print(*args, **kwargs)


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
        info(f"Saving video, {path} to {outputdir}")

        video = cv2.VideoCapture(path)

        if not video.isOpened():
            raise IOError(f"Could not open video, {path}")

        while True:
            result, frame = video.read()

            if not result:
                break

            if self.values[self.i] >= 0:
                cv2.imwrite(f"{outputdir}/img{str(self.written).rjust(width, '0')}.png", frame)
                self.written += 1
            self.i += 1

        return self

    def annotate(self, path):
        info(f"Starting annotation of video, {path}")

        video = cv2.VideoCapture(path)

        if not video.isOpened():
            raise IOError(f"Could not open video, {path}")

        self.mouseX = video.get(cv2.CAP_PROP_FRAME_WIDTH) / 2

        fps = int(video.get(cv2.CAP_PROP_FPS))
        mspf = int(1000 / fps * SLOW_DOWN_FACTOR)
        windowName = "Annotator"
        cv2.namedWindow(windowName)
        cv2.setMouseCallback(windowName, Annotator.mouseMove, self)

        delay = 0

        for i in itertools.count():
            result, frame = video.read()

            if not result:
                break

            if i % FRAME_SKIP:
                self.values.append(-1)
                continue

            frame = cv2.pyrDown(frame)
            cv2.imshow(windowName, frame)

            if self.capture:
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


def main():
    if not os.path.exists("annotator/dataset/input"):
        os.makedirs("annotator/dataset/input")
    else:
        warn("Are you sure you want to clear the annotator data? [yes/no]")
        answer = input("> ")
        if answer != "yes":
            return

        for file in os.listdir("annotator/dataset/input"):
            os.remove("annotator/dataset/input/" + file)

    annotator = Annotator()
    video_range = None

    for i, file in enumerate(os.listdir("annotator/data")):
        if video_range is not None and i in video_range:
            annotator.annotate("annotator/data/" + file)

    for i, file in enumerate(os.listdir("annotator/data")):
        if video_range is not None and i in video_range:
            annotator.save_video("annotator/data/" + file, "annotator/dataset/input")

    annotator.finish()


if __name__ == "__main__":
    main()
