import cv2
import numpy as np
from time import sleep
import math


def disp(img, w, h):
  return "".join("r" if img[h-1, i, 2] == 255 else "b" for i in range(w))

def colors_equal(c1, c2):
  # print(c1, c2)
  return all(a == b for a, b in zip(c1, c2))

def approach(v):
  return v*1
  
def dialate(image, n, w, h, color):
  # print(disp(image, w, h))
  for _ in range(n):
    tempRow = [None]*w
    for i in range(w):
      if i == 0:
        if colors_equal(image[h-1, i+1], color):
          tempRow[i] = color
      elif i == w-1:
        if colors_equal(image[h-1, i-1], color):
          tempRow[i] = color
      else:
        if colors_equal(image[h-1, i-1], color) or colors_equal(image[h-1, i+1], color):
          tempRow[i] = color
        # else:
          # print(tuple(image[h-1, i-1]))

    # print(tempRow)
    for i, c in enumerate(tempRow):
      if c is not None:
        for j in range(1, 5):
          image[h-j, i, 0] = c[0]
          image[h-j, i, 1] = c[1]
          image[h-j, i, 2] = c[2]
  # print(disp(image, w, h))



def find_turn_angle(file, window, lastColor, doDialate, prediction, truth):

  image = cv2.imread(file)
  
  image = cv2.GaussianBlur(image, (51, 51), cv2.BORDER_DEFAULT)

  h, w, _ = image.shape
  
  b = 0
  g = 0
  r = 0

  GRAY_THRESHOLD = 10
  THRESHOLD = 50
  NON_BLACK_THRESHOLD = 20
  Y_OFFSET = 0
  counter = 0
  for i in range(w):
    channels = (image[h-1, i, 0], image[h-1, i, 1], image[h-1, i, 2])
    if max(channels) - min(channels) <= GRAY_THRESHOLD and sum(channels) > NON_BLACK_THRESHOLD:
      b += image[h-1, i, 0]
      g += image[h-1, i, 1]
      r += image[h-1, i, 2]
      counter += 1
  if counter == 0:
    r, g, b = lastColor
  else:
    r = r//counter
    g = g//counter
    b = b//counter
  # print(b, g, r)
  

  left = 0
  for i in range(w//2):
    for j in range(h-10-Y_OFFSET, h-Y_OFFSET):
      dis_b = b-image[j, i, 0]
      dis_g = g-image[j, i, 1]
      dis_r = r-image[j, i, 2]
      dis = (dis_b**2 + dis_g**2 + dis_r**2)**(1/2)
      if(dis < THRESHOLD):
        left += 1

  right = 0
  for i in range(w//2, w):
    for j in range(h-10-Y_OFFSET, h-Y_OFFSET):
      dis_b = b-image[j, i, 0]
      dis_g = g-image[j, i, 1]
      dis_r = r-image[j, i, 2]
      dis = (dis_b**2 + dis_g**2 + dis_r**2)**(1/2)
      if(dis < THRESHOLD):
        right += 1

  angle = right / (left+right)
  xPos = int(prediction*w)
  xTruth = int(truth * w)

  average = []
  averageConuter = 0
  for i in range(w):
    dis_b = b-image[h-1-Y_OFFSET, i, 0]
    dis_g = g-image[h-1-Y_OFFSET, i, 1]
    dis_r = r-image[h-1-Y_OFFSET, i, 2]
    dis = (dis_b**2 + dis_g**2 + dis_r**2)**(1/2)
    for j in range(1, 10):
      if dis > THRESHOLD:
        image[h-j-Y_OFFSET, i, 0] = 0
        image[h-j-Y_OFFSET, i, 1] = 0
        image[h-j-Y_OFFSET, i, 2] = 0
      else:
        image[h-j-Y_OFFSET, i, 0] = 0
        image[h-j-Y_OFFSET, i, 1] = 0
        image[h-j-Y_OFFSET, i, 2] = 255
    if(dis < THRESHOLD):
      average.append(i)
      averageConuter += 1

  averageSize = len(average)
  cutoff = int(averageSize * 0.25)

  average = average[cutoff:-cutoff]

  average = sum(average)//len(average)

  turningBias = (average / w) * 0.8 - 0.4

  print(turningBias)

  for j in range(h):
    image[j, xPos, 0] = 0
    image[j, xPos, 1] = 0
    image[j, xPos, 2] = 255
    image[j, xTruth, 0] = 0
    image[j, xTruth, 1] = 255
    image[j, xTruth, 2] = 0
    image[j, average, 0] = 255
    image[j, average, 1] = 0
    image[j, average, 2] = 255

  
  if(doDialate):
    n = 50
    dialate(image, n, w, h, (0,0,0))
    dialate(image, n, w, h, (0,0,255))

  
  
  cv2.imshow(window, image)
  pressed = cv2.waitKey(0) & 0xFF
  if pressed == ord("q"):
    raise InterruptedError("WE DONT WANT TO DO THIS ANYMORE THANK YOU")
  return (r, g, b), (pressed == ord("d")), angle

def adaptive(file):
  img1 = cv2.imread(file, 0)

  img = cv2.medianBlur(img1,5)

  ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

  th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
              cv2.THRESH_BINARY,11,2)

  th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
              cv2.THRESH_BINARY,11,2)

  # titles = ['Original Image', 'Global Thresholding (v = 127)',
  #             'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']

  # images = [img, th1, th2, th3]


  cv2.imshow(file, img1)
  cv2.waitKey(0)
  cv2.imshow(file, th1)
  cv2.waitKey(0)
  cv2.imshow(file, th2)
  cv2.waitKey(0)
  cv2.imshow(file, th3)

  cv2.waitKey(0)
  cv2.destroyAllWindows()

windowName = "Image"
window = cv2.namedWindow(windowName)

truth = np.load("output.npy")
total_error = 0

lastColor = (10, 10, 10)
doDialate = False
prediction = 0.5
for i in range(3000, 4000):
  oh = "input\\"
  s = "img"
  t = f'{i:05d}'
  r = ".png"
  string = oh + s + t + r
  # print(string)
  lastColor, toggleDialate, angle = find_turn_angle(string, windowName, lastColor, doDialate, prediction, truth[i])
  prediction += approach(angle - prediction)
  total_error += (truth[i] - prediction)**2
  # print("total error", total_error / (i+1))
  if(toggleDialate):
    doDialate = not doDialate
  # adaptive(string)


cv2.destroyWindow(windowName)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
