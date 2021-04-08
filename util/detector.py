import cv2
import dlib
import os

detector = dlib.get_frontal_face_detector()

path = "./data/test_set/focus/"
cnt = 0
for file in os.listdir(path):
    if ".jpg" in file or "png" in file:
        image = cv2.imread(path + file)
        print(file)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detections = detector(gray, 0)

        for rect in detections:
            cnt += 1
            face = image[rect.top() if rect.top() >= 0 else 0 : rect.bottom(), rect.left() if rect.left() >= 0 else 0 : rect.right()]

            cv2.imwrite("../../../download/data/20201209{:05d}.jpg".format(cnt), face)
