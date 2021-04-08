# Command line parameters
# python demo.py --shape-predictor ./landmarks/shape_predictor_68_face_landmarks.dat --video ./4.mp4
# python demo.py --shape-predictor ./landmarks/shape_predictor_68_face_landmarks.dat
# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import torch
import torch.nn as nn
import torchvision
import os
import numpy as np
from torch.autograd import Variable
from torchvision.datasets import ImageFolder
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import ImageFile
from MobileNetV2 import mymobilenet

ImageFile.LOAD_TRUNCATED_IMAGES = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def data_loader(dataset_dir, batch_size):
    img_data = ImageFolder(dataset_dir, transform=transform)
    data_loader = DataLoader(img_data, batch_size=batch_size, shuffle=False)
    return data_loader


def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return ear


# resNet50训练
# model = torchvision.models.resnet50(num_classes=2)
# model.load_state_dict(torch.load("./ResNet50.pickle"))

# mobileNetV2效果
model = mymobilenet(num_classes=2)
model.load_state_dict(
    torch.load("./mobilenetv2_bs64_lr1e-4_epoch50_reg1e-2.pickle", map_location=device)
)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument(
    "-p", "--shape-predictor", required=True, help="path to facial landmark predictor"
)
ap.add_argument("-v", "--video", type=str, default="", help="path to input video file")
args = vars(ap.parse_args())

# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 2

# initialize the frame counters and the total number of blinks
COUNTER = 0
TOTAL = 0

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# start the video stream thread
print("[INFO] starting video stream thread...")
vs = FileVideoStream(args["video"]).start()
Video_Num = 0
tired = False
fileStream = True
# vs = VideoStream(src=0).start()
m = nn.Softmax(dim=1)
# fileStream = False
time.sleep(1.0)
classes = ("distracted", "focused")
# loop over frames from the video stream
while True:
    # if this is a file video stream, then we need to check if
    # there any more frames left in the buffer to process

    if fileStream and not vs.more():
        break

    # grab the frame from the threaded video file stream, resize
    # it, and convert it to grayscale
    # channels)
    frame = vs.read()
    if frame is None:
        break
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale frame
    rects = detector(gray, 0)

    if len(rects) == 0:
        cv2.putText(
            frame,
            "look forward!!!",
            (125, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
        )

    # loop over the face detections
    for rect in rects:
        top = rect.top() if rect.top() >= 0 else 0
        left = rect.left() if rect.left() >= 0 else 0

        faces = frame[top : rect.bottom(), left : rect.right()]
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        faces = Image.fromarray(cv2.cvtColor(faces, cv2.COLOR_BGR2RGB))
        testdata = transform(faces)
        testdata = Variable(
            torch.unsqueeze(testdata, dim=0).float(), requires_grad=False
        )
        with torch.no_grad():
            model.eval()

            outputs = model(testdata)
            # print(outputs)
            rate = m(outputs)
            _, predicted = torch.max(outputs.data, 1)
            result = classes[predicted]

        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # extract the left and right eye coordinates, then use the
        # coordinates to compute the eye aspect ratio for both eyes
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # average the eye aspect ratio together for both eyes
        ear = (leftEAR + rightEAR) / 2.0

        # compute the convex hull for the left and right eye, then
        # visualize each of the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        # cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        # cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # check to see if the eye aspect ratio is below the blink
        # threshold, and if so, increment the blink frame counter
        if ear < EYE_AR_THRESH:
            COUNTER += 1

        # otherwise, the eye aspect ratio is not below the blink
        # threshold
        # else:
        #     # if the eyes were closed for a sufficient number of
        #     # then increment the total number of blinks
        #     if COUNTER >= EYE_AR_CONSEC_FRAMES:
        #         TOTAL += 1

        #     # reset the eye frame counter
        #     COUNTER = 0
        Video_Num += 1

        if Video_Num == 15 and COUNTER >= 3:
            tired = True
            COUNTER = 0
            Video_Num = 0
        elif Video_Num == 15:
            Video_Num = 0
            tired = False
            COUNTER = 0
        # draw the total number of blinks on the frame along with
        # the computed eye aspect ratio for the frame
        cv2.putText(
            frame,
            "Blinks: {}".format(COUNTER),
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
        print(Video_Num)
        cv2.putText(
            frame,
            "EAR: {:.2f}".format(ear),
            (300, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
        cv2.putText(
            frame,
            "State: {}".format("Tired" if tired else "energetic"),
            (10, 275),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
        cv2.putText(
            frame,
            "Attention: {}".format(result),
            (220, 275),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
        # cv2.putText(frame, "COUNTER: {}".format(COUNTER), (140, 30),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
