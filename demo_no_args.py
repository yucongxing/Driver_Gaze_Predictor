# Command line parameters
# python demo.py --shape-predictor ./shape_predictor_68_face_landmarks.dat --video ./output.mp4
# python demo.py --shape-predictor ./shape_predictor_68_face_landmarks.dat

# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import imutils
import time
import dlib
import cv2
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.datasets import ImageFolder
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import ImageFile, ImageDraw, ImageFont
from MobileNetV2 import mymobilenet
import numpy as np


face_cascade = cv2.CascadeClassifier(
    r"C:\software\anaconda3\envs\distract\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml"
)
face_cascade.load(
    r"C:\software\anaconda3\envs\distract\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml"
)

ImageFile.LOAD_TRUNCATED_IMAGES = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def addtext(frame, text, size, left, top):
    if isinstance(frame, np.ndarray):
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("./simhei.ttf", size=size, encoding="utf-8")
    draw.text((left, top), text, fill=(255, 0, 0), font=font)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


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
    torch.load("./mobilenetv2_bs32_lr1e-4_epoch50_reg1e-2.pickle", map_location=device)
)

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
predictor = dlib.shape_predictor("./landmarks/shape_predictor_68_face_landmarks.dat")

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# start the video stream thread
print("[INFO] starting video stream thread...")
# TODO:如果用现有的视频测试的话修改下面的"./output.mp4"为需要的视频所在路径，把filestream变量改成true
# 如果调用摄像头则把下面这句注释掉，把vs = VideoStream(src=0).start()这句取消注释，并把fileStream改成false
vs = FileVideoStream("./3.mp4").start()
fileStream = True
# vs = VideoStream(src=0).start()
# fileStream = False

m = nn.Softmax(dim=1)
time.sleep(1.0)
classes = ("分心", "专心")
Video_Num = 0
tired = False
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
    # rects = detector(gray, 0)
    rects = face_cascade.detectMultiScale(
        gray, scaleFactor=1.2, flags=1, minSize=(32, 32)
    )

    if len(rects) == 0:
        frame = addtext(frame, "请注意前方！！", 60, frame.shape[1] / 2 - 200, frame.shape[0] / 2)
        # cv2.putText(
        #     frame,
        #     "look forward!!!",
        #     (125, 100),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     1.0,
        #     (0, 0, 255),
        #     2,
        # )
    else:
        rects = [max(rects, key=lambda x: x[2] * x[3])]
    # loop over the face detections
    for rect in rects:
        # top = rect.top() if rect.top() >= 0 else 0
        # left = rect.left() if rect.left() >= 0 else 0
        x, y, w, h = rect
        face = frame[y : y + h, x : x + w]
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        face = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        testdata = transform(face)
        testdata = Variable(
            torch.unsqueeze(testdata, dim=0).float(), requires_grad=False
        )
        with torch.no_grad():
            model.eval()

            outputs = model(testdata)
            # print(outputs)
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

        if Video_Num == 30 and COUNTER >= 3:
            tired = True
            COUNTER = 0
            Video_Num = 0
        elif Video_Num == 30:
            Video_Num = 0
            tired = False
            COUNTER = 0
        # draw the total number of blinks on the frame along with
        # the computed eye aspect ratio for the frame
        # cv2.putText(
        #     frame,
        #     "Blinks: {}".format(COUNTER),
        #     (10, 30),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.7,
        #     (0, 0, 255),
        #     2,
        # )
        frame = addtext(frame, "眨眼次数：{}".format(COUNTER), 30, 10, 30)
        # cv2.putText(
        #     frame,
        #     "EAR: {:.2f}".format(ear),
        #     (300, 30),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.7,
        #     (0, 0, 255),
        #     2,
        # )
        frame = addtext(frame, "EAR指数：{:.02f}".format(ear), 30, 250, 30)
        # cv2.putText(
        #     frame,
        #     "状态: {}".format("疲劳" if tired else "精神"),
        #     (10, frame.shape[0] - 50),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.7,
        #     (0, 0, 255),
        #     2,
        # )
        frame = addtext(frame, "状态: {}".format("疲劳" if tired else "精神"), 30, 10, frame.shape[0] - 50)
        # print(frame.shape) ---> [height*width* 3 channels]
        # cv2.putText(
        #     frame,
        #     "Attention: {}".format(result),
        #     (frame.shape[1] - 250, frame.shape[0] - 50),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     0.7,
        #     (0, 0, 255),
        #     2,
        # )
        frame = addtext(frame, "注意力：{}".format(result), 30, frame.shape[1] - 200, frame.shape[0] - 50)
        # cv2.putText(frame, "COUNTER: {}".format(COUNTER), (140, 30),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        print(frame.shape)

    # show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
