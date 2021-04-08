from PyQt5.QtWidgets import (
    QApplication,
    QLabel,
    QWidget,
    QRadioButton,
    QButtonGroup,
    QPushButton,
    QFileDialog,
    QLineEdit,
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from imutils.video import FileVideoStream
import sys
import cv2
from MobileNetV2 import mymobilenet
import torch
import dlib


class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Driving Engagement Predictor")
        self.resize(1000, 1000)
        self.begin = QPushButton(self)
        self.time = QTimer()
        self.begin.setText("开始")
        self.begin.move(200, 600)
        self.set_modelgroup()
        self.set_filegroup()
        self.set_fileselect()
        self.vs = FileVideoStream("./3.mp4").start()
        self.Video_Num = 0
        self.img = None
        self.flag = False
        self.camera_label = QLabel(self)
        self.camera_label.setText("视频播放...")
        self.camera_label.resize(500, 1000)
        self.camera_label.move(500, 0)
        self.camera_label.setStyleSheet("background-color:grey; font-size: 20px")
        self.camera_label.setAlignment((Qt.AlignCenter))
        self.face_cascade = cv2.CascadeClassifier(
            r"C:\software\anaconda3\envs\distract\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml"
        )
        self.face_cascade.load(
            r"C:\software\anaconda3\envs\distract\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml"
        )
        self.model = mymobilenet(num_classes=2)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.load_state_dict(
            torch.load(
                "./mobilenetv2_bs32_lr1e-4_epoch50_reg1e-2.pickle",
                map_location=self.device,
            )
        )
        self.predictor = dlib.shape_predictor(
            "./landmarks/shape_predictor_68_face_landmarks.dat"
        )
        self.begin.clicked.connect(self.change)
        self.time.timeout.connect(self.show_img)

    def change(self):
        if not self.flag:
            self.flag = not self.flag
            self.start()
            self.begin.setText("暂停")
        else:
            self.flag = not self.flag
            self.end()
            self.begin.setText("开始")

    def start(self):
        self.time.start(30)

    def end(self):
        self.time.stop()

    def set_fileselect(self):
        filepath = QLineEdit(self)
        filepath.resize(180, 24)
        filepath.move(120, 500)
        btn = QPushButton(self)
        btn.setText("选择文件")
        btn.move(300, 500)

        def selectfile(self):
            # dialog = QFileDialog()
            # dialog.setFileMode(QFileDialog.AnyFile)
            # dialog.setFilter(QDir.Files)

            # if dialog.exec_():
            filename, _ = QFileDialog.getOpenFileName(
                None, "Open file", ".", "Video Files (*.mp4 *.ts)"
            )
            print(filename)
            filepath.setText(filename)

        btn.clicked.connect(selectfile)

    def set_modelgroup(self):
        label = QLabel(self)
        label.setText("选择网络模型：")
        label.move(120, 300)
        modelgroup = QButtonGroup(self)
        radiobutton1 = QRadioButton(self)
        radiobutton1.setText("MobileNetV2")
        radiobutton1.setChecked(True)
        radiobutton1.move(200, 300)
        radiobutton2 = QRadioButton(self)
        radiobutton2.setText("ResNet18")
        radiobutton2.move(300, 300)
        modelgroup.addButton(radiobutton1, 0)
        modelgroup.addButton(radiobutton2, 1)

    def set_filegroup(self):
        label = QLabel(self)
        label.setText("选择视频源：")
        label.move(120, 400)
        filegroup = QButtonGroup(self)
        filebutton0 = QRadioButton(self)
        filebutton0.setText("filestream")
        filebutton0.move(200, 400)
        filebutton0.setChecked(True)
        filebutton1 = QRadioButton(self)
        filebutton1.setText("camerastream")
        filebutton1.move(300, 400)
        filegroup.addButton(filebutton0, 0)
        filegroup.addButton(filebutton1, 1)

    def show_img(self):
        from scipy.spatial import distance as dist
        from imutils import face_utils
        import imutils
        import time
        import cv2
        from torch.autograd import Variable
        from torchvision.datasets import ImageFolder
        from PIL import Image

        # from imutils.video import FileVideoStream
        # from imutils.video import VideoStream
        from torch.utils.data import DataLoader
        from torchvision import transforms
        from PIL import ImageFile, ImageDraw, ImageFont
        import numpy as np

        ImageFile.LOAD_TRUNCATED_IMAGES = True

        transform = transforms.Compose(
            [
                transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
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

        # define two constants, one for the eye aspect ratio to indicate
        # blink and then a second constant for the number of consecutive
        # frames the eye must be below the threshold
        EYE_AR_THRESH = 0.25

        # initialize the frame counters and the total number of blinks
        COUNTER = 0

        # initialize dlib's face detector (HOG-based) and then create
        # the facial landmark predictor
        # print("[INFO] loading facial landmark predictor...")

        # grab the indexes of the facial landmarks for the left and
        # right eye, respectively
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

        # start the video stream thread
        # print("[INFO] starting video stream thread...")
        # TODO:如果用现有的视频测试的话修改下面的"./output.mp4"为需要的视频所在路径，把filestream变量改成true
        # 如果调用摄像头则把下面这句注释掉，把vs = VideoStream(src=0).start()这句取消注释，并把fileStream改成false

        # fileStream = True
        # vs = VideoStream(src=0).start()
        # fileStream = False

        time.sleep(1.0)
        classes = ("分心", "专心")
        tired = False
        # loop over frames from the video stream
        # if this is a file video stream, then we need to check if
        # there any more frames left in the buffer to process

        # grab the frame from the threaded video file stream, resize
        # it, and convert it to grayscale
        # channels)
        frame = self.vs.read()

        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces in the grayscale frame
        # rects = detector(gray, 0)
        rects = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.2, flags=1, minSize=(32, 32)
        )

        if len(rects) == 0:
            frame = addtext(
                frame, "请注意前方！！", 60, frame.shape[1] / 2 - 200, frame.shape[0] / 2
            )

        else:
            rects = [max(rects, key=lambda x: x[2] * x[3])]
        # loop over the face detections
        for rect in rects:

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
                self.model.eval()

                outputs = self.model(testdata)
                # print(outputs)
                _, predicted = torch.max(outputs.data, 1)
                result = classes[predicted]

            shape = self.predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # extract the left and right eye coordinates, then use the
            # coordinates to compute the eye aspect ratio for both eyes
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]

            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            # average the eye aspect ratio together for both eyes
            ear = (leftEAR + rightEAR) / 2.0

            if ear < EYE_AR_THRESH:
                COUNTER += 1

            self.Video_Num += 1

            if self.Video_Num == 30 and COUNTER >= 3:
                tired = True
                COUNTER = 0
                self.Video_Num = 0
            elif self.Video_Num == 30:
                self.Video_Num = 0
                tired = False
                COUNTER = 0

            frame = addtext(frame, "眨眼次数：{}".format(COUNTER), 30, 10, 30)

            frame = addtext(frame, "EAR指数：{:.02f}".format(ear), 30, 250, 30)

            frame = addtext(
                frame,
                "状态: {}".format("疲劳" if tired else "精神"),
                30,
                10,
                frame.shape[0] - 50,
            )

            frame = addtext(
                frame,
                "注意力：{}".format(result),
                30,
                frame.shape[1] - 200,
                frame.shape[0] - 50,
            )
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = QImage(
                frame.data,
                frame.shape[1],
                frame.shape[0],
                frame.shape[1] * 3,
                QImage.Format_RGB888,
            )
            self.camera_label.setPixmap(QPixmap.fromImage(img))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())
