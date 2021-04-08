import cv2
import os

# import dlib
# import matplotlib.image as mp

face_cascade = cv2.CascadeClassifier(
    r"D:\software\anaconda3\envs\distract\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml"
)
face_cascade.load(
    r"D:\software\anaconda3\envs\distract\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml"
)

i = 0
imagePath = "../../download/pics/2002/09"
print(os.listdir(imagePath))

cnt = 00000
for dirs in os.listdir(imagePath):  # 开始一张一张索引目录中的图像
    dir = imagePath + "/" + dirs + "/big/"
    for file in os.listdir(dir):
        print(file)
        if ".jpg" in file or ".png" in file:
            img = cv2.imread(dir + file)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.2, flags=1, minSize=(32, 32)
            )

            if len(faces) > 0:  # 大于0则检测到人脸
                # count = len(faces)
                # if count > pre:
                for i, face in enumerate(faces):  # 单独框出每一张人脸
                    x, y, w, h = face
                    if (w < 100 or h < 100):
                        continue
                    cnt += 1
                    face = img[y : y + h, x : x + w]
                    cv2.imwrite(
                        "../../download/face_pics/20201206{:05}".format(cnt) + file[-4:],
                        face,
                    )
