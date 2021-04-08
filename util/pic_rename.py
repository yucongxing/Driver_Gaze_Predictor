import os

root = "./imagefiles/distract/"
for i, file in enumerate(os.listdir(root)):
    os.rename(root + file, root + "20202121{:05}".format(i) + file[-4:])
