## ~~version 1.0~~(已弃用，哈哈哈)

## version 1.1

### 扩充了一部分数据库，现在大约有两千张左右的图片

### 把数据预处理的一些脚本放到util文件夹，现util文件夹可以当作python的模块来调用其中的各种预处理工具了

### 改了一些逻辑，现在demo_no_args直接点击运行就行了（不用输入一大堆）

### 改了一些逻辑，现在时评行的提醒字母会跟据视频文件或者摄像头的分辨率自适应分布，不会出现换一个视频文字消失的情况啦！

### demo_no_args.py中采用CascadeClassifier来进行人脸检测，和数据集更加接近，可以提高准确率！

### 写了一个mymobilenet 在mobilenetV2的基础下加了一个dropout+全连接，可以使用预训练的参数，加快训练速度

### mobilenetv2_bs64_lr1e-4_epoch50_reg1e-2.pickle这个是训练好的模型，batch_size=64,learningrate=0.00001,num_epochs=50,reg=0.01，测试准确率为95%