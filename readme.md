#项目下载后请执行以下步骤
#准备工作
####Python3.x；
####tensorflow（深度学习框架，可以用anaconda安装）；
####opencv（Python图像处理库）；
####摄像头（用于捕获人脸数据）；

#执行顺序（源码在src目录下）
###获取人脸数据
python catchFace.py
###制作数据集
python mkDataset.py
###训练cnn
python trainFace.py
###测试
python testImageFace.py

###总运行文件
运行 run_model.py 依次执行1，2，3步骤生成模型，使用4或5进行图片或视频识别。

