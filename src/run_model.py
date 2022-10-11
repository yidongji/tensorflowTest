import catchFace
import mkDataset
import trainFace
import testImageFace

import numpy as np

#type取值
TYPE_TRAIN = 'train'
TYPE_TEST = 'test'

#模型储存地址
PATH_MODEL_SAVE = "./data/model"

def run_model(step, img_path=''):
    """
    依次执行步骤1，2，3生成数据模型，执行4或者5进行视频或者图片比对
    :param step:
    :param img_path:
    :return:
    """
    if step == 1:
        # 使用摄像头捕捉图像，生成训练和测试的数据集合
        people_name = 'jk'
        catchFace.catchFaceFromCamera(people_name)
    elif step == 2:
        # # 制作训练集
        images, labels, names_map = mkDataset.mkDataset(TYPE_TRAIN)
        mkDataset.saveDataset(images, labels, names_map, TYPE_TRAIN)
        # 制作测试集
        images, labels, names_map = mkDataset.mkDataset(TYPE_TEST)
        mkDataset.saveDataset(images, labels, names_map, TYPE_TEST)
    elif step == 3:
        # 训练模型
        # images, labels, names_map
        train_x, train_y, names_map = trainFace.loadDataset(TYPE_TRAIN)
        train_x = train_x.astype(np.float32) / 255.0
        test_x, test_y, names_map = trainFace.loadDataset(TYPE_TEST)
        test_x = test_x.astype(np.float32) / 255.0
        trainFace.train(train_x, train_y, test_x, test_y, "{}/model.ckpt".format(PATH_MODEL_SAVE))
    elif step == 4:
        # 摄像头识别
        testImageFace.testFaceFromCamera()
    elif step == 5:
        if len(img_path) != 0:
            testImageFace.displayImage(img_path)
        else:
            print('请输入有效的图片路径')

if __name__ == '__main__':
    img_path = '/Users/mac/Downloads/1.jpg'
    run_model(5, img_path)
    pass