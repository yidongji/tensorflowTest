import cv2
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

#type取值
TYPE_TRAIN = 'train'
TYPE_TEST = 'test'
#计算机摄像设备索引
CAMERA_IDX = 0
#cv2人脸识别分类器地址
PATH_CLASSFIER_CV2_FRONTALFACE_ALT2 = "./haarcascade_frontalface_alt2.xml"
#数据集储存地址
PATH_DATASET_SAVE = "./data/dataset"
#模型储存地址
PATH_MODEL_SAVE = "./data/model"
#识别出的人脸坐标要往外拓宽多少
FACE_GIRD_EXT_SIZE = 10
#人脸图片的大小
SIZE = 64
#cv2识别出人脸后使用的颜色
COLOR_CV2_FRONTALFACE = (0, 255, 0)
#调整图片大小时扩充的地方填充的颜色
RESIZE_FILL_COLOR = (0,0,0)

# 告诉OpenCV使用人脸识别分类器
classfier = cv2.CascadeClassifier(PATH_CLASSFIER_CV2_FRONTALFACE_ALT2)
#统计已经保存了多少张人脸
num = 0

x_data = tf.placeholder(tf.float32, [None, SIZE, SIZE, 3])
y_data = tf.placeholder(tf.float32, [None, None])
keep_prob_5 = tf.placeholder(tf.float32)
keep_prob_75 = tf.placeholder(tf.float32)

def weightVariable(shape):
    '''
    定义Weight变量，输入shape，返回变量的参数。
    其中我们使用了tf.random_normal产生随机变量来进行初始化
    '''
    init = tf.random_normal(shape, stddev=0.01)
    #init = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(init)

def biasVariable(shape):
    '''
    定义biase变量，输入shape，返回变量的一些参数。
    '''
    init = tf.random_normal(shape)
    #init = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(init)

def conv2d(x, W):
    '''
    定义卷积操作。
    tf.nn.conv2d函数是Tensorflow里面的二维的卷积函数，
    x是图片的所有参数，W是卷积层的权重，
    然后定义步长strides=[1,1,1,1]值。
    strides[0]和strides[3]的两个1是默认值，
    意思是不对样本个数和channel进行卷积，中间两个1代表padding是在x方向运动一步，y方向运动一步，padding采用的方式实“SAME”就是0填充。
    :param x:
    :param W:
    :return:
    '''
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def maxPool(x):
    '''
    定义池化操作。
    为了得到更多的图片信息，卷积时我们选择的是一次一步，
    也就是strides[1]=strides[2]=1,这样得到的图片尺寸没有变化，
    而我们希望压缩一下图片也就是参数能少一些从而减少系统的复杂度，
    因此我们采用pooling来稀疏化参数，也就是卷积神经网络中所谓的下采样层。
    '''
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def dropout(x, keep):
    '''为了防止过拟合的问题，可以加一个dropout的处理。'''
    return tf.nn.dropout(x, keep)

def cnnLayer(classnum):
    '''创建卷积层'''
    # 第一层
    W1 = weightVariable([3, 3, 3, 32]) # 卷积核大小(3,3)， 输入通道(3)， 输出通道(32)
    b1 = biasVariable([32])
    conv1 = tf.nn.relu(conv2d(x_data, W1) + b1)
    pool1 = maxPool(conv1)
    # 减少过拟合，随机让某些权重不更新
    drop1 = dropout(pool1, keep_prob_5) # 32 * 32 * 32 多个输入channel 被filter内积掉了

    # 第二层
    W2 = weightVariable([3, 3, 32, 64])
    b2 = biasVariable([64])
    conv2 = tf.nn.relu(conv2d(drop1, W2) + b2)
    pool2 = maxPool(conv2)
    drop2 = dropout(pool2, keep_prob_5) # 64 * 16 * 16

    # 第三层
    W3 = weightVariable([3, 3, 64, 64])
    b3 = biasVariable([64])
    conv3 = tf.nn.relu(conv2d(drop2, W3) + b3)
    pool3 = maxPool(conv3)
    drop3 = dropout(pool3, keep_prob_5) # 64 * 8 * 8

    # 全连接层
    Wf = weightVariable([8*16*32, 512])
    bf = biasVariable([512])
    drop3_flat = tf.reshape(drop3, [-1, 8*16*32])
    dense = tf.nn.relu(tf.matmul(drop3_flat, Wf) + bf)
    dropf = dropout(dense, keep_prob_75)

    # 输出层
    Wout = weightVariable([512, classnum])
    bout = weightVariable([classnum])
    resMat = tf.matmul(dropf, Wout)

    #out = tf.matmul(dropf, Wout) + bout
    # out = tf.add(tf.matmul(dropf, Wout), bout) # 原始数据输出
    # 输出层归一化
    # Sigmoid函数可以用来解决多标签问题，Softmax函数用来解决单标签问题
    # out = tf.add(tf.sigmoid(resMat), bout) # [array([[0.0795017 , 0.03605248, 0.9799969 ]]
    out = tf.add(tf.nn.softmax(resMat), bout) # [array([[0.00744988, 0.03517907, 0.979998  ]]

    print(f'tf.matmul(dropf, Wout):{tf.sigmoid(tf.matmul(dropf, Wout))}')
    return out

def loadDataset(type):
    '''
    读取数据集
    '''
    dir = PATH_DATASET_SAVE
    images = np.load('{}/{}_images.npy'.format(dir, type))
    labels = np.load('{}/{}_labels.npy'.format(dir, type))
    names_map = loadDict('{}/{}_names_map.npy'.format(dir, type))
    return images, labels, names_map

def loadDict(path):
    '''
    读取字典
    '''
    f = open(path, 'r')
    a = f.read()
    dict = eval(a)
    f.close()
    return dict

def getFaceGird(face_rect):
    '''
    获取人脸坐标
    '''
    x, y, w, h = face_rect
    t = y + h + FACE_GIRD_EXT_SIZE #top
    r = x + w + FACE_GIRD_EXT_SIZE #right
    b = y - FACE_GIRD_EXT_SIZE #bottom
    l = x - FACE_GIRD_EXT_SIZE #left
    return t, b, r, l

'''
识别摄像头里的人脸并打上名字
'''
def testFaceFromCamera() -> object:
    chkpoint = "{}/model.ckpt".format(PATH_MODEL_SAVE)
    # 调用摄像头，conf.CAMERA_IDX为摄像头索引，默认为0，也可以这样写cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(CAMERA_IDX)
    #
    test_x, test_y, names_map = loadDataset(TYPE_TEST)
    output = cnnLayer(test_y.shape[1])
    predict = output
    saver = tf.train.Saver()
    with tf.Session() as sess:
        #还原模型变量
        saver.restore(sess, chkpoint)
        while True:
            # 读取一帧数据
            if cap.isOpened()==False:
                break
            ok, frame = cap.read()
            if not ok:
                break
            #识别

            discernAndCallFace(frame, sess, predict, output, names_map)
            #显示图像
            cv2.imshow('testFace', frame)
            #监听输入，按esc退出
            c = cv2.waitKey(10)
            if c & 0xFF == 27:
                break
    # 释放摄像头并销毁所有窗口
    cap.release()
    cv2.destroyAllWindows()

def resizeImage(image, height, width):
    '''按照指定图像大小调整尺寸'''
    top, bottom, left, right = (0, 0, 0, 0)
    # 获取图像尺寸
    h, w, _ = image.shape
    # 对于长宽不相等的图片，找到最长的一边
    longest_edge = max(h, w)
    # 计算短边需要增加多上像素宽度使其与长边等长
    if h < longest_edge:
        dh = longest_edge - h
        top = dh // 2
        bottom = dh - top
    elif w < longest_edge:
        dw = longest_edge - w
        left = dw // 2
        right = dw - left
    else:
        pass
    # 给图像增加边界，是图片长、宽等长，cv2.BORDER_CONSTANT指定边界颜色由value指定
    constant = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=RESIZE_FILL_COLOR)
    # 调整图像大小并返回
    return cv2.resize(constant, (height, width))

'''
识别人脸并标注名字
'''
def discernAndCallFace(frame, sess, predict, output, names_map):
    # 将当前桢图像转换成灰度图像
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 人脸检测，scaleFactor和minNeighbors分别为图片缩放比例和需要检测的有效点数
    face_rects = classfier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
    #标记人脸
    if len(face_rects) > 0:
        for face_rect in face_rects:  # 单独框出每一张人脸
            t, b, r, l = getFaceGird(face_rect) #坐标
            # 框出人脸
            cv2.rectangle(frame, (l, b), (r, t), COLOR_CV2_FRONTALFACE, 2)
            #调整图像尺寸
            image = frame[b: t, l: r]
            image = resizeImage(image, SIZE, SIZE)
            #转换成模型输入变量
            test_x = np.array([image])
            test_x = test_x.astype(np.float32) / 255.0
            #计算分类概率，我们用到的是res[1][0]，res的结构为[([[-2.8184052,  2.8337383]], dtype=float32), ([1], dtype=int64)]
            res = sess.run([predict, tf.argmax(output, 1)], feed_dict={x_data: test_x, keep_prob_5: 1.0, keep_prob_75: 1.0})

            # res的内容为：[array([[-3.935829, -7.281601, 16.272253]], dtype=float32), array([2])]
            print(res)

            # 显示名字
            # 需要的names_map格式 ：
            # names_map:{0: 'chenpiaoqi', 1: 'jinkang', 2: 'shaobixing'}
            # if 0 not in names_map.keys(): # 转换成需要的格式
            #     names_map = dict([val, key] for key, val in names_map.items())
            # else:
            #     pass

            print(f'names_map:{names_map}')
            cv2.putText(frame, '{}'.format(names_map[res[1][0]]), (l + 30, b + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 4)
    else:
        print('没有人脸')

def displayImage(path):
    """
    识别单张图片
    :param path:
    :return:
    """
    image = cv2.imread(path)
    # cv2.imshow('img_window', image)
    # cv2.waitKey(0)

    chkpoint = "{}/model.ckpt".format(PATH_MODEL_SAVE)
    test_x, test_y, names_map = loadDataset(TYPE_TEST)
    output = cnnLayer(test_y.shape[1])
    predict = output
    saver = tf.train.Saver()
    with tf.Session() as sess:
        #还原模型变量
        saver.restore(sess, chkpoint)
        discernAndCallFace(image, sess, predict, output, names_map)
        #显示图像
        cv2.imshow('testFace', image)
        cv2.waitKey(0)

if __name__ == '__main__':
    # 摄像头识别
    testFaceFromCamera()

    # path='/Users/mac/Downloads/test.jpg'
    # displayImage(path)

