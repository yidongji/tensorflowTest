import sys
import cv2
import os
import shutil

#type取值
TYPE_TRAIN = 'train'
TYPE_TEST = 'test'
#cv2识别出人脸后使用的颜色
COLOR_CV2_FRONTALFACE = (0, 255, 0)
#识别出的人脸坐标要往外拓宽多少
FACE_GIRD_EXT_SIZE = 10
#图片储存地址
PATH_FACE_SAVE = "./data/faceImageData"
#一个人需要取多少张训练样本和测试样本
SUM_OF_FACE_TRAIN = 200
SUM_OF_FACE_TEST = 50

#cv2人脸识别分类器地址
#    人脸检测器（Haar_1）：haarcascade_frontalface_alt.xml
#    人脸检测器（Tree）：haarcascade_frontalface_alt_tree.xml
#    人脸检测器（快速的Haar）：haarcascade_frontalface_alt2.xml
#    人脸检测器（默认）：haarcascade_frontalface_default.xml
PATH_CLASSFIER_CV2_FRONTALFACE_ALT2 = "./haarcascade_frontalface_alt2.xml"
# 告诉OpenCV使用人脸识别分类器
classfier = cv2.CascadeClassifier(PATH_CLASSFIER_CV2_FRONTALFACE_ALT2)
#统计已经保存了多少张人脸
num = 0

def getFaceSavePath(name, num, type):
    '''
    获取该张人脸的存放路径
    :param name '人名'
    :param num '第几张人脸'
    :param type '训练类型'
    :return '存放路径'
    '''
    return '{}/{}/{}/{}.jpg'.format(PATH_FACE_SAVE, name, type, num+1)

def getFaceGird(face_rect):
    '''
    获取人脸坐标(并向外扩充10宽度)
    '''
    x, y, w, h = face_rect
    t = y + h + FACE_GIRD_EXT_SIZE  # top
    r = x + w + FACE_GIRD_EXT_SIZE  # right
    b = y - FACE_GIRD_EXT_SIZE  # bottom
    l = x - FACE_GIRD_EXT_SIZE  # left
    return t, b, r, l

def discernAndSaveFace(frame, name, type, sum):
    '''
    识别人脸并保存
    '''
    global num
    # 将当前桢图像转换成灰度图像
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 人脸检测，scaleFactor和minNeighbors分别为图片缩放比例和需要检测的有效点数
    face_rects = classfier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
    #如果只有一张人脸则保存当前帧
    if len(face_rects) == 1:
        t, b, r, l = getFaceGird(face_rects[0]) #坐标
        image = frame[b: t, l: r]
        cv2.imwrite(getFaceSavePath(name, num, type), image)
        num += 1
    #标记人脸
    if len(face_rects) > 0:
        for face_rect in face_rects:  # 单独框出每一张人脸
            t, b, r, l = getFaceGird(face_rect) #坐标
            # 框出人脸
            cv2.rectangle(frame, (l, b), (r, t), COLOR_CV2_FRONTALFACE, 2)
            # 统计当前捕捉的人脸数量
            cv2.putText(frame, '{} num:{}/{}'.format(type, num, sum), (l + 30, b + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 4)

def buildFaceDir(name):
    '''
    创建/清空目标文件夹
    :param name '人名'
    '''
    dir = '{}/{}'.format(PATH_FACE_SAVE, name)
    if os.path.isdir(dir):
        shutil.rmtree(dir)
    os.mkdir(dir)
    #存放训练数据
    os.mkdir('{}/{}'.format(dir, TYPE_TRAIN))
    #存放测试数据
    os.mkdir('{}/{}'.format(dir, TYPE_TEST))

def catchFaceFromCamera(name):
    '''
    从摄像头捕捉人脸并保存为训练/测试样本
    :param name 人名
    '''
    global num
    print('start to catch face of {}'.format(name))
    #要捕捉的人脸总数
    catch_sum = SUM_OF_FACE_TRAIN
    # 调用摄像头，conf.CAMERA_IDX为摄像头索引，默认为0
    cap = cv2.VideoCapture(0)
    #构建该人脸的存放目录
    buildFaceDir(name)
    #开始捕捉数据（人脸）
    current_type = TYPE_TRAIN
    while True:
        #训练数据捕捉完了，则捕捉测试数据
        if num >= catch_sum and current_type == TYPE_TRAIN:
            current_type = TYPE_TEST
            num = 0
            catch_sum = SUM_OF_FACE_TEST
        #测试数据也捕捉完了，则退出
        elif num >= catch_sum and current_type == TYPE_TEST:
            break
        #其他情况则退出
        elif num >= catch_sum:
            raise Exception('current_type error')
        # 读取一帧数据
        if cap.isOpened()==False:
            break
        ok, frame = cap.read()
        if not ok:
            break
        #识别人脸并保存
        if current_type == TYPE_TRAIN:
            discernAndSaveFace(frame, name, TYPE_TRAIN, SUM_OF_FACE_TRAIN)
        else:
            discernAndSaveFace(frame, name, TYPE_TEST, SUM_OF_FACE_TEST)
        # 显示图像
        cv2.imshow(name, frame)
        #监听输入，按esc退出
        # if cv2.waitKey(10) & 0xFF == 27:
        #     break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # 释放摄像头并销毁所有窗口
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # 开始截取脸部图像
    people_name = 'jk'
    catchFaceFromCamera(people_name)

