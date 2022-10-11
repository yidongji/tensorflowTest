import numpy as np
import logging as log
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import cv2

#type取值
TYPE_TRAIN = 'train'
TYPE_TEST = 'test'
#数据集储存地址
PATH_DATASET_SAVE = "./data/dataset"
#人脸图片的大小
SIZE = 64
#模型储存地址
PATH_MODEL_SAVE = "./data/model"
#循环一次训练多少样本
TRAIN_BATCH_SIZE = 100
#循环训练次数
TRAIN_TIMES = 20

def loadDict(path):
    '''
    读取字典
    '''
    f = open(path, 'r')
    a = f.read()
    dict = eval(a)
    f.close()
    return dict

def loadDataset(type):
    '''
    读取数据集
    '''
    dir = PATH_DATASET_SAVE
    images = np.load('{}/{}_images.npy'.format(dir, type))
    labels = np.load('{}/{}_labels.npy'.format(dir, type))
    names_map = loadDict('{}/{}_names_map.npy'.format(dir, type))
    return images, labels, names_map

x_data = tf.placeholder(tf.float32, [None, SIZE, SIZE, 3])
y_data = tf.placeholder(tf.float32, [None, None])

keep_prob_5 = tf.placeholder(tf.float32)
keep_prob_75 = tf.placeholder(tf.float32)

def weightVariable(shape):
    '''定义Weight变量，输入shape，返回变量的参数。其中我们使用了tf.random_normal产生随机变量来进行初始化'''
    init = tf.random_normal(shape, stddev=0.01)
    #init = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(init)

def biasVariable(shape):
    ''' 定义biase变量，输入shape，返回变量的一些参数。'''
    init = tf.random_normal(shape)
    #init = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(init)

def conv2d(x, W):
    '''
    定义卷积操作。tf.nn.conv2d函数是Tensorflow里面的二维的卷积函数，x是图片的所有参数，W是卷积层的权重，然后定义步长strides=[1,1,1,1]值。strides[0]和strides[3]的两个1是默认值，意思是不对样本个数和channel进行卷积，中间两个1代表padding是在x方向运动一步，y方向运动一步，padding采用的方式实“SAME”就是0填充。
    :param x:
    :param W:
    :return:
    '''
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def maxPool(x):
    '''定义池化操作。为了得到更多的图片信息，卷积时我们选择的是一次一步，也就是strides[1]=strides[2]=1,这样得到的图片尺寸没有变化，而我们希望压缩一下图片也就是参数能少一些从而减少系统的复杂度，因此我们采用pooling来稀疏化参数，也就是卷积神经网络中所谓的下采样层。'''
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
    #out = tf.matmul(dropf, Wout) + bout
    out = tf.add(tf.matmul(dropf, Wout), bout)
    return out

'''
开始训练模型
'''
def train(train_x, train_y, test_x, test_y, tfsavepath):
    log.debug('train')
    out = cnnLayer(train_y.shape[1])
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=y_data))
    train_step = tf.train.AdamOptimizer(0.01).minimize(cross_entropy)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out, 1), tf.argmax(y_data, 1)), tf.float32))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_times = TRAIN_TIMES
        batch_size = TRAIN_BATCH_SIZE
        num_batch = len(train_x) // batch_size
        for n in range(train_times):
            r = np.random.permutation(len(train_x))
            train_x = train_x[r, :]
            train_y = train_y[r, :]

            for i in range(num_batch):
                batch_x = train_x[i*batch_size : (i+1)*batch_size]
                batch_y = train_y[i*batch_size : (i+1)*batch_size]
                #loss:损失函数的值，当实际输出接近预期，那么损失函数应该接近0
                _, loss = sess.run([train_step, cross_entropy],\
                                   feed_dict={x_data:batch_x, y_data:batch_y,
                                              keep_prob_5:0.75, keep_prob_75:0.75})
                print('batch times: {}, loss: {}'.format(n*num_batch+i+1, loss))
            # 获取测试数据的准确率
            acc = accuracy.eval({x_data:test_x, y_data:test_y, keep_prob_5:1.0, keep_prob_75:1.0})
            print('after {} times run: accuracy is {}'.format(n+1, acc))

        # 获取测试数据的准确率
        # acc = accuracy.eval({x_data:test_x, y_data:test_y, keep_prob_5:1.0, keep_prob_75:1.0})
        # print('after {} times run: accuracy is {}'.format(train_times, acc))
        saver.save(sess, tfsavepath)

if __name__ == '__main__':
    # images, labels, names_map
    train_x, train_y, names_map = loadDataset(TYPE_TRAIN)
    train_x = train_x.astype(np.float32) / 255.0
    test_x, test_y, names_map = loadDataset(TYPE_TEST)
    test_x = test_x.astype(np.float32) / 255.0
    train(train_x, train_y, test_x, test_y, "{}/model.ckpt".format(PATH_MODEL_SAVE))