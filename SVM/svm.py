# -*- coding:UTF-8 -*-
# 这个svm分类器需要先训练再判别，但也可以修改边训练边判别的svm，简单说说思路
# 先获取视频流循环读帧，利用map(hog,trainpic.append(cv2.imread(f)))循环添加
import sys
import cv2
import numpy as np
from glob import glob
from os.path import dirname, join, basename


# 特征统计数量
bin_n = 16*16

# hog特征
def hog(img):
    # 图片长宽像素值
    x_pixel,y_pixel=320,320
    # Sobel函数（1图，2深，3x阶，4y阶）
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    # 笛卡尔坐标转极坐标
    mag, ang = cv2.cartToPolar(gx, gy)
    bins = np.int32(bin_n*ang/(2*np.pi))
    bin_cells = bins[:x_pixel/2,:y_pixel/2], bins[x_pixel/2:,:y_pixel/2], bins[:x_pixel/2,y_pixel/2:], bins[x_pixel/2:,y_pixel/2:]
    mag_cells = mag[:x_pixel/2,:y_pixel/2], mag[x_pixel/2:,:y_pixel/2], mag[:x_pixel/2,y_pixel/2:], mag[x_pixel/2:,y_pixel/2:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    # 64 bit向量x
    hist = np.hstack(hists)
    return hist

img = {}
num = 0

# 循环读猫图
for fn in glob(join('cat', '*.jpg')): #join 多路径组合后返回
    img[num] = cv2.imread(fn,0)#加0，只读取黑白，去0，彩色读取
    num = num + 1
print num,' num'
# 把第一种情况记为positive并输出
positive = num
print positive,' positive'

# 循环读狗图
for fn in glob(join('dog', '*.jpg')):
    img[num] = cv2.imread(fn,0)#加0，只读取黑白，去0，彩色读取
    num = num + 1
print num,' num'


# 训练图片数组
trainpic = []
for i in img:
    trainpic.append(img[i])

# 玄学，普遍推荐的参数配置
svm_params = dict( kernel_type = cv2.SVM_LINEAR, svm_type = cv2.SVM_C_SVC, C = 2.67, gamma = 5.383 )

temp=hog(img[0])


# 用map一一对应hog特征和训练图片
hogdata = map(hog,trainpic)
trainData = np.float32(hogdata).reshape(-1,bin_n*4)
responses = np.float32(np.repeat(1.0,trainData.shape[0])[:,np.newaxis])
responses[positive:trainData.shape[0]]=-1.0


# 初始化svm
svm = cv2.SVM()
# 训练
svm.train(trainData,responses, params=svm_params)
# 先输出svm，这样主要方便日后调整流程
svm.save('svm_data.dat')
svm.load('svm_data.dat')

# 加载你需要测试的照片
img = cv2.imread('test3.jpg',0)
# 取待测试图的hog特征
hogdata = hog(img)
testData = np.float32(hogdata).reshape(-1,bin_n*4)
# 预测

result = svm.predict(testData)
print result
if result > 0:
    print "it's a cat"
else:
    print "it's a dog"
