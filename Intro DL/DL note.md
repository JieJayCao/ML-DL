# 一、工欲善其事

> ## 1 jupyter notebook
>

人工智能：把人从生产力和生产关系的桎梏当中解放出来

工欲善其事，必先利其器，jupyter notebook 一款常用于机器学习与数据科学领域的工具，是一种 Web 应用，能让用户将说明文本、数学方程、代码和可视化内容全部组合到一个易于共享的文档中。Anaconda是一个集成了大量科学库的python环境，里面也内置了jupyter notebook。

![1544669781303](C:\Users\Albert-CJ\AppData\Roaming\Typora\typora-user-images\1544669781303.png)

> ## 2 TensorFlow
>

由Google brain 开发的一款开源深度学习框架

tensorboard：可以将深度学习的流程与数据结构、运算过程可视化的工具 内嵌于tensorflow



# 二 、神经网络中的数学问题与解决方法

> ## 1  从人神经网络至人造神经网络
>

​	人脑中轴突传输信号，神经末梢接受信号，细胞核接受信号，当收到刺激时电信号会在神经网络中传递并在细胞核内积累，当超过阈值时将会传递给下一个神经元。

<img src="C:\Users\Albert-CJ\AppData\Roaming\Typora\typora-user-images\1544669805987.png" width=50 height=10/>

​	通过神经元的处理模式，拓扑构造出人造神经网络

![1544669810897](C:\Users\Albert-CJ\AppData\Roaming\Typora\typora-user-images\1544669810897.png)

 ### （1）感知机

​	感知机模型可成功解决线性分类问题有清晰的决策边界，但是大部分神经网络与现实问题都属于非线性问题（比如XOR问题）。

![1544670012765](C:\Users\Albert-CJ\AppData\Roaming\Typora\typora-user-images\1544670012765.png)

​	

![1544670018877](C:\Users\Albert-CJ\AppData\Roaming\Typora\typora-user-images\1544670018877.png)

因此从1970后十年进入研究冰河期

### （2）多层感知机——双层神经网络

* 全连接：每一个输入都与下一层相连

* 通过引入hidden layer 与 active function来解决非线性问题。一个双层神经网络，相较于感知机模型多了一层隐藏层。公式内的所有w和a均是矩阵

* active function为神经网络模型引入非线性因素。

* bias：假如输入  的输入分布如图中蓝点（A集合）和红点（B集合）所示（在x轴上的分布），sigmoid函数中的W系数需要学的很大，才能保证尽可能的判断准确。但是如果一个测试样本在图中绿点所在的位置呢，很明显我们可以将绿点判为红点所在的B集合，但是通过训练学到的W是不能正确判断的。如果要通过学w去解决的话那w要学的很大，因此引入bias对函数进行简单平移即可，就是简单用于调整函数的。

![1544670147906](C:\Users\Albert-CJ\AppData\Roaming\Typora\typora-user-images\1544670147906.png)

​	通过多层神经网络可以拟合所有连续函数图像。

### （3）模型训练

1. y是正确的输出值,yp为预测值，loss就可以表达为w的函数，这就是损失函数。问题转化为：如何优化参数，使损失函数最少

2. 欧式距离其实是一个不佳的计算损失函数，所以绝大部分模型都会使用softmax（归一化指数函数）+cross entropy    

3. 我们知道max，假如说我有两个数，a和b，并且a>b，如果取max，那么就直接取a，没有第二种可能。但有的时候我不想这样，因为这样会造成分值小的那个饥饿。所以我希望分值大的那一项经常取到，分值小的那一项也偶尔可以取到，那么我用softmax就可以了 现在还是a和b，a>b，如果我们取按照softmax来计算取a和b的概率，那a的softmax值大于b的，所以a会经常取到，而b也会偶尔取到。

4. 不确定度越大熵值越大，根据熵的计算公式可知一个事件发生的概率越大熵越小，损失函数越小

5. Softmax 将所有输出映射到【0，1】，每一个输出点都对应一个概率，集合成概率向量。这样就可以将输出理解为概率，对概率进行信息量的计算这学期的信息论大家都上过，我们使用熵作为更好的损失函数

6. 因此我们需要找到一个新的损失函数——交叉熵，具体原因我也还在想没完全搞明白所以不讲，但是这是一个好的算损失函数的模型

### (4)优化问题

我们其实已经了解神经网络的基本构造，但是任何模型都离不开优化问题

1. 梯度下降

   损失函数最终将表示成为w与b的函数，求loss的最小值，也就是求（loss）导数为0时w与b的值，但是对于神  	经网络而言w数量非常多因此不可能一次求出，就引入了梯度下降与反向传播算法。



   ![1544670492803](C:\Users\Albert-CJ\AppData\Roaming\Typora\typora-user-images\1544670492803.png)

2. 我们同时可以假设这座山最陡峭的地方是无法通过肉眼立马观察出来的，而是需要一个复杂的工具来测量，同时，这个人此时正好拥有测量出最陡峭方向的能力。所以，此人每走一段距离，都需要一段时间来测量所在位置最陡峭的方向，这是比较耗时的。那么为了在太阳下山之前到达山底，就要尽可能的减少测量方向的次数。这是一个两难的选择，如果测量的频繁，可以保证下山的方向是绝对正确的，但又非常耗时，如果测量的过少，又有偏离轨道的风险。所以需要找到一个合适的测量方向的频率，来确保下山的方向不错误，同时又不至于耗时太多！

![img](https://upload-images.jianshu.io/upload_images/1234352-f20521a962005299.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1000/format/webp)

​	![img](https://upload-images.jianshu.io/upload_images/1234352-abb73822fb6d2a2c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/127/format/webp)

![img](https://upload-images.jianshu.io/upload_images/1234352-57538d21dbb34e65.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/281/format/webp)

3. 学习率取值

学习速率过大会导致参数进行剧烈摇摆，模型容易爆炸，而过小的学习率，会导致训练速度过慢，因为函数将考虑每一个局部的极小值。

![img](https://upload-images.jianshu.io/upload_images/1234352-ba3da0b06da97ddb.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/827/format/webp)



​	所以学习率不能一成不变，要根据训练情况动态调整

4. 过拟合情况

​	我们可能会以为节点数，层数越多越好其实不然，虽然模型会高度符合训练集数据，但是他不一定完全符合现实的实际情况。

​	节点与层数足够多，甚至大于输入数量的时候，就可以将所有点均进行拟合，连成函数曲线，这条线会过于严格。然而实际生活中所有的输入都会含有噪声，这其实是不用考虑的，所以函数拟合应该要符合整体发展的趋势。

![1544670590426](C:\Users\Albert-CJ\AppData\Roaming\Typora\typora-user-images\1544670590426.png)

因此采用正则化的方法防止过拟合（具体不解释）

#三、算力提高

其实于上个世纪神经网络的模型就已经发展的较为成熟,可能遇到的数学问题都有合适的处理方案,主要限制神经网络进一步发展的原因就是计算机的算力不够,神经网络的复杂计算给i计算机的性能提出了极高的要求.所以当高性能GPU快速发展的最近十年,以及移动终端设备性能的不断提高使得神经网络才有了用武之地.





