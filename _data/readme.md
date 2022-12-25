# This dir is used for saving raw data

## MNIST

MNIST 共包含四个 IDX 格式文件， IDX 格式是一种用来存储向量与多维度矩阵的文件格式:

* train-images-idx3-ubyte.gz: 训练集 55000 张, 验证集 5000 张
* train-labels-idx1-ubyte.gz: 训练集、验证集对应的数字标签
* t10k-images-idx3-ubyte.gz: 测试集 10000 张
* t10k-labels-idx1-ubyte.gz: 测试集对应的数字标签

每个集合包含图片和标签两部分内容，图片为28*28点阵图；标签为0-9之间数字。这些文件本身并没有使用标准的图片格式储存。使用时需要进行解压和重构