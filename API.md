# <center>A²PI²-API version2.2</center>

* 2.2版本添加了关于M1芯片的适配情况, 使用Rosetta 2实现的运行的也将标记为否.

# 1.catboost

| 版本   | 描述                 | 注意                 | 适配M1 |
| ------ | -------------------- | -------------------- | ------ |
| 0.24.4 | 梯度提升决策树(GBDT) | 可直接在sklearn使用. | 否     |

# 2.cv2

| 版本  | 描述           | 注意                                                         | 适配M1 |
| ----- | -------------- | ------------------------------------------------------------ | ------ |
| 4.5.0 | 图像处理软件库 | 1. M1目前需要使用conda安装, 使用conda install opencv                                                                                   2. OpenCV的图片格式是HWC, TensorFlow的是WHC.                                                                                          3. Intel-based Mac安装时使用pip install opencv-python | 是     |

## 2.1.imread()

加载指定路径的图片.|numpy.ndarray

```python
import cv2

image = cv2.imread(filename='./image.png',  # str|要加载的文件的路径.
                   flags=2)  # int|None|读入的色彩方式.
```

## 2.2.resize()

将图像调整到指定大小.|numpy.ndarray

```python
import cv2

image = cv2.imread('./image.png')
image = cv2.resize(src=image,  # numpy.ndarray|输入的图像.
                   dsize=(2000, 1000))  # tuple|修改后图像的尺寸.
```

# 3.Eigen

| 版本  | 描述              | 注意                                                         | 适配M1 |
| ----- | ----------------- | ------------------------------------------------------------ | ------ |
| 3.3.9 | C++线性代数模版库 | 1. 在ubuntu 16.04下请使用源码安装, apt的最高版本是3.3beta0, 存在BUG. | 是     |

## 3.1.ArrayXd

实例化一维(双精度)动态数组.

```c++
#include <iostream>
#include "Eigen/Dense"

int main() {
    Eigen::ArrayXd arr(4);
    arr << 1, 2, 3, 4;
  
    std::cout << arr << std::endl;
  
    return 0;
}
```

## 3.2.ArrayXXd

实例化二维(双精度)动态数组.

```c++
#include <iostream>
#include "Eigen/Dense"

int main() {
    Eigen::ArrayXXd arr(2, 2);
    arr << 1, 2, 3, 4;
  
    std::cout << arr << std::endl;
  
    return 0;
}
```

### 3.2.1.exp()

逐元素计算e的幂次.

```c++
#include <iostream>
#include "Eigen/Dense"

int main() {
    Eigen::ArrayXXd arr(2, 2);
    arr << 1, 2, 3, 4;
    
    std::cout << arr.exp() << std::endl;
    
    return 0;
}
```

### 3.2.2.log()

逐元素计算自然对数.

```c++
#include <iostream>
#include "Eigen/Dense"

int main() {
    Eigen::ArrayXXd arr(2, 2);
    arr << 2.718281, 7.38906, 20.0855, 54.5982;

    std::cout << arr.log() << std::endl;
    
    return 0;
}
```

### 3.2.3.pow()

逐元素计算指定的幂次.

```c++
#include <iostream>
#include "Eigen/Dense"

int main() {
    Eigen::ArrayXXd arr(2, 2);
    arr << 1, 2, 3, 4;

    std::cout << arr.pow(2) << std::endl;
    
    return 0;
}
```

### 3.2.4.tanh()

逐元素计算双曲正切.

```c++
#include <iostream>
#include "Eigen/Dense"

int main() {
    Eigen::ArrayXXd arr(2, 2);
    arr << 1, 2, 3, 4;

    std::cout << arr.tanh() << std::endl;
    
    return 0;
}
```

## 3.3.BDCSVD<>

实例化对角分治SVD.

```c++
#include "Eigen/Dense"

int main() {
    // 用法一(仅初始化).
    Eigen::BDCSVD<Eigen::MatrixXd> svd;
  	// 用法二(初始化并传入要分解的矩阵).
    Eigen::MatrixXd mat(2, 2);
    mat << 1, 2, 3, 4;
    Eigen::BDCSVD<Eigen::MatrixXd> bdc_svd(mat);
    
    return 0;
}
```

### 3.3.1.compute()

对要分解的矩阵进行计算.

```c++
#include "Eigen/Dense"

int main() {
    Eigen::BDCSVD<Eigen::MatrixXd> svd;
    Eigen::MatrixXd mat(2, 2);
    mat << 1, 2, 3, 4;
    svd.compute(mat);
    
    return 0;
}
```

### 3.3.2.matrixU()

获取U矩阵(左奇异向量).

```c++
#include <iostream>
#include "Eigen/Dense"

int main() {
    Eigen::MatrixXd mat(2, 2);
    mat << 1, 2, 3, 4;
    Eigen::BDCSVD<Eigen::MatrixXd> svd(mat, Eigen::ComputeFullU|Eigen::ComputeFullV);

    std::cout << svd.matrixU() << std::endl;
    
    return 0;
}
```

### 3.3.3.matrixV()

获取V矩阵(右奇异向量).

```c++
#include <iostream>
#include "Eigen/Dense"

int main() {
    Eigen::MatrixXd mat(2, 2);
    mat << 1, 2, 3, 4;
    Eigen::BDCSVD<Eigen::MatrixXd> svd(mat, Eigen::ComputeFullU|Eigen::ComputeFullV);

    std::cout << svd.matrixV() << std::endl;
    
    return 0;
}
```

### 3.3.4.singularValues()

获取奇异值向量(按照降序排序).

```c++
#include <iostream>
#include "Eigen/Dense"

int main() {
    Eigen::MatrixXd mat(2, 2);
    mat << 1, 2, 3, 4;
    Eigen::BDCSVD<Eigen::MatrixXd> svd(mat, Eigen::ComputeFullU|Eigen::ComputeFullV);

    std::cout << svd.singularValues() << std::endl;
    
    return 0;
}
```

## 3.4.Map<>

映射到现有的矩阵或向量.

```c++
#include <iostream>
#include "Eigen/Dense"

int main() {
    Eigen::Matrix<double, 2, 2> mat;
    // 映射矩阵或者向量的首地址的指针, 映射后的行数和列数
    Eigen::Map<Eigen::MatrixXd> map(mat.data(), 1, 4);
    
    mat << 1, 2, 3, 4;
    
    std::cout << mat << std::endl;
    std::cout << map << std::endl;
    
    return 0;
}
```

## 3.5.Matrix<>

实例化矩阵.

```c++
#include <iostream>
#include "Eigen/Dense"

int main() {
    // 元素类型, 矩阵的行数和列数
    Eigen::Matrix<int, 2, 2> mat;
    mat << 1, 2, 3, 4;
    
    std::cout << mat << std::endl;
    
    return 0;
}
```

## 3.6.MatrixXd

实例化(双精度)动态矩阵.

```c++
#include <iostream>
#include "Eigen/Dense"

int main() {
    Eigen::MatrixXd mat(2, 2);
    mat << 1, 2, 3, 4;
    
    std::cout << mat << std::endl;
    
    return 0;
}
```

### 3.6.1.adjoint()

获取矩阵的伴随(共轭转置)矩阵.

```c++
#include <iostream>
#include "Eigen/Dense"

int main() {
    Eigen::MatrixXd mat(2, 2);
    mat << 1, 2, 3, 4;
    
    std::cout << mat.adjoint() << std::endl;
    
    return 0;
}
```

### 3.6.2.array()

将矩阵修饰成数组, 便于执行逐元素的操作.

```c++
#include <iostream>
#include "Eigen/Dense"

int main() {
    Eigen::MatrixXd mat(2, 2);
    mat << 1, 2, 3, 4;
    
    // eigen3不能跟numpy直接对元素直接操作,
    // 使用array方法后将被修饰成一个ArrayWrapper对象,
    // 才能使用例如log, exp等方法.
    
    // 这样是无法编译的.
    /* std::cout << mat.log() << std::endl; */
  
    // 正确做法.
    std::cout << mat.array().log() << std::endl;
    
    return 0;
}
```

### 3.6.3.cols()

获取矩阵的列数.

```c++
#include <iostream>
#include "Eigen/Dense"

int main() {
    Eigen::MatrixXd mat(2, 3);
    mat << 1, 2, 3, 4, 5, 6;
    
    std::cout << mat.cols() << std::endl;
    
    return 0;
}
```

### 3.6.4.data()

获取矩阵首地址的指针.

```c++
#include <iostream>
#include "Eigen/Dense"

int main() {
    Eigen::MatrixXd mat(2, 2);
    mat << 1, 2, 3, 4;
    
    std::cout << mat.data() << std::endl;
    
    return 0;
}
```

### 3.6.5.inverse()

获取矩阵的逆矩阵.

```c++
#include <iostream>
#include "Eigen/Dense"

int main() {
    Eigen::MatrixXd mat(2, 2);
    mat << 1, 2, 3, 4;
    
    std::cout << mat.inverse() << std::endl;
    
    return 0;
}
```

### 3.6.6.maxCoeff()

获取矩阵元素的最大值和对应的下标.

```c++
#include <iostream>
#include "Eigen/Dense"

int main() {
    int row, col;
    Eigen::MatrixXd mat(2, 3);
    mat << 1, 2, 3, 4, 5, 6;
  
    // 用法一(仅返回最大值).
    std::cout << "max value:" << mat.maxCoeff() << std::endl;
    // 用法二(返回最大值和对应下标).
    std::cout << "max value:" << mat.maxCoeff(&row, &col) << " row:" << row << " col:" << col << std::endl;
    
    return 0;
}

```

### 3.6.7.row()

获取矩阵的某行.

```c++
#include <iostream>
#include "Eigen/Dense"

int main() {
    Eigen::MatrixXd mat(2, 2);
    mat << 1, 2, 3, 4;

    std::cout << mat.row(1) << std::endl;
    
    return 0;
}
```

### 3.6.8.rows()

获取矩阵的行数.

```c++
#include <iostream>
#include "Eigen/Dense"

int main() {
    Eigen::MatrixXd mat(3, 2);
    mat << 1, 2, 3, 4, 5, 6;
    
    std::cout << mat.rows() << std::endl;
    
    return 0;
}
```

### 3.6.9.rowwise()

将矩阵修饰成多个向量, 对矩阵进行逐行操作.

```c++
#include <iostream>
#include "Eigen/Dense"

int main() {
    Eigen::MatrixXd mat(3, 2);
    mat << 1, 2, 3, 4, 5, 6;
    
    std::cout << mat.rowwise().sum() << std::endl;
    
    return 0;
}
```

### 3.6.10.size()

获取矩阵的元素总数.

```c++
#include <iostream>
#include "Eigen/Dense"

int main() {
    Eigen::MatrixXd mat(3, 2);
    mat << 1, 2, 3, 4, 5, 6;
    
    std::cout << mat.size() << std::endl;
    
    return 0;
}
```

### 3.6.11.sum()

计算矩阵的元素和.

```c++
#include <iostream>
#include "Eigen/Dense"

int main() {
    Eigen::MatrixXd mat(2, 2);
    mat << 5, 6, 7, 8;
    
    std::cout << mat.sum() << std::endl;
    
    return 0;
}
```

### 3.6.12.transpose()

获取矩阵的转置矩阵.

```c++
#include <iostream>
#include "Eigen/Dense"

int main() {
    Eigen::MatrixXd mat(2, 2);
    mat << 1, 2, 3, 4;
    
    std::cout << mat.transpose() << std::endl;
    
    return 0;
}
```

## 3.7.RowVectorXd

实例化(双精度)动态行向量.

```c++
#include <iostream>
#include "Eigen/Dense"

int main() {
    Eigen::RowVectorXd vec(4);
    vec << 1, 2, 3, 4;
    
    std::cout << vec << std::endl;
    
    return 0;
}
```

### 3.7.1.size()

获取行向量的元素总数.

```c++
#include <iostream>
#include "Eigen/Dense"

int main() {
    Eigen::RowVectorXd vec(4);
    vec << 1, 2, 3, 4;
    
    std::cout << vec.size() << std::endl;
    
    return 0;
}
```

## 3.8.VectorXd

实例化(双精度)列态行向量.

```c++
#include <iostream>
#include "Eigen/Dense"

int main() {
    Eigen::VectorXd vec(4);
    vec << 1, 2, 3, 4;
    
    std::cout << vec << std::endl;
    
    return 0;
}
```

### 3.8.1.asDiagonal()

将向量转换对角阵.

```c++
#include <iostream>
#include "Eigen/Dense"

int main() {
    Eigen::VectorXd vec(4);
    vec << 1, 2, 3, 4;
    Eigen::MatrixXd diag_mat = vec.asDiagonal();
    
    std::cout << diag_mat << std::endl;
    
    return 0;
}
```

# 4.h5py

| 版本  | 描述                 | 注意                        | 适配M1 |
| ----- | -------------------- | --------------------------- | ------ |
| 3.1.0 | HDF5的Python操作接口 | 1. M1目前需要使用conda安装. | 是     |

## 4.1.File()

创建一个文件对象.|h5py._hl.files.File

```python
import h5py

fp = h5py.File(name='./file.h5',  # str or file-like object|硬盘上的文件名.
               mode='w')  # {'r', 'r+', 'w', 'w- or x', 'a'}|模式.
```

### 4.1.1.attrs[]

添加到文件对象的属性.

```python
import h5py

fp = h5py.File('./file.h5', 'w')
fp.attrs['a'] = 1

print(fp.attrs['a'])
```

### 4.1.2.close()

关闭文件对象.

```python
import h5py

fp = h5py.File('./file.h5', 'w')
fp.close()
```

### 4.1.3.create_dataset()

创建新的HDF5数据集.|h5py._hl.dataset.Dataset

```python
import h5py

fp = h5py.File('./file.h5', 'w')
dataset = fp.create_dataset(name='dataset',  # str|数据集的名称.
                            dtype=float)  # Numpy Dtype or str|数据集元素的类型.
```

### 4.1.4.create_group()

创建新的HDF5组.|h5py._hl.group.Group

```python
import h5py

fp = h5py.File(name='./file.h5', mode='w')
group = fp.create_group(name='group')  # str|组的名称.
```

# 5.imageio

| 版本  | 描述           | 注意 | 适配M1 |
| ----- | -------------- | ---- | ------ |
| 2.9.0 | 图像处理软件库 | -    | 是     |

## 5.1.imread()

加载指定路径的图片.|imageio.core.util.Array

```python
import imageio

image = imageio.imread(uri='./image.jpg')  # str or pathlib.Path or bytes or file|要加载的文件的路径.
```

# 6.lightgbm

| 版本  | 描述                 | 注意                                                         | 适配M1 |
| ----- | -------------------- | ------------------------------------------------------------ | ------ |
| 3.1.1 | 基于树的梯度提升框架 | 1. M1目前需要使用conda安装.                                                                                                                          2. 2.可直接在sklearn使用.                                                                                                                                       3. 3.Intel-based Mac需要先使用brew安装libomp | 是     |

## 6.1.LGBMClassifier()

实例化一个LGBM分类器.

```python
from lightgbm import LGBMClassifier

model = LGBMClassifier(boosting_type='gbdt',  # {'gbdt', 'dart', 'goss', 'rf'}(可选)|'gbdt'|集成方式.
                       max_depth=-1,  # int(可选)|-1|基学习器的最大深度，负值表示没有限制.
                       learning_rate=0.1,  # float(可选)|0.1|学习率.
                       n_estimators=100)  # int(可选)|100|树的数量.
```

### 6.1.1.fit()

训练LGBM分类器|self

```python
model.fit(X,  # array-like or 形状为[n_samples, n_features]的稀疏矩阵|特征数据.
          y,  # array-like|标签.
          eval_set)  # list of (X, y) tuple(可选)|None|验证集元组列表.
```

### 6.1.2.predict()

进行预测|numpy.ndarray

```python
result = model.predict(X)  # array-like or 形状为[n_samples, n_features]的稀疏矩阵|用于预测的数据.
```

# 7.matplotlib

| 版本  | 描述         | 注意 | 适配M1 |
| ----- | ------------ | ---- | ------ |
| 3.3.2 | Python绘图库 | -    | 是     |

## 7.1.axes

| 版本 | 描述                                              | 注意 |
| ---- | ------------------------------------------------- | ---- |
| -    | axes是matplotlib的图形接口, 提供设置坐标系的功能. | -    |

### 7.1.1.annotate()

为坐标点进行注释.

```python
import matplotlib.pyplot as plt

ax = plt.subplot()
ax.annotate(text='annotate',  # str|注释的内容.
            xy=(0.45, 0.5),  # (float, float)|注释点的坐标.
            xytext=(0.6, 0.6),  # (float, float)(可选)|None|注释内容的坐标.
            xycoords='data',  # str(可选)|'data'|注释点放置的坐标系.
            textcoords='data',  # str(未设置则和xycoords一致, 可选)|注释内容放置的坐标系.
            arrowprops=dict(arrowstyle='<-'),  # dict(可选)|None(即不绘制箭头)|绘制箭头的样式.
            size=20,  # int(可选)|10|注释文本字号.
            verticalalignment='baseline',  # str(可选)|'baseline'|垂直对齐.
            horizontalalignment='center',  # str(可选)|'left'|水平对齐.
            bbox=dict(fc='skyblue'))  # dict(可选)|None(即不绘制文本框)|绘制文本框.

plt.show()
```

### 7.1.2.axis()

坐标轴的设置选项.

```python
import matplotlib.pyplot as plt

ax = plt.subplot()
ax.axis('off')
```

### 7.1.3.clabel()

在等高线上显示高度.

```python
import numpy as np
import matplotlib.pyplot as plt

ax = plt.subplot()
x = np.linspace(0, 9, 10)
y = np.linspace(0, 9, 10)
X, Y = np.meshgrid(x, y)
cs = ax.contour(X, Y, X + Y, 'orange', 1)
ax.clabel(cs)

plt.show()
```

### 7.1.4.contour()

绘制等高线.|matplotlib.contour.QuadContourSet

```python
import numpy as np
import matplotlib.pyplot as plt

ax = plt.subplot()
x = np.linspace(0, 9, 10)
y = np.linspace(0, 9, 10)
X, Y = np.meshgrid(x, y)
cs = ax.contour(X,  # array-like|横坐标.
                Y,  # array-like|纵坐标.
                X + Y,  # array-like(必须是2D的)|横纵坐标的关系公式.
                colors='orange',  # str|等高线的颜色.
                linewidths=1)  # int|等高线的宽度.

plt.show()
```

### 7.1.5.grid()

绘制网格线.

```python
import matplotlib.pyplot as plt

ax = plt.subplot()
ax.grid(axis='x',  # {'both', 'x', 'y'}(可选)|'both'|绘制的范围.
        linestyle=':')  # {'-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted'}(可选)|'-'|网格线的样式.

plt.show()
```

### 7.1.6.legend()

放置图例.

```python
import matplotlib.pyplot as plt

ax = plt.subplot()
ax.legend(loc='center')  # {'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left', 'center right', 'lower center', 'upper center', 'center'}(可选)|'best'|放置的位置.

plt.show()
```

### 7.1.7.patch

| 版本 | 描述                                   | 注意 |
| ---- | -------------------------------------- | ---- |
| -    | patches是画布颜色和边框颜色的控制接口. | -    |

#### 7.1.7.1.set_alpha()

设置画布的透明度.

```python
import matplotlib.pyplot as plt

ax = plt.subplot()
ax.patch.set_alpha(alpha=0.1)  # float|透明度.
ax.patch.set_facecolor('green')

plt.show()
```

#### 7.1.7.2.set_facecolor()

设置画布的颜色.

```python
import matplotlib.pyplot as plt

ax = plt.subplot()
ax.patch.set_alpha(0.1)
ax.patch.set_facecolor(color='green')  # str|颜色.

plt.show()
```

### 7.1.8.set_title()

设置标题.

```python
import matplotlib.pyplot as plt

ax = plt.subplot()
ax.set_title('this is title')

plt.show()
```

### 7.1.9.set_xlabel()

设置x轴的内容.

```python
import matplotlib.pyplot as plt

ax = plt.subplot()
ax.set_xlabel(xlabel='this is x label')  # str|文本内容.

plt.show()
```

### 7.1.10.set_xticks()

设置x轴的刻度.

```python
import matplotlib.pyplot as plt

ax = plt.subplot()
ax.set_xticks(ticks=[1, 2, 3, 4])  # list(空列表就表示不显示刻度)|刻度.

plt.show()
```

### 7.1.11.set_yticks()

设置y轴的刻度.

```python
import matplotlib.pyplot as plt

ax = plt.subplot()
ax.set_yticks(ticks=[1, 2, 3, 4])  # list(空列表就表示不显示刻度)|刻度.

plt.show()
```

###  7.1.12.spines

| 版本 | 描述                          | 注意 |
| ---- | ----------------------------- | ---- |
| -    | 画布的边框, 包括上下左右四个. | -    |

#### 7.1.12.1.set_color()

设置画布边框的颜色.

```python
import matplotlib.pyplot as plt

ax = plt.subplot()
ax.spines['left'].set_color(c='red')  # str|颜色.
ax.spines['right'].set_color(c='yellow')
ax.spines['top'].set_color(c='blue')
ax.spines['bottom'].set_color(c='green')

plt.show()
```

### 7.1.13.text()

给点添加文本.|matplotlib.text.Text

```python
import matplotlib.pyplot as plt

ax = plt.subplot()
ax.text(x=0.5,  # float|注释点的x坐标.
        y=0.5,  # float|注释点的y坐标.
        s='this is text')  # str|注释的文本内容.

plt.show()
```

## 7.2.pyplot

| 版本 | 描述                                                         | 注意 |
| ---- | ------------------------------------------------------------ | ---- |
| -    | pyplot是matplotlib的state-based接口, 主要用于简单的交互式绘图和程序化绘图. | -    |

### 7.2.1.axis()

设置坐标轴.

```python
import matplotlib.pyplot as plt

xmin, ymin = 0, 0
xmax, ymax = 10, 10
plt.axis([xmin, xmax, ymin, ymax])

plt.show()
```

### 7.2.2.barh()

绘制水平方向的条形图.

```python
import matplotlib.pyplot as plt

plt.barh(y=['No.1', 'No.2', 'No.3', 'No.4'],  # float或者array-like|条形图的y轴坐标.
         width=[1, -0.5, 2, 6],  # float或者array-like|每个数据的值.
         height=0.8)  # float(可选)|0.8|每个数据条的的宽度.

plt.show()
```

### 7.2.3.clabel()

在等高线上显示高度.

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(1, 9, 10)
y = np.linspace(1, 9, 10)
X, Y = np.meshgrid(x, y)
cs = plt.contour(X, Y, X + Y, colors='orange', linewidths=1)
plt.clabel(cs)  # matplotlib.contour.QuadContourSet|等高线标签.

plt.show()
```

### 7.2.4.colorbar()

显示色彩条.|matplotlib.colorbar.Colorbar

```python
import matplotlib.pyplot as plt

x = [[1], [2], [3], [4]]
plt.matshow(A=x)
plt.colorbar()

plt.show()
```

### 7.2.5.figure()

创建画布.|matplotlib.figure.Figure

```python
import matplotlib.pyplot as plt

figure = plt.figure(figsize=(5, 5))  # (float, float)(可选)|(6.4, 4.8)|画布的尺寸.
```

### 7.2.6.imread()

加载指定路径的图片.|numpy.ndarray

```python
import matplotlib.pyplot as plt

image = plt.imread(fname='./image.jpg')  # str or file-like|要加载的文件的路径.
```

### 7.2.7.imshow()

将图片数组在画布上显示.|matplotlib.image.AxesImage

```python
import matplotlib.pyplot as plt

image = plt.imread('./image.jpg')
plt.imshow(X=image,  # array-like or PIL image|希望显示的图像数据.
           cmap=None)  # str or matplotlib.colors.Colormap|None|色图.

plt.show()
```

### 7.2.8.matshow()

将矩阵绘制成图像.

```python
import matplotlib.pyplot as plt

mat = [[1], [2], [3], [4]]
plt.matshow(A=mat)  # array-like(M, N)|要绘制的矩阵.

plt.show()
```

### 7.2.9.pcolormesh()

使用非规则的矩形创建网格背景图.

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(1, 9, 10)
y = np.linspace(1, 9, 10)
X, Y = np.meshgrid(x, y)
plt.pcolormesh(X,  # array-like|横坐标.
               Y,  # array-like|纵坐标.
               X + Y,  # array-like(必须是2D的)|横纵坐标的关系公式.
               alpha=0.75,  # float|None|透明度.
               cmap='GnBu',  # str|None|色图.
               shading='nearest')  # {'flat', 'nearest', 'gouraud', 'auto'}(可选)|'flat'|阴影.

plt.show()
```

### 7.2.10.plot()

绘制函数图像.

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(1, 9, 10)
y = np.linspace(1, 9, 10)
plt.plot(x, y)  # 1D array-like|函数的变量.

plt.show()
```

### 7.2.11.rcParams[]

实例化配置文件实例.

```python
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = 'Arial Unicode MS'  # 默认字体
```

### 7.2.12.savefig()

保存当前的画布.

```python
import matplotlib.pyplot as plt

plt.savefig(fname)  # str or path-like or file-like A path|要加载的文件的路径.
```

### 7.2.13.scatter()

绘制散点图.

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10, num=10)
y = np.linspace(0, 5, num=10)
plt.scatter(x=x,  # float or array-like 形状必须是(n, )|x坐标.
            y=y,  # float or array-like 形状必须是(n, )|y坐标.
            s=150,  # float or array-like, 形状必须是(n, )(可选)|点的大小.
            c='blue',  # str|点的颜色.
            marker='o',  # str|点的标记的形状.
            edgecolors='green')  # str|标记的颜色.

plt.show()
```

### 7.2.14.show()

显示画布.

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(1, 9, 10)
y = np.linspace(1, 9, 10)
plt.plot(x, y)

plt.show()
```

### 7.2.15.subplot()

在当前画布上创建子图.|matplotlib.axes._subplots.AxesSubplot

```python
import matplotlib.pyplot as plt

axes = plt.subplot()
```

### 7.2.16.subplots()

同时创建画布和一组子图.|matplotlib.figure.Figure和array of matplotlib.axes._subplots.AxesSubplot

```python
import matplotlib.pyplot as plt

figure, axes = plt.subplots(nrows=4,  # int|1|子图数量的行数.
                            ncols=4,  # int|1|子图数量的列数.
                            figsize=(10, 10))  # tuple(list) of int|画布的大小.
```

### 7.2.17.subplots_adjust()

调整子图布局.

```python
import matplotlib.pyplot as plt

plt.subplots_adjust(left=0.125,  # float(可选)|None|子图左边框距离画布的距离.
                    bottom=0.9,  # float(可选)|None|子图底边框距离画布的距离.
                    right=0.1,  # float(可选)|None|子图右边框距离画布的距离.
                    top=0.9,  # float(可选)|None|子图顶边框距离画布的距离.
                    wspace=0.2,  # float(可选)|None|两张子图之间的左右间隔.
                    hspace=0.2)  # float(可选)|None|两张子图之间的上下间隔.
```

### 7.2.18.tight_layout()

自动调整子图布局.

```python
import matplotlib.pyplot as plt

plt.tight_layout()
```

### 7.2.19.title()

设置标题.

```python
import matplotlib.pyplot as plt

plt.title(label='this is title')  # str|文本内容.

plt.show()
```

### 7.2.20.xlabel()

设置x轴的内容.

```python
import matplotlib.pyplot as plt

plt.xlabel(xlabel='x')  # str|文本内容.

plt.show()
```

### 7.2.21.xlim()

设置x轴的显示范围.

```python
import matplotlib.pyplot as plt

plt.xlim([1, 2])  # [left, right]|[左界, 右界].

plt.show()
```

### 7.2.22.ylabel()

设置y轴的内容.

```python
import matplotlib.pyplot as plt

plt.ylabel(ylabel='y')  # str|文本内容.

plt.show()
```

# 8.numpy

| 版本   | 描述           | 注意                        | 适配M1 |
| ------ | -------------- | --------------------------- | ------ |
| 1.19.4 | Python数值计算 | 1. M1目前需要使用conda安装. | 是     |

## 8.1.abs()

逐元素计算绝对值.|numpy.ndarray

```python
import numpy as np

arr = [1, 2, -1, 3]
x = np.abs(arr)  # array_like|输入的数据.
```

## 8.2.any()

判断数组是否存在某个元素为True, 如果有返回True, 否则返回False.|numpy.bool_

```python
import numpy as np

arr = [1, 0, 1, 1]
x = np.any(a=arr)  # array_like|输入的数据.
```

## 8.3.arange()

返回指定范围的整数数组.|numpy.ndarray

```python
import numpy as np

arr = np.arange(start=1,  # number(可选)|None|起始值.
                stop=10,  # number|结束值.
                step=2)  # number(可选)|None|步长.
```

## 8.4.argmax()

返回指定维度最大值的索引.|numpy.int64

```python
import numpy as np

arr = [1, 2, 3]
max_value = np.argmax(a=arr,  # array_like|输入的数据.
                			axis=None)  # int(可选)|None|筛选所沿的维度.
```

## 8.5.around()

逐元素进行四舍五入取整.|numpy.ndarray

```python
import numpy as np

arr = [1.4, 1.6]
x = np.around(a=arr)  # array_like|输入的数据.
```

## 8.6.asarray()

将输入转换为ndarray数组.|numpy.ndarray

```python
import numpy as np

arr = [1, 2, 3]
nd_arr = np.asarray(a=arr,  # array_like|输入的数据.
                    dtype=None)  # data-type(可选)|None|元素的数据类型.
```

## 8.7.asmatrix()

将输入转换为矩阵.|numpy.matrix

```python
import numpy as np

arr = [1, 2, 3]
mat = np.asmatrix(data=arr)  # array_like|输入的数据.
```

## 8.8.ceil()

逐元素进行向上取整.|numpy.ndarray

```python
import numpy as np

arr = [5.1, 4.9]
x = np.ceil(arr)  # array_like|输入的数据.
```

## 8.9.concatenate()

按照指定维度合并多个数组.|numpy.ndarray

```python
import numpy as np

arr0 = [[1], [1], [1]]
arr1 = [[2], [2], [2]]
arr2 = [[3], [3], [3]]
x = np.concatenate([arr0, arr1, arr2],  # array_like|要合并的数组.
                   axis=1)  # int(可选)|0|沿指定维度合并.
```

## 8.10.c_[]

将第二个数组沿水平方向与第一个数组连接.|numpy.ndarray

```python
import numpy as np

arr0 = [[1, 2], [1, 2], [1, 2]]
arr1 = [[3], [3], [3]]
x = np.c_[arr0, arr1]
```

## 8.11.diag()

提取对角线的值, 或构建对角阵.|numpy.ndarray

```python
import numpy as np

arr = [1, 2, 3]
x = np.diag(v=arr)  # array_like|输入的数据.
```

## 8.12.dot()

计算两个数组的点乘.|numpy.ndarray

```python
import numpy as np

arr0 = [[1, 2, 3]]
arr1 = [[1], [2], [3]]
x = np.dot(a=arr0,  # array_like|第一个元素.
           b=arr1)  # array_like|第二个元素.
```

## 8.13.equal()

逐元素判断元素值是否一致.|numpy.ndarray

```python
import numpy as np

arr1 = [1, 2, 3]
arr2 = [1, 2, 2]
x = np.equal(arr1, arr2)  # array_like|输入的数据.
```

## 8.14.exp()

逐元素计算e的幂次.|numpy.ndarray

```python
import numpy as np

arr = [1, 2, 3]
x = np.exp(arr)  # array_like|输入的数据.
```

## 8.15.expm1()

逐元素计算e的幂次并减1.|numpy.ndarray

```python
import numpy as np

arr = [1, 2, 3]
x = np.expm1(arr)  # array_like|输入的数据.
```

## 8.16.expand_dims()

增加数组的维度.|numpy.ndarray

```python
import numpy as np

arr = [1, 2, 3]
x = np.expand_dims(a=arr,  # array_like|输入的数组.
                   axis=0)  # int or tuple of ints|添加新维度的位置.
```

## 8.17.eye()

生成单位阵.|numpy.ndarray

```python
import numpy as np

mat = np.eye(N=3)  # int|矩阵的行数.
```

## 8.18.hstack()

按照水平顺序合并数组.|numpy.ndarray

```python
import numpy as np

arr0 = [[1, 2], [1, 2]]
arr1 = [[3], [3]]
x = np.hstack(tup=[arr0, arr1])  # array-like|数组序列.
```

## 8.19.linalg

| 版本 | 描述               | 注意 |
| ---- | ------------------ | ---- |
| -    | numpy的线性代数库. | -    |

### 8.19.1.inv()

获取矩阵的逆矩阵.|numpy.ndarray

```python
import numpy as np

mat = [[1, 2],
       [3, 4]]
x = np.linalg.inv(a=mat)  # array_like|输入的矩阵.
```

### 8.19.2.norm()

计算矩阵或向量范数.|numpy.float64

```python
import numpy as np

arr = [[1, 2],
       [3, 4]]
x = np.linalg.norm(x=arr,  # array_like|输入的数据.
                   ord=2)  # {non-zero int, inf, -inf, 'fro', 'nuc'}(可选)|None|范数选项.
```

### 8.19.3.svd()

奇异值分解.|tuple of numpy.ndarray

```python
import numpy as np

arr = [[1, 2],
       [3, 4]]
u, s, vh = np.linalg.svd(a=arr)  # array_like|输入的数据.
```

## 8.20.linspace()

返回指定间隔内的等差数列.|numpy.ndarray

```python
import numpy as np

x = np.linspace(start=1,  # array_like|起始值.
                stop=10,  # array_like|结束值.
                num=10)  # int(可选)|50|生成序列的样本的总数.
```

## 8.21.load()

从npy npz或者序列化文件加载数组或序列化的对象.|numpy.ndarray

```python
import numpy as np

arr = np.load(file='./arr.npy',  # file-like object, string, or pathlib.Path|读取的文件路径.
              allow_pickle=True,  # bool(可选)|False|允许加载序列化的数组.
              encoding='ASCII')  # str(可选)|'ASCII'|解码方式.
```

## 8.22.log()

逐元素计算自然对数.|numpy.ndarray

```python
import numpy as np

arr = [1, 2, 3]
x = np.log(arr)  # array_like|输入的数据.
```

## 8.23.log1p()

逐元素计算本身加1的自然对数.|numpy.ndarray

```python
import numpy as np

arr = [1, 2, 3]
x = np.log1p(arr)  # array_like|输入的数据.
```

## 8.24.log2()

逐元素计算以2为底对数.|numpy.ndarray

```python
import numpy as np

arr = [1, 2, 3]
x = np.log2(arr)  # array_like|输入的数据.
```

## 8.25.mat()

将输入转换为矩阵.|numpy.matrix

```python
import numpy as np

arr = [[1, 2, 3]]
mat = np.mat(data=arr,  # array_like|输入的数据.
             dtype=None)  # data-type|None|矩阵元素的数据类型.
```

## 8.26.matmul()

两个数组的矩阵乘积|numpy.ndarray

```python
import numpy as np

arr0 = [[1, 2, 3]]
arr1 = [[1], [2], [3]]
x = np.matmul(arr0,  # array_like|第一个元素.
              arr1)  # array_like|第二个元素.
```

## 8.27.max()

返回沿指定维度的最大值.|numpy.float64

```python
import numpy as np

arr = [1, 2, 5, 3, 4]
max_value = np.max(a=arr,  # array_like|输入的数据.
                   axis=None)  # int(可选)|None|所沿的维度.
```

## 8.28.maximum()

返回数组逐元素的最大值.|numpy.ndarray

```python
import numpy as np

arr0 = [2, 3, 4]
arr1 = [1, 5, 2]
x = np.maximum(arr0, arr1)  # array_like|输入的数据.
```

## 8.29.mean()

沿指定维度计算均值.|numpy.float64

```python
import numpy as np

arr = [1, 2, 3]
x = np.mean(a=arr,  # array_like|输入的数据.
            axis=None)  # int(可选)|None|所沿的维度.
```

## 8.30.meshgrid()

生成坐标矩阵.|list of numpy.ndarray

```python
import numpy as np

x_crood = np.linspace(0, 4, 5)
y_crood = np.linspace(0, 4, 5)
vec_mat = np.meshgrid(x_crood, y_crood)  # array_like|坐标向量.
```

## 8.31.nonzero()

返回非零元素索引.|tuple_of_arrays

```python
import numpy as np

arr = np.asarray([1, 2, 3, 4, 0, 0, 5])
x = np.nonzero(a=arr)  # array_like|输入的数据.
```

## 8.32.ones()

生成全一数组.|numpy.ndarray

```python
import numpy as np

x = np.ones(shape=[2, 3],  # int or sequence of ints|数组的形状.
            dtype=np.int8)  # data-type(可选)|numpy.float64|矩阵元素的数据类型.
```

## 8.33.power()

逐元素计算指定幂次.|numpy.ndarray

```python
import numpy as np

x = np.power([1, 2], [1, 3])   # array_like|底数和指数.
```

## 8.34.random

| 版本 | 描述                     | 注意 |
| ---- | ------------------------ | ---- |
| -    | numpy的随机数生成函数库. | -    |

### 8.34.1.multinomial()

从多项分布中抽取样本.|numpy.ndarray

```python
import numpy as np

x = np.random.multinomial(n=1,  # int|实验次数.
                          pvals=[1/2, 1/3, 1/6],  # sequence of floats|每个部分的概率(概率和为1).
                          size=1)  # int or tuple of ints(可选)|None|数组的形状.
```

### 8.34.2.normal()

生成正态分布样本.|numpy.ndarray

```python
import numpy as np

x = np.random.normal(size=[2, 3])  # int or tuple of ints(可选)|None|数组的形状.
```

### 8.34.3.permutation()

随机打乱序列.|numpy.ndarray

```python
import numpy as np

arr = [1, 2, 3, 4]
x = np.random.permutation(arr)  # array_like|输入的数据.
```

### 8.34.4.rand()

生成随机数组.|numpy.ndarray

```python
import numpy as np

x = np.random.rand(2, 3)  # int(可选)|None|数组的形状.
```

### 8.34.5.randint()

返回指定区间[low, high)随机整数.|int

```python
import numpy as np

x = np.random.randint(low=1,  # int or array-like of ints|左边界.
                      high=10)  # int or array-like of ints(可选)|None|右边界.
```

### 8.34.6.randn()

生成正态分布随机数组.|numpy.ndarray

```python
import numpy as np

x = np.random.randn(2, 3)  # int(可选)|None|数组的形状.
```

### 8.34.7.RandomState()

实例化伪随机数生成器.|numpy.random.mtrand.RandomState

```python
import numpy as np

rs = np.random.RandomState(seed=2021)  # None|随机种子.
```

#### 8.34.7.1.shuffle()

打乱数据.|numpy.ndarray

```python
import numpy as np

rs = np.random.RandomState(seed=2021)
x = np.asarray([1, 2, 3, 4])
rs.shuffle(x)
```

### 8.34.8.seed()

设置随机种子.

```python
import numpy as np

np.random.seed(seed=2021)  # None|随机种子.
```

## 8.35.ravel()

展平数组.|numpy.ndarray

```python
import numpy as np

x = np.asarray([[1, 2], [3, 4]])
x = np.ravel(a=x)  # array_like|输入的数据.
```

## 8.36.reshape()

改变数组的形状.|numpy.ndarray

```python
import numpy as np

arr = [1, 2, 3, 4]
x = np.reshape(a=arr,  # array_like|要改变形状的数组.
               newshape=[2, 2])  # int or tuple of ints|新的形状.
```

## 8.37.save()

将数组保存进二进制的npy文件.

```python
import numpy as np

arr = [1, 2, 3]
np.save(file='arr.npy',  # file, str, or pathlib.Path|文件保存的路径.
        arr=arr,  # array_like|要保存的数据.
        allow_pickle=True)  # |bool(可选)|True|允许使用序列化保存数组.
```

## 8.38.sort()

返回排序(升序)后的数组.|numpy.ndarray

```python
import numpy as np

arr = [1, 3, 2, 4]
x = np.sort(a=arr)  # array_like|要排序的数组.
```

## 8.39.split()

拆分数组.|list of ndarrays

```python
import numpy as np

arr = np.asarray([[1, 2, 5, 6], [3, 4, 7, 8]])
arr_list = np.split(ary=arr,  # numpy.ndarray|要拆分的数组.
                    indices_or_sections=2,  # int or 1-D array|拆分方式.
                    axis=1)  # int(可选)|0|所沿的维度.
```

## 8.40.sqrt()

逐元素计算平方根.|numpy.ndarray

```python
import numpy as np

arr = [1, 2, 3]
x = np.sqrt(arr)  # array_like|输入的数据.
```

## 8.41.squeeze()

删除维度为一的维度.|numpy.ndarray

```python
import numpy as np

arr = [[1, 2, 3]]
x = np.squeeze(arr)  # array_like|输入的数据.
```

## 8.42.std()

沿指定维度计算标准差.|numpy.float64

```python
import numpy as np

arr = [1, 2, 5, 3, 4]
std_value = np.std(a=arr,  # array_like|输入的数据.
                   axis=None)  # None or int or tuple of ints(可选)|None|所沿的维度.
```

## 8.43.sum()

沿指定维度求和.|numpy.ndarray

```python
import numpy as np

arr = [1, 2, 5, 3, 4]
sum_value = np.sum(a=arr,  # array_like|输入的数据.
                   axis=None)  # None or int or tuple of ints(可选)|None|所沿的维度.
```

## 8.44.transpose()

转置数组.|numpy.ndarray

```python
import numpy as np

arr = np.asarray([[1, 2], [3, 4]])
# 方法一
x0 = np.transpose(a=arr,  # 输入的数组|array-like
                  axes=None)  # tuple or list of ints(可选)|None|轴的排列顺序.

# 方法二
x1 = arr.T
```

## 8.45.var()

沿指定维度计算方差.|numpy.float64

```python
import numpy as np

arr = [1, 2, 5, 3, 4]
var_value = np.var(a=arr,  # array_like|输入的数据.
                   axis=None)  # None or int or tuple of ints(可选)|None|所沿的维度.
```

## 8.46.void()

实例化numpy.void对象.

```python
import numpy as np

x = np.void(b'abc')  # bytes|输入的数据.
```

## 8.47.zeros()

生成全零数组.|numpy.ndarray

```python
import numpy as np

x = np.zeros(shape=[2, 3],  # int or sequence of ints|数组的形状.
             dtype=np.int8)  # data-type(可选)|numpy.float64|矩阵元素的数据类型.
```

# 9.pandas

| 版本  | 描述                  | 注意                        | 适配M1 |
| ----- | --------------------- | --------------------------- | ------ |
| 1.1.4 | 结构化数据分析软件库. | 1. M1目前需要使用conda安装. | 是     |

## 9.1.concat()

沿特定轴连接pandas对象.|pandas.core.frame.DataFrame

```python
import pandas as pd

sr0 = pd.Series([1, 4, 7])
sr1 = pd.Series([2, 5, 8])
sr2 = pd.Series([3, 6, 9])
df = pd.concat([sr0, sr1, sr2],  # Series or DataFrame|要连接的pandas对象.
               axis=1)  # {0/'index', 1/'columns'}|0|所沿的维度.
```

## 9.2.DataFrame()

实例化DataFrame对象.|pandas.core.frame.DataFrame

```python
import pandas as pd

df_map = {'key': [0, 1, 2], 'values': [0.1, 0.5, 1.0]}
df = pd.DataFrame(data=df_map,  # ndarray (structured or homogeneous), Iterable, dict, or DataFrame|输入的数据.
                  index=['一', '二', '三'],  # Index or array-like|None(0, 1, 2, ..., n)|索引名.
                  columns=None)  # Index or array-like|None(0, 1, 2, ..., n)|列名.
```

### 9.2.1.columns

DataFrame的列标签.|pandas.core.indexes.base.Index

```python
import pandas as pd

df_map = {'key': [0, 1, 2], 'values': [0.1, 0.5, 1.0]}
df = pd.DataFrame(df_map)

print(df.columns)
```

### 9.2.2.corr()

计算列成对相关度.|pandas.core.frame.DataFrame

```python
import pandas as pd

df_map = {'key': [0, 1, 2], 'values': [0.1, 0.5, 1.0]}
df = pd.DataFrame(df_map)
correlation_value = df.corr()
```

### 9.2.3.drop()

根据指定的标签删除行或者列.|pandas.core.frame.DataFrame

```python
import pandas as pd

df_map = {'key': [0, 1, 2], 'values': [0.1, 0.5, 1.0]}
df = pd.DataFrame(df_map)
df = df.drop(labels=1,  # single label or list-like|None|要删除的行或者列的标签.
             axis=0)  # {0/'index', 1/'columns'}|0|所沿的维度.
```

