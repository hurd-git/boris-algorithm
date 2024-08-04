## 简介

一个使用`boris`算法推动粒子轨迹的高性能计算库，使用python3和c++。

`boris`算法来源于[Why is Boris algorithm so good?](https://pubs.aip.org/aip/pop/article-abstract/20/8/084503/317652/Why-is-Boris-algorithm-so-good?redirectedFrom=fulltext)一文，它能计算给定位置、速度的粒子在三维电磁场下的推进。

这个库实现了`boris`算法，并支持多线程并行计算，可以极大的加快轨迹的推动速度。使用本库进行计算，在单线程下，3万个粒子推进1万步仅需12.3秒。使用32线程，时间将缩短至1秒。该库的内存耗费很小，上述情景的内存耗费约14G，这与`30000x10000x6`的`numpy`矩阵大小一致，所有内存几乎都用于数据存储。

本库的计算假定电磁场为托卡马克电磁场，粒子为$\alpha$粒子，存储有默认参数，并可自行调节。

## 项目环境

**!! Python3.8 Required !!**

```shell
pip install numpy matplotlib tqdm scipy
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## 调用实现

引入相关库

```python
import os
import numpy as np
from boris import Boris
import matplotlib.pyplot as plt
```

初始化Boris实例

```
boris = Boris()
```

随机生成初始粒子的位置和速度，这里指定粒子的个数为30000个。粒子的位置在整个托卡马克环空间内随机。

```python
x0, v0 = boris.random_x_v(30000)
print(x0.shape, v0.shape)
```

```
>>> (30000, 3) (30000, 3)
```

调用boris库完成30000个粒子各自的10000步推进，打开性能分析`info=True`查看详细信息：

```python
threads = os.cpu_count()
data = boris.run(x0, v0, steps=10000, threads=threads, info=True)
print(data.shape)
```

```
>>> prepare memory...done (1.723s)
>>> Using threads: 32, start calculating...done (0.992s)
```

展示其中一个粒子的计算轨迹：

```python
i = np.random.randint(low=0, high=data.shape[0] - 1)

plt.figure()

ax1 = plt.subplot(1, 2, 1)
plt.plot(data[i, :, 0], data[i, :, 1])
plt.xlabel('X')
plt.ylabel('Y')
ax1.set_aspect('equal')
plt.plot(data[i, 0, 0], data[i, 0, 1], marker='o', color='r', label='Original')
plt.legend()

ax2 = plt.subplot(1, 2, 2, projection='3d')
plt.plot(data[i, :, 0], data[i, :, 1], data[i, :, 2])
plt.xlabel('X')
plt.ylabel('Y')
ax2.set_zlabel('Z')
ax2.set_aspect('equal')
plt.plot(data[i, 0, 0], data[i, 0, 1], data[i, 0, 2], marker='o', color='r', label='Original')
plt.legend()

plt.show()
```



## 香蕉轨道示例

![img.png](img.png)