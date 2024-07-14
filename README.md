## 简介

一个使用`boris`算法推动粒子轨迹的计算库。

`boris`算法来源于https://pubs.aip.org/aip/pop/article-abstract/20/8/084503/317652/Why-is-Boris-algorithm-so-good?redirectedFrom=fulltext一文，它能计算给定位置、速度的粒子在三维电磁场下的推进。

这个库实现了boris算法，并可用cpu或gpu进行多轨迹并行计算。计算假定电磁场为托卡马克电磁场，粒子为$\alpha$粒子，粒子速度为$12926625\text{m/s}$。

## 项目环境

```
pip install numpy matplotlib tqdm scipy
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## 调用实现

引入相关库

```
import numpy as np
import boris
import matplotlib.pyplot as plt
```

随机生成初始粒子的位置和速度，这里指定粒子的个数为30000个。这里粒子的位置是在整个托卡马克环空间内随机的。

```
x0, v0 = boris.random_x_v(30000)
```

调用boris库完成30000个粒子各自的1000步推进：

```
data = boris.boris(x0, v0, steps=1000, device='cpu')
```

- 使用gpu版本的调用：

  ```
  data1 = boris.boris(x0, v0, steps=1000, device='cuda')
  ```

展示其中一个粒子的计算轨迹：

```
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

