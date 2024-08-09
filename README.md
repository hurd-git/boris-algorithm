# Boris Algorithm with Eigen3

A high-performance computing library for calculating particle trajectories of charged particles using the boris algorithm, based on Python Numpy and C++ Eigen3.

## Introduce

`boris` algorithm \([Why is Boris algorithm so good?](https://pubs.aip.org/aip/pop/article-abstract/20/8/084503/317652/Why-is-Boris-algorithm-so-good?redirectedFrom=fulltext)\) can calculate the advance of a particle with a given position and velocity under a three-dimensional electromagnetic field.

This library implements `boris` algorithm and supports multi-threaded parallel computation, which can greatly accelerate the speed of trajectory promotion. Using this library, it takes 12.3 seconds for 30,000 particles to advance 10,000 steps in 1 thread. With 32 threads, the time is reduced to 1 second. The memory consumption of the library is small, about 14G for the above scenario, which is consistent with the `numpy` matrix size of `30000x10000x6`, and almost all of the memory is used for data storage.

The calculation of this library assumes that the electromagnetic field is a tokamak electromagnetic field, the particle is a <img src="https://latex.codecogs.com/svg.image?\alpha" /> particle, and the default parameters are stored and can be adjusted. The expression of the magnetic field is

<p align="center">
    <img src="https://latex.codecogs.com/svg.image?\mathbf{B}=\frac{B_0}{q_0}\left(\frac{z}{r}\mathbf{e}_r+\frac{q_0%20R_0}{r}\mathbf{e}_\phi+(-1+\frac{R_0}{r})\mathbf{e}_z\right)" />
</p>

It is ring symmetric. Converted to Cartesian coordinates, its expression is

<p align="center">
    <img src="https://latex.codecogs.com/svg.image?\mathbf{B}=\frac{B_0}{q_0}\left(\frac{-q_0%20R_0%20y+z%20x}{x^2+y^2}\mathbf{e}_x+\frac{q_0%20R_0%20x+z%20y}{x^2+y^2}\mathbf{e}_y+\left(-1+\frac{R_0}{\sqrt{x^2+y^2}}\right)\mathbf{e}_z\right)" />
</p>

We take the ITER parameter, so by default:

<p align="center">
    <img src="https://latex.codecogs.com/svg.image?B_0=5\;\mathrm{T}\quad%20q_0=2.5\quad%20R_0=6.2\;\mathrm{m}\quad%20r_0=2.0\;\mathrm{m}" />
</p>

Where <img src="https://latex.codecogs.com/svg.image?B_0" /> is the magnetic field at the magnetic axis, <img src="https://latex.codecogs.com/svg.image?q_0" /> is the safety factor, <img src="https://latex.codecogs.com/svg.image?R_0" /> is the large radius, and <img src="https://latex.codecogs.com/svg.image?r_0" /> is the small radius.



## Environment

**Python >= 3.8 Required**

```shell
pip install numpy<2.0.0 matplotlib tqdm
```

## Example

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
>>> Using threads: 32, prepare memory + start calculating...done (1.138s)
>>> (30000, 10001, 6)
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

![img.png](img.png)
