
import numpy as np
import boris
import matplotlib.pyplot as plt


if __name__ == '__main__':
    device = 'cpu'

    x0, v0 = boris.random_x_v(30000)
    print(x0.shape, v0.shape)

    data = boris.boris(x0, v0, steps=1000, device='cpu')
    print(data.shape)

    data1 = boris.boris(x0, v0, steps=1000, device='cuda')
    print(data.shape)

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

