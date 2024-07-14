import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # 为1时最快
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import math
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import scipy
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# 磁场信息
R0 = 6.2
q0 = 2.5
B0 = 5
r0 = 2

q = 1.602 * 10 ** (-19)
m = 2 * 1.661 * 10 ** (-27)
t = 2.9 * 10 ** (-9)  # 十分之一的回旋周期

normv = 12926625  # 回旋半径最大0.05左右，小半径是2m


def task_assign(tasks, workers):
    worker_task = np.zeros(shape=(workers,), dtype=np.int32)
    avg_task = tasks // workers
    remain_task = tasks % workers
    worker_task += avg_task
    worker_task[:remain_task] += 1

    assign_ = [(rank, task_num) for rank, task_num in enumerate(worker_task) if task_num > 0]
    assign = []
    for rank, task_num in assign_:
        if rank == 0:
            assign.append((rank, np.arange(0, task_num), task_num))
        else:
            last_stop = assign[-1][1][-1] + 1
            assign.append((rank, np.arange(last_stop, last_stop + task_num), task_num))

    return assign


def get_magnetic(x, y, z, device='cpu'):
    x, y, z = np.array(x, ndmin=1), np.array(y, ndmin=1), np.array(z, ndmin=1)

    Bx = B0 * (-1 * q0 * R0 * y + z * x) / (q0 * (x ** 2 + y ** 2))
    By = B0 * (q0 * R0 * x + z * y) / (q0 * (x ** 2 + y ** 2))
    Bz = B0 * (-1 + R0 / (x ** 2 + y ** 2) ** (1 / 2)) / q0
    B = np.array([Bx, By, Bz]).transpose()
    normB = (Bx ** 2 + By ** 2 + Bz ** 2) ** (1 / 2)
    b = np.array([Bx / normB, By / normB, Bz / normB]).transpose()
    e1 = np.vstack([By, -Bx, np.zeros(shape=(len(Bx)))]).transpose()
    e1 = (e1.T / (Bx ** 2 + By ** 2) ** (1 / 2)).transpose()
    e2 = np.cross(b, e1)
    return e1, e2, b, normB, B


def get_velocity(normv, theta1, theta2, x, y, z):
    e1, e2, b, *_ = get_magnetic(x, y, z)
    v_para = np.array((b.T * np.cos(theta1)).T * normv)
    normv_para = np.linalg.norm(v_para, axis=-1)
    normv_prep = (normv ** 2 - normv_para ** 2) ** (1 / 2)
    v_prep1 = np.array(normv_prep * np.cos(theta2) * e1.T).T
    v_prep2 = np.array(normv_prep * np.sin(theta2) * e2.T).T
    v = v_para + v_prep1 + v_prep2
    return v


def sampling_circle(sample_size):
    a = np.random.uniform(size=sample_size)
    b = np.random.uniform(size=sample_size)
    c = np.random.uniform(size=sample_size)
    x = (R0 + r0 * a ** (1 / 2) * np.cos(2 * math.pi * b))*np.cos(2 * math.pi * c)
    y = (R0 + r0 * a ** (1 / 2) * np.cos(2 * math.pi * b))*np.sin(2 * math.pi * c)
    z = r0 * a ** (1 / 2) * np.sin(2 * math.pi * b)
    return np.vstack((x, y, z))


def get_P(x, y, z, device='cpu'):
    ex, ey, ez, _, B = get_magnetic(x, y, z, device)
    B_len = B.shape[0]
    ze = np.zeros(shape=(B_len,))
    Omega = np.array([[ze, B[:, 2], -B[:, 1]],
                      [-B[:, 2], ze, B[:, 0]],
                      [B[:, 1], -B[:, 0], ze]]) * (t / 2) * (q / m)
    Omega = Omega.transpose(2, 0, 1)
    I = np.identity(3)

    inv_IOmega = np.linalg.inv(I - Omega)
    P = (I + Omega) @ inv_IOmega
    return P


def _get_magnetic(x, y, z, device='cpu'):
    Bx = B0 * (-1 * q0 * R0 * y + z * x) / (q0 * (x ** 2 + y ** 2))
    By = B0 * (q0 * R0 * x + z * y) / (q0 * (x ** 2 + y ** 2))
    Bz = B0 * (-1 + R0 / (x ** 2 + y ** 2) ** (1 / 2)) / q0

    B = torch.vstack([Bx, By, Bz]).T
    normB = (Bx ** 2 + By ** 2 + Bz ** 2) ** (1 / 2)
    b = torch.vstack([Bx / normB, By / normB, Bz / normB]).T
    e1 = torch.vstack(
        [By, -Bx, torch.zeros(size=(len(Bx),), device=device, dtype=torch.float64)]
    ).T
    e1 = (e1.T / (Bx ** 2 + By ** 2) ** (1 / 2)).T
    e2 = torch.cross(b, e1, dim=1)

    return e1, e2, b, normB, B


def _get_P(x, y, z, device='cpu'):
    ex, ey, ez, _, B = _get_magnetic(x, y, z, device)
    B_len = B.shape[0]
    Omega = torch.zeros(size=(B_len, 3, 3), device=device, dtype=torch.float64)
    Omega[:, 0, 1] = B[:, 2]
    Omega[:, 1, 0] = -B[:, 2]
    Omega[:, 0, 2] = -B[:, 1]
    Omega[:, 2, 0] = B[:, 1]
    Omega[:, 1, 2] = B[:, 0]
    Omega[:, 2, 1] = -B[:, 0]
    Omega = Omega * (t / 2) * (q / m)
    I = torch.eye(3, device=device)
    inv_IOmega = torch.linalg.inv(I - Omega)
    P = (I + Omega) @ inv_IOmega
    return P


def random_x_v(sample_num):
    x0 = sampling_circle(sample_size=(sample_num,)).transpose()
    theta1 = np.random.uniform(0, 2 * np.pi, size=(sample_num,))  # pinch angle
    theta2 = np.random.uniform(0, 2 * np.pi, size=(sample_num,))  # 在R平面投影角度
    v0 = get_velocity(normv, theta1, theta2, x0[:, 0], x0[:, 1], x0[:, 2])
    return x0, v0


def boris_step(x0, v0, device='cpu'):
    v1 = _get_P(x0[:, 0], x0[:, 1], x0[:, 2], device) @ v0[..., None]
    v1 = v1.reshape(v0.shape[0], -1)
    x1 = x0 + v1 * t
    return x1, v1


def _boris_partial(idx, x0, v0, steps, device, pbar):
    data_len = x0.shape[0]
    data = torch.empty(size=(data_len, steps + 1, 6), device=device, dtype=torch.float64)
    data[:, 0, :3] = x0
    data[:, 0, 3:] = v0
    for step in range(steps):
        x1, v1 = boris_step(x0, v0, device)
        x0, v0 = x1, v1
        data[:, step + 1, :3] = x0
        data[:, step + 1, 3:] = v0
        pbar.update(data_len)
    return idx, data


def boris(x0, v0, steps=1, device='cpu'):
    x0 = torch.tensor(x0, device=device, dtype=torch.float64)
    v0 = torch.tensor(v0, device=device, dtype=torch.float64)
    data_len = x0.shape[0]
    data = torch.empty(size=(data_len, steps + 1, 6), device=device, dtype=torch.float64)
    pbar = tqdm(total=data_len * steps, desc='Push steps')
    pbar.reset()
    if device == 'cuda':
        _, data_partial = _boris_partial(0, x0, v0, steps, device, pbar)
        data[:] = data_partial[:]
    elif device == 'cpu':
        tasks = []
        workers = min(os.cpu_count(), 3)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for rank, idx, _ in task_assign(data_len, workers):
                tasks.append(
                    executor.submit(_boris_partial,
                                    idx, x0[idx, :], v0[idx, :], steps, device, pbar)
                )

            for task in as_completed(tasks):
                idx, data_partial = task.result()
                data[idx, :, :] = data_partial
    else:
        raise ValueError('Unknown device')
    return data


def save(d):
    if isinstance(d, torch.Tensor):
        d = d.detach().cpu().numpy()
    else:
        d = np.array(d, dtype=np.float64)
    if not os.path.exists('data'):
        os.makedirs('data')
    scipy.io.savemat(os.path.join('./data', 'data.mat'), {'data': d})

