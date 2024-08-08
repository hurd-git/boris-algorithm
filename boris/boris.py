import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # 为1时最快
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
from tqdm import tqdm
import scipy
import time
import ctypes
import multiprocessing as mp

import boris.boris_cpp as boris_cpp

# class MagneticField:
#     def get_magnetic(self, *args, **kwargs):
#         pass


# class Tokamak1(MagneticField):
#     def __init__(self):
#         self.R0 = 6.2
#         self.q0 = 2.5
#         self.B0 = 5
#         self.r0 = 2
#
#     def get_magnetic(self, x, y, z):
#         pass


# magnetics = {'tokamak1': Tokamak1}

cpu_num = os.cpu_count()


def tqdm_guard_process(shared_array_base, barrier, total_progress):
    thread_progress = np.frombuffer(shared_array_base, dtype=shared_array_base._type_)

    tbar = tqdm(total=total_progress, ncols=100, mininterval=0.05)
    barrier.wait()
    while True:
        up_num = sum(thread_progress) - tbar.n
        tbar.update(up_num)
        if tbar.n >= tbar.total:
            break
        time.sleep(0.04)


e0 = 1.602176634 * 10 ** (-19)
c0 = 299792458


class Boris:
    def __init__(self):
        self.cpp = boris_cpp.Boris()
        self.mag = self.cpp.mag

        # 磁场信息
        self.R0 = self.mag.R0
        self.B0 = self.mag.B0
        self.r0 = self.mag.r0
        self.q0 = self.mag.q0

        self.Q = self.mag.Q
        self.m = self.mag.m
        self.dt = self.mag.dt

        self.E = 3.5  # MeV, when E=3.5MeV, v=12991758m/s, r=0.05388m

    @property
    def R0(self) -> np.float64:
        return self.mag.R0

    @R0.setter
    def R0(self, value: np.float64):
        self.mag.R0 = value

    @property
    def B0(self) -> np.float64:
        return self.mag.B0

    @B0.setter
    def B0(self, value: np.float64):
        self.mag.B0 = value

    @property
    def r0(self) -> np.float64:
        return self.mag.r0

    @r0.setter
    def r0(self, value: np.float64):
        self.mag.r0 = value

    @property
    def q0(self) -> np.float64:
        return self.mag.q0

    @q0.setter
    def q0(self, value: np.float64):
        self.mag.q0 = value

    @property
    def Q(self) -> np.float64:
        return self.mag.Q

    @Q.setter
    def Q(self, value: np.float64):
        self.mag.Q = value

    @property
    def m(self) -> np.float64:
        return self.mag.m

    @m.setter
    def m(self, value: np.float64):
        self.mag.m = value

    @property
    def dt(self) -> np.float64:
        return self.mag.dt

    @dt.setter
    def dt(self, value: np.float64):
        self.mag.dt = value

    @property
    def q0_recip(self) -> np.float64:
        return self.mag.q0_recip

    @property
    def Qdt_2m(self) -> np.float64:
        return self.mag.Qdt_2m

    @property
    def Qdt_2m_2(self) -> np.float64:
        return self.mag.Qdt_2m_2

    @property
    def T0(self) -> np.float64:
        return self.mag.T0

    #@property
    #def normv(self):
    #    return np.sqrt(2 * e0 * self.E * 10 ** 6 / self.m)

    @property
    def normv(self):
        return c0 * np.sqrt(1 - (self.m*c0**2/(self.E*e0*10**6 + self.m*c0**2))**2)

    @property
    def r(self):  # Cyclotron radius
        return self.T0 * self.normv / (2 * np.pi)

    def get_magnetic(self, x, y, z, device='cpu'):
        x, y, z = np.array(x, ndmin=1), np.array(y, ndmin=1), np.array(z, ndmin=1)
        B = np.empty(shape=(len(x), 3), dtype=np.float64)

        Bx = self.B0 * (-1 * self.q0 * self.R0 * y + z * x) / (self.q0 * (x ** 2 + y ** 2))
        By = self.B0 * (self.q0 * self.R0 * x + z * y) / (self.q0 * (x ** 2 + y ** 2))
        Bz = self.B0 * (-1 + self.R0 / (x ** 2 + y ** 2) ** (1 / 2)) / self.q0

        B = np.array([Bx, By, Bz]).transpose()
        normB = (Bx ** 2 + By ** 2 + Bz ** 2) ** (1 / 2)
        b = np.array([Bx / normB, By / normB, Bz / normB]).transpose()
        e1 = np.vstack([By, -Bx, np.zeros(shape=(len(Bx)))]).transpose()
        e1 = (e1.T / (Bx ** 2 + By ** 2) ** (1 / 2)).transpose()
        e2 = np.cross(b, e1)
        return e1, e2, b, normB, B

    def get_velocity(self, normv, theta1, theta2, x, y, z):
        e1, e2, b, *_ = self.get_magnetic(x, y, z)
        v_para = np.array((b.T * np.cos(theta1)).T * normv)
        normv_para = np.linalg.norm(v_para, axis=-1)
        normv_prep = (normv ** 2 - normv_para ** 2) ** (1 / 2)
        v_prep1 = np.array(normv_prep * np.cos(theta2) * e1.T).T
        v_prep2 = np.array(normv_prep * np.sin(theta2) * e2.T).T
        v = v_para + v_prep1 + v_prep2
        return v

    def sampling_circle(self, sample_size):
        a = np.random.uniform(size=sample_size)
        b = np.random.uniform(size=sample_size)
        c = np.random.uniform(size=sample_size)
        x = (self.R0 + self.r0 * a ** (1 / 2) * np.cos(2 * np.pi * b)) * np.cos(2 * np.pi * c)
        y = (self.R0 + self.r0 * a ** (1 / 2) * np.cos(2 * np.pi * b)) * np.sin(2 * np.pi * c)
        z = self.r0 * a ** (1 / 2) * np.sin(2 * np.pi * b)
        return np.vstack((x, y, z))

    def random_x_v(self, sample_num):
        x0 = self.sampling_circle(sample_size=(sample_num,)).transpose()
        theta1 = np.random.uniform(0, 2 * np.pi, size=(sample_num,))  # pinch angle
        theta2 = np.random.uniform(0, 2 * np.pi, size=(sample_num,))  # 在R平面投影角度
        v0 = self.get_velocity(self.normv, theta1, theta2, x0[:, 0], x0[:, 1], x0[:, 2])
        return x0, v0

    def get_random_v(self, x0):
        sample_num = x0.shape[0]
        theta1 = np.random.uniform(0, 2 * np.pi, size=(sample_num,))  # pinch angle
        theta2 = np.random.uniform(0, 2 * np.pi, size=(sample_num,))  # 在R平面投影角度
        v0 = self.get_velocity(self.normv, theta1, theta2, x0[:, 0], x0[:, 1], x0[:, 2])
        return v0

    def run(self, x0: np.ndarray, v0: np.ndarray, steps=1, threads=cpu_num, info=False) -> np.ndarray:
        # copy x0, v0, !!do not change this code!!, it is stable and perfect and have no strange bugs
        position0 = np.empty(shape=(x0.shape[0], 3), dtype=np.float64)
        velocity0 = np.empty(shape=(v0.shape[0], 3), dtype=np.float64)
        position0[:] = x0[:]
        velocity0[:] = v0[:]

        # 检查并纠正数据合法性
        if threads is None:
            threads = 1
        steps = int(steps)
        threads = int(threads)
        if steps < 0:
            steps = 0
        if steps == 0:
            return np.concatenate([x0, v0], axis=-1)
        if threads <= 0:
            threads = 1

        # 参数初始化
        # cpp参数

        particle_num = x0.shape[0]
        total_progress, actual_threads = self.cpp.task_evaluation(particle_num, steps, threads)
        # tqdm伴随参数
        shared_progress_base = mp.Array(ctypes.c_int64, actual_threads, lock=False)
        # barrier = mp.Barrier(2)
        thread_progress = np.frombuffer(shared_progress_base, dtype=shared_progress_base._type_)
        # 结果
        # 传入2维矩阵
        result = np.empty(shape=(particle_num, steps + 1, 6), dtype=np.float64)
        result = result.reshape(-1, 6)

        # position0 = np.random.rand(particle_num, 3)
        # velocity0 = np.random.rand(particle_num, 3) / 10 ** -9

        # 提交到cpp
        # if info:
        #     tic = time.time()
        #     print("prepare memory...", end='')
        self.cpp.submit(thread_progress, result, position0, velocity0, actual_threads)
        # if info:
        #     toc = time.time()
        #     print('done (' + '{:.3f}'.format(toc - tic) + 's)')
        # 开启tqdm伴随
        # subprocess = mp.Process(target=tqdm_guard_process, args=(shared_progress_base, barrier, total_progress))
        # subprocess.start()

        # 与tqdm保持同步
        # barrier.wait()
        # 开始计算
        if info:
            tic = time.time()
            print("Using threads: {}, prepare memory + start calculating...".format(actual_threads), end='')
        self.cpp.run()
        if info:
            toc = time.time()
            print('done (' + '{:.3f}'.format(toc - tic) + 's)')
        # 结束tqdm
        # subprocess.join()
        # cpp运行参数重置
        self.cpp.reset()

        result = result.reshape(particle_num, steps + 1, 6)
        return result
