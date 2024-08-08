from typing import Any

import numpy as np
from numpy import ndarray

class Tokamak1:

    def __init__(
        self,
        R0: np.float64 = ...,  # Large radius
        B0: np.float64 = ...,  # Magnetic field at large radius
        r0: np.float64 = ...,  # Small radius
        q0: np.float64 = ...,  # Safety factor
        dt: np.float64 = ...   # T=2π/QB, dt=T/10
    ): ...

    def get_magnetic(
        self,
        position: ndarray[Any, np.dtype[np.float64]]
    ) -> ndarray[Any, np.dtype[np.float64]]: ...

    def get_v_from_P(
        self,
        magnetic: ndarray[Any, np.dtype[np.float64]],
        velocity: ndarray[Any, np.dtype[np.float64]]
    ) -> ndarray[Any, np.dtype[np.float64]]: ...

    # read and write
    R0: np.float64
    B0: np.float64
    r0: np.float64

    # read and write
    q0: np.float64
    # readonly
    @property
    def q0_recip(self) -> np.float64: ...

    # read and write
    Q: np.float64
    m: np.float64
    dt: np.float64

    # readonly
    @property
    def Qdt_2m(self) -> np.float64: ...
    @property
    def Qdt_2m_2(self) -> np.float64: ...

    @property
    def T0(self) -> np.float64: ...


class Boris:
    # readonly
    @property
    def mag(self) -> Tokamak1: ...

    def __init__(self): ...

    @staticmethod
    def task_evaluation(
        particle_num_: np.int64,
        steps_: np.int64,
        thread_num_: np.int64
    ) -> tuple[np.int64, np.int64]: ...

    def submit(
        self,
        thread_progress_: ndarray[Any, np.dtype[np.int64]],
        result_: ndarray[Any, np.dtype[np.float64]],
        position0_: ndarray[Any, np.dtype[np.float64]],
        velocity0_: ndarray[Any, np.dtype[np.float64]],
        thread_num_: np.int64
    ) -> None: ...

    def reset(self) -> None: ...

    def run(self) -> None: ...



