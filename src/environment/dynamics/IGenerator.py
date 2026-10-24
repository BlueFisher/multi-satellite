import numpy as np


class IGenerator:
    def __call__(
        self,
        state: np.ndarray[(6,), np.float64],
        accel: np.ndarray[(3,), np.float64],
        step: float = 10,
    ) -> np.ndarray[(6,), np.float64]:
        raise NotImplementedError("This is an abstract class")
