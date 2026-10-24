import h5py
import numpy as np
import os

class ExperienceDatasetWriter:
    def __init__(self, save_path: str, max_steps: int):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.file = h5py.File(save_path, "w")
        self.ptr = 0
        self.max_steps = max_steps
        self.datasets = None

    def _init_datasets(self, experience_kwargs):
        self.datasets = {}
        for key, value in experience_kwargs.items():
            shape = (self.max_steps,) + value.shape
            self.datasets[key] = self.file.create_dataset(
                key,
                shape=shape,
                dtype=value.dtype,
                chunks=(1,) + value.shape,   # step-wise chunk
                compression="gzip",
            )

    def append(self, experience_kwargs):
        if self.datasets is None:
            self._init_datasets(experience_kwargs)

        for key, value in experience_kwargs.items():
            self.datasets[key][self.ptr] = value

        self.ptr += 1

    def close(self):
        self.file.attrs["num_steps"] = self.ptr
        self.file.close()