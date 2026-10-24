import h5py

class ExperienceDatasetReader:
    def __init__(self, dataset_path: str):
        self.file = h5py.File(dataset_path, "r")
        self.length = self.file.attrs["num_steps"]

    def __len__(self):
        return self.length

    def get_step(self, idx: int):
        return {
            "obs": self.file["obs"][idx],
            "actions": self.file["actions"][idx],
            "rewards": self.file["rewards"][idx],
            "next_obs": self.file["next_obs"][idx],
            "dones": self.file["dones"][idx],
        }

    def close(self):
        self.file.close()
