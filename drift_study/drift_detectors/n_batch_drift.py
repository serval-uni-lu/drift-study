import numpy as np


class NBatchDrift:
    def __init__(
        self,
        drift_detector,
        batch_size: int,
        **kwargs,
    ) -> None:
        self.drift_detector = drift_detector
        self.batch_size = batch_size
        self.counter = 0
        self.mem = []

    def fit(self, **kwargs):
        self.drift_detector.fit(**kwargs)
        self.mem = []
        self.counter = 0

    def update(self, **kwargs):
        self.mem.append(kwargs)
        self.counter += 1
        if self.counter >= self.batch_size:
            mem = {k: [dic[k] for dic in self.mem] for k in self.mem[0]}
            self.mem = []
            self.counter = 0
            return self.drift_detector.update(**mem)

        return False, False, np.nan, np.nan

    def needs_label(self):
        return self.drift_detector.needs_label()
