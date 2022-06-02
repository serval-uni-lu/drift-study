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

    def fit(self, x, t, y, y_scores):
        self.drift_detector.fit(self, x, t, y, y_scores)
        self.mem = []

    def update(self, x, t, y, y_scores):
        self.mem.append((x, t, y, y_scores))
        self.counter += 1
        if self.counter >= self.batch_size:
            x, t, y, y_scores = self.drift_detector.update(zip(*self.mem))
            self.mem = []
            return self.drift_detector.update(x, t, y, y_scores)

        return False, False, np.nan, np.nan

    def needs_label(self):
        return self.drift_detector.needs_label()
