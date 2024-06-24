import numpy as np
import logging

logger = logging.getLogger(__name__)


class NormalizerMixin:
    def normalize(self, values: np.ndarray) -> np.ndarray:
        if self.norm is None:
            logger.debug("Node weights are not normalized.")
            return values
        if self.norm == "l1":
            return values / np.sum(values)
        if self.norm == "l2":
            return values / np.linalg.norm(values)
        if self.norm == "unit-max":
            return values / np.amax(values)
        if self.norm == "unit-sum":
            return values / np.sum(values)
        if self.norm == "unit-std":
            return values / np.std(values)
        raise ValueError("Weight normalization must be 'l1', 'l2', 'unit-max' 'unit-sum' or 'unit-std'.")
