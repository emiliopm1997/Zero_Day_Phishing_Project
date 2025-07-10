import numpy as np
import pandas as pd

from sklearn.svm import OneClassSVM
from typing import Any, Dict

from .abstract import AbstractModel


class OCSVMReconstruction(AbstractModel):
    """Reconstruction class for OC-SVM."""

    name = "OC-SVM"

    def __init__(self, *args, **kwargs):
        """Initialize attributes."""
        super().__init__(*args, **kwargs)
        self.kernel = kwargs.get("kernel")
        self.nu = kwargs.get("nu")
        self.gamma = kwargs.get("gamma")

        if not self.kernel:
            raise ValueError("Missing 'kernel' key-argument.")
        if not self.nu:
            raise ValueError("Missing 'nu' key-argument.")
        if not self.gamma:
            raise ValueError("Missing 'gamma' key-argument.")

        self._model = OneClassSVM(
            kernel=self.kernel,
            nu=self.nu,
            gamma=self.gamma
        )

    @property
    def hyperparameters(self) -> Dict[str, Any]:
        """Get hyperparameters."""
        return {
            "kernel": self.kernel,
            "nu": self.nu,
            "gamma": self.gamma,
            "re_quantile": self.re_quantile
        }

    def _calculate_reconstruction_errors(
        self, data: pd.DataFrame
    ) -> np.ndarray:
        errors = -self._model.decision_function(data)
        return errors
