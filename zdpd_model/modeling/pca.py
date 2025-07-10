import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from typing import Any, Dict

from .abstract import AbstractModel


class PCAReconstruction(AbstractModel):
    """Reconstruction class for PCA."""

    name = "PCA"

    def __init__(self, *args, **kwargs):
        """Initialize attributes."""
        super().__init__(*args, **kwargs)
        self.n_components = kwargs.get("n_components")

        if not self.n_components:
            raise ValueError("Missing 'n_components' key-argument.")
        self._model = PCA(n_components=self.n_components)

    @property
    def hyperparameters(self) -> Dict[str, Any]:
        """Get hyperparameters."""
        return {
            "n_components": self.n_components,
            "re_quantile": self.re_quantile
        }

    def _calculate_reconstruction_errors(
        self, data: pd.DataFrame
    ) -> np.ndarray:
        rec_data = self._reconstruct(data)
        # Euclidean (L2) norm
        return np.linalg.norm(data.values - rec_data.values, axis=1)

    def _reconstruct(self, data: pd.DataFrame) -> pd.DataFrame:
        data_proj = self._model.transform(data)
        data_reconstructed = self._model.inverse_transform(data_proj)
        return pd.DataFrame(
            data_reconstructed, index=data.index, columns=data.columns
        )
