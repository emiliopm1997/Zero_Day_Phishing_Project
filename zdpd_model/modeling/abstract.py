import joblib
import json
import numpy as np
import pandas as pd

from abc import ABC
from pathlib import Path
from typing import Optional, Union

from .encoding import BERTEncoder

MODELS_PATH = Path().cwd().parent / "models"


class AbstractModel(ABC):
    """Abstract class for models."""

    def __init__(self, re_quantile: float, *args, **kwargs):
        """Initialize attributes."""
        if (re_quantile > 1) or (re_quantile < 0):
            raise ValueError(
                "Reconstruction error quantile ('re_quantile') must be "
                "between [0, 1]."
            )
        self.re_quantile = re_quantile

    def encode_data(
        self,
        data: pd.Series,
        encoder: BERTEncoder,
        **kwargs
    ) -> pd.DataFrame:
        """Encode data with encoder.

        Parameters
        ----------
        data : pd.Series
            The data to encode.
        encoder : BERTEncoder
            The encoding instance.

        Returns
        -------
        pd.DataFrame :
            The encoded data.
        """
        return pd.DataFrame(data.apply(
            lambda x: encoder.encode(x, **kwargs)
        ).tolist())

    def fit(
        self,
        data: Union[pd.DataFrame, pd.Series],
        encoder: Optional[BERTEncoder] = None,
        **kwargs
    ):
        """Fit data to model.

        Parameters
        ----------
        data : Union[pd.DataFrame, pd.Series]
            The data to fit the model. The type should be a pd.DataFrame if it
            is already encoded, and a pd.Series if it needs to be encoded.
        encoder : Optional[BERTEncoder]
            An encoder instance, by default None.
        """
        if encoder and isinstance(data, pd.Series):
            data = self.encode_data(data, encoder, **kwargs)
        self._model.fit(data)
        errors = self._calculate_reconstruction_errors(data)
        self.re_threshold = np.quantile(errors, self.re_quantile)

    def predict(
        self,
        data: Union[pd.DataFrame, pd.Series],
        encoder: Optional[BERTEncoder] = None,
        **kwargs
    ) -> pd.Series:
        """Predict classifications using the model.

        Parameters
        ----------
        data : Union[pd.DataFrame, pd.Series]
            The data for prediction. The type should be a pd.DataFrame if it
            is already encoded, and a pd.Series if it needs to be encoded.
        encoder : Optional[BERTEncoder]
            An encoder instance, by default None.

        Returns
        -------
        pd.Series
            The classified data.
        """
        if encoder and isinstance(data, pd.Series):
            data = self.encode_data(data, encoder, **kwargs)
        errors = self._calculate_reconstruction_errors(data)
        errors_ser = pd.Series(errors, index=data.index)
        return (errors_ser >= self.re_threshold).astype(int)

    @classmethod
    def _read_json_metadata(cls):
        with open(MODELS_PATH / "model_metadata.json", "r") as f:
            cls.metadata = json.load(f)

    @classmethod
    def _write_metadata(cls):
        with open(MODELS_PATH / "model_metadata.json", "w") as f:
            json.dump(cls.metadata, f)

    @classmethod
    def load(cls, model_name: str):
        """Load a model.

        Parameters
        ----------
        model_name : str
            The name of the model one wants to load.
        """
        cls._read_json_metadata()
        file_path = cls.metadata[model_name]["file_path"]

        model = cls(**cls.metadata[model_name]["hyperparameters"])
        model._model = joblib.load(file_path)
        model.re_threshold = cls.metadata[model_name]["re_threshold"]
        return model

    def save(self, model_name: str, encoding_method: str):
        """Save a model.

        Parameters
        ----------
        model_name : str
            The name of the model.
        encoding_method : str
            The encoding method used for training.
        """
        self._read_json_metadata()
        metadata = {
            "model_type": self.name,
            "hyperparameters": self.hyperparameters,
            "re_threshold": self.re_threshold,
            "encoding_method": encoding_method,
            "file_path": str(MODELS_PATH / f"{model_name}.joblib")
        }

        joblib.dump(self._model, Path(metadata["file_path"]))
        self.metadata[model_name] = metadata
        self._write_metadata()
