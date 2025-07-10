import re
import pandas as pd
from abc import ABC, abstractmethod


class AbstractPreprocessor(ABC):
    """Abstract base class for preprocessing steps."""

    @classmethod
    @abstractmethod
    def apply(cls, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Apply specific preprocessing steps to a data source.

        Parameters
        ----------
        raw_data : pd.DataFrame
            Data to preprocess.

        Returns
        -------
        pd.DataFrame
            Preprocessed data.

        Raises
        ------
        NotImplementedError
            When method is missing.
        """
        raise NotImplementedError(
            "'apply' method missing in {}.".format(cls.__name__)
        )


class AbstractPatternRemover(ABC):
    """Abstract class for forward and reply pattern removers."""

    pattern = ""
    search_kwargs = {"flags": re.IGNORECASE}

    @classmethod
    def remove(cls, text: str) -> str:
        """Remove a specific forward and reply pattern.

        Parameters
        ----------
        text : str
            The email body.

        Returns
        -------
        str
            The email body with the forward and reply patterns removed.

        Raises
        ------
        ValueError
            When pattern attribute is empty.
        """
        if not cls.pattern:
            raise ValueError(
                "a 'pattern' attribute is needed to run this method."
            )

        match = re.search(cls.pattern, text, **cls.search_kwargs)
        if match:
            return text[:match.start()].strip()
        return text
