import pandas as pd
import re

from ftfy import fix_text

from .abstract import AbstractPreprocessor


class AbstractEmptyFieldToStr(AbstractPreprocessor):
    """Converts NaN fields to empty strings."""

    field = ""

    @classmethod
    def apply(cls, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Apply transformation of an empty field to an empty string.

        Parameters
        ----------
        raw_data : pd.DataFrame
            The data to transform.

        Returns
        -------
        pd.DataFrame
            The transformed data.

        Raises
        ------
        ValueError
            When field attribute is empty.
        """
        data = raw_data.copy(deep=True)

        if not cls.field:
            raise ValueError(
                "a 'field' attribute is needed to run this method."
            )

        data[cls.field] = data[cls.field].fillna("")
        return data


class EmptySubjectToStr(AbstractEmptyFieldToStr):
    """Converts NaN subject to empty strings."""

    field = "subject"


class EmptyBodyToStr(AbstractEmptyFieldToStr):
    """Converts NaN body to empty strings."""

    field = "body"


class ReplaceAttachedFilePatternEnron(AbstractEmptyFieldToStr):
    """Replaces attached file mentions in Enron emails with a tag."""

    @classmethod
    def apply(cls, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Convert attached files to tags.

        Parameters
        ----------
        raw_data : pd.DataFrame
            The data with explicit attached file text.

        Returns
        -------
        pd.DataFrame
            The data with attached file text converted to a tag.
        """
        pattern = (
            r'(?:\( ?see attached file ?: ?.+? ?\) ?)+'
            r'(?:\r?\n- ?.+)+'
        )
        data = raw_data.copy(deep=True)
        
        if "source" not in data.columns:
            return data
        elif "Enron" not in data["source"].unique():
            return data
        data["body"] = data["body"].apply(
            lambda x: re.sub(
                pattern, '<attached_file>', x, flags=re.IGNORECASE
            )
        )
        return data


class FixUnicodeErrors(AbstractPreprocessor):
    """Fix common Unicode errors."""

    @classmethod
    def apply(cls, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Fix common Unicode errors and inconsistencies.

        Parameters
        ----------
        raw_data : pd.DataFrame
            The data set with Unicode errors.

        Returns
        -------
        pd.DataFrame
            The data set with fixed Unicode errors.
        """
        data = raw_data.copy(deep=True)
        data["body"] = data["body"].apply(fix_text)
        return data
