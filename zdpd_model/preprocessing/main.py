import pandas as pd

from .abstract import AbstractPreprocessor
from .re_fw_removals import RemoveCopiedInformationEnron
from .text_removals import (
    RemoveEmptyBodyEmails, RemoveHeadersNigerianFraud, RemoveMultiSpacing,
    RemoveUnwantedCharacters, RemoveLeadingTrailingNL, RemoveMultipleNL,
    RemoveLeadingGreaterThan, RemoveDuplicateTextNazario
)
from .text_replacements import (
    EmptyBodyToStr, EmptySubjectToStr, FixUnicodeErrors,
    ReplaceAttachedFilePatternEnron
)


class MainPreprocessor(AbstractPreprocessor):
    """Main class to run all preprocessors."""

    preprocessors = [
        EmptySubjectToStr,
        EmptyBodyToStr,
        RemoveUnwantedCharacters,
        RemoveLeadingGreaterThan,
        RemoveMultiSpacing,
        RemoveMultipleNL,
        RemoveLeadingTrailingNL,
        ReplaceAttachedFilePatternEnron,
        RemoveCopiedInformationEnron,
        RemoveHeadersNigerianFraud,
        RemoveDuplicateTextNazario,
        FixUnicodeErrors,
        RemoveEmptyBodyEmails
    ]

    @classmethod
    def apply(cls, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Apply all preprocessing steps to the data.

        Parameters
        ----------
        raw_data : pd.DataFrame
            Data to preprocess.

        Returns
        -------
        pd.DataFrame
            Preprocessed data
        """
        data = raw_data.copy(deep=True)
        for preprocessor in cls.preprocessors:
            print("Running {} ...".format(preprocessor.__name__))
            data = preprocessor.apply(data)

        return data
