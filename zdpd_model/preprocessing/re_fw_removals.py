import pandas as pd
import re

from .abstract import AbstractPatternRemover, AbstractPreprocessor


class OriginalMsgPatternRemover(AbstractPatternRemover):
    """Removes pattern: '- - - - original message - - - -'."""

    pattern = r'(?:-+\s*)*original message(?:\s*-+)*'


class NameOnDatePatternRemover(AbstractPatternRemover):
    """Removes pattern: '"<name>" on mm/dd/yyyy hh:mm:ss am/pm'."""

    pattern = (
        r'"[^"]+"\s+on\s+'
        r'\d{1,2}\s*/\s*\d{1,2}\s*/\s*\d{2,4}'
        r'\s+\d{1,2}\s*:\s*\d{2}'
        r'(?:\s*:\s*\d{2})?\s*(?:am|pm)'
    )


class ForwardedByPatternRemover(AbstractPatternRemover):
    """Removes pattern: 'forwarded by'."""

    pattern = r'(?i)(?:-+\s*){3,}forwarded by.*'
    search_kwargs = {"flags": re.IGNORECASE | re.DOTALL}


class NameNLDatePatternRemover(AbstractPatternRemover):
    """Removes pattern: '<name><new_line>mm/dd/yyyy hh:mm am/pm'."""

    pattern = (
        r'[^\n]+\n\s*'
        r'\d{1,2}\s*/\s*\d{1,2}\s*/\s*\d{2,4}'
        r'\s+\d{1,2}\s*:\s*\d{2}'
        r'(?:\s*:\s*\d{2})?\s*(?:am|pm)'
    )


class FromEmailPatternRemover(AbstractPatternRemover):
    """Removes pattern: 'from: <email> mm/dd/yyyy hh:mm am/pm'."""

    pattern = (
        r'from\s*:\s*[\w\.\s@]+'
        r'\s+\d{1,2}\s*/\s*\d{1,2}\s*/\s*\d{2,4}'
        r'\s+\d{1,2}\s*:\s*\d{2}'
        r'(?:\s*:\s*\d{2})?\s*(?:am|pm)'
    )


class EmailOnDatePatternRemover(AbstractPatternRemover):
    """Removes pattern: '<email> on mm/dd/yyyy hh:mm am/pm'."""

    pattern = (
        r'[a-z0-9]+(?:\s*[_\-.]\s*[a-z0-9]+)*'   # email local part
        r'\s*@\s*'
        r'[a-z0-9]+(?:\s*[-.]\s*[a-z0-9]+)*'     # email domain
        r'\s*\.\s*[a-z]{2,}'                     # TLD with optional space
        r'\s+on\s+'                              # 'on' separator
        r'\d{1,2}\s*/\s*\d{1,2}\s*/\s*\d{2,4}'   # date with flexible spacing
        r'\s+\d{1,2}\s*:\s*\d{2}'                # hour:min
        r'(?:\s*:\s*\d{2})?'                     # optional :ss
        r'\s*(?:am|pm)'                          # am/pm
    )


class FromNameEmailDatePatternRemover(AbstractPatternRemover):
    """Removes pattern: 'from: <name> / <email> mm/dd/yyyy hh:mm am/pm'."""

    pattern = (
        r'from\s*:\s*.+?'
        r'\s+on\s+'
        r'\d{1,2}\s*/\s*\d{1,2}\s*/\s*\d{2,4}'
        r'\s+\d{1,2}\s*:\s*\d{2}'
        r'(?:\s*:\s*\d{2})?\s*(?:am|pm)'
    )


class NameEmailWrotePatternRemover(AbstractPatternRemover):
    """Removes pattern: '- - - <name/email> wrote :'."""

    pattern = r'\n[-\s]*[^:\n]+wrote\s*:'


class NLCSSNLPatternRemover(AbstractPatternRemover):
    """Removes pattern: '<new_line>css<new_line>'."""

    pattern = r'(?im)^ccs\s*:?.*$'


class RemoveCopiedInformationEnron(AbstractPreprocessor):
    """Removes copied information often found in Enron replies or forwards."""

    pattern_removers = [
        OriginalMsgPatternRemover,
        NameOnDatePatternRemover,
        ForwardedByPatternRemover,
        NameNLDatePatternRemover,
        FromEmailPatternRemover,
        EmailOnDatePatternRemover,
        FromNameEmailDatePatternRemover,
        NameEmailWrotePatternRemover,
        NLCSSNLPatternRemover
    ]

    @classmethod
    def apply(cls, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Remove all forward and reply patterns from Enron.

        Parameters
        ----------
        raw_data : pd.DataFrame
            The data with foward and reply patterns included.

        Returns
        -------
        pd.DataFrame
            The data with forward and reply patterns removed.
        """
        data = raw_data.copy(deep=True)

        if "source" not in data.columns:
            return data
        elif "Enron" not in data["source"].unique():
            return data

        enron_indices = data[data['source'] == "Enron"].index.to_list()
        non_enron_df = data[~data.index.isin(enron_indices)]
        enron_df = data[data.index.isin(enron_indices)]

        enron_df["body"] = enron_df["body"].apply(cls._remove_all)
        return pd.concat([non_enron_df, enron_df], axis=0).sort_index()

    @classmethod
    def _remove_all(cls, text: str) -> str:
        """Remove all forward and reply patterns.

        Parameters
        ----------
        text : str
            The raw email body.

        Returns
        -------
        str
            The text with all forward and reply patterns removed.
        """
        for pattern_remover in cls.pattern_removers:
            text = pattern_remover.remove(text)

        return text
