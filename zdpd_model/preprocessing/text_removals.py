import re
import pandas as pd

from .abstract import AbstractPreprocessor


class RemoveEmptyBodyEmails(AbstractPreprocessor):
    """Removes rows where the email body is empty or null."""

    @classmethod
    def apply(cls, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Remove emails with empty bodies.

        Parameters
        ----------
        raw_data : pd.DataFrame
            The full data set.

        Returns
        -------
        pd.DataFrame
            The cropped data set.
        """
        data = raw_data.copy(deep=True)
        data = data[~data["body"].isnull()]
        data = data[data["body"] != ""]
        return data


class RemoveUnwantedCharacters(AbstractPreprocessor):
    """Removes unwanted characters from the body."""

    unwanted_characters = [
        "\r",  # carriage returns
        "\t",  # tabs
        "\\"  # backslash
    ]

    @classmethod
    def apply(cls, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Remove unwanted characters.

        Parameters
        ----------
        raw_data : pd.DataFrame
            The data set with all sorts of characters.

        Returns
        -------
        pd.DataFrame
            The data set with unwanted characters removed.
        """
        data = raw_data.copy(deep=True)

        for uc in cls.unwanted_characters:
            data["body"] = data["body"].str.replace(uc, "", regex=False)
        return data


class RemoveLeadingTrailingNL(AbstractPreprocessor):
    """Removes leading and trailing new lines."""

    @classmethod
    def apply(cls, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Remove new lines at the beginning and end of text.

        Parameters
        ----------
        raw_data : pd.DataFrame
            The data set with leading and trailing new lines in some texts.

        Returns
        -------
        pd.DataFrame
            The data set with leading and trailing new lines removed.
        """
        data = raw_data.copy(deep=True)
        data["body"] = data["body"].apply(
            lambda x: re.sub(r'^\n+|\n+$', '', x)
        )
        return data


class RemoveMultipleNL(AbstractPreprocessor):
    """Remove multiple new lines ."""

    @classmethod
    def apply(cls, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Remove multiple consecutive new lines.

        Parameters
        ----------
        raw_data : pd.DataFrame
            The data set with multiple new lines together.

        Returns
        -------
        pd.DataFrame
            The data set with multiple new lines together removed.
        """
        data = raw_data.copy(deep=True)
        data["body"] = data["body"].apply(
            lambda x: re.sub(r'\n{2,}', '\n', x)
        )
        return data


class RemoveMultiSpacing(AbstractPreprocessor):
    """Removes multi-spacing from email bodies."""

    @classmethod
    def apply(cls, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Remove multi-spacing.

        Parameters
        ----------
        raw_data : pd.DataFrame
            The data set with continuous spacing.

        Returns
        -------
        pd.DataFrame
            The data set with continuous spacing removed and converted into
            single spacing.
        """
        data = raw_data.copy(deep=True)
        data["body"] = data["body"].apply(
            lambda x: re.sub(r'\s+', ' ', x).strip()
        )
        return data


class RemoveHeadersNigerianFraud(AbstractPreprocessor):
    """Removes headers from the Nigerian Fraud emails."""

    prefixes = [
        "FROM",
        "TEL/FAX",
        "E-MAIL",
        "TEL",
        "Email",
        "ATTENTION",
        "CONFIDENTIAL TEL"
        "DATE",
        "PRIVATE TEL",
        "FAX",
        "ATTN",
        "CONFIDENTIAL FAX",
        "REPLY TO",
        "WEBSIDE",
        "WEBSITE",
        "MOBILE",
        "NAME",
        "PHONE",
        "ACCOUNTANT",
        "INTERNET FAX",
        "TO",
        "RE",
        "CC",
        "CONFIDENTIAL LINE",
        "CORPORATE LINE",
        "Present Direct Tel",
        "Direct Fax",
        "World Tel",
        "World Fax",
        "Private Tel/Fax",
        "Private E-mail",
        "DIRECT TEL",
        "SATELLITE TEL",
        "SATELLITE FAX"
    ]

    @classmethod
    def apply(cls, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Remove headers that don't are not part of the body.

        Parameters
        ----------
        raw_data : pd.DataFrame
            The data set with headers.

        Returns
        -------
        pd.DataFrame
            The data set with headers removed.
        """
        data = raw_data.copy(deep=True)

        if "source" not in data.columns:
            return data
        elif "Nigerian_Fraud" not in data["source"].unique():
            return data

        concat_prefixes = [
            re.escape(prefix.strip()) for prefix in cls.prefixes
        ]
        pattern = r'^\s*(?:' + '|'.join(concat_prefixes) + r').*$'

        nf_indices = data[data['source'] == "Nigerian_Fraud"].index.to_list()
        non_nf_df = data[~data.index.isin(nf_indices)]
        nf_df = data[data.index.isin(nf_indices)]

        nf_df["body"] = nf_df["body"].apply(
            lambda x: re.sub(
                pattern, '', x, flags=re.IGNORECASE | re.MULTILINE
            ).strip()
        )
        return pd.concat([non_nf_df, nf_df], axis=0).sort_index()


class RemoveLeadingGreaterThan(AbstractPreprocessor):
    """Removes leading greater than symbol in new lines."""

    @classmethod
    def apply(cls, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Remove leading greater than symbol in new lines.

        Parameters
        ----------
        raw_data : pd.DataFrame
            The data set with leading '>' symbols in email bodies.

        Returns
        -------
        pd.DataFrame
            The data set with no leading '>' symbols in email bodies.
        """
        data = raw_data.copy(deep=True)
        data["body"] = data["body"].apply(
            lambda x: re.sub(r'(?m)^(?:\s*>+\s*)+', '', x)
        )
        return data


class RemoveDuplicateTextNazario(AbstractPreprocessor):
    """Removes duplicate text in email bodies."""

    @classmethod
    def apply(cls, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate text in email bodies.

        Parameters
        ----------
        raw_data : pd.DataFrame
            The data set with duplicate information.

        Returns
        -------
        pd.DataFrame
            The data set with duplicate information removed.
        """
        data = raw_data.copy(deep=True)

        if "source" not in data.columns:
            return data
        elif "Nazario" not in data["source"].unique():
            return data

        data["body"] = data["body"].apply(cls._cut_duplicate_text)

        nazario_indices = data[data['source'] == "Nazario"].index.to_list()
        non_nazario_df = data[~data.index.isin(nazario_indices)]
        nazario_df = data[data.index.isin(nazario_indices)]

        nazario_df["body"] = nazario_df["body"].apply(cls._cut_duplicate_text)
        return pd.concat([non_nazario_df, nazario_df], axis=0).sort_index()

    @staticmethod
    def _normalize(txt):
        # Lowercase and remove all non-alphanumeric characters except spaces.
        return re.sub(r'\W+', '', txt.lower())

    @classmethod
    def _cut_duplicate_text(cls, txt: str) -> str:
        word_count = 10
        # Extract first 'word_count' words
        words = re.findall(r'\b\w+\b', txt)
        if len(words) < word_count:
            return txt  # Not enough words to check duplication

        # Build normalized snippet to look for
        snippet_words = words[:word_count]
        snippet_normalized = cls._normalize(''.join(snippet_words))

        # Normalize the entire text for matching
        full_normalized = cls._normalize(txt)

        # Find first and second occurrence of normalized snippet
        first = full_normalized.find(snippet_normalized)
        second = full_normalized.find(
            snippet_normalized, first + len(snippet_normalized)
        )

        if second == -1:
            return txt  # No duplication found

        # Now find the corresponding raw-text index for where the duplication
        # begins.
        compact_index = 0
        char_index = 0
        while compact_index < second and char_index < len(txt):
            if re.match(r'\w', txt[char_index]):
                compact_index += 1
            char_index += 1

        return txt[:char_index]  # Truncate at the start of the duplicate
