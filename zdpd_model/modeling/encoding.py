import numpy as np
import torch

from abc import ABC, abstractmethod
from transformers import BertTokenizer, BertModel


class BERTEncoder(ABC):
    """Abstract class for encoders."""

    def __init__(self):
        """Initialize attributes."""
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased').eval().to(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

    @abstractmethod
    def encode(self, text: str, **kwargs) -> np.array:
        """Encode the text into an array.

        Parameters
        ----------
        text : str
            The text to encode.

        Returns
        -------
        np.array
            The encoded text as an array of shape (768,).
        """
        raise NotImplementedError("Encoder requires 'encode' method.")


class CLSEncoder(BERTEncoder):
    """CLS BERT encoder class."""

    def encode(
        self, text: str, window_size=256, stride=128, max_length=512, **kwargs
    ) -> np.array:
        """Encode the text using [CLS] tokenization method.

        Parameters
        ----------
        text : str
            The text to encode.
        window_size : int, optional
            The size of the sliding window to average tokens of texts longer
            than 512 tokens, by default 256
        stride : int, optional
            The size of stride used to create the windows, by default 128
        max_length : int, optional
            The maximum tokens allowed for direct encoding, by default 512

        Returns
        -------
        np.array
            The encoded text as an array of shape (768,).
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=False)
        tokens = inputs["input_ids"][0].to(self.model.device)

        # Short text (single pass)
        if len(tokens) <= max_length:
            with torch.no_grad():
                # Forward pass
                outputs = self.model(tokens.unsqueeze(0))
                # Extract [CLS] token
                cls_embedding = outputs.last_hidden_state[:, 0, :]
                return cls_embedding.cpu().numpy()[0]  # Convert to NumPy

        # Long text (sliding window)
        cls_embeddings = []
        for i in range(0, len(tokens), stride):
            chunk = tokens[i:i + window_size]
            if len(chunk) > 0:
                with torch.no_grad():
                    # Forward pass
                    outputs = self.model(chunk.unsqueeze(0))
                    # Extract [CLS]
                    cls_embedding = outputs.last_hidden_state[:, 0, :]
                    cls_embeddings.append(cls_embedding)

        return (
            torch.mean(
                torch.stack(cls_embeddings), dim=0
            ).cpu().numpy().ravel()
            if cls_embeddings
            else np.zeros(768)
        )


class MeanPoolingEncoder(BERTEncoder):
    """Mean pooling BERT encoder class."""

    def __init__(self):
        """Initialize attributes."""
        super().__init__()
        self.device = next(self.model.parameters()).device

    def encode(
        self, text: str, window_size=256, stride=128, max_length=512, **kwargs
    ) -> np.array:
        """Encode the text using mean pooling methodology.

        Parameters
        ----------
        text : str
            The text to encode.
        window_size : int, optional
            The size of the sliding window to average tokens of texts longer
            than 512 tokens, by default 256
        stride : int, optional
            The size of stride used to create the windows, by default 128
        max_length : int, optional
            The maximum tokens allowed for direct encoding, by default 512

        Returns
        -------
        np.array
            The encoded text as an array of shape (768,).
        """
        # Tokenize without special tokens
        tokens = self.tokenizer.encode(text, add_special_tokens=False)

        # Short text processing
        if len(tokens) <= max_length:
            inputs = {
                'input_ids': torch.tensor([tokens]).to(self.device),
                'attention_mask': torch.ones(
                    (1, len(tokens)), dtype=torch.long
                ).to(self.device)
            }

            with torch.no_grad():
                outputs = self.model(**inputs)
                return torch.mean(
                    outputs.last_hidden_state[0], dim=0
                ).cpu().numpy()

        # Sliding window for long texts
        window_embeds = []
        for i in range(0, len(tokens), stride):
            chunk = tokens[i:i + window_size]
            window_input = {
                'input_ids': torch.tensor([chunk]).to(self.device),
                'attention_mask': torch.ones(
                    (1, len(chunk)), dtype=torch.long
                ).to(self.device)
            }

            with torch.no_grad():
                outputs = self.model(**window_input)
                window_embeds.append(
                    torch.mean(outputs.last_hidden_state[0], dim=0)
                )

        return torch.mean(torch.stack(window_embeds), dim=0).cpu().numpy()
