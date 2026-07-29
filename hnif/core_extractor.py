import torch
import numpy as np
from typing import Dict, Any, Tuple


class TensorExtractor:
    """
    Handles the intrinsic extraction of neural tensors from
    Transformer-based models (DeBERTa, RoBERTa, etc.)
    """

    def __init__(self, model: torch.nn.Module, tokenizer: Any):
        self.model = model
        self.tokenizer = tokenizer

        # must ensure the model outputs attention weights
        if not self.model.config.output_attentions:
            print(
                "⚠️ Warning: Model config 'output_attentions' was False. Forcing to True for HNIF extraction."
            )
            self.model.config.output_attentions = True

        self.device = next(self.model.parameters()).device

    def extract_attention(self, text: str) -> Tuple[np.ndarray, list]:
        """
        Runs a single forward pass and extracts the final layer's [CLS] attention.

        Args:
            text (str): The input sentence to analyze.

        Returns:
            Tuple[np.ndarray, list]: The attention weights array and the token list.
        """
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # HuggingFace standard: outputs.attentions is a tuple of all layers
        # We want the last layer [-1]
        last_layer_attention = outputs.attentions[-1]

        # Shape is usually (batch_size, num_heads, sequence_length, sequence_length)
        # We average across all attention heads (dim=1) for the [CLS] token (index 0)
        # Resulting shape: (sequence_length,)
        cls_attention = last_layer_attention[0].mean(dim=0)[0].cpu().numpy()

        # Get the actual string tokens
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

        return cls_attention, tokens
