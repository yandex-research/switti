import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer


class FrozenCLIPEmbedder(nn.Module):
    """Uses the CLIP transformer encoder for text (from huggingface)"""

    def __init__(
        self,
        version="openai/clip-vit-large-patch14",
        device="cuda",
        max_length=77,
        freeze=True,
    ):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version).to(device)
        self.device = device
        self.hidden_size = self.transformer.config.hidden_size
        self.max_length = max_length
        if freeze:
            self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        ).to(self.device)

        outputs = self.transformer(**batch_encoding)

        attn_bias = batch_encoding["attention_mask"].to(outputs["last_hidden_state"].dtype)
        attn_bias[attn_bias == 0] = -float("inf")
        attn_bias[attn_bias == 1] = 0.0
        outputs["attn_bias"] = attn_bias
        return outputs

    @torch.no_grad()
    def encode(self, text):
        return self(text)
