"""ESM-2 embedding tool for peptide sequences."""

from __future__ import annotations

import time
from typing import Any

import torch
from transformers import AutoModel, AutoTokenizer

from peptagent.tools.base import PeptideTool, ToolResult


class ESM2Embedder(PeptideTool):
    """Compute ESM-2 embeddings for peptide sequences.

    Returns per-residue embeddings (mean-pooled to sequence-level by default).
    Used both as a feature extractor for property prediction and as the
    embedding space for OOD detection in the reliability layer.
    """

    name = "esm2_embed"
    description = (
        "Compute protein language model embeddings for a peptide sequence. "
        "Returns a fixed-length vector representation useful for similarity "
        "search and property prediction."
    )

    def __init__(
        self,
        model_name: str = "facebook/esm2_t33_650M_UR50D",
        device: str = "cuda",
        pool: str = "mean",
    ) -> None:
        self.device = device if torch.cuda.is_available() else "cpu"
        self.pool = pool
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()

    @torch.no_grad()
    def run(self, sequence: str, **kwargs: Any) -> ToolResult:
        t0 = time.time()
        inputs = self.tokenizer(sequence, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        hidden = outputs.last_hidden_state  # (1, seq_len, hidden_dim)

        if self.pool == "mean":
            # Exclude BOS/EOS tokens
            embedding = hidden[0, 1:-1, :].mean(dim=0).cpu()
        elif self.pool == "cls":
            embedding = hidden[0, 0, :].cpu()
        else:
            embedding = hidden[0, 1:-1, :].cpu()

        return ToolResult(
            value=embedding,
            confidence=None,  # embeddings don't have confidence
            metadata={
                "model": self.model.config.name_or_path,
                "pool": self.pool,
                "embedding_dim": embedding.shape[-1],
                "elapsed_s": round(time.time() - t0, 3),
            },
        )

    @torch.no_grad()
    def batch_run(self, sequences: list[str], **kwargs: Any) -> list[ToolResult]:
        t0 = time.time()
        inputs = self.tokenizer(sequences, return_tensors="pt", padding=True).to(self.device)
        outputs = self.model(**inputs)
        hidden = outputs.last_hidden_state
        mask = inputs["attention_mask"]

        results = []
        for i in range(len(sequences)):
            seq_len = mask[i].sum().item()
            # Exclude BOS/EOS
            if self.pool == "mean":
                emb = hidden[i, 1 : seq_len - 1, :].mean(dim=0).cpu()
            else:
                emb = hidden[i, 0, :].cpu()
            results.append(
                ToolResult(
                    value=emb,
                    confidence=None,
                    metadata={
                        "model": self.model.config.name_or_path,
                        "pool": self.pool,
                        "embedding_dim": emb.shape[-1],
                    },
                )
            )

        elapsed = round(time.time() - t0, 3)
        for r in results:
            r.metadata["elapsed_s"] = elapsed
        return results
