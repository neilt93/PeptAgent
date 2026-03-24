"""Property prediction tools for peptide sequences.

Wraps models for hemolysis, solubility, toxicity, permeability, and stability.
Supports ensemble mode for reliability estimation.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from peptagent.tools.base import PeptideTool, ToolResult


class PropertyPredictor(PeptideTool):
    """Predict a single peptide property using a fine-tuned transformer.

    Each property (hemolysis, toxicity, etc.) is a separate instance with its
    own model checkpoint. When ensemble_size > 1, multiple checkpoints are loaded
    and predictions are averaged, with disagreement used as a confidence signal.
    """

    def __init__(
        self,
        property_name: str,
        model_paths: list[str],
        device: str = "cuda",
        threshold: float = 0.5,
    ) -> None:
        self.name = f"predict_{property_name}"
        self.description = (
            f"Predict {property_name} for a peptide sequence. "
            f"Returns probability (0-1) and binary classification."
        )
        self.property_name = property_name
        self.device = device if torch.cuda.is_available() else "cpu"
        self.threshold = threshold

        # Load ensemble of models
        self.models = []
        self.tokenizers = []
        for path in model_paths:
            tokenizer = AutoTokenizer.from_pretrained(path)
            model = AutoModelForSequenceClassification.from_pretrained(path)
            model = model.to(self.device).eval()
            self.models.append(model)
            self.tokenizers.append(tokenizer)

    @torch.no_grad()
    def run(self, sequence: str, **kwargs: Any) -> ToolResult:
        t0 = time.time()
        probabilities = []

        for model, tokenizer in zip(self.models, self.tokenizers):
            inputs = tokenizer(sequence, return_tensors="pt").to(self.device)
            logits = model(**inputs).logits
            prob = torch.softmax(logits, dim=-1)[0, 1].item()
            probabilities.append(prob)

        mean_prob = float(np.mean(probabilities))
        std_prob = float(np.std(probabilities)) if len(probabilities) > 1 else None

        # Confidence: high when ensemble agrees (low std), low when they disagree
        if std_prob is not None:
            confidence = max(0.0, 1.0 - 2.0 * std_prob)  # scale std to 0-1
        else:
            confidence = None

        return ToolResult(
            value={
                "probability": round(mean_prob, 4),
                "prediction": mean_prob >= self.threshold,
                "ensemble_std": round(std_prob, 4) if std_prob is not None else None,
            },
            confidence=confidence,
            metadata={
                "property": self.property_name,
                "ensemble_size": len(self.models),
                "threshold": self.threshold,
                "individual_probs": [round(p, 4) for p in probabilities],
                "elapsed_s": round(time.time() - t0, 3),
            },
        )


class PropertySuite(PeptideTool):
    """Run all property predictors on a sequence and return a combined profile.

    Convenience tool that the agent can call to get a full property assessment
    in a single tool call rather than calling each predictor individually.
    """

    name = "predict_all_properties"
    description = (
        "Predict all available properties (hemolysis, solubility, toxicity, "
        "permeability, stability) for a peptide sequence. Returns a property "
        "profile with predictions and confidence scores."
    )

    def __init__(self, predictors: list[PropertyPredictor]) -> None:
        self.predictors = {p.property_name: p for p in predictors}

    def run(self, sequence: str, **kwargs: Any) -> ToolResult:
        t0 = time.time()
        profile = {}
        confidences = []

        for prop_name, predictor in self.predictors.items():
            result = predictor.run(sequence)
            profile[prop_name] = result.value
            if result.confidence is not None:
                confidences.append(result.confidence)

        overall_confidence = float(np.mean(confidences)) if confidences else None

        return ToolResult(
            value=profile,
            confidence=overall_confidence,
            metadata={
                "properties_evaluated": list(profile.keys()),
                "elapsed_s": round(time.time() - t0, 3),
            },
        )
