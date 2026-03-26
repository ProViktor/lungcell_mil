from typing import Optional, Literal
from pydantic import BaseModel, Field, model_validator
import json


class RunParams(BaseModel):
    aggregator: Literal[
        "MeanAggergation",
        "MaxAggergation",
        "AttentionAggregation",
        "GatedAttentionAggregation",
    ]
    n_hidden: int = Field(..., gt=0)
    hidden_size: int = Field(..., gt=0)
    encoding_size: int = Field(..., gt=0)
    seed: int
    lr: float = Field(..., gt=0)
    decay: float = Field(..., ge=0)
    attention_hidden_size: Optional[int] = Field(None, gt=0)
    sparse: bool = True
    num_epochs: int = 30

    @model_validator(mode="after")
    def check_attention_hidden_size(self):
        if (
            self.aggregator in ["AttentionAggregation", "GatedAttentionAggregation"]
            and self.attention_hidden_size is None
        ):
            raise ValueError(f"attention_hidden_size is required for {self.aggregator}")
        return self

    def save_json(self, path: str):
        with open(path, "w") as f:
            f.write(self.model_dump_json(indent=4))
