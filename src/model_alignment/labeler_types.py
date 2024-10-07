#  Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Labeler types."""

import re
from typing import Dict, Literal, Optional, Sequence, TypedDict, cast


class TrainingHyperparameters(TypedDict):
  """Hperparameters for training"""

  # Max number of examples to train on.
  n_incorrect_predictions_to_sample: int
  # How many mutations to evaluate.
  n_mutations_to_try: int
  # Number of prompts to return in between training rounds.
  n_best: int
  # 'accuracy' | 'f1_binary' | 'f1' (average fscore across labels for which
  #   there is at least one example)
  score_fn: str
  # If score_fn is f1_binary, which label f1 should be evaluated.
  pos_label: Optional[str]
  # Max number of examples to run evaluation on.
  n_validation_examples: int
  # Number of tokens to truncate input features to.
  truncation_length: int


class Mutation(TypedDict):
  """A mutation to be performed on a checkpoint."""

  # Unique identifier
  key: str
  # Type of action to be performed
  action: str
  # Label that the mutation would be performed on
  label_value: str
  # Attribute that the mutation would be performed on
  attribute_id: Optional[str]
  # Description of the mutation to be performed
  prompt: str


class Scorecard(TypedDict):
  """A set of metrics to characterize labeler performance."""
  precision: list[float]
  recall: list[float]
  fscore: list[float]
  support: list[int]
  n_empty: int
  accuracy: float


class PredictionResult(TypedDict):
  """Wrapper for the result of calling inference."""

  request: str
  raw_prediction: str
  prediction: str


class LabelerAttribute(TypedDict):
  id: str
  value: str
  label: str


class LabelerCheckpoint:

  def __init__(self):
    self.attributes: list[LabelerAttribute] = []
