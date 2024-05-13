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
"""Utilities for labeler."""

import collections
import copy
import enum
import json
import random
from typing import Any, Dict, Optional, TypedDict


def generate_attribute_id() -> str:
  return str(random.randint(10000, 99999))


def truncate_from_middle(text: str, n_words: Optional[int]) -> str:
  if n_words is None:
    return text

  split = text.split(" ")
  if len(split) < n_words:
    return text

  midpoint_index = int(n_words // 2)
  truncated_text_l = split[:midpoint_index]
  truncated_text_r = split[-midpoint_index:]
  return f"{' '.join(truncated_text_l)}...\n...{' '.join(truncated_text_r)}"
