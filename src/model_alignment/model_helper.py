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
"""Helper classes for calling various LLMs."""

import abc
from typing import Optional, Union

import google.generativeai as genai


class ModelHelper(abc.ABC):
  """Class for managing calling LLMs."""

  def predict(
      self,
      prompt: str,
      temperature: float,
      stop_sequences: Optional[list[str]] = None,
      candidate_count: int = 1,
  ) -> Union[list[str], str]:
    raise NotImplementedError()


class GeminiModelHelper(ModelHelper):
  """Gemini model calls."""

  def __init__(self, api_key):
    genai.configure(api_key=api_key)
    self.model = genai.GenerativeModel('gemini-pro')

  def predict(
      self,
      prompt: str,
      temperature: float,
      stop_sequences: Optional[list[str]] = None,
      candidate_count: int = 1,
  ) -> Union[list[str], str]:
    response = self.model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            candidate_count=candidate_count,
            stop_sequences=stop_sequences,
            temperature=temperature,
        ),
        safety_settings={
            'HARASSMENT': 'block_none',
            'SEXUAL': 'block_none',
            'HATE_SPEECH': 'block_none',
            'DANGEROUS': 'block_none',
        },
    )
    if candidate_count == 1:
      return response.text
    else:
      return [candidate.parts[0].text for candidate in response.candidates]
