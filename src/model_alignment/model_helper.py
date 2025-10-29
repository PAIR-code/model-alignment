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
import time
from typing import Optional, Union

from google import genai
from google.genai import types

import keras_nlp

MAX_NUM_RETRIES = 10


class ModelHelper(abc.ABC):
  """Class for managing calling LLMs."""

  def predict(
      self,
      prompt: str,
      temperature: float,
      stop_sequences: Optional[list[str]] = None,
      candidate_count: int = 1,
      max_output_tokens: Optional[int] = None,
  ) -> Union[list[str], str]:
    raise NotImplementedError()


class GeminiModelHelper(ModelHelper):
  """Gemini model calls."""

  def __init__(self, api_key, model_name='gemini-2.5-pro'):
    self.api_key = api_key
    self.model_name = model_name
    self.client = genai.Client(api_key=self.api_key)

  def predict(
      self,
      prompt: str,
      temperature: float,
      stop_sequences: Optional[list[str]] = None,
      candidate_count: int = 1,
      max_output_tokens: Optional[int] = None,
  ) -> Union[list[str], str]:
    num_attempts = 0
    response = None
    
    generation_config = types.GenerateContentConfig(
      candidate_count=candidate_count,
      max_output_tokens=max_output_tokens,
      temperature=temperature,
      stop_sequences=stop_sequences,
      safety_settings=[
        types.SafetySetting(
            category="HARM_CATEGORY_HARASSMENT",
            threshold="BLOCK_NONE",
        ),
        types.SafetySetting(
            category="HARM_CATEGORY_HATE_SPEECH",
            threshold="BLOCK_NONE",
        ),
        types.SafetySetting(
            category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
            threshold="BLOCK_NONE",
        ),
        types.SafetySetting(
            category="HARM_CATEGORY_DANGEROUS_CONTENT",
            threshold="BLOCK_NONE",
        ),
      ],
    )
    if max_output_tokens is not None:
      generation_config["max_output_tokens"] = max_output_tokens

    while num_attempts < MAX_NUM_RETRIES and response is None:
      try:
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=generation_config
        )
        num_attempts += 1
      except Exception as e:  # pylint: disable=broad-except
        if 'quota' in str(e):
          print('\033[31mQuota limit exceeded. Waiting to retry...\033[0m')
          time.sleep(2**num_attempts)

    if response is None:
      raise ValueError('Failed to generate content.')

    # Handle recitation policy blocks and other safety issues.
    if not response.candidates:
        if response.prompt_feedback.block_reason == 'SAFETY':
            print('\033[31mResponse blocked due to safety settings.\033[0m')
            return ''
        # The recitation policy is a specific type of safety block.
        if response.prompt_feedback.block_reason == 'RECITATION':
            print('\033[31mResponse blocked due to recitation policy.\033[0m')
            return ''
        return ''

    texts = []
    for candidate in response.candidates:
      if candidate.content and candidate.content.parts:
        texts.append(candidate.content.parts[0].text)

    if not texts:
      return ''

    if candidate_count == 1:
      return texts[0]
    else:
      return texts


class GemmaModelHelper(ModelHelper):
  """Gemma model calls through Keras."""

  def __init__(self, model_name='gemma2_instruct_2b_en'):
    self.model = keras_nlp.models.GemmaCausalLM.from_preset(model_name)

  def predict(
      self,
      prompt: str,
      temperature: float,
      stop_sequences: Optional[list[str]] = None,
      candidate_count: int = 1,
      max_output_tokens: Optional[int] = None,
  ) -> Union[list[str], str]:
    # Add control tokens to the prompt.
    prompt = ('<start_of_turn>user\n' + prompt +
              '<end_of_turn>\n<start_of_turn>model\n')
    responses = []
    for _ in range(candidate_count):
      response = self.model.generate(prompt, max_length=max_output_tokens)
      responses.append(response)

    if not responses:
      return ''

    # Remove input prompt and ending control tokens from the response.
    start_index = len(prompt)
    def process_response(response):
      ret = response[start_index:]
      ret = ret.split('<end_of_turn>')[0]
      if stop_sequences:
        for stop_sequence in stop_sequences:
          idx = ret.find(stop_sequence)
          if idx != -1:
            ret = ret[:idx]
      return ret.strip()

    responses = [process_response(r) for r in responses]

    if candidate_count == 1:
      return responses[0]
    else:
      return responses
