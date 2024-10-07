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
"""Helper methods for creating an alginable agent."""

import re
from typing import cast, Dict, Literal, Optional, Sequence, TypedDict
from model_alignment import model_helper
from model_alignment import prompts

LOWEST_TEMP = 0.1
LOW_TEMP = 0.25
MEDIUM_TEMP = 0.5

MutationType = Literal['add', 'edit', 'delete']
FeedbackType = Literal['kudos', 'critique', 'rewrite']

ModelHelper = model_helper.ModelHelper


class Mutation(TypedDict):
  text: str
  operation: MutationType
  principle_idx: int


class ConversationTurn():
  def __init__(self):
    self.is_user: bool = False
    self.text: str = ''
    self.vars: Dict[str, str] = {}


class ConstitutionalPrompt():
  def __init__(self):
    self.preamble: str = ''
    self.original_preamble: str = ''
    self.principles: list[str] = []
    self.conversations: list[list[ConversationTurn]] = []


class AlignableModelCalls():
  """Contains model calls for building an alignable agent."""

  def __init__(self, prompt_model: ModelHelper, alignment_model: ModelHelper):
    self.prompt_model = prompt_model
    self.alignment_model = alignment_model

  def _call_model(
      self,
      helper: ModelHelper,
      prompt: str,
      temperature: float,
      stop_sequences: Optional[list[str]] = None,
      candidate_count: int = 1,
  ):
    return helper.predict(
        prompt=prompt,
        temperature=temperature,
        stop_sequences=stop_sequences,
        candidate_count=candidate_count,
    )

  def candidate_to_turn(self, candidate: str) -> ConversationTurn:
    """Convert a candidate into a turn."""
    turn = ConversationTurn()
    turn.is_user = False
    turn.text = candidate
    return turn

  def rewrite_to_principle_single_run(
      self,
      last_user_input: ConversationTurn,
      last_model_output: str,
      preamble: str,
      rewrite: str,
  ) -> str:
    """Convert a rewritten response into a principle."""
    vars_string = self.get_vars_string(last_user_input)
    prompt = prompts.USER_REWRITE_TO_PRINCIPLE_SINGLE_RUN.format(
        instructions=preamble,
        vars=vars_string,
        result=last_model_output,
        rewrite=rewrite,
    )
    principle = cast(str, self._call_model(
        self.alignment_model, prompt=prompt, temperature=LOWEST_TEMP,
        stop_sequences=['}']
    ))
    return principle

  def critique_to_principle_single_run(
      self,
      last_user_input: ConversationTurn,
      last_model_output: str,
      preamble: str,
      critique: str,
  ) -> str:
    """Convert a critique into a principle."""
    vars_string = self.get_vars_string(last_user_input)
    prompt = prompts.CRITIQUE_TO_PRINCIPLE_SINGLE_RUN.format(
        instructions=preamble,
        vars=vars_string,
        result=last_model_output,
        critique=critique,
    )
    principle = cast(str, self._call_model(
        self.alignment_model, prompt=prompt, temperature=LOWEST_TEMP,
        stop_sequences=['}']
    ))
    # Strip out the prefix if it exists in the model response.
    principle = principle.removeprefix('INSTRUCTION_ADDITION: {')
    return principle

  def kudos_to_principle_single_run(
      self,
      last_user_input: ConversationTurn,
      last_model_output: str,
      preamble: str,
      kudos: str,
  ) -> str:
    """Convert a kudos into a principle."""
    vars_string = self.get_vars_string(last_user_input)
    prompt = prompts.KUDOS_TO_PRINCIPLE_SINGLE_RUN.format(
        instructions=preamble,
        vars=vars_string,
        result=last_model_output,
        kudos=kudos,
    )
    principle = cast(str, self._call_model(
        self.alignment_model, prompt=prompt, temperature=LOWEST_TEMP,
        stop_sequences=['}']
    ))
    return principle

  def single_run(
      self,
      single_run_prompt: str,
      generate_multiple_candidates: bool = False,
  ) -> Sequence[ConversationTurn] | ConversationTurn:
    """Run a single-run prompt based on the input text."""
    num_candidates = 3 if generate_multiple_candidates else 1
    result = self._call_model(
        self.prompt_model,
        prompt=single_run_prompt,
        temperature=MEDIUM_TEMP,
        candidate_count=num_candidates,
    )
    result = [result] if num_candidates == 1 else result

    turns = [
        self.candidate_to_turn(cand)
        for cand in result
    ]
    return (
        turns[0]
        if (not generate_multiple_candidates and len(turns) == 1)
        else turns
    )

  def get_vars_string(self, turn: ConversationTurn) -> str:
    """Generate a string of variable definitions."""
    if turn.vars:
      vars_list = []
      for var_name in turn.vars:
        var_value = turn.vars[var_name]
        vars_list.append(f'{var_name}: {{ {var_value} }}')
      vars_string = '. '.join(vars_list)
      return vars_string
    else:
      return ''

  def generate_critiques_single_run(
      self, preamble: str, convo: Sequence[ConversationTurn]
  ) -> list[str]:
    """Generate possible critiques for a single-run prompt's output."""
    vars_string = self.get_vars_string(convo[0])
    prompt = prompts.EXEMPLAR_TO_NEGATIVE_RATIONALES_SINGLE_RUN.format(
        instructions=preamble, vars=vars_string, result=convo[1].text
    )
    model_response = cast(str, self._call_model(
        self.alignment_model,
        prompt=prompt,
        temperature=LOWEST_TEMP,
        stop_sequences=['CONVERSATION_CONTEXT:'],
    ))
    rationales = model_response.split('}')
    rationales_parsed = []
    for rationale in rationales:
      rationale_split = rationale.split('{')
      rationale_parsed = (
          rationale_split[1] if len(rationale_split) > 1 else rationale_split[0]
      )
      if rationale_parsed:
        rationales_parsed.append(rationale_parsed)
    return rationales_parsed

  def generate_kudos_single_run(
      self, preamble: str, convo: Sequence[ConversationTurn]
  ) -> list[str]:
    """Generate possible kudos for a single-run prompt's output."""
    vars_string = self.get_vars_string(convo[0])
    prompt = prompts.EXEMPLAR_TO_POSITIVE_RATIONALES_SINGLE_RUN.format(
        instructions=preamble, vars=vars_string, result=convo[1].text
    )
    model_response = cast(str, self._call_model(
        self.alignment_model,
        prompt=prompt,
        temperature=LOWEST_TEMP,
        stop_sequences=['CONVERSATION_CONTEXT:'],
    ))
    rationales = model_response.split('}')
    rationales_parsed = []
    for rationale in rationales:
      rationale_split = rationale.split('{')
      rationale_parsed = (
          rationale_split[1] if len(rationale_split) > 1 else rationale_split[0]
      )
      if rationale_parsed:
        rationales_parsed.append(rationale_parsed)
    return rationales_parsed

  def edit_single_run_model_description(
      self, model_description: str, princple: str
  ) -> str:
    """Revises a model description given a new principle."""
    prompt = prompts.EDIT_SINGLE_RUN_MODEL_DESCRIPTION.format(
        model_description=model_description, principle=princple
    )
    response = cast(
        str,
        self._call_model(
            self.alignment_model,
            prompt=prompt,
            temperature=LOW_TEMP,
            stop_sequences=['<END_PROMPT>']))
    response_split = response.split('<BEGIN_PROMPT>')
    result = (
        response_split[1] if len(response_split) > 1 else response_split[0]
    )
    return result

  def create_mutation_choices(
      self, principles: Sequence[str]) -> list[Mutation]:
    """Create mutation choices for responding to feedback."""
    add_mutation: Mutation = {
        'text': f'1: {prompts.ADD_NEW_PRINCIPLE_MUTATION}',
        'operation': 'add',
        'principle_idx': -1}
    mutations = [add_mutation]
    for i, principle in enumerate(principles):
      delete_str = prompts.DELETE_PRINCIPLE_MUTATION.format(
          principle=principle)
      edit_str = prompts.EDIT_PRINCIPLE_MUTATION.format(
          principle=principle)
      mutations.append({
          'text': f"""{len(mutations) + 1}: {delete_str}""",
          'operation': 'delete',
          'principle_idx': i})
      mutations.append({
          'text': f"""{len(mutations) + 1}: {edit_str}""",
          'operation': 'edit',
          'principle_idx': i})
    return mutations

  def create_mutation_decision_prompt_single_run(
      self, principles: Sequence[str],
      convo: Sequence[ConversationTurn], model_description: str, feedback: str,
      feedback_type: FeedbackType) -> tuple[str, list[Mutation]]:
    """Create possible mutations and the corresponding prompt string."""
    mutations = self.create_mutation_choices(principles)
    mutations_str = '\n'.join([mutation['text'] for mutation in mutations])
    prompt_to_fill_out = (prompts.MUTATION_CHOICE_SINGLE_RUN_REWRITE
                          if feedback_type == 'rewrite'
                          else prompts.MUTATION_CHOICE_SINGLE_RUN_FEEDBACK)
    prompt = prompt_to_fill_out.format(
        model_description=model_description,
        input_text=convo[0].text,
        response=convo[1].text,
        feedback=feedback,
        mutations=mutations_str)
    return prompt, mutations

  def perform_mutation_single_run(
      self, principles: list[str],
      convo: Sequence[ConversationTurn], model_description: str, feedback: str,
      feedback_type: FeedbackType, mutation: Mutation) -> Sequence[str]:
    """Perform the appropriate mutation to the principles."""
    if mutation['operation'] == 'add':
      if feedback_type == 'kudos':
        principles.append(self.kudos_to_principle_single_run(
            convo[0], convo[1].text, model_description, feedback))
      elif feedback_type == 'critique':
        principles.append(self.critique_to_principle_single_run(
            convo[0], convo[1].text, model_description, feedback))
      else:
        principles.append(self.rewrite_to_principle_single_run(
            convo[0], convo[1].text, model_description, feedback))
    elif mutation['operation'] == 'delete':
      principles.pop(mutation['principle_idx'])
    else:
      prompt_to_fill_out = (prompts.SINGLE_RUN_REWRITE_PRINCIPLE_REWRITE
                            if feedback_type == 'rewrite'
                            else prompts.SINGLE_RUN_REWRITE_PRINCIPLE_FEEDBACK)
      prompt = prompt_to_fill_out.format(
          model_description=model_description,
          input_text=convo[0].text,
          response=convo[1].text,
          feedback=feedback,
          principle=principles[mutation['principle_idx']])
      response = cast(str, self._call_model(
          self.alignment_model, prompt, LOWEST_TEMP))
      end_tag_idx = response.find(prompts.END_PRINCIPLE_TAG)
      if end_tag_idx != -1:
        response = response[:end_tag_idx]
      principles[mutation['principle_idx']] = response
    return principles

  def run_mutation_single_run(
      self, principles: Sequence[str],
      convo: Sequence[ConversationTurn], model_description: str, feedback: str,
      feedback_type: FeedbackType) -> Sequence[str]:
    """Determine and make the appropriate principle update based on feedback."""
    # Determine the right principle mutation to make (adding/editing/deleting).
    principles = list(principles)
    prompt, mutations = self.create_mutation_decision_prompt_single_run(
        principles, convo, model_description, feedback, feedback_type)
    response = self._call_model(self.alignment_model, prompt, LOWEST_TEMP)

    # Subtract one from the response number as the mutations list is 0-based
    # but the prompt used is 1-based for better LLM performance.
    re_result = re.search(r'\d+', response)
    if re_result:
      response_idx = int(re_result.group())
    else:
      response_idx = 1
    mutation = mutations[response_idx - 1]

    # Perform the appropriate mutation.
    return self.perform_mutation_single_run(
        principles, convo, model_description, feedback, feedback_type, mutation)

  def update_model_description_from_feedback_single_run(
      self, model_description: str, feedback: str) -> str:
    """Update the model description directly based on feedback."""
    prompt = prompts.MASTER_UPDATE_PROMPT.format(
        model_prompt=model_description, feedback=feedback)
    response = cast(
        str, self._call_model(self.alignment_model, prompt, LOW_TEMP))
    # Strip out any <PROMPT> or <REVISED PROMPT> opening and closing tags.
    split = re.split('<(?:REVISED )?PROMPT>', response)
    if len(split) > 1:
      response = split[1]
    split = re.split('</(?:REVISED )?PROMPT>', response)
    if len(split) > 1:
      response = split[0]
    return response.strip()
