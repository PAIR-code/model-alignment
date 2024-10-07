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
"""For training classification prompts."""

import copy
import json
import random
import re
from typing import Any, Dict, Optional, Tuple
import jinja2
from model_alignment import alignable_model_calls
from model_alignment import labeler_prompt_templates
from model_alignment import labeler_types
from model_alignment import model_helper
from model_alignment import utils
import numpy as np
import pandas as pd
from sklearn import metrics
import tqdm

MAX_EVAL_OUTPUT_TOKENS = 8
EVAL_TEMPERATURE = 0.1
N_VALIDATION_EXAMPLES = 100
N_INCORRECT_PREDICTIONS_TO_SAMPLE = 3
N_MUTATIONS_TO_TRY = 2
N_BEST = 3
SCORE_FN = 'accuracy'
TRAIN_TEMPERATURE = 0.9
TRUNCATION_LENGTH = 100

GeminiModelHelper = model_helper.GeminiModelHelper
ConstitutionalPrompt = alignable_model_calls.ConstitutionalPrompt
ConversationTurn = alignable_model_calls.ConversationTurn
ModelHelper = model_helper.ModelHelper
AlignableModelCalls = alignable_model_calls.AlignableModelCalls

LabelerCheckpoint = labeler_types.LabelerCheckpoint
LabelerAttribute = labeler_types.LabelerAttribute
PredictionResult = labeler_types.PredictionResult
Mutation = labeler_types.Mutation
Scorecard = labeler_types.Scorecard
TrainingHyperparameters = labeler_types.TrainingHyperparameters

ExampleWithPrediction = Tuple[pd.Series, PredictionResult]
CheckpointWithScore = Tuple[LabelerCheckpoint, float]

DEFAULT_HPARAMS: TrainingHyperparameters = {
    'n_incorrect_predictions_to_sample': N_INCORRECT_PREDICTIONS_TO_SAMPLE,
    'n_mutations_to_try': N_MUTATIONS_TO_TRY,
    'n_best': N_BEST,
    'score_fn': SCORE_FN,
    'pos_label': None,
    'n_validation_examples': N_VALIDATION_EXAMPLES,
    'truncation_length': TRUNCATION_LENGTH,
}


def delete_attribute_from_checkpoint(
    checkpoint: LabelerCheckpoint,
    attribute_id_to_delete: str,
) -> LabelerCheckpoint:
  """Deletes an attribute from the checkpoint."""
  attributes = checkpoint.attributes
  checkpoint.attributes = [
      item for item in attributes if item['id'] != attribute_id_to_delete
  ]
  return checkpoint


def get_attributes_for_label(
    checkpoint: LabelerCheckpoint, label: str
) -> list[LabelerAttribute]:
  if not label:
    return list(checkpoint.attributes)
  attributes = []
  for attr in checkpoint.attributes:
    if attr['label'] == label:
      attributes.append(attr)
  return attributes


def get_attribute_for_id(
    checkpoint: LabelerCheckpoint, attribute_id: str
) -> Optional[LabelerAttribute]:
  for attr in checkpoint.attributes:
    if attr['id'] == attribute_id:
      return attr
  return None


def print_prediction(
    prediction, input_names, label_name, truncation_length=1000
):
  """Prints the prediction in a readable format."""
  print('\n====================')
  for input_feature in input_names:
    print(
        f'{input_feature}:'
        f' {utils.truncate_from_middle(prediction[0][input_feature], truncation_length)}'
    )
  print(f'LLM request: {prediction[1]["request"]}')
  print(f'raw prediction: {prediction[1]["raw_prediction"]}')
  print(f'prediction: {prediction[1]["prediction"]}')
  if label_name in prediction[0]:
    print(f'label: {prediction[0][label_name]}')


def print_checkpoint(checkpoint: LabelerCheckpoint):
  """Prints the checkpoint in a readable format."""
  labels = {}
  for attribute in checkpoint.attributes:
    if attribute['label'] not in labels:
      labels[attribute['label']] = []
    labels[attribute['label']].append(attribute)

  for label in labels:
    print('\n====================')
    print(f'Attributes for: \033[1m{label}\033[0m')
    print('====================\n')
    attributes = labels[label]
    for attribute in attributes:
      print(f'{attribute["id"]}: {attribute["value"]}')


def get_checkpoint_templating_params(
    checkpoint: LabelerCheckpoint,
    shuffle_classes: bool = True,
) -> Dict[str, Any]:
  """Gets the templating parameters for the checkpoint."""
  classes_dict = {}
  for attribute in checkpoint.attributes:
    if attribute['label'] not in classes_dict:
      classes_dict[attribute['label']] = ''
    classes_dict[attribute['label']] += f'{attribute["value"]} '

  classes = list(classes_dict.items())
  if shuffle_classes:
    random.shuffle(classes)
  return {
      'classes': [{'id': key, 'description': value} for key, value in classes]
  }


class Labeler:
  """A class for training text prompts on data."""

  def __init__(
      self,
      input_names: list[str],
      label_name: str,
      label_values: list[str],
      task_description: str,
      train_model_helper: ModelHelper,
      eval_model_helper: ModelHelper,
      max_eval_output_tokens: Optional[int] = MAX_EVAL_OUTPUT_TOKENS,
      eval_temperature: Optional[float] = EVAL_TEMPERATURE,
      hparams: TrainingHyperparameters = DEFAULT_HPARAMS,
      debug_mode: bool = False,
  ):
    """Constructor.

    Args:
      input_names: a list of input features (e.g. 'comment')
      label_name: the output feature to be labeled (e.g. 'isToxic')
      label_values: the values the output feature can assume (e.g. 'yes', 'no')
      task_description: a one-sentence description of the task, used to seed the
        classifier (e.g. 'Does the example contain toxic speech?')
      train_model_helper: LLM model helper for training (can use larger model
        here)
      eval_model_helper: LLM model helper for eval (the model that will
        ultimately be using the labeler prompt)
      max_eval_output_tokens: max number of tokens to output when predicting
      eval_temperature: temperature when calling LLM for eval
      hparams: Training hyperparameters.
      debug_mode: Whether to print intermediate results
    """
    self.train_model_helper = train_model_helper
    self.eval_model_helper = eval_model_helper

    self.input_names = [str(name) for name in input_names]
    self.label_name = str(label_name)
    self.label_values = [str(val) for val in label_values]
    self.task_description = task_description

    self.max_eval_output_tokens = max_eval_output_tokens
    self.eval_temperature = eval_temperature

    self.hparams = hparams
    self.debug_mode = debug_mode

    self.classifier_prompt_stop_sequences = [' ', '\n']

    self.classifier_prompt = jinja2.Template(
        labeler_prompt_templates.CLASSIFIER_TEXT_DIRECT_LABEL
    )
    self.initial_label_attribute_prompt = jinja2.Template(
        labeler_prompt_templates.INITIAL_LABEL_ATTRIBUTE_TEXT
    )
    self.get_feedback_prompt = jinja2.Template(
        labeler_prompt_templates.GET_FEEDBACK_TEXT
    )
    self.identify_mutation_prompt = jinja2.Template(
        labeler_prompt_templates.IDENTIFY_MUTATION_TEXT
    )
    self.get_edit_instructions_prompt = jinja2.Template(
        labeler_prompt_templates.GET_EDIT_INSTRUCTIONS_TEXT
    )
    self.execute_edit_mutation_prompt = jinja2.Template(
        labeler_prompt_templates.EXECUTE_EDIT_MUTATION_TEXT
    )
    self.execute_add_mutation_prompt = jinja2.Template(
        labeler_prompt_templates.EXECUTE_ADD_MUTATION_TEXT
    )

  def initialize_checkpoint(self, train_examples: pd.DataFrame):
    """Creates initial constitution, given few-shot examples."""
    checkpoint = LabelerCheckpoint()
    exemplars = list(labeler_prompt_templates.INITIAL_LABEL_ATTRIBUTE_EXEMPLARS)
    for value in self.label_values:
      train_examples_for_value = train_examples[
          train_examples[self.label_name].astype(str) == value
      ]
      n_matches = train_examples_for_value.shape[0]
      if n_matches == 0:
        raise ValueError(f'No examples found where {self.label_name}={value}')

      train_example_records = (
          train_examples_for_value[self.input_names]
          .sample(n=min(2, n_matches))
          .to_dict('records')
      )
      params = {
          **self.get_global_templating_inputs(),
          'exemplars': exemplars,
          'input_list': json.dumps(train_example_records),
          'label_value': value,
      }
      templated = self.initial_label_attribute_prompt.render(params)
      attribute: str = self.train_model_helper.predict(  # pytype: disable=annotation-type-mismatch
          templated,
          temperature=TRAIN_TEMPERATURE,
          stop_sequences=['\n\n', 'input:', 'inputs:'],
          candidate_count=1,
      )
      exemplars.append({
          'label': self.label_name,
          'label_value': value,
          'input_list': json.dumps(train_example_records),
          'criteria': attribute,
      })

      checkpoint.attributes.append(
          LabelerAttribute(
              id=utils.generate_attribute_id(),
              value=attribute,
              label=value,
          )
      )
    return checkpoint

  def train_on_incorrect_predictions(
      self,
      checkpoint: LabelerCheckpoint,
      incorrect_predictions: list[Tuple[pd.Series, PredictionResult]],
      n_mutations_to_try: int,
  ) -> list[LabelerCheckpoint]:
    """Trains on an incorrect prediction.

    Args:
      checkpoint: the checkpoint.
      incorrect_predictions: incorrect predictions.
      n_mutations_to_try: the number of mutations to evaluate.

    Returns:
      a list of improved checkpoints, or the original checkpoint if none of the
      mutation candidates outperform it.
    """
    feedback = self.get_feedback_on_incorrect_prediction(
        checkpoint, incorrect_predictions[0]
    )
    mutations = self.get_mutations(
        checkpoint, incorrect_predictions, feedback, n_mutations_to_try
    )
    candidates = []
    for mutation in mutations:
      if mutation['action'] == 'DELETE':
        executed = ''
      else:
        executed = self.execute_mutation(
            checkpoint, incorrect_predictions, feedback, mutation
        )
      evolved = self.evolve_checkpoint(checkpoint, mutation, executed)
      if self.debug_mode:
        print('Mutation candidate:')
        print_checkpoint(evolved)
      candidates.append(evolved)
    return candidates

  def execute_mutation(
      self,
      checkpoint: LabelerCheckpoint,
      incorrect_predictions: list[ExampleWithPrediction],
      feedback: str,
      mutation: Mutation,
  ) -> str:
    """Perform the requested mutation."""
    runner = self.execute_edit_mutation_prompt
    predictions = self.get_incorrect_predictions_templating_input(
        incorrect_predictions
    )
    inputs = {
        **self.get_global_templating_inputs(),
        **self.get_rulebook_templating_inputs(checkpoint),
        'incorrect_predictions': predictions,
        'feedback': feedback,
        'to_edit_label_value': mutation['label_value'],
        'golden_label': predictions[0]['golden_label'],
    }
    if mutation['action'] == 'ADD':
      runner = self.execute_add_mutation_prompt
      inputs = {
          **inputs,
          'existing_attributes': get_attributes_for_label(
              checkpoint, mutation['label_value']
          ),
          'new_attribute_id': utils.generate_attribute_id(),
      }
    elif mutation['action'] == 'EDIT':
      edit_instructions_inputs = {
          **inputs,
          'attribute_to_edit': get_attribute_for_id(
              checkpoint, mutation['attribute_id']
          ),
      }
      templated = self.get_edit_instructions_prompt.render(
          edit_instructions_inputs
      )
      edit_instructions: str = self.train_model_helper.predict(  # pytype: disable=annotation-type-mismatch
          templated,
          temperature=TRAIN_TEMPERATURE,
          stop_sequences=['END_ANSWER', '\n\n'],
          candidate_count=1,
          max_output_tokens=256,
      )
      inputs = {
          **inputs,
          'edit_instructions': edit_instructions,
          'attribute_to_edit': get_attribute_for_id(
              checkpoint, mutation['attribute_id']
          ),
      }
    templated = runner.render(inputs)
    prediction: str = self.train_model_helper.predict(  # pytype: disable=annotation-type-mismatch
        templated,
        temperature=TRAIN_TEMPERATURE,
        stop_sequences=['END_ANSWER', '\n\n'],
        candidate_count=1,
        max_output_tokens=256,
    )
    return prediction

  def get_feedback_on_incorrect_prediction(
      self,
      checkpoint: LabelerCheckpoint,
      incorrect_prediction: Tuple[pd.Series, PredictionResult],
  ) -> str:
    """Get explanation for why prediction was wrong."""
    feedback_inputs = {
        **self.get_global_templating_inputs(),
        **self.get_rulebook_templating_inputs(checkpoint),
        'incorrect_predictions': (
            self.get_incorrect_predictions_templating_input(
                [incorrect_prediction]
            )
        ),
        'golden_label': incorrect_prediction[0][self.label_name],
    }
    templated = self.get_feedback_prompt.render(feedback_inputs)
    feedback: str = self.train_model_helper.predict(  # pytype: disable=annotation-type-mismatch
        templated,
        temperature=TRAIN_TEMPERATURE,
        stop_sequences=['END_ANSWER', '\n\n'],
        candidate_count=1,
        max_output_tokens=256,
    )
    return feedback

  def evolve_checkpoint(
      self,
      checkpoint: LabelerCheckpoint,
      mutation: Mutation,
      attribute_value: str,
  ) -> LabelerCheckpoint:
    """Given a checkpoint and a realized mutation, return an evolved checkpoint."""
    evolved = LabelerCheckpoint()
    evolved.attributes = copy.deepcopy(checkpoint.attributes)

    if mutation['action'] == 'ADD':
      evolved.attributes.append(
          LabelerAttribute(
              id=utils.generate_attribute_id(),
              value=attribute_value,
              label=mutation['label_value'],
          )
      )
    elif mutation['action'] == 'EDIT':
      for attr in evolved.attributes:
        if attr['id'] == mutation['attribute_id']:
          attr['value'] = attribute_value
    elif mutation['action'] == 'DELETE':
      delete_attribute_from_checkpoint(evolved, mutation['attribute_id'])

    return evolved

  def get_score(
      self,
      labeled_examples: pd.DataFrame,
      predictions: list[PredictionResult],
  ) -> float:
    """Scores the predictions.

    Args:
      labeled_examples: examples with golden labels dataframe
      predictions: a list of PredictionResult wrappers

    Returns:
      Score.
    """
    scorecard = self.get_scorecard(labeled_examples, predictions)

    score_fn = self.hparams['score_fn']
    if score_fn == 'accuracy':
      return scorecard['accuracy']

    if score_fn == 'f1_binary':
      pos_label = self.hparams['pos_label']
      if pos_label is None:
        pos_label = self.label_values[0]
      index_of_pos_label = self.label_values.index(pos_label)
      return scorecard['fscore'][index_of_pos_label]

    if score_fn == 'f1':
      fscores = []
      for item in zip(scorecard['fscore'], scorecard['support']):
        if item[1] > 0:
          fscores.append(item[0])
      return np.average(fscores, axis=0)

    raise ValueError(f'Unknown score_fn: {score_fn}')

  def train_step(
      self,
      checkpoint: Optional[LabelerCheckpoint],
      train_examples: pd.DataFrame,
      test_examples: Optional[pd.DataFrame] = None,
  ) -> LabelerCheckpoint:
    """Perform one optimization step.

    Args:
      checkpoint: the checkpoint to improve
      train_examples: training examples
      test_examples: test examples to evaluate mutation candidates on. If none,
        uses a random subset of train examples.

    Returns:
      improved checkpoint, or the original checkpoint if none of the mutation
    candidates outperform it.
    """
    candidates = self.get_best_mutations(
        checkpoint,
        train_examples,
        test_examples,
        n_best=self.hparams['n_best'],
    )
    return candidates[0][0]

  def get_mutations(
      self,
      checkpoint: LabelerCheckpoint,
      incorrect_predictions: list[ExampleWithPrediction],
      feedback: str,
      n_mutations_to_try: int,
  ) -> list[Mutation]:
    """Get mutation prediction from LLM."""
    mutation_options = self.get_mutation_options(checkpoint)
    mutation_options_dict = {item['key']: item for item in mutation_options}

    predictions = self.get_incorrect_predictions_templating_input(
        incorrect_predictions
    )

    mutations_to_try = []
    for _ in range(n_mutations_to_try):
      mutation_inputs = {
          **self.get_global_templating_inputs(),
          **self.get_rulebook_templating_inputs(checkpoint),
          'incorrect_predictions': predictions,
          'golden_label': predictions[0]['golden_label'],
          'feedback': feedback,
          'mutations': mutation_options,
          'mutation_keys': [item['key'] for item in mutation_options],
      }
      templated = self.identify_mutation_prompt.render(mutation_inputs)
      prediction = self.train_model_helper.predict(
          templated,
          temperature=TRAIN_TEMPERATURE,
          stop_sequences=['END_ANSWER', '\n\n'],
          candidate_count=1,
          max_output_tokens=2,
      )
      mutation_key = re.search(r'\d+', prediction)
      if mutation_key is not None:
        mutation_key = mutation_key.group(0)
        # In case the model predicts a number that doesn't correspond to a
        # mutation key.
        if mutation_key not in mutation_options_dict:
          mutation_key = list(mutation_options_dict.keys())[0]
      else:
        mutation_key = mutation_options[0]['key']

      mutations_to_try.append(mutation_options_dict[mutation_key])
      mutation_options = [
          item for item in mutation_options if item['key'] != mutation_key
      ]
    return mutations_to_try

  def get_best_mutations(
      self,
      checkpoint: Optional[LabelerCheckpoint],
      train_examples: pd.DataFrame,
      test_examples: Optional[pd.DataFrame] = None,
      n_mutations_to_try: int = 2,
      n_best: int = 1,
  ) -> list[CheckpointWithScore]:
    """Perform one optimization step.

    Args:
      checkpoint: the checkpoint to improve
      train_examples: training examples
      test_examples: test examples to evaluate mutation candidates on
      n_mutations_to_try: number of mutation candidates to try
      n_best: how many candidates to return

    Returns:
      The n best checkpoints with scores, ordered descending.
    """
    mutation_candidates = self.get_mutation_candidates(
        checkpoint,
        train_examples,
        n_mutations_to_try,
    )

    if test_examples is None:
      test_examples = train_examples.sample(
          min(train_examples.shape[0], self.hparams['n_validation_examples'])
      )
    return self.evaluate_candidates_bruteforce(
        mutation_candidates, test_examples, n_best
    )

  def get_mutation_candidates(
      self,
      checkpoint: Optional[LabelerCheckpoint],
      train_examples: pd.DataFrame,
      n_mutations_to_try: int,
  ) -> list[LabelerCheckpoint]:
    """Create a set of mutated candidates.

    Args:
      checkpoint: the checkpoint to improve
      train_examples: training examples
      n_mutations_to_try: the number of mutations to generate

    Returns:
      n mutated checkpoints.
    """

    if checkpoint is None:
      return [
          self.initialize_checkpoint(train_examples)
          for _ in range(n_mutations_to_try)
      ]

    train_examples = train_examples.sample(frac=1)

    counter = 0
    n_empty_predictions_encountered = 0
    incorrect_predictions = []
    while (
        len(incorrect_predictions)
        < self.hparams['n_incorrect_predictions_to_sample']
        and counter < train_examples.shape[0]
    ):
      prediction = self.infer_checkpoint(
          checkpoint, train_examples.iloc[[counter]]
      )[0]

      if len(prediction['prediction']) == 0:  # pylint: disable=g-explicit-length-test
        n_empty_predictions_encountered += 1
      elif not self.prediction_is_correct(
          train_examples.iloc[counter], prediction
      ):
        incorrect_predictions.append((train_examples.iloc[counter], prediction))
      counter += 1

    print(
        'Number of empty predictions encountered:'
        f' {n_empty_predictions_encountered}'
    )
    if self.debug_mode:
      for index, item in enumerate(incorrect_predictions):
        print(f'\n{index + 1} / {len(incorrect_predictions)}:')
        print_prediction(item, self.input_names, self.label_name)

    if len(incorrect_predictions) == 0:  # pylint: disable=g-explicit-length-test
      print('No nonempty incorrect predictions.')
      return [checkpoint]

    mutation_candidates = []
    for item in incorrect_predictions:
      mutation_candidates.extend(
          self.train_on_incorrect_predictions(
              checkpoint, [item], n_mutations_to_try
          )
      )
    return mutation_candidates

  def evaluate_candidates_bruteforce(
      self,
      candidates: list[LabelerCheckpoint],
      test_examples: pd.DataFrame,
      n_best: int,
  ) -> list[CheckpointWithScore]:
    """Evaluate mutation candidates by scoring against the entire test set.

    Args:
      candidates: Mutation candidates to evaluate.
      test_examples: Test examples to evaluate candidates on.
      n_best: Number of candidates to return.

    Returns:
      The n best checkpoints with scores, ordered descending.
    """

    scores = []
    for candidate in candidates:
      predictions = self.infer_checkpoint(candidate, test_examples)
      scores.append(
          self.get_score(
              test_examples,
              predictions,
          )
      )
    print('Scores:', scores)
    top_k_indices = np.argsort(scores)[::-1][:n_best]
    print('Top indices:', top_k_indices)
    return [(candidates[i], scores[i]) for i in top_k_indices]

  def prediction_is_correct(
      self, example: pd.Series, prediction: PredictionResult
  ) -> bool:
    label = str(example[self.label_name])
    return prediction['prediction'] == label

  def get_incorrect_predictions(
      self,
      labeled_examples: pd.DataFrame,
      predictions: list[PredictionResult],
  ) -> list[ExampleWithPrediction]:
    """Gets incorrect predictions."""
    incorrect_predictions = []
    for example_obj, prediction in zip(
        labeled_examples.iterrows(), predictions
    ):
      if not self.prediction_is_correct(example_obj[1], prediction):
        incorrect_predictions.append((example_obj[1], prediction))
    return incorrect_predictions

  def get_incorrect_predictions_nonempty(
      self,
      labeled_examples: pd.DataFrame,
      predictions: list[PredictionResult],
  ) -> list[ExampleWithPrediction]:
    """Gets incorrect predictions that are not empty."""
    incorrect_predictions = self.get_incorrect_predictions(
        labeled_examples, predictions
    )
    filtered = []
    for item in incorrect_predictions:
      prediction = item[1]['prediction']
      if len(prediction) > 0:  # pylint: disable=g-explicit-length-test
        filtered.append(item)
    return filtered

  def get_mutation_options(
      self, checkpoint: LabelerCheckpoint
  ) -> list[Mutation]:
    """Get the mutation options.

    For each class, include an option to add an attribute. For each existing
    attribute, include an option to edit it.

    Args:
      checkpoint: The current labeler.

    Returns:
      List of mutations that can be performed.
    """
    mutations: list[Mutation] = []
    for value in self.label_values:
      value_str = f'<{self.label_name}>{value}</{self.label_name}>'
      mutations.append({
          'key': str(len(mutations) + 1),
          'action': 'ADD',
          'label_value': value,
          'attribute_id': None,
          'prompt': f'Add a new rule to {value_str}',
      })
      attributes = get_attributes_for_label(checkpoint, value)
      for attribute in attributes:
        attribute_id = attribute['id']
        mutations.append({
            'key': str(len(mutations) + 1),
            'action': 'EDIT',
            'label_value': value,
            'attribute_id': attribute_id,
            'prompt': f'Edit rule_{attribute_id} of {value_str}',
        })
        if len(attributes) > 1:
          mutations.append({
              'key': str(len(mutations) + 1),
              'action': 'DELETE',
              'label_value': value,
              'attribute_id': attribute_id,
              'prompt': f'Delete rule_{attribute_id} from {value_str}',
          })
    return mutations

  def get_scorecard(
      self,
      labeled_examples: pd.DataFrame,
      predictions: list[PredictionResult],
  ) -> Scorecard:
    """Get detailed performance numbers."""
    y_true = labeled_examples[self.label_name].astype(str)
    n_empty = [item['raw_prediction'] for item in predictions].count('')
    y_true = y_true.to_numpy()
    y_pred = np.array([item['prediction'] for item in predictions])

    precision, recall, fscore, support = (
        metrics.precision_recall_fscore_support(
            y_true,
            y_pred,
            average=None,
            labels=self.label_values,
        )
    )
    accuracy = metrics.accuracy_score(y_true, y_pred)
    return Scorecard(
        precision=precision,
        recall=recall,
        fscore=fscore,
        support=support,
        n_empty=n_empty,
        accuracy=accuracy,
    )

  def infer_checkpoint(
      self,
      checkpoint: LabelerCheckpoint,
      examples: pd.DataFrame,
  ) -> list[PredictionResult]:
    """Batch inference over examples given a constitutional checkpoint."""
    # Get subset of columns from input_names that are actually found in the
    # examples.
    columns_to_use = [
        col for col in self.input_names if col in examples.columns
    ]
    missing_cols = [
        col for col in self.input_names if col not in examples.columns
    ]
    if missing_cols:
      print(
          'The following input features were not found in the examples:'
          f' {missing_cols}'
      )

    # Construct input arguments to classifier prompt template.
    train_example_inputs = examples[columns_to_use].to_dict('records')

    predictions = []
    for example in tqdm.tqdm(train_example_inputs):
      items = []
      for col in columns_to_use:
        if example[col] is not None:
          items.append((col, example[col]))
      params = {
          **self.get_global_templating_inputs(),
          **get_checkpoint_templating_params(checkpoint),
          'input_features': [
              {
                  'name': key,
                  'value': utils.truncate_from_middle(
                      value, self.hparams['truncation_length']
                  ),
              }
              for key, value in items
          ],
      }
      templated = self.classifier_prompt.render(params)
      prediction: str = (  # pytype: disable=annotation-type-mismatch
          self.eval_model_helper.predict(
              templated,
              temperature=self.eval_temperature,
              stop_sequences=self.classifier_prompt_stop_sequences,
              max_output_tokens=self.max_eval_output_tokens,
          )
      )
      # TODO: This checks for presence of correct answer as substring in the
      # prediction, to get around unexpected Gemini behavior where the answer
      # includes "answer_" even though it's in the prompt. We should check for
      # exact match.
      cleaned = ''
      for cand in self.label_values:
        if cand in prediction:
          cleaned = cand
          break
      predictions.append(
          PredictionResult(
              request=templated,
              raw_prediction=prediction,
              prediction=cleaned,
          )
      )
    return predictions

  def get_incorrect_predictions_templating_input(
      self, incorrect_predictions: list[ExampleWithPrediction]
  ) -> list[dict[str, Any]]:
    """Get templating argument for incorrect predictions prompt sections."""
    predictions = []
    for item in incorrect_predictions:
      example, prediction_result = item
      predictions.append({
          'index': len(predictions),
          'input_features': [
              {'name': feature, 'value': example[feature]}
              for feature in self.input_names
          ],
          'raw_prediction': prediction_result['raw_prediction'],
          'predicted_label': prediction_result['prediction'],
          'golden_label': example[self.label_name],
      })
    return predictions

  def get_rulebook_templating_inputs(
      self, checkpoint: LabelerCheckpoint
  ) -> dict[str, Any]:
    """Get templating arguments for rulebook prompt sections."""
    labels = []
    for label_value in self.label_values:
      labels.append({
          'attributes': get_attributes_for_label(checkpoint, label_value),
          'label_value': label_value,
      })
    return {'labels': labels}

  def get_global_templating_inputs(self) -> dict[str, Any]:
    """Get shared templating arguments."""
    return {
        'task_description': self.task_description,
        'question_delims': {'start': 'QUESTION:', 'end': 'END_QUESTION'},
        'response_delims': {'start': 'ANSWER:', 'end': 'END_ANSWER'},
        'label_name': self.label_name,
        'label_values': self.label_values,
    }
