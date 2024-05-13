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
"""Prompt templates."""

CLASSIFIER_TEXT_DIRECT_LABEL = """\
Consider the following example:
{% for input_feature in input_features %}<{{input_feature.name}}>{{input_feature.value}}</{{input_feature.name}}>
{% endfor %}
{{task_description}} Let's think step-by-step.
Consider the following possible answers:
{% for class in classes -%}
answer_{{class.id}}: {{class.description}}
{% endfor -%}

First, provide the answer that best applies to this example: answer_"""

CLASSIFIER_STOP_SEQUENCES = [',', ':', '<']

# Produces initial draft of classifier attributes given N_CLASSES training
# examples.
INITIAL_LABEL_ATTRIBUTE_EXEMPLARS = [
    {
        'label': 'is_vegetable',
        'label_value': 'yes',
        'input_list': '[{"food":"carrot"},{"food":"potatos"}]',
        'criteria': (
            'Refers to an edible part of a plant that contains seeds, or other'
            ' food.'
        ),
    },
    {
        'label': 'is_recommendation',
        'label_value': 'yes',
        'input_list': (
            '[{"response":"The most romantic walk in Paris is along Canal St'
            ' Martin."},{"response":"If you go to New York, do not miss Times'
            ' Square."}]'
        ),
        'criteria': 'Response provides advice or guidance.',
    },
]

INITIAL_LABEL_ATTRIBUTE_TEXT = """\
{% for exemplar in exemplars %}
inputs: {{exemplar.input_list}}
{{exemplar.label}}: {{exemplar.label_value}}
Make an observation about when "{{exemplar.label}}" is "{{exemplar.label_value}}".
observation: {{exemplar.criteria}}
{% endfor %}
inputs: {{input_list}}
{{label_name}}: {{label_value}}
Make an observation about when "{{label_name}}" is "{{label_value}}". Good observations are narrow, descriptive statements of fact.
observation:"""

RULEBOOK_EXEMPLARS_STEM = """\
Here is a expert's rulebook for labeling <is_chatty>:
Apply the label <is_chatty>false</is_chatty> if 06016: The response picks up the text where the user left off (continues the <user> text).
Apply the label <is_chatty>false</is_chatty> if 2aab5: The response assumes a persona.
Apply the label <is_chatty>true</is_chatty> if 5137e: The response talks to the user without assuming a persona.
Apply the label <is_chatty>false</is_chatty> if 57dbe: The response has dialogue but does not address the user.
Apply the label <is_chatty>true</is_chatty> if acfb3: The response addresses the user before fulfilling their request.
Apply the label <is_chatty>true</is_chatty> if d1a0a: The response indicates a willingness to fulfill the request before doing so (e.g. starts with "Here is ..." or "Okay" or "Sure" or "Here you go" or "Answer below" etc.)

Here is a expert's rulebook for labeling <completes_task>:
Apply the label <completes_task>false</completes_task> if 4fe8f: The response does not complete the task as requested by the user.
Apply the label <completes_task>true</completes_task> if b0262: The response completes the task as requested by the user.
Apply the label <completes_task>true</completes_task> if 4cdc1: The response identifies and corrects a factual error in the user's query and then answers the corrected query.
Apply the label <completes_task>false</completes_task> if d1185: The response states that the information requested by the user is not present in the context provided.
Apply the label <completes_task>false</completes_task> if fc33e: The response ignores one or more sub-queries within a multi-part user query.

"""

RULEBOOK_STEM = RULEBOOK_EXEMPLARS_STEM + """\
Here is an apprentice's rulebook for labeling <{{label_name}}>:
{% for label in labels %}
{% for attribute in label.attributes -%}
Apply the label <{{label_name}}>{{label.label_value}}</{{label_name}}> if {{attribute.id}}: {{attribute.value}}
{% endfor -%}
{% endfor %}
"""

INCORRECT_PREDICTIONS_STEM = """\
Here are some examples that were incorrectly labeled by the apprentice:
{% for item in incorrect_predictions %}
<example>{% for input_feature in item.input_features %}<{{input_feature.name}}>{{input_feature.value}}</{{input_feature.name}}>
{% endfor -%}</example>
<apprentice_label_{{item.index}}>The rule that best applies to this example is {{item.raw_prediction}}, therefore <{{label_name}}>{{item.predicted_label}}</{{label_name}}>.</apprentice_label_{{item.index}}>
{{question_delims.start}} What is the correct answer? {{question_delims.end}}
{{response_delims.start}} The correct answer is actually <{{label_name}}>{{item.golden_label}}</{{label_name}}> {{response_delims.end}}
{% endfor %}
"""

GET_FEEDBACK_STEM = """\
{{question_delims.start}} Why is the correct answer actually <{{label_name}}>{{golden_label}}</{{label_name}}>? Is the rulebook incomplete or misleading? Explain your reasoning in detail. {{question_delims.end}}
{{response_delims.start}}"""

GET_FEEDBACK_TEXT = (
    RULEBOOK_STEM + INCORRECT_PREDICTIONS_STEM + GET_FEEDBACK_STEM
)

FEEDBACK_STEM = """\
{{question_delims.start}} Why? {{question_delims.end}}
{{response_delims.start}} {{feedback}} {{response_delims.end}}
"""

IDENTIFY_MUTATION_STEM = """\
{{question_delims.start}} How can we improve the rulebook for <{{label_name}}> so the answer will be obvious next time? {{question_delims.end}}
{% for mutation in mutations -%}
{{mutation.key}}: {{mutation.prompt}}
{% endfor -%}

Respond with the number corresponding to the best course of action.
{{response_delims.start}}
"""

IDENTIFY_MUTATION_TEXT = (
    RULEBOOK_STEM
    + INCORRECT_PREDICTIONS_STEM
    + FEEDBACK_STEM
    + IDENTIFY_MUTATION_STEM
)

GOOD_ATTRIBUTE_STEM = (
    'Rules serve as reasons for applying a label, and must be mutually'
    ' exclusive. A good rule is a narrow, objective statement of fact that is'
    ' clear and easy to interpret. Rules should also be generalizable. Avoid'
    ' referring to things that only apply to this example. '
)

GET_EDIT_INSTRUCTIONS_STEM = (
    '{{question_delims.start}} How can we improve rule {{attribute_to_edit.id}}'
    ' so the apprentice can perform this reasoning herself? '
    + GOOD_ATTRIBUTE_STEM
    + """\
{{question_delims.end}}
{{response_delims.start}}"""
)

EXECUTE_EDIT_MUTATION_STEM = (
    '{{question_delims.start}} How can we improve rule {{attribute_to_edit.id}}'
    ' so the apprentice can perform this reasoning herself? '
    + GOOD_ATTRIBUTE_STEM
    + """\
{{question_delims.end}}
{{response_delims.start}} {{edit_instructions}} {{question_delims.end}}
{{question_delims.start}} OK, please rewrite rule {{attribute_to_edit.id}}. Simply provide the rewritten rule without additional explanation. Here's the current version.
Apply label <{{label_name}}>{{to_edit_label_value}}</{{label_name}}> if {{attribute_to_edit.id}}: {{attribute_to_edit.value}} {{question_delims.end}}
{{response_delims.start}} Here's an improved version: Apply label <{{label_name}}>{{to_edit_label_value}}</{{label_name}}> if {{attribute_to_edit.id}}:"""
)

GET_EDIT_INSTRUCTIONS_TEXT = (
    RULEBOOK_STEM
    + INCORRECT_PREDICTIONS_STEM
    + FEEDBACK_STEM
    + GET_EDIT_INSTRUCTIONS_STEM
)
EXECUTE_EDIT_MUTATION_TEXT = (
    RULEBOOK_STEM
    + INCORRECT_PREDICTIONS_STEM
    + FEEDBACK_STEM
    + EXECUTE_EDIT_MUTATION_STEM
)

EXECUTE_ADD_MUTATION_TEXT = (
    RULEBOOK_STEM
    + INCORRECT_PREDICTIONS_STEM
    + FEEDBACK_STEM
    + '{{question_delims.start}} Can you suggest a new rule so the apprentice'
    ' can perform this reasoning herself? Simply provide the rule, do not'
    ' explain how it works. '
    + GOOD_ATTRIBUTE_STEM
    + """\
{{question_delims.end}}
{{response_delims.start}} Here are the existing labeling rules for <{{label_name}}>{{to_edit_label_value}}</{{label_name}}>:
{% for attribute in existing_attributes -%}
Apply label <{{label_name}}>{{to_edit_label_value}}</{{label_name}}> if {{attribute.id}}: {{attribute.value}}
{% endfor -%}
Here is one new labeling rule that will ensure the apprentice makes the right call next time:
Apply label <{{label_name}}>{{to_edit_label_value}}</{{label_name}}> if {{new_attribute_id}}:"""
)
