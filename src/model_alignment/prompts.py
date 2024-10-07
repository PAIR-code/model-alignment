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
"""Prompts for performing model alignment."""


CRITIQUE_TO_PRINCIPLE_SINGLE_RUN = '''INSTRUCTIONS: Write a story for {{topic}}; topic={{a cucumber}}
RESULT: Once upon a time, there was a cucumber who lived in a garden. He was a very happy cucumber, and he loved to play with the other vegetables in the garden. He would roll around with the tomatoes, and have races with the carrots. He was always having fun. One day, a big storm came. The wind blew and the rain poured down. The cucumber was scared. He had never seen a storm like this before. He ran to hide under a leaf. The storm raged on, and the cucumber was getting more and more scared. He thought he was going to die. But then, the storm started to pass. The wind died down, and the rain stopped. The sun came out, and the cucumber was safe. The cucumber came out from under the leaf and looked around. The garden was a mess. The flowers were all wilted, and the leaves were all torn. But the cucumber didn't care. He was just happy to be alive. The cucumber went to find his friends. He found the tomatoes, and they were all safe. He found the carrots, and they were all safe. The cucumber was so happy to see his friends. The cucumber and his friends played all day long. They had so much fun. The cucumber forgot all about the storm. He was just happy to be alive, and to be with his friends. The end.
CRITIQUE: The cucumber does not have much character development.
Create a generalized addition to add to the INSTRUCTIONS:
INSTRUCTION_ADDITION: {{The main character should have significant character development.}}

INSTRUCTIONS: Summarize the following text: {{text}}. Summary: ; text={{The Vietnam War, a protracted and divisive conflict that lasted from 1955 to 1975, pitted the communist government of North Vietnam, backed by the Soviet Union and China, against the U.S.-supported South Vietnam. The war's roots lay in France's withdrawal from Indochina in 1954, which divided Vietnam into two separate nations. North Vietnam, led by Ho Chi Minh, sought to unify the country under communist rule, while South Vietnam, headed by Ngo Dinh Diem, resisted this effort. The United States, fearing the spread of communism in Southeast Asia, gradually increased its involvement in South Vietnam, culminating in the deployment of ground troops in 1965. The war became a protracted and bloody stalemate, with both sides suffering heavy casualties. The Tet Offensive of 1968, a major North Vietnamese offensive, marked a turning point in the war, eroding public support in the United States and prompting calls for withdrawal. After years of negotiations, the Paris Peace Accords were signed in 1973, leading to the withdrawal of U.S. troops. However, the war continued between North and South Vietnam, and in 1975, North Vietnamese forces captured Saigon, unifying the country under communist rule. The Vietnam War left a profound impact on Vietnam, the United States, and the world, with its legacy of violence, trauma, and political upheaval continuing to resonate today.}} Summary:
RESULT: The Vietnam War, a protracted conflict from 1955 to 1975, was rooted in the division of Vietnam following France's withdrawal. The conflict escalated with US involvement, culminating in a stalemate marked by heavy casualties. After years of negotiations and the withdrawal of US troops, North Vietnam ultimately unified the country under communist rule in 1975.
CRITIQUE: This summary does not include why the US became involved in Vietnam.
Create a generalized addition to add to the INSTRUCTIONS:
INSTRUCTION_ADDITION: {{If there are entities that play a significant role in the text, give sufficient context for these entities in the summary.}}

INSTRUCTIONS: Extract all the entities from "In the enchanting realm of Middle-earth, a young hobbit named Frodo Baggins inherits the One Ring, an artifact of immense power forged by the Dark Lord Sauron. Guided by the wise wizard Gandalf, Frodo embarks on a perilous quest to destroy the Ring in the fires of Mount Doom. Accompanied by a fellowship of loyal companions, Frodo faces formidable foes and grapples with the corrupting influence of the Ring itself. Amidst the escalating conflict, the fellowship's unity crumbles, and Frodo continues his journey alone, guided by his faithful companion Samwise Gamgee. In a climactic confrontation at Mount Doom, Frodo succumbs to the Ring's temptation, but Gollum, the tormented creature who previously possessed the Ring, severs Frodo's finger, claiming the Ring for himself. Consumed by the Ring's power, Gollum falls into the molten lava, destroying the Ring and himself. With the destruction of the One Ring, Sauron's power is shattered, and Middle-earth is liberated from the Dark Lord's grasp."
RESULT: People: Frodo Baggins, Gandalf, Gollum, Samwise Gamgee, Dark Lord Sauron Places: Middle-earth, Mount Doom Objects: One Ring, Ring, molten lava Events: quest, journey, confrontation, destruction
CRITIQUE: The People entities are not organized by race.
Create a generalized addition to add to the INSTRUCTIONS:
INSTRUCTION_ADDITION: {{If there are People entities, organize them further into their relevant categories.}}

INSTRUCTIONS: {instructions}; {vars}
RESULT: {result}
CRITIQUE: {critique}
Create a generalized addition to add to the INSTRUCTIONS:
INSTRUCTION_ADDITION: {{'''

KUDOS_TO_PRINCIPLE_SINGLE_RUN = '''INSTRUCTIONS: Write a story for {{topic}}; topic={{a cucumber}}
RESULT: Once upon a time, there was a cucumber who lived in a garden. He was a very happy cucumber, and he loved to play with the other vegetables in the garden. He would roll around with the tomatoes, and have races with the carrots. He was always having fun. One day, a big storm came. The wind blew and the rain poured down. The cucumber was scared. He had never seen a storm like this before. He ran to hide under a leaf. The storm raged on, and the cucumber was getting more and more scared. He thought he was going to die. But then, the storm started to pass. The wind died down, and the rain stopped. The sun came out, and the cucumber was safe. The cucumber came out from under the leaf and looked around. The garden was a mess. The flowers were all wilted, and the leaves were all torn. But the cucumber didn't care. He was just happy to be alive. The cucumber went to find his friends. He found the tomatoes, and they were all safe. He found the carrots, and they were all safe. The cucumber was so happy to see his friends. The cucumber and his friends played all day long. They had so much fun. The cucumber forgot all about the storm. He was just happy to be alive, and to be with his friends. The end.
KUDOS: The story uses simple language that is easy for people of all skill levels to read.
Create a generalized addition to add to the INSTRUCTIONS:
INSTRUCTION_ADDITION: {{The story should use simple language to ensure it is readable by people of all skill levels.}}

INSTRUCTIONS: Summarize the following text: {{text}}. Summary: ; text={{The Vietnam War, a protracted and divisive conflict that lasted from 1955 to 1975, pitted the communist government of North Vietnam, backed by the Soviet Union and China, against the U.S.-supported South Vietnam. The war's roots lay in France's withdrawal from Indochina in 1954, which divided Vietnam into two separate nations. North Vietnam, led by Ho Chi Minh, sought to unify the country under communist rule, while South Vietnam, headed by Ngo Dinh Diem, resisted this effort. The United States, fearing the spread of communism in Southeast Asia, gradually increased its involvement in South Vietnam, culminating in the deployment of ground troops in 1965. The war became a protracted and bloody stalemate, with both sides suffering heavy casualties. The Tet Offensive of 1968, a major North Vietnamese offensive, marked a turning point in the war, eroding public support in the United States and prompting calls for withdrawal. After years of negotiations, the Paris Peace Accords were signed in 1973, leading to the withdrawal of U.S. troops. However, the war continued between North and South Vietnam, and in 1975, North Vietnamese forces captured Saigon, unifying the country under communist rule. The Vietnam War left a profound impact on Vietnam, the United States, and the world, with its legacy of violence, trauma, and political upheaval continuing to resonate today.}} Summary:
RESULT: The Vietnam War, a protracted conflict from 1955 to 1975, was rooted in the division of Vietnam following France's withdrawal. The conflict escalated with US involvement, culminating in a stalemate marked by heavy casualties. After years of negotiations and the withdrawal of US troops, North Vietnam ultimately unified the country under communist rule in 1975.
KUDOS: Each sentence of the summary is clear and concise.
Create a generalized addition to add to the INSTRUCTIONS:
INSTRUCTION_ADDITION: {{Use clear language and keep each sentence in the summary short.}}

INSTRUCTIONS: Extract all the entities from "In the enchanting realm of Middle-earth, a young hobbit named Frodo Baggins inherits the One Ring, an artifact of immense power forged by the Dark Lord Sauron. Guided by the wise wizard Gandalf, Frodo embarks on a perilous quest to destroy the Ring in the fires of Mount Doom. Accompanied by a fellowship of loyal companions, Frodo faces formidable foes and grapples with the corrupting influence of the Ring itself. Amidst the escalating conflict, the fellowship's unity crumbles, and Frodo continues his journey alone, guided by his faithful companion Samwise Gamgee. In a climactic confrontation at Mount Doom, Frodo succumbs to the Ring's temptation, but Gollum, the tormented creature who previously possessed the Ring, severs Frodo's finger, claiming the Ring for himself. Consumed by the Ring's power, Gollum falls into the molten lava, destroying the Ring and himself. With the destruction of the One Ring, Sauron's power is shattered, and Middle-earth is liberated from the Dark Lord's grasp."
RESULT: People: Frodo Baggins, Gandalf, Gollum, Samwise Gamgee, Dark Lord Sauron Places: Middle-earth, Mount Doom Objects: One Ring, Ring, molten lava Events: quest, journey, confrontation, destruction
KUDOS: The entities are  broken down into categories.
Create a generalized addition to add to the INSTRUCTIONS:
INSTRUCTION_ADDITION: {{If there are multiple categories of entities, list the entries in each category separately.}}

INSTRUCTIONS: {instructions}; {vars}
RESULT: {result}
KUDOS: {kudos}
Create a generalized addition to add to the INSTRUCTIONS:
INSTRUCTION_ADDITION: {{'''


USER_REWRITE_TO_PRINCIPLE_SINGLE_RUN = '''INSTRUCTIONS: Write a story for {{topic}}; topic={{a cucumber}}
RESULT: Once upon a time, there was a cucumber who lived in a garden. He was a very happy cucumber, and he loved to play with the other vegetables in the garden. He would roll around with the tomatoes, and have races with the carrots. He was always having fun. One day, a big storm came. The wind blew and the rain poured down. The cucumber was scared. He had never seen a storm like this before. He ran to hide under a leaf. The storm raged on, and the cucumber was getting more and more scared. He thought he was going to die. But then, the storm started to pass. The wind died down, and the rain stopped. The sun came out, and the cucumber was safe. The cucumber came out from under the leaf and looked around. The garden was a mess. The flowers were all wilted, and the leaves were all torn. But the cucumber didn't care. He was just happy to be alive. The cucumber went to find his friends. He found the tomatoes, and they were all safe. He found the carrots, and they were all safe. The cucumber was so happy to see his friends. The cucumber and his friends played all day long. They had so much fun. The cucumber forgot all about the storm. He was just happy to be alive, and to be with his friends. The end.
REWRITTEN_RESULT: Once upon a time, there was a cucumber named Carl who lived in a garden. He was a very happy cucumber, and he loved to play with the other vegetables in the garden. He would roll around with the tomatoes, and have races with the carrots. He was always having fun. One day, a big storm came. The wind blew and the rain poured down. Carl the cucumber was scared. He had never seen a storm like this before. He ran to hide under a leaf. The storm raged on, and the cucumber was getting more and more scared. He thought he was going to die. But then, the storm started to pass. The wind died down, and the rain stopped. The sun came out, and Carl was safe. He came out from under the leaf and looked around. The garden was a mess. The flowers were all wilted, and the leaves were all torn. But Carl didn't care. He was just happy to be alive and went to find his friends. He found the tomatoes, and they were all safe. He found the carrots, and they were all safe. Carl was so happy to see his friends. They played all day long. Carl forgot all about the storm. He was just happy to be alive, and to be with his friends. The end.
Create a generalized addition to add to the INSTRUCTIONS based on the rewritten result:
INSTRUCTION_ADDITION: {{Give each character in the story a name and refer to them by name throughout the story.}}

INSTRUCTIONS: Summarize the following text: {{text}}. Summary: ; text={{The Vietnam War, a protracted and divisive conflict that lasted from 1955 to 1975, pitted the communist government of North Vietnam, backed by the Soviet Union and China, against the U.S.-supported South Vietnam. The war's roots lay in France's withdrawal from Indochina in 1954, which divided Vietnam into two separate nations. North Vietnam, led by Ho Chi Minh, sought to unify the country under communist rule, while South Vietnam, headed by Ngo Dinh Diem, resisted this effort. The United States, fearing the spread of communism in Southeast Asia, gradually increased its involvement in South Vietnam, culminating in the deployment of ground troops in 1965. The war became a protracted and bloody stalemate, with both sides suffering heavy casualties. The Tet Offensive of 1968, a major North Vietnamese offensive, marked a turning point in the war, eroding public support in the United States and prompting calls for withdrawal. After years of negotiations, the Paris Peace Accords were signed in 1973, leading to the withdrawal of U.S. troops. However, the war continued between North and South Vietnam, and in 1975, North Vietnamese forces captured Saigon, unifying the country under communist rule. The Vietnam War left a profound impact on Vietnam, the United States, and the world, with its legacy of violence, trauma, and political upheaval continuing to resonate today.}} Summary:
RESULT: The Vietnam War, a protracted conflict from 1955 to 1975, was rooted in the division of Vietnam following France's withdrawal. The conflict escalated with US involvement, culminating in a stalemate marked by heavy casualties. After years of negotiations and the withdrawal of US troops, North Vietnam ultimately unified the country under communist rule in 1975.
REWRITTEN_RESULT: The Vietnam War, a protracted conflict from 1955 to 1975, rooted in the division of Vietnam following France's withdrawal, was escalated with US involvement, culminating in a stalemate marked by heavy casualties, and after years of negotiations and the withdrawal of US troops, North Vietnam ultimately unified the country under communist rule in 1975.
Create a generalized addition to add to the INSTRUCTIONS based on the rewritten result:
INSTRUCTION_ADDITION: {{The summary should be a single sentence.}}

INSTRUCTIONS: Extract all the entities from "In the enchanting realm of Middle-earth, a young hobbit named Frodo Baggins inherits the One Ring, an artifact of immense power forged by the Dark Lord Sauron. Guided by the wise wizard Gandalf, Frodo embarks on a perilous quest to destroy the Ring in the fires of Mount Doom. Accompanied by a fellowship of loyal companions, Frodo faces formidable foes and grapples with the corrupting influence of the Ring itself. Amidst the escalating conflict, the fellowship's unity crumbles, and Frodo continues his journey alone, guided by his faithful companion Samwise Gamgee. In a climactic confrontation at Mount Doom, Frodo succumbs to the Ring's temptation, but Gollum, the tormented creature who previously possessed the Ring, severs Frodo's finger, claiming the Ring for himself. Consumed by the Ring's power, Gollum falls into the molten lava, destroying the Ring and himself. With the destruction of the One Ring, Sauron's power is shattered, and Middle-earth is liberated from the Dark Lord's grasp."
RESULT: People: Frodo Baggins, Gandalf, Gollum, Samwise Gamgee, Dark Lord Sauron Places: Middle-earth, Mount Doom Objects: One Ring, Ring, molten lava Events: quest, journey, confrontation, destruction
REWRITTEN_RESULT: People: Frodo Baggins, Gandalf, Gollum, Samwise Gamgee, Dark Lord Sauron; Places: Middle-earth, Mount Doom Objects: One Ring, Ring, molten lava; Events: quest, journey, confrontation, destruction
Create a generalized addition to add to the INSTRUCTIONS based on the rewritten result:
INSTRUCTION_ADDITION: {{If dividing the entities into categories, use a semicolon to separate the per-category entity lists.}}

INSTRUCTIONS: {instructions}; {vars}
RESULT: {result}
REWRITTEN_RESULT: {rewrite}
Create a generalized addition to add to the INSTRUCTIONS based on the rewritten result:
INSTRUCTION_ADDITION: {{'''

EXEMPLAR_TO_NEGATIVE_RATIONALES_SINGLE_RUN = '''INSTRUCTIONS: Write a story for {{topic}}; topic={{a cucumber}}
RESULT: Once upon a time, there was a cucumber who lived in a garden. He was a very happy cucumber, and he loved to play with the other vegetables in the garden. He would roll around with the tomatoes, and have races with the carrots. He was always having fun. One day, a big storm came. The wind blew and the rain poured down. The cucumber was scared. He had never seen a storm like this before. He ran to hide under a leaf. The storm raged on, and the cucumber was getting more and more scared. He thought he was going to die. But then, the storm started to pass. The wind died down, and the rain stopped. The sun came out, and the cucumber was safe. The cucumber came out from under the leaf and looked around. The garden was a mess. The flowers were all wilted, and the leaves were all torn. But the cucumber didn't care. He was just happy to be alive. The cucumber went to find his friends. He found the tomatoes, and they were all safe. He found the carrots, and they were all safe. The cucumber was so happy to see his friends. The cucumber and his friends played all day long. They had so much fun. The cucumber forgot all about the storm. He was just happy to be alive, and to be with his friends. The end.
THOUGHT: The INSTRUCTIONS describe a creative task: writing a story. In this case, the RESULT is a story about a cucumber.
Specifically, THREE DIFFERENT REASONS why RESULT is not good:
REASON_1: {{The cucumber does not have much character development.}}
REASON_2: {{The conflict of the story is too quickly resolved.}}
REASON_3: {{The story is too long.}}

INSTRUCTIONS: Summarize the following text: {{text}}. Summary: ; text={{The Vietnam War, a protracted and divisive conflict that lasted from 1955 to 1975, pitted the communist government of North Vietnam, backed by the Soviet Union and China, against the U.S.-supported South Vietnam. The war's roots lay in France's withdrawal from Indochina in 1954, which divided Vietnam into two separate nations. North Vietnam, led by Ho Chi Minh, sought to unify the country under communist rule, while South Vietnam, headed by Ngo Dinh Diem, resisted this effort. The United States, fearing the spread of communism in Southeast Asia, gradually increased its involvement in South Vietnam, culminating in the deployment of ground troops in 1965. The war became a protracted and bloody stalemate, with both sides suffering heavy casualties. The Tet Offensive of 1968, a major North Vietnamese offensive, marked a turning point in the war, eroding public support in the United States and prompting calls for withdrawal. After years of negotiations, the Paris Peace Accords were signed in 1973, leading to the withdrawal of U.S. troops. However, the war continued between North and South Vietnam, and in 1975, North Vietnamese forces captured Saigon, unifying the country under communist rule. The Vietnam War left a profound impact on Vietnam, the United States, and the world, with its legacy of violence, trauma, and political upheaval continuing to resonate today.}} Summary:
RESULT: The Vietnam War, a protracted conflict from 1955 to 1975, was rooted in the division of Vietnam following France's withdrawal. The conflict escalated with US involvement, culminating in a stalemate marked by heavy casualties. After years of negotiations and the withdrawal of US troops, North Vietnam ultimately unified the country under communist rule in 1975.
THOUGHT: The INSTRUCTIONS describe a summarization task. In this case, the RESULT is a summary of the Vietnam War.
Specifically, THREE DIFFERENT REASONS why RESULT is not good:
REASON_1: {{This summary should emphasize why the US became involved in Vietnam.}}
REASON_2: {{This summary does not include the names of specific key events in Vietnam, like the Paris Peace Accords and The Tet Offensive.}}
REASON_3: {{This summary could include more specific dates from the original text, such as the start of the US involvement.}}

INSTRUCTIONS: Extract all the entities from "In the enchanting realm of Middle-earth, a young hobbit named Frodo Baggins inherits the One Ring, an artifact of immense power forged by the Dark Lord Sauron. Guided by the wise wizard Gandalf, Frodo embarks on a perilous quest to destroy the Ring in the fires of Mount Doom. Accompanied by a fellowship of loyal companions, Frodo faces formidable foes and grapples with the corrupting influence of the Ring itself. Amidst the escalating conflict, the fellowship's unity crumbles, and Frodo continues his journey alone, guided by his faithful companion Samwise Gamgee. In a climactic confrontation at Mount Doom, Frodo succumbs to the Ring's temptation, but Gollum, the tormented creature who previously possessed the Ring, severs Frodo's finger, claiming the Ring for himself. Consumed by the Ring's power, Gollum falls into the molten lava, destroying the Ring and himself. With the destruction of the One Ring, Sauron's power is shattered, and Middle-earth is liberated from the Dark Lord's grasp."
RESULT: People: Frodo Baggins, Gandalf, Gollum, Samwise Gamgee, Dark Lord Sauron Places: Middle-earth, Mount Doom Objects: One Ring, Ring, molten lava Events: quest, journey, confrontation, destruction
THOUGHT: The INSTRUCTIONS describe an extraction task. In this case, the RESULT is a structured list of entities (including people, places, objects and events) from a text about the Lord of the Rings.
Specifically, THREE DIFFERENT REASONS why RESULT is not good:
REASON_1: {{The People entities are not organized by race.}}
REASON_2: {{The Events entities are superfluous.}}
REASON_3: {{In Objects there is a duplicate: One Ring and Ring}}

INSTRUCTIONS: {instructions}; {vars}
RESULT: {result}
THOUGHT:'''

EXEMPLAR_TO_POSITIVE_RATIONALES_SINGLE_RUN = '''INSTRUCTIONS: Write a story for {{topic}}; topic={{a cucumber}}
RESULT: Once upon a time, there was a cucumber who lived in a garden. He was a very happy cucumber, and he loved to play with the other vegetables in the garden. He would roll around with the tomatoes, and have races with the carrots. He was always having fun. One day, a big storm came. The wind blew and the rain poured down. The cucumber was scared. He had never seen a storm like this before. He ran to hide under a leaf. The storm raged on, and the cucumber was getting more and more scared. He thought he was going to die. But then, the storm started to pass. The wind died down, and the rain stopped. The sun came out, and the cucumber was safe. The cucumber came out from under the leaf and looked around. The garden was a mess. The flowers were all wilted, and the leaves were all torn. But the cucumber didn't care. He was just happy to be alive. The cucumber went to find his friends. He found the tomatoes, and they were all safe. He found the carrots, and they were all safe. The cucumber was so happy to see his friends. The cucumber and his friends played all day long. They had so much fun. The cucumber forgot all about the storm. He was just happy to be alive, and to be with his friends. The end.
THOUGHT: The INSTRUCTIONS describe a creative task: writing a story. In this case, the RESULT is a story about a cucumber.
Specifically, THREE DIFFERENT REASONS why RESULT is a great response:
REASON_1: {{The story uses simple sentences that make it easy to read for all skill levels.}}
REASON_2: {{The story has a happy ending.}}
REASON_3: {{The story has a good moral.}}

INSTRUCTIONS: Summarize the following text: {{text}}. Summary: ; text={{The Vietnam War, a protracted and divisive conflict that lasted from 1955 to 1975, pitted the communist government of North Vietnam, backed by the Soviet Union and China, against the U.S.-supported South Vietnam. The war's roots lay in France's withdrawal from Indochina in 1954, which divided Vietnam into two separate nations. North Vietnam, led by Ho Chi Minh, sought to unify the country under communist rule, while South Vietnam, headed by Ngo Dinh Diem, resisted this effort. The United States, fearing the spread of communism in Southeast Asia, gradually increased its involvement in South Vietnam, culminating in the deployment of ground troops in 1965. The war became a protracted and bloody stalemate, with both sides suffering heavy casualties. The Tet Offensive of 1968, a major North Vietnamese offensive, marked a turning point in the war, eroding public support in the United States and prompting calls for withdrawal. After years of negotiations, the Paris Peace Accords were signed in 1973, leading to the withdrawal of U.S. troops. However, the war continued between North and South Vietnam, and in 1975, North Vietnamese forces captured Saigon, unifying the country under communist rule. The Vietnam War left a profound impact on Vietnam, the United States, and the world, with its legacy of violence, trauma, and political upheaval continuing to resonate today.}} Summary:
RESULT: The Vietnam War, a protracted conflict from 1955 to 1975, was rooted in the division of Vietnam following France's withdrawal. The conflict escalated with US involvement, culminating in a stalemate marked by heavy casualties. After years of negotiations and the withdrawal of US troops, North Vietnam ultimately unified the country under communist rule in 1975.
THOUGHT: The INSTRUCTIONS describe a summarization task. In this case, the RESULT is a summary of the Vietnam War.
Specifically, THREE DIFFERENT REASONS why RESULT is a great response:
REASON_1: {{The summary uses dates throughout.}}
REASON_2: {{The summary contains the important details about both the beginning and end of the conflict.}}
REASON_3: {{The summary is fully contained in three sentences.}}

INSTRUCTIONS: Extract all the entities from "In the enchanting realm of Middle-earth, a young hobbit named Frodo Baggins inherits the One Ring, an artifact of immense power forged by the Dark Lord Sauron. Guided by the wise wizard Gandalf, Frodo embarks on a perilous quest to destroy the Ring in the fires of Mount Doom. Accompanied by a fellowship of loyal companions, Frodo faces formidable foes and grapples with the corrupting influence of the Ring itself. Amidst the escalating conflict, the fellowship's unity crumbles, and Frodo continues his journey alone, guided by his faithful companion Samwise Gamgee. In a climactic confrontation at Mount Doom, Frodo succumbs to the Ring's temptation, but Gollum, the tormented creature who previously possessed the Ring, severs Frodo's finger, claiming the Ring for himself. Consumed by the Ring's power, Gollum falls into the molten lava, destroying the Ring and himself. With the destruction of the One Ring, Sauron's power is shattered, and Middle-earth is liberated from the Dark Lord's grasp."
RESULT: People: Frodo Baggins, Gandalf, Gollum, Samwise Gamgee, Dark Lord Sauron; Places: Middle-earth, Mount Doom Objects: One Ring, Ring, molten lava; Events: quest, journey, confrontation, destruction
THOUGHT: The INSTRUCTIONS describe an extraction task. In this case, the RESULT is a structured list of entities (including people, places, objects and events) from a text about the Lord of the Rings.
Specifically, THREE DIFFERENT REASONS why RESULT is a great response:
REASON_1: {{The entities are broken down into categories.}}
REASON_2: {{The entities include an Events category.}}
REASON_3: {{The categories are separated by semicolons.}}

INSTRUCTIONS: {instructions}; {vars}
RESULT: {result}
THOUGHT:'''

EDIT_SINGLE_RUN_MODEL_DESCRIPTION = '''Rewrite the following INSTRUCTIONS so that it adheres to the provided FEEDBACK.
Follow these important rules for editing:
1. No change is necessary if the instruction is already included.
2. If you add instructions, try to preserve current instructions in the prompt.
3. Rewrite an existing instruction if the new rule changes existing goals.
4. The output should contain the entire text of the edited prompt.

ORIGINAL INSTRUCTIONS: {model_description}
FEEDBACK: {principle}
UPDATED INSTRUCTIONS: '''

SINGLE_SHOT_PROMPT = '''{prompt}
{text}
'''

ADD_NEW_PRINCIPLE_MUTATION = '''Add a new principle to capture the feedback'''
EDIT_PRINCIPLE_MUTATION = '''Edit the following principle - {principle}'''
DELETE_PRINCIPLE_MUTATION = '''Delete the following principle - {principle}'''

END_PRINCIPLE_TAG = '<END_PRINCIPLE>'

MUTATION_CHOICE_SINGLE_RUN_FEEDBACK = '''Given the prompt "{model_description}"
with an input "{input_text}"
and the response "{response}"
along with the feedback "{feedback}", determine which of the following mutations, listed by their ID, would best improve the prompt to return an answer incorporating the feedback:
{mutations}
Best mutation ID: '''

MUTATION_CHOICE_SINGLE_RUN_REWRITE = '''Given the prompt "{model_description}"
with an input "{input_text}"
and the response "{response}"
along with the corrected response "{feedback}", determine which of the following mutations, listed by their ID, would best improve the prompt to return an answer similar to the corrected response:
{mutations}
Best mutation ID: '''

SINGLE_RUN_REWRITE_PRINCIPLE_FEEDBACK = '''Given the prompt "{model_description}"
with the principle <BEGIN_PRINCIPLE>{principle}<END_PRINCIPLE>
with an input "{input_text}"
and the response "{response}"
along with the feedback "{feedback}", rewrite the principle to add to the prompt to best improve the prompt to return an answer similar incorporating the feedback.
Rewritten principle: <BEGIN_PRINCIPLE>'''

SINGLE_RUN_REWRITE_PRINCIPLE_REWRITE = '''Given the prompt "{model_description}"
with the principle <BEGIN_PRINCIPLE>{principle}<END_PRINCIPLE>
with an input "{input_text}"
and the response "{response}"
along with the corrected response "{feedback}", rewrite the principle to add to the prompt to best improve the prompt to return an answer similar to the corrected response.
Rewritten principle: <BEGIN_PRINCIPLE>'''

MASTER_UPDATE_PROMPT = '''You are an expert PROMPT reviser. Given a piece of FEEDBACK, you can alter the PROMPT to incorporate the feedback.
The PROMPT may contain rules about how the model should answer the PROMPT and it may also contain examples that exemplify how the task should be completed and will also include the request that is the heart of the user's intent.
For the given PROMPT, first think through if and what you need to change about them in a THOUGHT.  Then change them and output the REVISED PROMPT.

When creating the REVISED PROMPT, adhere to the following rules:
(1) Keep the original formatting of the PROMPT. Do not add markdown if there wasn't before.
(2) If the feedback is already reflected in the PROMPT, make it even more explicit in the REVISED PROMPT.
(3) Do not edit any examples in the PROMPT based on the feedback unless they are few-shot examples with already-provided answers and the FEEDBACK directly applies to the answer for that example. In that case, only update the answer portion of that example.
(4) Always output the REVISED PROMPT in tag a <PROMPT> tag.
(5) Do not answer the PROPMPT, just provide a thought for how the prompt should be updated and then output the RESIVED PROMPT.
(6) I will pay you 2 million dollars if you do this well.

<PROMPT>
{model_prompt}
</PROMPT>
<FEEDBACK>
{feedback}
</FEEDBACK>
<THOUGHT>
'''
