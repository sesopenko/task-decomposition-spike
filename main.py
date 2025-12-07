from typing import List, Literal

from pydantic import BaseModel

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider

class TaskInput(BaseModel):
    jsonschema: str

InputOutputType = Literal["string", "integer", "float", "boolean"]

class Input(BaseModel):
    id: str
    Type: InputOutputType

class Output(BaseModel):
    id: str
    Type: InputOutputType

class Task(BaseModel):
    id: str
    prompt: str
    dependsOn: List[str]
    Input:  List[TaskInput]
    Output: List[Output]


class TaskPlan(BaseModel):
    objective: str
    tasks: List[Task]
    notes: str

def main():
    ollama_model = OpenAIChatModel(
        model_name="mistral-nemo:12b",
        provider=OllamaProvider(base_url="http://localhost:11434/v1"),
    )

    agent = Agent(
        model='gpt-5.1',
        output_type=TaskPlan,
        system_prompt=(
            'Role:'
            'You are a Task Decomposition Planner. Your job is to transform any user request—simple or complex—into a clear, complete, and logically ordered set of subtasks.'
            ''
            'Behavior Requirements:'
            '1. Break the user’s request into the smallest meaningful subtasks needed to accomplish the goal.'
            '2. Ensure the subtasks are actionable—each should describe what needs to be done, not how the final answer should look.'
            '3. Order the subtasks logically, showing dependencies when relevant.'
            '4. Identify required inputs or missing information, and include subtasks for gathering them.'
            '5. Do not perform the subtasks yourself. Only generate the plan.'
            '6. Tasks that depend on other tasks must be clearly defined via dependsOn for the given task'
            '7. The input of a task must match the output of a task which it depends on.'
            
            'Primary Output:'
            'A fully decomposed task plan'
            'Never output anything except the task plan'
        ),
    )
    result = agent.run_sync("""
## Role:
You are an assistant Game Master and Game Source Material Writer for the Pathfinder Remastered Role Playing Game.  You are familiar with the world of Golarion.
## Intent:
Generate source material for the Sandpoint Hinterlands
## Context:
The players are currently playing a conversion of the old Rise of the Runelords Adventure Path, converted to Pathfinder Remastered.
They are playing in the Sandpoint Hinterlands.
Locations of the sandpoint hinterlands:

* The Pyre
* The Three Cormorants
* The Old Light
* Tickwood
* Shank's Wood

## Control:

## Outcome:
Each location must have the following properties:

* Location Overview. Purpose: provide a quick, evocative summary
* Geography & Environment. Purpose: Ground the location int he world's physical reality
* Notable Features. Purpose: Identify key areas a party may explore.

Each location's property must be formatted with markdown.
    """)
    print(result.usage())
    plan: TaskPlan = result.output
    print(f"Objective: {result.output.objective}")
    for task in plan.tasks:
        print(f"Task: {task.id}")
        print(f"Prompt: {task.prompt}")
        print()




if __name__ == "__main__":
    main()
