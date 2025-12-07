from typing import List, Literal

from pydantic import BaseModel, Field

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider

from cost_calculator import calculate_cost

InputOutputType = Literal["string", "integer", "float", "boolean"]


class Input(BaseModel):
    description: str = Field(
        ...,
        description="A description of the input which helps the prompted LLM understand what's is being input",
    )
    type: InputOutputType = Field(
        ...,
        description="The basic type of the input, used for validation of the input",
    )

class Output(BaseModel):
    description: str = Field(
        ...,
        description="A description of the input which helps the prompted LLM understand what's required to output"
    )
    type: InputOutputType = Field(
        ...,
        description="The basic type of the input, used for validation of the output",
    )

class Task(BaseModel):
    id: str = Field(
        ...,
        description="A unique identifier for the task, referred to by dependencies."
    )
    prompt: str = Field(
        ...,
        description="The LLM prompt for the agent to run. Must follow the format: Role:, Intent:, Context:, Constraints:, Output:"
    )
    dependsOn: List[str] = Field(
        default_factory=list,
        description="A list of task ids this task depends on. Used to build a dependency tree and ensures tasks are ran in the correct order of the graph"
    )
    inputs: List[Input] = Field(
        default_factory=list,
        description="A list of input parameters required by this task. Empty if this task doesn't have a dependency. Must match the Output of the dependant task."
    )
    outputs: List[Output] = Field(
        default_factory=list,
        description="A list of output properties output by this task. Empty if this task doesn't have any output (ie: calls a tool). Inputs of tasks must match this task's output."
    )


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

Must produce 1 document for each location.

## Output:
Each location must have the following sections in the document:

* Location Overview. Purpose: provide a quick, evocative summary
* Geography & Environment. Purpose: Ground the location int he world's physical reality
* Notable Features. Purpose: Identify key areas a party may explore.

Each location's document must be formatted with markdown.

    """)
    usage = result.usage()
    print(calculate_cost(usage))

    plan: TaskPlan = result.output
    print(f"Objective: {result.output.objective}")
    print()
    for task in plan.tasks:
        print(f"Task: {task.id}")
        print(f"Prompt:")
        print("```")
        print(task.prompt)
        print("```")
        for dep in task.dependsOn:
            print(f"Depends on: {dep}")
        for input in task.inputs:
            print(f"Input ({input.type}): {input.description}")
        for output in task.outputs:
            print(f"Output ({output.type}): {output.description}")
        print()




if __name__ == "__main__":
    main()
