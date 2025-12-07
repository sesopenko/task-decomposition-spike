from typing import List, Literal

from pydantic import BaseModel, Field


DataType = Literal["string", "integer", "float", "boolean"]


class Input(BaseModel):
    description: str = Field(
        ...,
        description="A description of the input which helps the prompted LLM understand what's is being input",
    )
    type: DataType = Field(
        ...,
        description="The basic type of the input, used for validation of the input",
    )


class Output(BaseModel):
    description: str = Field(
        ...,
        description="A description of the input which helps the prompted LLM understand what's required to output"
    )
    type: DataType = Field(
        ...,
        description="The basic type of the input, used for validation of the output",
    )


class Dependency(BaseModel):
    taskId: str = Field(
        ...,
        description="The id of the task for this dependency",
    )
    inputs: List[Input] = Field(
        default_factory=list,
        description="A list of inputs required by the dependency"
    )


class Task(BaseModel):
    id: str = Field(
        ...,
        description="A unique identifier for the task, referred to by dependencies."
    )
    prompt: str = Field(
        ...,
        description="The LLM prompt for the agent to run. Must follow the format: Role:, Intent:, Context:, Constraints:, Output:, must explain the dependencies and outputs."
    )
    dependsOn: List[Dependency] = Field(
        default_factory=list,
        description="A list of tasks this task depends on and their outputs. Used to build a dependency tree and ensures tasks are ran in the correct order of the graph"
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
    objective: str = Field(
        ...,
        description="The overall objective of the task set"
    )
    tasks: List[Task] = Field(
        default_factory=list,
        description="The tasks which must be complete to complete the overall task. Will form a graph of prompts executed by agents and fed into dependant tasks."
    )
