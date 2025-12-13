from dataclasses import dataclass
from typing import Any, Dict

from pydantic_ai import Agent, StructuredDict, format_as_xml

from task_decomposition.models_schema import Task
from task_decomposition.task_graph_builder import DelegateRunResult


@dataclass
class DelegateContext:
    """
    Context required by a DelegateRunner to form its inputs for a task run.

    Attributes:
        dependency_tasks: A mapping from task ID to the corresponding Task
            object for all tasks that this task depends on.
        dependency_results: A mapping from task ID to the DelegateRunResult
            produced when executing that dependency task.
    """
    dependency_tasks: Dict[str, Task]
    dependency_results: Dict[str, DelegateRunResult]


class DelegateRunner:
    """
    Abstraction for executing a single Task with prepared inputs.

    This interface exists to support dependency inversion: TaskPlanExecutor
    depends only on this abstraction, so concrete implementations that talk
    to an LLM or other agents can be swapped out or mocked in tests.
    """

    def run(self, task: Task, delegate_context: DelegateContext) -> DelegateRunResult:
        """
        Execute the given Task using the provided prepared_inputs and return
        a DelegateRunResult.

        Implementations are responsible for:
        - Constructing the actual prompt / tool call from `task` and `prepared_inputs`
        - Executing the delegate agent (e.g., LLM, tool, service)
        - Mapping the raw outputs into a DelegateRunResult that matches
          the task's declared outputs.
        """
        task_output = StructuredDict(
            task.OutputsToSchema(),
            name="OutputSpecification",
        )

        # TODO: Use DelegateContext and prepared_inputs to build a richer prompt.
        prompt_dict: Dict[str, Any] = {}


        prompt = format_as_xml(prompt_dict)

        system_prompt = (
            "Execute the given task using the provided inputs and return according "
            "to the OutputSpecification.\n"
        ) + task.prompt

        agent = Agent(
            "openai:gpt-5.1",
            output_type=task_output,
            system_prompt=system_prompt,
        )
        result = agent.run_sync(prompt)

        # NOTE: This is a placeholder; actual mapping from `result` to
        # DelegateRunResult should be implemented as needed.
        return DelegateRunResult(
            id=task.id,
            output_types=[o.type for o in task.outputs],
            outputs=list(result.data.values()) if hasattr(result, "data") else [],
        )
