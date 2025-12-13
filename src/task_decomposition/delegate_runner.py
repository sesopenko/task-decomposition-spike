from dataclasses import dataclass
from typing import Any, Dict, List
import logging
from pprint import pformat

from pydantic_ai import Agent, StructuredDict, format_as_xml

from task_decomposition.models_schema import Task
from task_decomposition.task_graph_builder import DelegateRunResult

logger = logging.getLogger(__name__)


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
        Execute the given Task using the provided delegate_context and return
        a DelegateRunResult.

        Implementations are responsible for:
        - Constructing the actual prompt / tool call from `task` and `delegate_context`
        - Executing the delegate agent (e.g., LLM, tool, service)
        - Mapping the raw outputs into a DelegateRunResult that matches
          the task's declared outputs.
        """
        task_output = StructuredDict(
            task.OutputsToSchema(),
            name="OutputSpecification",
        )

        # Build a rich, structured prompt that explains:
        # - Which dependency each value came from (by task id)
        # - The description of each output from the producing task
        # - The actual value produced
        # - How the current task describes the inputs it expects from each dependency
        prompt_dict = self.build_prompt_dict(delegate_context, task)

        prompt = format_as_xml(prompt_dict)

        system_prompt = (
            "You are executing a task in a dependency graph.\n"
            "You are given the current task and the results of its dependency tasks.\n"
            "Use the dependency outputs, guided by their descriptions and the\n"
            "current task's input descriptions, as inputs to complete the current\n"
            "task. Return your answer strictly according to the OutputSpecification.\n\n"
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

    def build_prompt_dict(self, delegate_context: DelegateContext, task: Task) -> dict[str, Any]:
        """
        Build a structured dictionary that will be converted to XML and sent
        to the delegate agent.

        It includes:
        - The current task metadata.
        - For each dependency:
          - The dependency task id.
          - Any available run result outputs, with their descriptions and types.
          - The current task's declared input descriptions for that dependency.
        """
        prompt_dict: Dict[str, Any] = {
            "current_task": {
                "id": task.id,
                "description": "This is the task you must now execute.",
            },
            "dependencies": [],
        }

        dependencies_list: List[Dict[str, Any]] = []

        # Build a quick lookup from dependency taskId -> list of Input models
        # as declared on the *current* task.
        dependency_input_specs: Dict[str, List[Any]] = {}
        for dep in task.dependsOn:
            dependency_input_specs.setdefault(dep.taskId, []).extend(dep.inputs)

        for dep_task_id, dep_task in delegate_context.dependency_tasks.items():
            dep_result = delegate_context.dependency_results.get(dep_task_id)

            # Map the current task's declared inputs for this dependency, if any.
            input_specs_for_dep = dependency_input_specs.get(dep_task_id, [])
            mapped_inputs: List[Dict[str, Any]] = []
            for index, input_spec in enumerate(input_specs_for_dep):
                mapped_inputs.append(
                    {
                        "index": index,
                        "description": input_spec.description,
                        "declared_type": input_spec.type,
                    }
                )

            if dep_result is None:
                # If we have a task but no result, just record that fact for the agent,
                # along with how the current task *expects* to use this dependency.
                dependencies_list.append(
                    {
                        "task_id": dep_task_id,
                        "note": "No run result is available for this dependency. It may not have had an output.",
                        "expected_inputs_from_this_dependency": mapped_inputs,
                    }
                )
                continue

            # Pair each declared output with the corresponding value from the run result.
            # We rely on DelegateRunResult.__post_init__ to have validated lengths/types.
            outputs_with_context: List[Dict[str, Any]] = []
            for index, output_spec in enumerate(dep_task.outputs):
                value = dep_result.outputs[index] if index < len(dep_result.outputs) else None
                outputs_with_context.append(
                    {
                        "index": index,
                        "description": output_spec.description,
                        "declared_type": output_spec.type,
                        "value": value,
                    }
                )

            dependencies_list.append(
                {
                    "task_id": dep_task_id,
                    "description": "Outputs from a dependency task that you may use as inputs.",
                    "outputs": outputs_with_context,
                    "expected_inputs_from_this_dependency": mapped_inputs,
                }
            )

        prompt_dict["dependencies"] = dependencies_list

        # Log the final prompt_dict for debugging/inspection in a human-readable way.
        logger.debug(
            "DelegateRunner.build_prompt_dict for task %s:\n%s",
            task.id,
            pformat(prompt_dict),
        )

        return prompt_dict
