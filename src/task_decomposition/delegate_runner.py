import json
from dataclasses import dataclass
from typing import Any, Dict, List
import logging
from pathlib import Path
from pprint import pformat

from pydantic_ai import Agent, StructuredDict, Tool, format_as_xml

from task_decomposition.models_schema import Task
from task_decomposition.task_graph_builder import DelegateRunResult

logger = logging.getLogger(__name__)


# Base directory where all run outputs are stored (project_root/output)
OUTPUT_ROOT = Path.cwd() / "output"

# Subdirectory for this specific run, created at startup in main.py
# and set via `set_run_output_dir`. Defaults to OUTPUT_ROOT if not set.
RUN_OUTPUT_DIR: Path = OUTPUT_ROOT


def set_run_output_dir(path: Path) -> None:
    """
    Set the base directory used by the save_file tool for this run.

    This should be called once at system startup (for example, in main.py)
    after creating a timestamped subdirectory under the project-level
    'output' directory.

    Args:
        path: Absolute path to the directory where all relative paths
              passed to save_file will be rooted.
    """
    global RUN_OUTPUT_DIR
    RUN_OUTPUT_DIR = path


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


def save_file(relative_path: str, content: str) -> str:
    """
    Save the given content to a file at the provided relative path.

    The path is interpreted as relative to a per-run output directory.
    Any missing parent directories are created automatically.

    Args:
        relative_path: A relative file path and filename, such as
                       "docs/foo/blah.md".
        content: The text content to write to the file.

    Returns:
        The absolute path of the file that was written, as a string.
    """
    # Normalise and ensure we only work with relative paths
    rel_path = Path(relative_path)
    if rel_path.is_absolute():
        raise ValueError(
            f"save_file tool only accepts relative paths, got absolute path: {relative_path!r}"
        )

    # Ensure the run output directory exists
    RUN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    target_path = RUN_OUTPUT_DIR / rel_path
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(content, encoding="utf-8")

    logger.info(
        "save_file tool wrote %d bytes to %s (run output dir: %s)",
        len(content),
        target_path,
        RUN_OUTPUT_DIR,
    )
    return str(target_path.resolve())


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
        schema_dict = json.loads(task.OutputsToSchema())
        task_output = StructuredDict(
            schema_dict,
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
            "You also have access to a tool named `save_file` that allows you to write\n"
            "text content to files on disk. Use this tool whenever the task's intent\n"
            "or outputs describe creating or updating documents or other file-based\n"
            "artifacts.\n\n"
            "Tool: save_file(relative_path: string, content: string) -> string\n"
            "- `relative_path` is a relative file path and filename, such as\n"
            "  'docs/foo/blah.md'. It must NOT be an absolute path.\n"
            "- `content` is the full text content to write into the file.\n"
            "- The tool will create any missing parent directories automatically.\n"
            "- The tool returns the absolute path of the file that was written.\n\n"
            "Guidance for using save_file:\n"
            "- Choose clear, descriptive relative paths that reflect the purpose of\n"
            "  the file (for example, 'docs/locations/sandpoint_hinterlands.md').\n"
            "- Ensure that the content you pass is exactly what should appear in the\n"
            "  file, including any required formatting such as Markdown.\n"
            "- If the task requires multiple files, call `save_file` once for each\n"
            "  file you need to create.\n\n"
        ) + task.prompt

        agent = Agent(
            "openai:gpt-5.1",
            output_type=task_output,
            system_prompt=system_prompt,
            tools=[Tool(save_file, takes_ctx=False)],
        )
        result = agent.run_sync(prompt)

        # pydantic-ai returns the structured output on `result.output`,
        # mirroring how main.py accesses the TaskPlan result.
        # The StructuredDict output is an object with a `.data` dict that
        # matches the JSON schema keys (item_0, item_1, ...).
        if not hasattr(result, "output"):
            raise RuntimeError(
                f"DelegateRunner.run expected result to have an 'output' attribute, "
                f"got {type(result).__name__} with attributes {dir(result)}"
            )

        structured_output = result.output

        # For a StructuredDict, the actual values are on `.data`.
        # Fall back to treating `structured_output` as a mapping if `.data`
        # is not present, to be defensive against version differences.
        if hasattr(structured_output, "data"):
            output_mapping = structured_output.data  # type: ignore[attr-defined]
        else:
            if not isinstance(structured_output, dict):
                raise RuntimeError(
                    "DelegateRunner.run expected structured_output to have a 'data' "
                    "attribute or be a dict-like mapping, "
                    f"got {type(structured_output).__name__}"
                )
            output_mapping = structured_output

        # Ensure deterministic ordering that matches the schema: item_0, item_1, ...
        expected_count = len(task.outputs)
        expected_keys = [f"item_{i}" for i in range(expected_count)]
        outputs: List[Any] = []

        for key in expected_keys:
            if key not in output_mapping:
                raise RuntimeError(
                    f"DelegateRunner.run: missing expected key '{key}' in delegate "
                    f"output for task '{task.id}'. Available keys: {list(output_mapping.keys())}"
                )
            outputs.append(output_mapping[key])

        return DelegateRunResult(
            id=task.id,
            output_types=[o.type for o in task.outputs],
            outputs=outputs,
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
