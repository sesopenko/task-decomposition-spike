from typing import Dict, List, Any

from task_decomposition.models import TaskPlan, Task, Dependency, Input
from task_decomposition.task_graph_builder import TaskGraphBuilder, DelegateRunResult


class TaskPlanExecutor:
    """
    Executes a TaskPlan in dependency order.

    - Uses TaskGraphBuilder to obtain a topologically sorted list of task IDs.
    - For each task ID, finds the corresponding Task in the TaskPlan.
    - Prepares inputs for the task from dependency results (self.results).
    - Calls self.run(task, prepared_inputs) to execute the task.
    - Stores each result as a DelegateRunResult in self.results, keyed by task ID.
    """

    def __init__(self, task_plan: TaskPlan) -> None:
        self._task_plan = task_plan
        self._builder = TaskGraphBuilder(task_plan)
        self.results: Dict[str, DelegateRunResult] = {}

    def execute(self) -> None:
        """
        Execute all tasks in the TaskPlan in topological order.

        The results are stored in self.results, keyed by task.id.
        """
        sorted_ids = self._builder.get_sorted_id_list()

        # Build a quick lookup from id -> Task
        tasks_by_id: Dict[str, Task] = {task.id: task for task in self._task_plan.tasks}

        for task_id in sorted_ids:
            task = tasks_by_id.get(task_id)
            if task is None:
                raise KeyError(f"Task with id {task_id!r} not found in TaskPlan")

            # Prepare inputs from dependency results
            prepared_inputs = self._prepare_inputs_from_dependencies(task)

            # Execute the task and store the result
            result = self.run(task, prepared_inputs)
            if not isinstance(result, DelegateRunResult):
                raise TypeError(
                    f"run() must return a DelegateRunResult, got {type(result).__name__}"
                )
            self.results[task_id] = result

    def _prepare_inputs_from_dependencies(self, task: Task) -> Dict[str, Any]:
        """
        Build a structure of inputs for `task` based on its dependencies and
        previously stored DelegateRunResult objects in self.results.

        For now this is a simple stub that demonstrates how dependency results
        would be gathered and made available to `run(...)`.

        Returns a dict keyed by dependency taskId, each value being the
        corresponding DelegateRunResult.
        """
        dependency_inputs: Dict[str, DelegateRunResult] = {}

        for dep in task.dependsOn:
            if dep.taskId not in self.results:
                raise KeyError(
                    f"Dependency result for task {dep.taskId!r} not found in executor results"
                )
            dependency_inputs[dep.taskId] = self.results[dep.taskId]

        return dependency_inputs

    def run(self, task: Task, prepared_inputs: Dict[str, Any]) -> DelegateRunResult:
        """
        Execute a single Task and return a DelegateRunResult.

        `prepared_inputs` contains data derived from this task's dependencies,
        gathered from self.results by _prepare_inputs_from_dependencies().

        This is currently a stub implementation that returns an empty
        DelegateRunResult for the given task.
        """
        # NOTE: In a real implementation, you would use `task`, `prepared_inputs`,
        # and possibly `task.inputs` / `task.outputs` metadata to construct the
        # actual LLM/tool call and map outputs back into a DelegateRunResult.
        return DelegateRunResult(
            id=task.id,
            output_types=[],
            outputs=[],
        )
