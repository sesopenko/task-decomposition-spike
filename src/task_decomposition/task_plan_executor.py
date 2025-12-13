from typing import Dict, Any

from task_decomposition.models_schema import TaskPlan, Task
from task_decomposition.task_graph_builder import TaskGraphBuilder, DelegateRunResult
from task_decomposition.delegate_runner import DelegateRunner


class TaskPlanExecutor:
    """
    Executes a TaskPlan in dependency order.

    - Uses TaskGraphBuilder to obtain a topologically sorted list of task IDs.
    - For each task ID, finds the corresponding Task in the TaskPlan.
    - Prepares inputs for the task from dependency results (self.results).
    - Calls a DelegateRunner to execute the task.
    - Stores each result as a DelegateRunResult in self.results, keyed by task ID.
    """

    def __init__(self, task_plan: TaskPlan, delegate_runner: DelegateRunner) -> None:
        self._task_plan = task_plan
        self._builder = TaskGraphBuilder(task_plan)
        self._delegate_runner = delegate_runner
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

            # Execute the task via the delegate runner and store the result
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

        Delegates the actual execution to the injected DelegateRunner instance.
        """
        return self._delegate_runner.run(task, prepared_inputs)
