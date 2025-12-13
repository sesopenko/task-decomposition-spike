from typing import Dict

from task_decomposition.models_schema import TaskPlan, Task
from task_decomposition.task_graph_builder import TaskGraphBuilder, DelegateRunResult
from task_decomposition.delegate_runner import DelegateRunner, DelegateContext


class TaskPlanExecutor:
    """
    Executes a TaskPlan in dependency order.

    - Uses TaskGraphBuilder to obtain a topologically sorted list of task IDs.
    - For each task ID, finds the corresponding Task in the TaskPlan.
    - Prepares a DelegateContext for the task from dependency results (self.results).
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

            # Build DelegateContext for this task based on its dependencies
            delegate_context = self._build_delegate_context(task, tasks_by_id)

            # Execute the task via the delegate runner and store the result
            result = self.run(task, delegate_context)
            if not isinstance(result, DelegateRunResult):
                raise TypeError(
                    f"run() must return a DelegateRunResult, got {type(result).__name__}"
                )
            self.results[task_id] = result

    def _build_delegate_context(
        self,
        task: Task,
        tasks_by_id: Dict[str, Task],
    ) -> DelegateContext:
        """
        Build a DelegateContext for `task` based on its declared dependencies
        and the previously stored DelegateRunResult objects in self.results.

        - dependency_tasks: all Task objects this task depends on, keyed by taskId.
        - dependency_results: the corresponding DelegateRunResult objects, keyed by taskId.

        Raises KeyError if a dependency task or its result cannot be found.
        """
        dependency_tasks: Dict[str, Task] = {}
        dependency_results: Dict[str, DelegateRunResult] = {}

        for dep in task.dependsOn:
            dep_id = dep.taskId

            # Look up the dependency Task definition
            dep_task = tasks_by_id.get(dep_id)
            if dep_task is None:
                raise KeyError(
                    f"Dependency task with id {dep_id!r} not found in TaskPlan when "
                    f"building DelegateContext for task {task.id!r}"
                )
            dependency_tasks[dep_id] = dep_task

            # Look up the dependency run result, if it exists
            if dep_id not in self.results:
                raise KeyError(
                    f"Dependency result for task {dep_id!r} not found in executor results "
                    f"when building DelegateContext for task {task.id!r}"
                )
            dependency_results[dep_id] = self.results[dep_id]

        return DelegateContext(
            dependency_tasks=dependency_tasks,
            dependency_results=dependency_results,
        )

    def run(self, task: Task, delegate_context: DelegateContext) -> DelegateRunResult:
        """
        Execute a single Task and return a DelegateRunResult.

        Delegates the actual execution to the injected DelegateRunner instance.
        """
        return self._delegate_runner.run(task, delegate_context)
