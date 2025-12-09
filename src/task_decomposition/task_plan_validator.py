from __future__ import annotations

from typing import Dict, List, Set

from .models import Dependency, Task, TaskPlan


class TaskPlanValidator:
    """
    Validates a TaskPlan instance.

    This class is responsible for checking structural and semantic correctness of
    a TaskPlan, including:

    - Task IDs are unique.
    - Dependencies reference existing tasks.
    - Dependency inputs are compatible with the outputs of the referenced tasks.
    - The dependency graph is acyclic.
    """

    def validate(self, task_plan: TaskPlan) -> bool:
        """
        Validate the given TaskPlan.

        Args:
            task_plan: The TaskPlan instance to validate.

        Returns:
            bool: True if the TaskPlan is considered valid, False otherwise.
        """
        # Empty plan is trivially valid
        if not task_plan.tasks:
            return True

        # Build a mapping from task id to Task, and ensure uniqueness
        task_by_id: Dict[str, Task] = {}
        for task in task_plan.tasks:
            if task.id in task_by_id:
                # Duplicate task IDs are invalid
                return False
            task_by_id[task.id] = task

        # 1. Check that all dependencies reference existing tasks
        if not self._dependencies_reference_existing_tasks(task_plan.tasks, task_by_id):
            return False

        # 2. Check input/output compatibility for each dependency
        if not self._validate_input_output_compatibility(task_plan.tasks, task_by_id):
            return False

        # 3. Check that the dependency graph is acyclic
        if self._has_cycles(task_plan.tasks):
            return False

        return True

    # -------------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------------

    def _dependencies_reference_existing_tasks(
        self, tasks: List[Task], task_by_id: Dict[str, Task]
    ) -> bool:
        """
        Ensure every Dependency.taskId refers to an existing Task.id.
        """
        for task in tasks:
            for dep in task.dependsOn:
                if dep.taskId not in task_by_id:
                    return False
        return True

    def _validate_input_output_compatibility(
        self, tasks: List[Task], task_by_id: Dict[str, Task]
    ) -> bool:
        """
        For each dependency, ensure:

        - The number of inputs equals the number of outputs of the referenced task.
        - Each input.type matches the corresponding output.type.
        """
        for task in tasks:
            for dep in task.dependsOn:
                producer: Task = task_by_id[dep.taskId]
                producer_outputs = producer.outputs
                dep_inputs = dep.inputs

                # Count must match
                if len(producer_outputs) != len(dep_inputs):
                    return False

                # Types must match positionally
                for out, inp in zip(producer_outputs, dep_inputs):
                    if out.type != inp.type:
                        return False

        return True

    def _has_cycles(self, tasks: List[Task]) -> bool:
        """
        Detect cycles in the dependency graph using DFS.

        We treat an edge as: task -> dependency_task (i.e., a task depends on
        another task). A cycle exists if we can reach a node that is already
        on the current DFS recursion stack.
        """
        # Build adjacency list: task_id -> list of task_ids it depends on
        adjacency: Dict[str, List[str]] = {
            task.id: [dep.taskId for dep in task.dependsOn] for task in tasks
        }

        visited: Set[str] = set()
        in_stack: Set[str] = set()

        def dfs(node: str) -> bool:
            if node in in_stack:
                # Found a back edge -> cycle
                return True
            if node in visited:
                return False

            visited.add(node)
            in_stack.add(node)

            for neighbor in adjacency.get(node, []):
                if dfs(neighbor):
                    return True

            in_stack.remove(node)
            return False

        for task in tasks:
            if task.id not in visited:
                if dfs(task.id):
                    return True

        return False
