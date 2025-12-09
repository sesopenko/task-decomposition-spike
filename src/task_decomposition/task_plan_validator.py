from .models import TaskPlan


class TaskPlanValidator:
    """
    Interface for validating a TaskPlan instance.

    This class is responsible for checking structural and semantic correctness of
    a TaskPlan, such as (non-exhaustive examples only, not yet implemented):

    - Task IDs are unique.
    - Dependencies reference existing tasks.
    - Dependency inputs are compatible with the outputs of the referenced tasks.
    - The dependency graph is valid (e.g., no cycles).
    - Tasks have prompts and outputs consistent with their declared interfaces.
    """

    def validate(self, task_plan: TaskPlan) -> bool:
        """
        Validate the given TaskPlan.

        Args:
            task_plan: The TaskPlan instance to validate.

        Returns:
            bool: True if the TaskPlan is considered valid, False otherwise.

        Note:
            The actual validation logic is intentionally not implemented yet.
        """
        # not implemented
        return True
