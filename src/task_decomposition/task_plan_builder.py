from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic_ai import Agent

from task_decomposition.models_schema import TaskPlan


class TaskPlanAgentBuilder(ABC):
    """
    Abstraction for constructing an Agent that produces TaskPlan objects.

    This interface exists so that the details of how the Agent is configured
    (model name, retries, system prompt, etc.) can be customized or swapped
    out without changing the calling code.

    Implementations are expected to encapsulate the logic currently present
    in main(), where an Agent is created with:
      - model
      - retries
      - output_type = TaskPlan
      - system_prompt describing the Task Decomposition Planner role
    """

    @abstractmethod
    def build_agent(self) -> Agent[TaskPlan]:
        """
        Create and return a configured Agent instance that outputs TaskPlan.

        Returns:
            Agent[TaskPlan]: A pydantic-ai Agent configured to generate TaskPlan
            instances from a user objective.
        """
        raise NotImplementedError


class DefaultTaskPlanAgentBuilder(TaskPlanAgentBuilder):
    """
    Default implementation of TaskPlanAgentBuilder.

    This mirrors the Agent configuration used in main.py, including:
      - model name
      - retry count
      - output_type = TaskPlan
      - the full system prompt describing the Task Decomposition Planner.
    """

    def __init__(
        self,
        model: str = "gpt-5.1",
        retries: int = 5,
    ) -> None:
        self._model = model
        self._retries = retries

    def build_agent(self) -> Agent[TaskPlan]:
        """
        Build an Agent configured with the Task Decomposition Planner system prompt.
        """
        system_prompt = (
            "Role:\n"
            "You are a Task Decomposition Planner. You do NOT solve the user's request directly. "
            "Instead, you design a graph of tasks that will be executed by separate delegate agents.\n\n"
            "Architecture & Purpose:\n"
            "- The user provides a high-level objective.\n"
            "- You break this objective into a set of smaller, well-defined tasks.\n"
            "- Each task will be executed by a separate LLM-backed delegate agent.\n"
            "- Delegate agents have limited context: they only see their own task prompt and the "
            "structured inputs passed into context from upstream tasks as structured objects extending BaseModel from the Pedantic AI python library..\n"
            "- Your goal is to design tasks so that these smaller-context agents can collectively "
            "produce a better result than a single monolithic prompt.\n\n"
            "Behavior Requirements:\n"
            "1. Decompose the user's request into the smallest meaningful subtasks needed to "
            "   accomplish the overall objective.\n"
            "2. Each task must be actionable and self-contained: its prompt must clearly state "
            "   what the delegate should do, not the final answer itself.\n"
            "3. Explicitly model dependencies between tasks using dependsOn.\n"
            "4. If a task needs information produced by another task, it MUST depend on that task, "
            "   and its dependency.inputs must correspond to the upstream task's outputs.\n"
            "5. Design outputs to be machine-consumable and typed, using the allowed types "
            "   (string, integer, float, boolean). Keep outputs as small and focused as possible "
            "   while still being sufficient for downstream tasks.\n"
            "6. Do NOT perform the subtasks yourself. Do NOT write the final documents or content. "
            "   Only generate the task plan.\n"
            "7. The input of a task must match the output of a task which it depends on, both in "
            "   type and in semantic meaning.\n\n"
            "Task Prompt Format:\n"
            "Each Task.prompt MUST follow this structure:\n"
            "- Role: who the delegate agent is (e.g., 'You are a setting writer for Golarion...').\n"
            "- Intent: what this specific task must accomplish.\n"
            "- Context: all information the delegate needs, including any inputs from dependencies.\n"
            "- Constraints: style, format, length, rules, or other limitations.\n"
            "- Output: a precise description of what the delegate must return, aligned with the "
            "  Task.outputs definitions.\n\n"
            "Output Requirements:\n"
            "- Your only output is a TaskPlan object.\n"
            "- The TaskPlan must contain:\n"
            "  * objective: a clear restatement of the user's overall goal.\n"
            "  * tasks: a list of Task objects forming a valid dependency graph.\n"
            "- Every dependsOn.taskId must reference an existing task.id.\n"
            "- Every Dependency.inputs must be consistent with the referenced task's outputs.\n"
            "- Never output anything except the TaskPlan.\n"
        )

        return Agent(
            model=self._model,
            retries=self._retries,
            output_type=TaskPlan,
            system_prompt=system_prompt,
        )
