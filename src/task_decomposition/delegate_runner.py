from abc import ABC, abstractmethod
from typing import Any, Dict, List

from task_decomposition.models import Task, Output
from task_decomposition.task_graph_builder import DelegateRunResult


class DelegateRunner(ABC):
    """
    Abstraction for executing a single Task with prepared inputs.

    This interface exists to support dependency inversion: TaskPlanExecutor
    depends only on this abstraction, so concrete implementations that talk
    to an LLM or other agents can be swapped out or mocked in tests.
    """

    @abstractmethod
    def run(self, task: Task, prepared_inputs: List[Output]) -> DelegateRunResult:
        """
        Execute the given Task using the provided prepared_inputs and return
        a DelegateRunResult.

        Implementations are responsible for:
        - Constructing the actual prompt / tool call from `task` and `prepared_inputs`
        - Executing the delegate agent (e.g., LLM, tool, service)
        - Mapping the raw outputs into a DelegateRunResult that matches
          the task's declared outputs.
        """
        raise NotImplementedError
