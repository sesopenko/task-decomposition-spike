from typing import Any, Dict

from task_decomposition.delegate_runner import DelegateRunner
from task_decomposition.models_schema import Task
from task_decomposition.task_graph_builder import DelegateRunResult


class DelegateRunnerAgent(DelegateRunner):
    """
    """

    def __init__(self, model: str = "gpt-5.1", retries: int = 5) -> None:
        """
        Create a DelegateRunnerAgent.

        Args:
            model: The model identifier to be used by the underlying agent.
            retries: How many times to retry the delegate call on failure.
        """
        self.model = model
        self.retries = retries


    def run(self, task: Task, prepared_inputs: Dict[str, Any]) -> DelegateRunResult:
        """
        Execute the given Task using the provided prepared_inputs.

        Current implementation is a stub and must be filled in with real logic.
        """
        # TODO: Implement delegate execution logic.
        # For now, this raises NotImplementedError to make it obvious at runtime.
        raise NotImplementedError("DelegateRunnerAgent.run is not yet implemented")
