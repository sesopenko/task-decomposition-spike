from typing import Any, Dict

from task_decomposition.delegate_runner import DelegateRunner
from task_decomposition.models_schema import Task
from task_decomposition.task_graph_builder import DelegateRunResult


class NoOpDelegateRunner(DelegateRunner):
    """
    A minimal concrete implementation of DelegateRunner with an empty body.

    This is intended as a development stub so that other components can be
    wired up and tested without needing a real LLM or external service.
    """

    def run(self, task: Task, prepared_inputs: Dict[str, Any]) -> DelegateRunResult:
        """
        Execute the given Task using the provided prepared_inputs.

        Current implementation is a stub and must be filled in with real logic.
        """
        # TODO: Implement delegate execution logic.
        # For now, this raises NotImplementedError to make it obvious at runtime.
        raise NotImplementedError("NoOpDelegateRunner.run is not yet implemented")
