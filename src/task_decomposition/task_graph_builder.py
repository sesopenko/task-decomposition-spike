from dataclasses import dataclass, field
from typing import List, Literal, Union
from graphlib import TopologicalSorter

from task_decomposition.models import TaskPlan


OutputPrimitiveType = Literal["string", "integer", "float"]
OutputPrimitiveValue = Union[str, int, float]


@dataclass(frozen=True)
class DelegateRunResult:
    """
    Immutable container for the result of a delegate agent run.

    Attributes:
        id: A unique identifier for this run result.
        output_types: A list describing the type of each corresponding value in `outputs`.
        outputs: The list of output values produced by the delegate agent.
    """
    id: str
    output_types: List[OutputPrimitiveType] = field()
    outputs: List[OutputPrimitiveValue] = field()

    def __post_init__(self) -> None:
        # Ensure lengths match
        if len(self.output_types) != len(self.outputs):
            raise ValueError(
                f"output_types length ({len(self.output_types)}) does not match "
                f"outputs length ({len(self.outputs)})"
            )

        # Validate each output against its declared type
        for index, (declared_type, value) in enumerate(zip(self.output_types, self.outputs)):
            if declared_type == "string":
                if not isinstance(value, str):
                    raise TypeError(
                        f"Output at index {index} expected type 'string' but got "
                        f"{type(value).__name__}: {value!r}"
                    )
            elif declared_type == "integer":
                # Reject bools, which are subclasses of int
                if not (isinstance(value, int) and not isinstance(value, bool)):
                    raise TypeError(
                        f"Output at index {index} expected type 'integer' but got "
                        f"{type(value).__name__}: {value!r}"
                    )
            elif declared_type == "float":
                # Allow ints to be treated as valid floats
                if not (isinstance(value, (float, int)) and not isinstance(value, bool)):
                    raise TypeError(
                        f"Output at index {index} expected type 'float' but got "
                        f"{type(value).__name__}: {value!r}"
                    )
            else:
                # This should be unreachable due to the Literal type, but guard anyway
                raise ValueError(f"Unsupported output type: {declared_type!r}")


class TaskGraphBuilder:
    _taskPlan: TaskPlan

    def __init__(self, task_plan: TaskPlan):
        self._taskPlan = task_plan

    def get_sorted_id_list(self):
        sorted_id_list: list[str] = []
        topology = {}
        for task in self._taskPlan.tasks:
            deps: list[str] = []
            for dep in task.dependsOn:
                deps.append(dep.taskId)
            topology[task.id] = deps
        ts: TopologicalSorter = TopologicalSorter(topology)
        sorted_id_list = list(ts.static_order())
        return sorted_id_list
