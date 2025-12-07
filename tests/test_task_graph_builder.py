import pytest

from task_decomposition.models import TaskPlan, Task, Dependency, Input, Output
from task_decomposition.task_graph_builder import TaskGraphBuilder, DelegateRunResult


def make_input(description: str = "desc", type_: str = "string") -> Input:
    return Input(description=description, type=type_)


def make_output(description: str = "desc", type_: str = "string") -> Output:
    return Output(description=description, type=type_)


def test_get_sorted_id():
    """
    For now TaskGraphBuilder.get_sorted_id_list is intentionally not implemented.
    This test documents the current behavior (raising an Exception) so we have a
    failing test once we start implementing it in the next step.
    """
    # Build a simple, valid TaskPlan with three tasks and linear dependencies:
    # T1 -> T2 -> T3
    task1 = Task(
        id="T1",
        prompt="Role: Test\nIntent: Task 1\nContext:\nConstraints:\nOutput:",
        dependsOn=[],
        inputs=[],
        outputs=[make_output("Output of T1")],
    )

    task2 = Task(
        id="T2",
        prompt="Role: Test\nIntent: Task 2\nContext:\nConstraints:\nOutput:",
        dependsOn=[
            Dependency(
                taskId="T1",
                inputs=[make_input("Input from T1")],
            )
        ],
        inputs=[make_input("Input from T1")],
        outputs=[make_output("Output of T2")],
    )

    task3 = Task(
        id="T3",
        prompt="Role: Test\nIntent: Task 3\nContext:\nConstraints:\nOutput:",
        dependsOn=[
            Dependency(
                taskId="T2",
                inputs=[make_input("Input from T2")],
            )
        ],
        inputs=[make_input("Input from T2")],
        outputs=[make_output("Output of T3")],
    )

    plan = TaskPlan(
        objective="Test objective",
        tasks=[task1, task2, task3],
    )

    builder = TaskGraphBuilder(plan)

    # At this stage of "red, green, refactor" we expect the method to raise,
    # because it's explicitly not implemented yet.
    sorted: list[str] = builder.get_sorted_id_list()

    # Once implemented, we expect a topologically sorted list that respects
    # the dependencies T1 -> T2 -> T3.
    assert sorted == ["T1", "T2", "T3"]


def test_delegate_run_result_valid():
    result = DelegateRunResult(
        id="run-1",
        output_types=["string", "integer", "float"],
        outputs=["hello", 42, 3.14],
    )

    assert result.id == "run-1"
    assert result.output_types == ["string", "integer", "float"]
    assert result.outputs == ["hello", 42, 3.14]


def test_delegate_run_result_invalid_length_mismatch():
    with pytest.raises(ValueError):
        DelegateRunResult(
            id="run-2",
            output_types=["string", "integer"],
            outputs=["only-one-value"],
        )


@pytest.mark.parametrize(
    "output_types, outputs",
    [
        (["string"], [123]),          # not a string
        (["integer"], ["not-int"]),   # not an int
        (["float"], ["not-float"]),   # not a float/int
        (["integer"], [True]),        # bool should be rejected for integer
        (["float"], [False]),         # bool should be rejected for float
    ],
)
def test_delegate_run_result_invalid_type_mismatch(output_types, outputs):
    with pytest.raises(TypeError):
        DelegateRunResult(
            id="run-3",
            output_types=output_types,
            outputs=outputs,
        )
