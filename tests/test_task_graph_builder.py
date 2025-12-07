import pytest

from task_decomposition.models import TaskPlan, Task, Dependency, Input, Output
from task_decomposition.task_graph_builder import TaskGraphBuilder


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
