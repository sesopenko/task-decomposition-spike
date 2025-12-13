import pytest

from task_decomposition.models_schema import TaskPlan, Task, Dependency, Input, Output
from task_decomposition.task_plan_validator import TaskPlanValidator


class TestTaskPlanValidator:
    """
    Tests for TaskPlanValidator.

    These tests define the expected behaviour of TaskPlanValidator before its
    implementation (red/green/refactor). They assume:

    - validate(...) returns True when the TaskPlan is structurally valid.
    - validate(...) returns False when the TaskPlan is invalid in any of the
      ways described by the tests.
    """

    def setup_method(self) -> None:
        self.validator = TaskPlanValidator()

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _make_input(description: str = "desc", type_: str = "string") -> Input:
        return Input(description=description, type=type_)

    @staticmethod
    def _make_output(description: str = "desc", type_: str = "string") -> Output:
        return Output(description=description, type=type_)

    @staticmethod
    def _make_task(
        id_: str,
        prompt: str = "Role: X\nIntent: Y\nContext: Z\nConstraints: C\nOutput: O\n",
        depends_on: list[Dependency] | None = None,
        outputs: list[Output] | None = None,
    ) -> Task:
        return Task(
            id=id_,
            prompt=prompt,
            dependsOn=depends_on or [],
            outputs=outputs or [],
        )

    # -------------------------------------------------------------------------
    # Cyclic dependency tests
    # -------------------------------------------------------------------------

    @pytest.mark.parametrize(
        "name, task_plan, expected_valid",
        [
            pytest.param(
                "no_tasks_is_trivially_acyclic",
                TaskPlan(objective="empty", tasks=[]),
                True,
                id="no_tasks_is_trivially_acyclic",
            ),
            pytest.param(
                "single_task_no_dependencies",
                TaskPlan(
                    objective="single",
                    tasks=[
                        _make_task.__func__("t1"),  # type: ignore[attr-defined]
                    ],
                ),
                True,
                id="single_task_no_dependencies",
            ),
            pytest.param(
                "two_tasks_linear_dependency",
                TaskPlan(
                    objective="linear",
                    tasks=[
                        _make_task.__func__("t1"),  # type: ignore[attr-defined]
                        _make_task.__func__(
                            "t2",
                            depends_on=[
                                Dependency(taskId="t1", inputs=[]),
                            ],
                        ),  # type: ignore[attr-defined]
                    ],
                ),
                True,
                id="two_tasks_linear_dependency",
            ),
            pytest.param(
                "simple_two_node_cycle",
                TaskPlan(
                    objective="cycle",
                    tasks=[
                        _make_task.__func__(
                            "t1",
                            depends_on=[Dependency(taskId="t2", inputs=[])],
                        ),  # type: ignore[attr-defined]
                        _make_task.__func__(
                            "t2",
                            depends_on=[Dependency(taskId="t1", inputs=[])],
                        ),  # type: ignore[attr-defined]
                    ],
                ),
                False,
                id="simple_two_node_cycle",
            ),
            pytest.param(
                "three_node_cycle",
                TaskPlan(
                    objective="3-cycle",
                    tasks=[
                        _make_task.__func__(
                            "t1",
                            depends_on=[Dependency(taskId="t2", inputs=[])],
                        ),  # type: ignore[attr-defined]
                        _make_task.__func__(
                            "t2",
                            depends_on=[Dependency(taskId="t3", inputs=[])],
                        ),  # type: ignore[attr-defined]
                        _make_task.__func__(
                            "t3",
                            depends_on=[Dependency(taskId="t1", inputs=[])],
                        ),  # type: ignore[attr-defined]
                    ],
                ),
                False,
                id="three_node_cycle",
            ),
            pytest.param(
                "diamond_shape_no_cycle",
                TaskPlan(
                    objective="diamond",
                    tasks=[
                        _make_task.__func__("root"),  # type: ignore[attr-defined]
                        _make_task.__func__(
                            "left",
                            depends_on=[Dependency(taskId="root", inputs=[])],
                        ),  # type: ignore[attr-defined]
                        _make_task.__func__(
                            "right",
                            depends_on=[Dependency(taskId="root", inputs=[])],
                        ),  # type: ignore[attr-defined]
                        _make_task.__func__(
                            "leaf",
                            depends_on=[
                                Dependency(taskId="left", inputs=[]),
                                Dependency(taskId="right", inputs=[]),
                            ],
                        ),  # type: ignore[attr-defined]
                    ],
                ),
                True,
                id="diamond_shape_no_cycle",
            ),
        ],
    )
    def test_cyclic_dependencies(
        self, name: str, task_plan: TaskPlan, expected_valid: bool
    ) -> None:
        """
        TaskPlanValidator should return False when the dependency graph
        contains any cycle, and True when it is acyclic.
        """
        result = self.validator.validate(task_plan)
        assert result is expected_valid

    # -------------------------------------------------------------------------
    # Input/output compatibility tests
    # -------------------------------------------------------------------------

    @pytest.mark.parametrize(
        "name, task_plan, expected_valid",
        [
            pytest.param(
                "valid_matching_counts_and_types",
                TaskPlan(
                    objective="valid-io",
                    tasks=[
                        _make_task.__func__(
                            "producer_valid",
                            outputs=[
                                _make_output.__func__(type_="string"),
                                _make_output.__func__(type_="integer"),
                            ],
                        ),  # type: ignore[attr-defined]
                        _make_task.__func__(
                            "consumer_valid",
                            depends_on=[
                                Dependency(
                                    taskId="producer_valid",
                                    inputs=[
                                        _make_input.__func__(type_="string"),
                                        _make_input.__func__(type_="integer"),
                                    ],
                                )
                            ],
                        ),  # type: ignore[attr-defined]
                    ],
                ),
                True,
                id="valid_matching_counts_and_types",
            ),
            pytest.param(
                "invalid_fewer_inputs_than_outputs",
                TaskPlan(
                    objective="fewer-inputs",
                    tasks=[
                        _make_task.__func__(
                            "producer_more_outputs",
                            outputs=[
                                _make_output.__func__(type_="string"),
                                _make_output.__func__(type_="integer"),
                            ],
                        ),  # type: ignore[attr-defined]
                        _make_task.__func__(
                            "consumer_fewer_inputs",
                            depends_on=[
                                Dependency(
                                    taskId="producer_more_outputs",
                                    inputs=[
                                        _make_input.__func__(type_="string"),
                                    ],
                                )
                            ],
                        ),  # type: ignore[attr-defined]
                    ],
                ),
                False,
                id="invalid_fewer_inputs_than_outputs",
            ),
            pytest.param(
                "invalid_more_inputs_than_outputs",
                TaskPlan(
                    objective="more-inputs",
                    tasks=[
                        _make_task.__func__(
                            "producer_fewer_outputs",
                            outputs=[
                                _make_output.__func__(type_="string"),
                            ],
                        ),  # type: ignore[attr-defined]
                        _make_task.__func__(
                            "consumer_more_inputs",
                            depends_on=[
                                Dependency(
                                    taskId="producer_fewer_outputs",
                                    inputs=[
                                        _make_input.__func__(type_="string"),
                                        _make_input.__func__(type_="integer"),
                                    ],
                                )
                            ],
                        ),  # type: ignore[attr-defined]
                    ],
                ),
                False,
                id="invalid_more_inputs_than_outputs",
            ),
            pytest.param(
                "invalid_mismatched_types",
                TaskPlan(
                    objective="type-mismatch",
                    tasks=[
                        _make_task.__func__(
                            "producer_type_mismatch",
                            outputs=[
                                _make_output.__func__(type_="string"),
                            ],
                        ),  # type: ignore[attr-defined]
                        _make_task.__func__(
                            "consumer_type_mismatch",
                            depends_on=[
                                Dependency(
                                    taskId="producer_type_mismatch",
                                    inputs=[
                                        _make_input.__func__(type_="integer"),
                                    ],
                                )
                            ],
                        ),  # type: ignore[attr-defined]
                    ],
                ),
                False,
                id="invalid_mismatched_types",
            ),
        ],
    )
    def test_invalid_input_outputs(
        self, name: str, task_plan: TaskPlan, expected_valid: bool
    ) -> None:
        """
        For each dependency, the declared inputs must be compatible with the
        outputs of the referenced task:

        - The number of inputs must equal the number of outputs.
        - Each input.type must match the corresponding output.type.

        If any dependency violates these rules, validate(...) should return False.
        """
        result = self.validator.validate(task_plan)
        assert result is expected_valid

    # -------------------------------------------------------------------------
    # Undefined dependency tests
    # -------------------------------------------------------------------------

    @pytest.mark.parametrize(
        "name, task_plan, expected_valid",
        [
            pytest.param(
                "no_dependencies_is_valid",
                TaskPlan(
                    objective="no-deps",
                    tasks=[
                        _make_task.__func__("t1"),  # type: ignore[attr-defined]
                    ],
                ),
                True,
                id="no_dependencies_is_valid",
            ),
            pytest.param(
                "all_dependencies_defined",
                TaskPlan(
                    objective="all-defined",
                    tasks=[
                        _make_task.__func__(
                            "producer",
                            outputs=[_make_output.__func__()],
                        ),  # type: ignore[attr-defined]
                        _make_task.__func__(
                            "consumer_valid",
                            depends_on=[
                                Dependency(
                                    taskId="producer",
                                    inputs=[_make_input.__func__()],
                                )
                            ],
                        ),  # type: ignore[attr-defined]
                    ],
                ),
                True,
                id="all_dependencies_defined",
            ),
            pytest.param(
                "undefined_dependency_task_id",
                TaskPlan(
                    objective="undefined-dep",
                    tasks=[
                        _make_task.__func__(
                            "producer",
                            outputs=[_make_output.__func__()],
                        ),  # type: ignore[attr-defined]
                        _make_task.__func__(
                            "consumer_invalid",
                            depends_on=[
                                Dependency(
                                    taskId="missing_task",
                                    inputs=[_make_input.__func__()],
                                )
                            ],
                        ),  # type: ignore[attr-defined]
                    ],
                ),
                False,
                id="undefined_dependency_task_id",
            ),
            pytest.param(
                "multiple_dependencies_one_undefined",
                TaskPlan(
                    objective="mixed-deps",
                    tasks=[
                        _make_task.__func__(
                            "producer",
                            outputs=[_make_output.__func__()],
                        ),  # type: ignore[attr-defined]
                        _make_task.__func__(
                            "another",
                            outputs=[_make_output.__func__()],
                        ),  # type: ignore[attr-defined]
                        _make_task.__func__(
                            "consumer_mixed",
                            depends_on=[
                                Dependency(
                                    taskId="producer",
                                    inputs=[_make_input.__func__()],
                                ),
                                Dependency(
                                    taskId="non_existent",
                                    inputs=[_make_input.__func__()],
                                ),
                            ],
                        ),  # type: ignore[attr-defined]
                    ],
                ),
                False,
                id="multiple_dependencies_one_undefined",
            ),
        ],
    )
    def test_undefined_dependency(
        self, name: str, task_plan: TaskPlan, expected_valid: bool
    ) -> None:
        """
        Every Dependency.taskId must reference an existing Task.id in the same
        TaskPlan. If any dependency refers to a non-existent task, validate(...)
        should return False.
        """
        result = self.validator.validate(task_plan)
        assert result is expected_valid
