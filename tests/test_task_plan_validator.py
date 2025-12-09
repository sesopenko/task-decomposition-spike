import pytest

from src.task_decomposition.models import TaskPlan, Task, Dependency, Input, Output
from src.task_decomposition.task_plan_validator import TaskPlanValidator


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

    def test_cyclic_dependencies(self) -> None:
        """
        TaskPlanValidator should return False when the dependency graph
        contains any cycle, and True when it is acyclic.
        """
        scenarios = [
            {
                "name": "no_tasks_is_trivially_acyclic",
                "task_plan": TaskPlan(objective="empty", tasks=[]),
                "expected_valid": True,
            },
            {
                "name": "single_task_no_dependencies",
                "task_plan": TaskPlan(
                    objective="single",
                    tasks=[self._make_task("t1")],
                ),
                "expected_valid": True,
            },
            {
                "name": "two_tasks_linear_dependency",
                "task_plan": TaskPlan(
                    objective="linear",
                    tasks=[
                        self._make_task("t1"),
                        self._make_task(
                            "t2",
                            depends_on=[
                                Dependency(taskId="t1", inputs=[]),
                            ],
                        ),
                    ],
                ),
                "expected_valid": True,
            },
            {
                "name": "simple_two_node_cycle",
                "task_plan": TaskPlan(
                    objective="cycle",
                    tasks=[
                        self._make_task(
                            "t1",
                            depends_on=[Dependency(taskId="t2", inputs=[])],
                        ),
                        self._make_task(
                            "t2",
                            depends_on=[Dependency(taskId="t1", inputs=[])],
                        ),
                    ],
                ),
                "expected_valid": False,
            },
            {
                "name": "three_node_cycle",
                "task_plan": TaskPlan(
                    objective="3-cycle",
                    tasks=[
                        self._make_task(
                            "t1",
                            depends_on=[Dependency(taskId="t2", inputs=[])],
                        ),
                        self._make_task(
                            "t2",
                            depends_on=[Dependency(taskId="t3", inputs=[])],
                        ),
                        self._make_task(
                            "t3",
                            depends_on=[Dependency(taskId="t1", inputs=[])],
                        ),
                    ],
                ),
                "expected_valid": False,
            },
            {
                "name": "diamond_shape_no_cycle",
                "task_plan": TaskPlan(
                    objective="diamond",
                    tasks=[
                        self._make_task("root"),
                        self._make_task(
                            "left",
                            depends_on=[Dependency(taskId="root", inputs=[])],
                        ),
                        self._make_task(
                            "right",
                            depends_on=[Dependency(taskId="root", inputs=[])],
                        ),
                        self._make_task(
                            "leaf",
                            depends_on=[
                                Dependency(taskId="left", inputs=[]),
                                Dependency(taskId="right", inputs=[]),
                            ],
                        ),
                    ],
                ),
                "expected_valid": True,
            },
        ]

        for scenario in scenarios:
            with self.subtests(msg=scenario["name"]):
                result = self.validator.validate(scenario["task_plan"])
                assert result is scenario["expected_valid"]

    # -------------------------------------------------------------------------
    # Input/output compatibility tests
    # -------------------------------------------------------------------------

    def test_invalid_input_outputs(self) -> None:
        """
        For each dependency, the declared inputs must be compatible with the
        outputs of the referenced task:

        - The number of inputs must equal the number of outputs.
        - Each input.type must match the corresponding output.type.

        If any dependency violates these rules, validate(...) should return False.
        """

        scenarios = [
            {
                "name": "valid_matching_counts_and_types",
                "task_plan": TaskPlan(
                    objective="valid-io",
                    tasks=[
                        self._make_task(
                            "producer_valid",
                            outputs=[
                                self._make_output(type_="string"),
                                self._make_output(type_="integer"),
                            ],
                        ),
                        self._make_task(
                            "consumer_valid",
                            depends_on=[
                                Dependency(
                                    taskId="producer_valid",
                                    inputs=[
                                        self._make_input(type_="string"),
                                        self._make_input(type_="integer"),
                                    ],
                                )
                            ],
                        ),
                    ],
                ),
                "expected_valid": True,
            },
            {
                "name": "invalid_fewer_inputs_than_outputs",
                "task_plan": TaskPlan(
                    objective="fewer-inputs",
                    tasks=[
                        self._make_task(
                            "producer_more_outputs",
                            outputs=[
                                self._make_output(type_="string"),
                                self._make_output(type_="integer"),
                            ],
                        ),
                        self._make_task(
                            "consumer_fewer_inputs",
                            depends_on=[
                                Dependency(
                                    taskId="producer_more_outputs",
                                    inputs=[
                                        self._make_input(type_="string"),
                                    ],
                                )
                            ],
                        ),
                    ],
                ),
                "expected_valid": False,
            },
            {
                "name": "invalid_more_inputs_than_outputs",
                "task_plan": TaskPlan(
                    objective="more-inputs",
                    tasks=[
                        self._make_task(
                            "producer_fewer_outputs",
                            outputs=[
                                self._make_output(type_="string"),
                            ],
                        ),
                        self._make_task(
                            "consumer_more_inputs",
                            depends_on=[
                                Dependency(
                                    taskId="producer_fewer_outputs",
                                    inputs=[
                                        self._make_input(type_="string"),
                                        self._make_input(type_="integer"),
                                    ],
                                )
                            ],
                        ),
                    ],
                ),
                "expected_valid": False,
            },
            {
                "name": "invalid_mismatched_types",
                "task_plan": TaskPlan(
                    objective="type-mismatch",
                    tasks=[
                        self._make_task(
                            "producer_type_mismatch",
                            outputs=[
                                self._make_output(type_="string"),
                            ],
                        ),
                        self._make_task(
                            "consumer_type_mismatch",
                            depends_on=[
                                Dependency(
                                    taskId="producer_type_mismatch",
                                    inputs=[
                                        self._make_input(type_="integer"),
                                    ],
                                )
                            ],
                        ),
                    ],
                ),
                "expected_valid": False,
            },
        ]

        for scenario in scenarios:
            with self.subtests(msg=scenario["name"]):
                result = self.validator.validate(scenario["task_plan"])
                assert result is scenario["expected_valid"]

    # -------------------------------------------------------------------------
    # Undefined dependency tests
    # -------------------------------------------------------------------------

    def test_undefined_dependency(self) -> None:
        """
        Every Dependency.taskId must reference an existing Task.id in the same
        TaskPlan. If any dependency refers to a non-existent task, validate(...)
        should return False.
        """
        producer = self._make_task("producer", outputs=[self._make_output()])

        consumer_with_valid_dep = self._make_task(
            "consumer_valid",
            depends_on=[
                Dependency(
                    taskId="producer",
                    inputs=[self._make_input()],
                )
            ],
        )

        consumer_with_undefined_dep = self._make_task(
            "consumer_invalid",
            depends_on=[
                Dependency(
                    taskId="missing_task",
                    inputs=[self._make_input()],
                )
            ],
        )

        scenarios = [
            {
                "name": "no_dependencies_is_valid",
                "task_plan": TaskPlan(
                    objective="no-deps",
                    tasks=[self._make_task("t1")],
                ),
                "expected_valid": True,
            },
            {
                "name": "all_dependencies_defined",
                "task_plan": TaskPlan(
                    objective="all-defined",
                    tasks=[producer, consumer_with_valid_dep],
                ),
                "expected_valid": True,
            },
            {
                "name": "undefined_dependency_task_id",
                "task_plan": TaskPlan(
                    objective="undefined-dep",
                    tasks=[producer, consumer_with_undefined_dep],
                ),
                "expected_valid": False,
            },
            {
                "name": "multiple_dependencies_one_undefined",
                "task_plan": TaskPlan(
                    objective="mixed-deps",
                    tasks=[
                        producer,
                        self._make_task("another", outputs=[self._make_output()]),
                        self._make_task(
                            "consumer_mixed",
                            depends_on=[
                                Dependency(
                                    taskId="producer",
                                    inputs=[self._make_input()],
                                ),
                                Dependency(
                                    taskId="non_existent",
                                    inputs=[self._make_input()],
                                ),
                            ],
                        ),
                    ],
                ),
                "expected_valid": False,
            },
        ]

        for scenario in scenarios:
            with self.subtests(msg=scenario["name"]):
                result = self.validator.validate(scenario["task_plan"])
                assert result is scenario["expected_valid"]
